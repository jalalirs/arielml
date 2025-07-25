"""
Baseline Pipeline - Simple and interpretable transit depth estimation.

This pipeline implements a simplified approach based on the baseline methodology:
1. Preprocess calibrated data (time binning + spatial aggregation)
2. Detect transit phases using gradient analysis
3. Optimize constant transit depth to minimize polynomial fit error
4. Apply same prediction to all wavelengths with scaling
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable, Union
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.stats import sigmaclip

# Import our existing infrastructure
from ..data.observation import DataObservation
from ..utils.observable import Observable
from ..utils.signals import DetrendingProgress, DetrendingStep
from ..backend import get_backend

# Base class for pipelines
class BasePipeline:
    """Base class for analysis pipelines."""
    
    def __init__(self, **kwargs):
        self.pipeline_params = kwargs
        self.is_fitted = False
    
    def get_params(self) -> Dict[str, Any]:
        """Get pipeline parameters."""
        return self.pipeline_params.copy()
    
    def set_params(self, **params) -> 'BasePipeline':
        """Set pipeline parameters."""
        self.pipeline_params.update(params)
        return self
    
    def predict_with_uncertainties(self, time: np.ndarray, flux: np.ndarray, transit_mask: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainties (required for Ariel assignment).
        
        Returns:
            predictions: Predicted values (e.g., transit depths)
            uncertainties: Standard deviations for predictions
            covariance_matrix: Full covariance matrix for uncertainties
        """
        raise NotImplementedError("Subclasses must implement predict_with_uncertainties")


class BaselinePipeline(BasePipeline, Observable):
    """
    Baseline pipeline implementing simple transit depth estimation.
    
    This pipeline uses a simplified approach:
    1. Preprocess calibrated data with time binning and spatial aggregation
    2. Detect transit phases using gradient analysis
    3. Optimize constant transit depth to minimize polynomial fit error
    4. Apply same prediction to all wavelengths with scaling
    """
    
    def __init__(self, 
                 instrument: str = "AIRS-CH0",
                 use_gpu: bool = False,
                 scale: float = 0.95,
                 sigma: float = 0.0009,
                 cut_inf: int = 39,
                 cut_sup: int = 250,
                 binning: int = 30,
                 phase_detection_slice: Tuple[int, int] = (30, 140),
                 optimization_delta: int = 7,
                 polynomial_degree: int = 3,
                 **kwargs):
        super().__init__(**kwargs)
        Observable.__init__(self)
        
        # Set up backend (xp arrays)
        self.xp, self.backend_name = get_backend('gpu' if use_gpu else 'cpu')
        self.use_gpu = use_gpu
        
        self.instrument = instrument
        self.scale = scale
        self.sigma = sigma
        self.cut_inf = cut_inf
        self.cut_sup = cut_sup
        self.binning = binning
        self.phase_detection_slice = slice(phase_detection_slice[0], phase_detection_slice[1])
        self.optimization_delta = optimization_delta
        self.polynomial_degree = polynomial_degree
        self.progress_callback = None
        
        # Store fitted components for analysis
        self.fitted_components = {}
        self.transit_phases = None
        self.optimized_transit_depth = None
        
    def set_progress_callback(self, callback: Callable[[str], None]):
        """Set a callback function for progress updates."""
        self.progress_callback = callback
    
    def _emit_progress(self, message: str):
        """Emit progress message if callback is set."""
        if self.progress_callback:
            self.progress_callback(message)
    
    def _preprocess_signal(self, observation: DataObservation) -> np.ndarray:
        """Preprocess calibrated signal with time binning and spatial aggregation."""
        self._emit_progress("Preprocessing signal...")
        
        # Get calibrated data
        if observation.processed_signal is None:
            raise ValueError("Observation must be calibrated before preprocessing")
        
        # Convert to backend-appropriate arrays
        signal = observation.get_data(return_type=self.backend_name)
        
        # Apply wavelength cuts for AIRS-CH0
        if self.instrument == "AIRS-CH0":
            signal = signal[:, :, self.cut_inf:self.cut_sup]
            self._emit_progress(f"Applied wavelength cuts: {self.cut_inf}-{self.cut_sup}")
        
        # Extract ROI (Region of Interest) - center spatial pixels
        if self.instrument == "AIRS-CH0":
            # Use center 12x12 spatial pixels (rows 10-22, all wavelength columns)
            signal_roi = signal[:, 10:22, :]
        else:  # FGS1
            # Use center 12x12 spatial pixels
            signal_roi = signal[:, 10:22, 10:22]
            signal_roi = signal_roi.reshape(signal_roi.shape[0], -1)
        
        # Aggregate spatial dimension (mean)
        mean_signal = self.xp.nanmean(signal_roi, axis=1)
        
        # Apply Correlated Double Sampling (CDS)
        cds_signal = mean_signal[1::2] - mean_signal[0::2]
        
        # Time binning with instrument-specific binning
        if self.instrument == "FGS1":
            # FGS1 needs more aggressive binning (30 * 12 = 360) to match AIRS-CH0 time dimension
            binning_factor = 360
        else:
            # AIRS-CH0 uses standard binning (30)
            binning_factor = 30
            
        n_bins = cds_signal.shape[0] // binning_factor
        binned = self.xp.array([
            cds_signal[j*binning_factor : (j+1)*binning_factor].mean(axis=0) 
            for j in range(n_bins)
        ])
        
        # Ensure consistent shape: (time_bins, wavelengths)
        if len(binned.shape) == 1:
            # If 1D, reshape to (time_bins, 1)
            binned = binned.reshape((binned.shape[0], 1))
        elif len(binned.shape) > 2:
            # If more than 2D, flatten wavelength dimensions
            binned = binned.reshape((binned.shape[0], -1))
        
        self._emit_progress(f"Preprocessing complete: {binned.shape} (instrument: {self.instrument}, binning_factor: {binning_factor})")
        return binned
    
    def _phase_detector(self, signal: np.ndarray) -> Tuple[int, int]:
        """Detect transit phases using gradient analysis."""
        self._emit_progress("Detecting transit phases...")
        
        # Use the specified slice for phase detection
        search_slice = self.phase_detection_slice
        signal_slice = signal[search_slice]
        
        # Find minimum point (transit center)
        min_index = self.xp.argmin(signal_slice) + search_slice.start
        
        # Split signal into before and after transit
        signal1 = signal[:min_index]
        signal2 = signal[min_index:]
        
        # Calculate gradients
        grad1 = self.xp.gradient(signal1)
        grad1 /= self.xp.max(self.xp.abs(grad1)) if self.xp.max(self.xp.abs(grad1)) != 0 else 1
        
        grad2 = self.xp.gradient(signal2)
        grad2 /= self.xp.max(self.xp.abs(grad2)) if self.xp.max(self.xp.abs(grad2)) != 0 else 1
        
        # Find phase boundaries
        phase1 = self.xp.argmin(grad1)  # Start of transit (steepest negative gradient)
        phase2 = self.xp.argmax(grad2) + min_index  # End of transit (steepest positive gradient)
        
        self._emit_progress(f"Detected phases: {phase1} -> {phase2}")
        return phase1, phase2
    
    def _objective_function(self, s: float, signal: np.ndarray, phase1: int, phase2: int) -> float:
        """Objective function for transit depth optimization."""
        import numpy as np
        
        delta = self.optimization_delta
        power = self.polynomial_degree
        
        # Ensure we have enough data points
        if phase1 - delta <= 0 or phase2 + delta >= len(signal) or phase2 - delta - (phase1 + delta) < 5:
            delta = 2
        
        # Create modified signal with transit depth applied
        y = np.concatenate([
            signal[:phase1 - delta],
            signal[phase1 + delta:phase2 - delta] * (1 + s),
            signal[phase2 + delta:]
        ])
        x = np.arange(len(y))
        
        # Fit polynomial and calculate error
        coeffs = np.polyfit(x, y, deg=power)
        poly = np.poly1d(coeffs)
        error = np.mean(np.abs(poly(x) - y))
        
        return error
    
    def fit(self, observation: Union[DataObservation, Dict[str, DataObservation]], **kwargs) -> 'BaselinePipeline':
        """Fit the baseline pipeline."""
        if isinstance(observation, dict):
            # Both instruments loaded - process both
            self._emit_progress(f"Fitting Baseline Pipeline for both instruments...")
            
            # Get both observations
            airs_obs = observation["AIRS-CH0"]
            fgs_obs = observation["FGS1"]
            
            # Preprocess both signals with correct instrument settings
            original_instrument = self.instrument
            
            # Process AIRS-CH0
            self.instrument = "AIRS-CH0"
            airs_signal = self._preprocess_signal(airs_obs)
            
            # Process FGS1
            self.instrument = "FGS1"
            fgs_signal = self._preprocess_signal(fgs_obs)
            
            # Restore original instrument setting
            self.instrument = original_instrument
            
            # Concatenate signals: FGS1 (1 wavelength) + AIRS-CH0 (282 wavelengths) = 283 total
            print(f"DEBUG: FGS signal shape: {fgs_signal.shape}")
            print(f"DEBUG: AIRS signal shape: {airs_signal.shape}")
            preprocessed_signal = self.xp.concatenate([fgs_signal, airs_signal], axis=1)
            self.fitted_components['preprocessed_signal'] = preprocessed_signal
            self.fitted_components['airs_signal'] = airs_signal
            self.fitted_components['fgs_signal'] = fgs_signal
            
            # Create 1D signal for phase detection (mean across wavelengths 1-283, excluding FGS1)
            # FGS1 is the first column (index 0), AIRS-CH0 is columns 1-282
            signal_1d = preprocessed_signal[:, 1:].mean(axis=1)
            
        else:
            # Single instrument loaded
            self._emit_progress(f"Fitting Baseline Pipeline for {self.instrument}...")
            
            # Preprocess signal
            preprocessed_signal = self._preprocess_signal(observation)
            self.fitted_components['preprocessed_signal'] = preprocessed_signal
            
            # Create 1D signal for phase detection (mean across wavelengths)
            if self.instrument == "FGS1":
                # FGS1 has only 1 wavelength, use it directly
                signal_1d = preprocessed_signal.flatten()
            else:
                # AIRS-CH0 has multiple wavelengths, use mean across all
                signal_1d = preprocessed_signal.mean(axis=1)
        
        # Apply Savitzky-Golay smoothing (convert to numpy for scipy compatibility)
        signal_1d_np = self.xp.asnumpy(signal_1d) if hasattr(self.xp, 'asnumpy') else signal_1d
        signal_1d_smooth = savgol_filter(signal_1d_np, 20, 2)
        # Convert back to xp array
        signal_1d = self.xp.array(signal_1d_smooth)
        self.fitted_components['smoothed_signal'] = signal_1d
        
        # Detect transit phases
        phase1, phase2 = self._phase_detector(signal_1d)
        self.transit_phases = (phase1, phase2)
        
        # Ensure phases are within bounds
        phase1 = self.xp.maximum(self.optimization_delta, phase1)
        phase2 = self.xp.minimum(len(signal_1d) - self.optimization_delta - 1, phase2)
        
        self._emit_progress("Optimizing transit depth...")
        
        # Optimize transit depth (convert to numpy for scipy compatibility)
        signal_1d_np = self.xp.asnumpy(signal_1d) if hasattr(self.xp, 'asnumpy') else signal_1d
        phase1_np = int(self.xp.asnumpy(phase1)) if hasattr(self.xp, 'asnumpy') else int(phase1)
        phase2_np = int(self.xp.asnumpy(phase2)) if hasattr(self.xp, 'asnumpy') else int(phase2)
        
        print(f"DEBUG: Starting optimization with signal shape: {signal_1d_np.shape}, phases: {phase1_np} -> {phase2_np}")
        result = minimize(
            fun=self._objective_function,
            x0=[0.0001],
            args=(signal_1d_np, phase1_np, phase2_np),
            method="Nelder-Mead"
        )
        print(f"DEBUG: Optimization completed, result: {result.x[0]}")
        
        self.optimized_transit_depth = result.x[0] * self.scale
        self.fitted_components['optimized_transit_depth'] = self.optimized_transit_depth
        self.fitted_components['optimization_result'] = result
        
        self.is_fitted = True
        self._emit_progress(f"Baseline Pipeline fitting completed! Transit depth: {self.optimized_transit_depth:.6f}")
        
        return self
    
    def predict(self, observation: Union[DataObservation, Dict[str, DataObservation]], **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions using the fitted baseline pipeline.
        
        Returns:
            predictions: Transit depths per wavelength (same value for all wavelengths)
            uncertainties: Standard deviations per wavelength (constant sigma)
            covariance_matrix: Diagonal covariance matrix
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions.")
        
        # Get number of wavelengths based on observation type
        if isinstance(observation, dict):
            # Both instruments loaded - 283 wavelengths total (1 FGS1 + 282 AIRS-CH0)
            n_wavelengths = 283
        else:
            # Single instrument loaded
            if observation.has_wavelengths():
                n_wavelengths = len(observation.get_wavelengths())
            else:
                # Fallback: use the shape of processed signal
                signal = observation.get_data(return_type=self.backend_name)
                if self.instrument == "AIRS-CH0":
                    n_wavelengths = signal.shape[2] - (self.cut_sup - self.cut_inf)
                else:
                    n_wavelengths = 1
        
        # Create predictions (same value for all wavelengths)
        predictions = self.xp.full(n_wavelengths, self.optimized_transit_depth)
        
        # Create uncertainties (constant sigma for all wavelengths)
        uncertainties = self.xp.full(n_wavelengths, self.sigma)
        
        # Create diagonal covariance matrix
        covariance_matrix = self.xp.eye(n_wavelengths) * (self.sigma ** 2)
        
        # Convert to numpy for return (for compatibility with existing code)
        if hasattr(self.xp, 'asnumpy'):
            predictions = self.xp.asnumpy(predictions)
            uncertainties = self.xp.asnumpy(uncertainties)
            covariance_matrix = self.xp.asnumpy(covariance_matrix)
        
        self._emit_progress(f"Baseline predictions generated: {n_wavelengths} wavelengths")
        
        return predictions, uncertainties, covariance_matrix
    
    def predict_with_uncertainties(self, time: np.ndarray, flux: np.ndarray, transit_mask: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Alias for predict() to match the required interface."""
        # This method requires an observation object, so we'll need to create one
        # For now, we'll use the fitted components directly
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions.")
        
        # Use the fitted components to generate predictions
        n_wavelengths = self.fitted_components['preprocessed_signal'].shape[1] - 1  # Exclude FGS1 column
        
        predictions = self.xp.full(n_wavelengths, self.optimized_transit_depth)
        uncertainties = self.xp.full(n_wavelengths, self.sigma)
        covariance_matrix = self.xp.eye(n_wavelengths) * (self.sigma ** 2)
        
        return predictions, uncertainties, covariance_matrix
    
    def get_transit_depths(self) -> np.ndarray:
        """Get the fitted transit depths."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first.")
        return self.xp.full(283, self.optimized_transit_depth)  # 283 wavelengths
    
    def get_uncertainties(self) -> np.ndarray:
        """Get the fitted uncertainties."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first.")
        return self.xp.full(283, self.sigma)  # 283 wavelengths
    
    def get_component(self, component_name: str) -> Any:
        """Get a specific fitted component."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first.")
        return self.fitted_components.get(component_name)
    
    def get_backend_info(self) -> str:
        """Get information about the pipeline configuration."""
        backend_info = f"BaselinePipeline using {self.instrument} with scale={self.scale}, sigma={self.sigma}"
        if self.use_gpu:
            backend_info += f" | GPU Backend: {self.backend_name}"
        else:
            backend_info += f" | CPU Backend: {self.backend_name}"
        return backend_info 