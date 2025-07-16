"""
Bayesian Pipeline - Complete re-implementation of ariel_gp approach.

This pipeline implements the full Bayesian framework: (star_spectrum * drift * transit) + noise
and returns transit depths directly, just like the arielgp code.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable
from scipy.optimize import minimize
from scipy import sparse
import pickle
from pathlib import Path

# Import drift components from detrending
from ..data.detrending import AIRSDriftDetrender, FGSDriftDetrender
from ..data.detrending import KISSGP2D
from ..utils.observable import Observable
from ..utils.signals import DetrendingProgress, DetrendingStep

# Import model components from models
from ..models import StellarSpectrumModel, TransitDepthModel, TransitWindowModel, NoiseModel, MCMCSampler

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

# Import arielgp hyperparameters
try:
    from ...arielgp import ariel_support as ars
    ARIELGP_AVAILABLE = True
except ImportError:
    ARIELGP_AVAILABLE = False
    print("Warning: arielgp not available. Using default hyperparameters.")


class BayesianPipeline(BasePipeline):
    """
    Complete Bayesian pipeline implementing the ariel_gp approach.
    
    This pipeline does both detrending and transit modeling in one integrated framework:
    (star_spectrum * drift * transit) + noise
    
    Returns transit depths directly, just like the arielgp code.
    """
    
    def __init__(self, 
                 instrument: str = "AIRS-CH0",
                 n_pca: int = 1,
                 n_iter: int = 7,
                 n_samples: int = 100,
                 use_gpu: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.instrument = instrument
        self.n_pca = n_pca
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.use_gpu = use_gpu
        self.progress_callback = None
        
        # Determine backend
        backend = 'gpu' if use_gpu else 'cpu'
        
        # Initialize component models with backend support
        self.stellar_model = StellarSpectrumModel(backend=backend)
        self.transit_depth_model = TransitDepthModel(n_pca=n_pca, backend=backend)
        self.transit_window_model = TransitWindowModel()
        self.noise_model = NoiseModel(backend=backend)
        
        # Initialize MCMC sampler with backend support
        self.mcmc_sampler = MCMCSampler(backend=backend)
        
        # Initialize drift model based on instrument
        drift_batch_size = kwargs.get('drift_batch_size', 8)
        
        # Reduce batch sizes for FGS due to higher wavelength count
        if instrument == "FGS1":
            drift_batch_size = min(drift_batch_size, 2)  # FGS needs smaller batches
            if 'batch_size' in kwargs:
                kwargs['batch_size'] = min(kwargs['batch_size'], 10)  # Reduce MCMC batch size for FGS
        
        if instrument == "AIRS-CH0":
            self.drift_model = AIRSDriftDetrender(use_gpu=use_gpu, batch_size=drift_batch_size)
        else:  # FGS
            self.drift_model = FGSDriftDetrender(use_gpu=use_gpu, batch_size=drift_batch_size)
        
        # Store fitted components
        self.fitted_components = {}
        
        # Store MCMC samples for analysis
        self.mcmc_samples = None
        
        # Store backend info for reporting
        self.backend_info = {
            'stellar': self.stellar_model.get_backend_info(),
            'transit_depth': self.transit_depth_model.get_backend_info(),
            'transit_window': 'TransitWindowModel using cpu backend',  # No backend support yet
            'noise': self.noise_model.get_backend_info(),
            'mcmc': self.mcmc_sampler.get_backend_info(),
            'drift': f"Drift model using {'gpu' if use_gpu else 'cpu'} backend"
        }
    
    def set_progress_callback(self, callback: Callable[[str], None]):
        """Set a callback function for progress updates."""
        self.progress_callback = callback
        # Also set the callback for the MCMC sampler
        self.mcmc_sampler.set_progress_callback(callback)
        # Connect drift model progress signals
        if hasattr(self.drift_model, 'add_observer'):
            self.drift_model.add_observer(self._on_drift_progress)
    
    def _on_drift_progress(self, progress):
        """Handle progress updates from the drift model."""
        if hasattr(progress, 'message'):
            self._emit_progress(progress.message)
        else:
            self._emit_progress("Fitting drift model...")
    
    def _emit_progress(self, message: str):
        """Emit progress message if callback is set."""
        if self.progress_callback:
            self.progress_callback(message)
    
    def fit(self, time: np.ndarray, flux: np.ndarray, transit_mask: np.ndarray, **kwargs) -> 'BayesianPipeline':
        """Fit the complete Bayesian pipeline."""
        self._emit_progress(f"Fitting Bayesian Pipeline for {self.instrument}...")
        self._emit_progress(f"Backend configuration:")
        for component, info in self.backend_info.items():
            self._emit_progress(f"  {component}: {info}")
        
        # Step 1: Fit stellar spectrum
        self._emit_progress("Fitting stellar spectrum...")
        stellar_spectrum = self.stellar_model.fit(flux)
        self.fitted_components['stellar'] = stellar_spectrum
        self._emit_progress("✓ Stellar spectrum fitted")
        
        # Step 2: Fit drift model
        self._emit_progress("Fitting drift model...")
        drift_model, drift_noise = self.drift_model.detrend(time, flux, transit_mask, np)
        self.fitted_components['drift'] = drift_model
        self._emit_progress("✓ Drift model fitted")
        
        # Clean up drift model observer
        if hasattr(self.drift_model, 'remove_observer'):
            self.drift_model.remove_observer(self._on_drift_progress)
        
        # Step 3: Fit transit depth variation
        self._emit_progress("Fitting transit depth variation...")
        transit_depths = self.transit_depth_model.fit(flux, transit_mask)
        self.fitted_components['transit_depth'] = transit_depths
        self._emit_progress("✓ Transit depth variation fitted")
        
        # Step 4: Fit transit window
        self._emit_progress("Fitting transit window...")
        transit_window = self.transit_window_model.fit(time, transit_mask)
        self.fitted_components['transit_window'] = transit_window
        self._emit_progress("✓ Transit window fitted")
        
        # Step 5: Fit noise model
        self._emit_progress("Fitting noise model...")
        points_per_bin = kwargs.get('points_per_bin', None)
        noise_levels = self.noise_model.fit(flux, points_per_bin)
        self.fitted_components['noise'] = noise_levels
        self._emit_progress("✓ Noise model fitted")
        
        self.is_fitted = True
        self._emit_progress("Bayesian Pipeline fitting completed!")
        
        return self
    
    def predict(self, time: np.ndarray, flux: np.ndarray, transit_mask: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions and return transit depths with uncertainties using MCMC sampling.
        
        Returns:
            predictions: Transit depths per wavelength (main output)
            uncertainties: Standard deviations per wavelength
            covariance_matrix: Full covariance matrix for uncertainties
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions.")
        
        # Get fitted components
        stellar = self.fitted_components['stellar']
        drift = self.fitted_components['drift']
        transit_depth = self.fitted_components['transit_depth']
        transit_window = self.fitted_components['transit_window']
        noise = self.fitted_components['noise']
        
        # Use GPU-accelerated MCMC sampling with progress tracking
        self._emit_progress(f"Generating {self.n_samples} MCMC samples for uncertainty estimation...")
        
        # Set batch size for memory management
        batch_size = kwargs.get('batch_size', 20)
        
        predictions, uncertainties, covariance_matrix = self.mcmc_sampler.sample_transit_depths(
            stellar=stellar,
            drift=drift,
            transit_depth=transit_depth,
            transit_window=transit_window,
            noise=noise,
            n_samples=self.n_samples,
            batch_size=batch_size
        )
        
        # Store samples for analysis
        self.mcmc_samples = predictions
        
        self._emit_progress(f"MCMC uncertainty estimation completed!")
        self._emit_progress(f"  Predictions shape: {predictions.shape}")
        self._emit_progress(f"  Uncertainties shape: {uncertainties.shape}")
        self._emit_progress(f"  Covariance matrix shape: {covariance_matrix.shape}")
        
        # Return predictions, uncertainties, and covariance matrix
        return predictions, uncertainties, covariance_matrix
    
    def predict_with_uncertainties(self, time: np.ndarray, flux: np.ndarray, transit_mask: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Alias for predict() to match the required interface."""
        return self.predict(time, flux, transit_mask, **kwargs)
    
    def get_transit_depths(self) -> np.ndarray:
        """Get the fitted transit depths."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first.")
        return self.fitted_components['transit_depth']
    
    def get_uncertainties(self) -> np.ndarray:
        """Get the fitted uncertainties."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first.")
        return self.fitted_components.get('noise', np.zeros_like(self.fitted_components['transit_depth']))
    
    def get_component(self, component_name: str) -> np.ndarray:
        """Get a specific fitted component."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first.")
        return self.fitted_components.get(component_name)
    
    def get_mcmc_samples(self) -> Optional[np.ndarray]:
        """Get the MCMC samples used for uncertainty estimation."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first.")
        return self.mcmc_samples if self.mcmc_samples is not None else None
    
    def get_backend_info(self) -> Dict[str, str]:
        """Get information about which backend each component is using."""
        return self.backend_info.copy()


# Convenience function
def create_bayesian_pipeline(instrument: str = "AIRS-CH0", use_gpu: bool = True, **kwargs) -> BayesianPipeline:
    """Create a Bayesian pipeline with default settings."""
    return BayesianPipeline(instrument=instrument, use_gpu=use_gpu, **kwargs) 