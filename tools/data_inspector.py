import sys
from pathlib import Path

# Add the project root to the Python path to allow importing 'arielml'
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QPushButton, QSlider, QLabel, QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox, QStatusBar,
    QTextEdit, QTabWidget, QStackedWidget, QProgressBar, QFormLayout, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# Import all necessary components from our library
from arielml.data.observation import DataObservation
from arielml.data import loaders, detrending
from arielml.pipelines.bayesian_pipeline import BayesianPipeline
from arielml.pipelines.baseline_pipeline import BaselinePipeline
from arielml.config import DATASET_DIR, PHOTOMETRY_APERTURES, WAVELENGTH_RANGES
from arielml.backend import GPU_ENABLED, GP_GPU_ENABLED
from arielml.utils.signals import DetrendingProgress

def force_gpu_cleanup():
    """Force cleanup of GPU memory to prevent OOM errors."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("DEBUG: Forced GPU cleanup")
    except ImportError:
        pass

# Check if sklearn GP is available
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    SKLEARN_GP_ENABLED = True
except ImportError:
    SKLEARN_GP_ENABLED = False


class BayesianPlotter:
    """Handles plotting of Bayesian analysis results with support for single and multiple points."""
    
    def __init__(self, ax, canvas):
        self.ax = ax
        self.canvas = canvas
        
    def plot_results(self, bayesian_results, observation, show_uncertainties=True, 
                    show_ground_truth=True, log_scale=False):
        """Plot Bayesian results with appropriate visualization for data type."""
        if not bayesian_results:
            self._plot_no_results(observation)
            return
            
        # Clear the plot
        self.ax.clear()
        
        # Get results
        predictions = bayesian_results['predictions']
        uncertainties = bayesian_results['uncertainties']
        
        # Determine if we have single or multiple points
        if len(predictions) == 1:
            self._plot_single_point(predictions, uncertainties, bayesian_results, 
                                  observation, show_uncertainties, show_ground_truth, log_scale)
        else:
            self._plot_multiple_points(predictions, uncertainties, bayesian_results, 
                                     observation, show_uncertainties, show_ground_truth, log_scale)
        
        # Add backend info
        self._add_backend_info(bayesian_results)
        
        # Final formatting
        self._format_plot(observation, log_scale)
        self.canvas.draw()
    
    def _plot_no_results(self, observation):
        """Handle case when no Bayesian results are available."""
        if observation and observation.has_ground_truth():
            self._show_ground_truth_preview(observation)
        else:
            self.ax.clear()
            self.ax.set_title("No Bayesian Results Available")
            self.ax.set_xlabel("Wavelength (μm)" if observation and observation.has_wavelengths() else "Wavelength Index")
            self.ax.set_ylabel("Transit Depth")
            self.canvas.draw()
    
    def _plot_single_point(self, predictions, uncertainties, bayesian_results, 
                          observation, show_uncertainties, show_ground_truth, log_scale):
        """Plot single point with appropriate visualization."""
        # Create a simple x-axis for single point
        x_pos = 0.5
        
        # Plot prediction as a point
        self.ax.scatter([x_pos], predictions, s=100, c='blue', alpha=0.8, 
                       label='Prediction', zorder=3)
        
        # Plot uncertainty as error bar
        if show_uncertainties:
            self.ax.errorbar([x_pos], predictions, yerr=[uncertainties], 
                           fmt='none', capsize=10, capthick=2, 
                           color='blue', alpha=0.6, label='±1σ Uncertainty')
        
        # Plot ground truth if available
        if observation and observation.has_ground_truth() and show_ground_truth:
            ground_truth = observation.get_ground_truth()
            if len(ground_truth) >= 1:
                gt_value = ground_truth[0] if len(ground_truth) > 0 else ground_truth
                self.ax.scatter([x_pos], [gt_value], s=100, c='red', alpha=0.8, 
                               marker='s', label='Ground Truth', zorder=3)
                
                # Calculate and display metrics
                metrics = self._calculate_metrics(predictions, [gt_value])
                self._add_metrics_text(metrics)
        
        # Set appropriate x-axis
        self.ax.set_xlim(0, 1)
        self.ax.set_xticks([x_pos])
        self.ax.set_xticklabels(['FGS1'])
        
        # Set y-axis limits with appropriate padding
        all_values = list(predictions)
        if observation and observation.has_ground_truth() and show_ground_truth:
            ground_truth = observation.get_ground_truth()
            if len(ground_truth) >= 1:
                all_values.append(ground_truth[0] if len(ground_truth) > 0 else ground_truth)
        
        if all_values:
            min_val, max_val = min(all_values), max(all_values)
            range_val = max_val - min_val
            padding = range_val * 0.2 if range_val > 0 else 0.1
            self.ax.set_ylim(min_val - padding, max_val + padding)
    
    def _plot_multiple_points(self, predictions, uncertainties, bayesian_results, 
                             observation, show_uncertainties, show_ground_truth, log_scale):
        """Plot multiple points as a line chart."""
        # Create wavelength array
        wavelengths = self._get_wavelengths(observation, len(predictions))
        
        # Plot transit depths
        self.ax.plot(wavelengths, predictions, 'b-', linewidth=2, alpha=0.8, label='Prediction')
        
        # Show uncertainties
        if show_uncertainties:
            self.ax.fill_between(wavelengths, 
                               predictions - uncertainties, 
                               predictions + uncertainties, 
                               alpha=0.3, color='blue', label='±1σ Uncertainty')
        
        # Show ground truth
        if observation and observation.has_ground_truth() and show_ground_truth:
            ground_truth = observation.get_ground_truth()
            if len(wavelengths) >= len(ground_truth):
                gt_wavelengths = wavelengths[:len(ground_truth)]
            else:
                ground_truth = ground_truth[:len(wavelengths)]
                gt_wavelengths = wavelengths
            
            self.ax.plot(gt_wavelengths, ground_truth, 'r--', linewidth=3, alpha=0.9, label='Ground Truth')
            
            # Calculate and display metrics
            metrics = self._calculate_metrics(predictions[:len(ground_truth)], ground_truth)
            self._add_metrics_text(metrics)
    
    def _get_wavelengths(self, observation, n_predictions):
        """Get appropriate wavelength array for plotting."""
        if observation and observation.has_wavelengths():
            wavelengths = observation.get_wavelengths()
            if len(wavelengths) >= n_predictions:
                return wavelengths[:n_predictions]
        return np.arange(n_predictions)
    
    def _calculate_metrics(self, predictions, ground_truth):
        """Calculate metrics between predictions and ground truth."""
        if len(predictions) != len(ground_truth):
            return {}
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Calculate metrics
        mse = np.mean((predictions - ground_truth) ** 2)
        rms_error = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - ground_truth))
        
        # Calculate correlation coefficient
        if len(predictions) > 1:
            correlation = np.corrcoef(predictions, ground_truth)[0, 1]
        else:
            correlation = 1.0 if predictions[0] == ground_truth[0] else 0.0
        
        return {
            'rms_error': rms_error,
            'mae': mae,
            'correlation': correlation
        }
    
    def _add_metrics_text(self, metrics):
        """Add metrics text to the plot."""
        if not metrics:
            return
            
        metrics_text = f"Metrics:\n"
        metrics_text += f"RMSE: {metrics['rms_error']:.4f}\n"
        metrics_text += f"MAE: {metrics['mae']:.4f}\n"
        metrics_text += f"R²: {metrics['correlation']:.4f}"
        
        self.ax.text(0.02, 0.02, metrics_text, transform=self.ax.transAxes, 
                    verticalalignment='bottom', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    def _add_backend_info(self, bayesian_results):
        """Add backend information to the plot."""
        backend_info = bayesian_results.get('backend_info', {})
        calibration_info = bayesian_results.get('calibration_info', {})
        
        info_text = "Backend Info:\n"
        for component, info in backend_info.items():
            if component not in ['sigma_fudger', 'mean_bias_fitter']:  # Skip these, we'll show them separately
                info_text += f"{component}: {info}\n"
        
        # Add calibration info
        if calibration_info:
            info_text += "\nCalibration:\n"
            if calibration_info.get('sigma_fudger_enabled', False):
                fudge_factor = calibration_info.get('sigma_fudge_factor', 'N/A')
                info_text += f"Sigma Fudger: {fudge_factor}\n"
            if calibration_info.get('mean_bias_fitter_enabled', False):
                bias_correction = calibration_info.get('mean_bias_correction', 'N/A')
                info_text += f"Bias Correction: {bias_correction}\n"
        
        # Add ground truth availability
        ground_truth_available = bayesian_results.get('ground_truth_available', False)
        info_text += f"\nGround Truth: {'Available' if ground_truth_available else 'Not Available'}"
        
        # Position the text box
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _format_plot(self, observation, log_scale):
        """Apply final formatting to the plot."""
        self.ax.set_title("Bayesian Pipeline Results")
        
        # Set appropriate labels
        if observation and observation.has_wavelengths():
            self.ax.set_xlabel("Wavelength (μm)")
        else:
            self.ax.set_xlabel("Wavelength Index")
            
        self.ax.set_ylabel("Transit Depth")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        # Apply log scale if requested
        if log_scale:
            self.ax.set_yscale('log')
    
    def _show_ground_truth_preview(self, observation):
        """Show ground truth preview when no Bayesian results are available."""
        ground_truth = observation.get_ground_truth()
        wavelengths = self._get_wavelengths(observation, len(ground_truth))
        
        self.ax.clear()
        self.ax.plot(wavelengths, ground_truth, 'r-', linewidth=2, label='Ground Truth')
        self.ax.set_title("Ground Truth Preview (No Bayesian Results)")
        self.ax.set_xlabel("Wavelength (μm)" if observation.has_wavelengths() else "Wavelength Index")
        self.ax.set_ylabel("Transit Depth")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

class CovariancePlotter:
    """Handles plotting of covariance matrices."""
    
    @staticmethod
    def plot_covariance(covariance, title="Covariance Matrix"):
        """Plot covariance matrix in a separate window."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(covariance, cmap='viridis', aspect='auto')
            ax.set_title(title)
            ax.set_xlabel('Wavelength Index')
            ax.set_ylabel('Wavelength Index')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Covariance')
            fig.tight_layout()
            fig.show()
        except Exception as e:
            print(f"Warning: Could not display covariance matrix: {e}")

class DataLoadingWorker(QThread):
    """Worker thread for loading data asynchronously."""
    
    progress_signal = pyqtSignal(str)  # Simple string messages for loading
    finished_signal = pyqtSignal(object)  # Emit the loaded observation(s)
    error_signal = pyqtSignal(str)
    
    def __init__(self, planet_id, instrument, obs_id, split, backend):
        super().__init__()
        self.planet_id = planet_id
        self.instrument = instrument
        self.obs_id = obs_id
        self.split = split
        self.backend = backend
        self._should_stop = False
    
    def run(self):
        """Load data in a separate thread."""
        observations = None
        try:
            if self.instrument == "Both":
                # Load both instruments
                self.progress_signal.emit("Loading AIRS-CH0 data...")
                self.check_stop()
                
                airs_observation = DataObservation(self.planet_id, "AIRS-CH0", int(self.obs_id), self.split)
                airs_observation.load(backend=self.backend)
                
                self.progress_signal.emit("Loading FGS1 data...")
                self.check_stop()
                
                fgs_observation = DataObservation(self.planet_id, "FGS1", int(self.obs_id), self.split)
                fgs_observation.load(backend=self.backend)
                
                observations = {"AIRS-CH0": airs_observation, "FGS1": fgs_observation}
                self.progress_signal.emit("Both instruments loaded successfully")
            else:
                # Load single instrument
                self.progress_signal.emit("Creating observation object...")
                self.check_stop()
                
                observation = DataObservation(self.planet_id, self.instrument, int(self.obs_id), self.split)
                
                self.progress_signal.emit("Loading raw data...")
                self.check_stop()
                
                observation.load(backend=self.backend)
                observations = observation
            
            # Final check before emitting success
            self.check_stop()
            
            self.progress_signal.emit("Data loading completed")
            self.finished_signal.emit(observations)
            
        except InterruptedError:
            # Operation was stopped by user - don't emit error
            pass
        except Exception as e:
            self.error_signal.emit(str(e))
    
    def check_stop(self):
        """Check if a stop has been requested."""
        if self._should_stop:
            raise InterruptedError("Data loading stopped by user")
    
    def stop(self):
        """Request the worker to stop."""
        self._should_stop = True

class BayesianPipelineWorker(QThread):
    """Worker thread for running the Bayesian pipeline asynchronously."""
    
    progress_signal = pyqtSignal(str)  # Simple string messages for pipeline steps
    finished_signal = pyqtSignal(object)  # Emit the pipeline results
    error_signal = pyqtSignal(str)
    
    def __init__(self, observation, pipeline_params):
        super().__init__()
        self.observation = observation
        self.pipeline_params = pipeline_params
        self._should_stop = False
    
    def run(self):
        """Run the Bayesian pipeline in a separate thread."""
        try:
            print("DEBUG: Starting Bayesian pipeline in worker thread")
            
            # Force GPU cleanup before starting
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("DEBUG: Forced GPU cleanup before starting Bayesian pipeline")
            except ImportError:
                pass
            
            # Create Bayesian pipeline
            self.progress_signal.emit("Creating Bayesian pipeline...")
            self.check_stop()
            
            pipeline = BayesianPipeline(**self.pipeline_params)
            
            # Set up progress callback
            pipeline.set_progress_callback(self.progress_signal.emit)
            
            # Ensure photometry has been run (needed for light curves)
            if self.observation.light_curves is None:
                self.progress_signal.emit("Running photometry for Bayesian pipeline...")
                self.check_stop()
                self.observation.run_photometry()
            
            # Get data from observation using correct methods
            time = self.observation.get_time_array()
            flux = self.observation.get_light_curves(return_type='numpy')
            transit_mask = self.observation.get_transit_mask(method='empirical')
            
            # Convert to numpy if needed
            if hasattr(time, 'get'):
                time = time.get()
            if hasattr(transit_mask, 'get'):
                transit_mask = transit_mask.get()
            
            # Fit the pipeline
            self.progress_signal.emit("Fitting Bayesian pipeline...")
            self.check_stop()
            pipeline.fit(time, flux, transit_mask)
            
            # Make predictions
            self.progress_signal.emit("Making predictions with MCMC...")
            self.check_stop()
            
            # Add batch size for memory management
            predictions, uncertainties, covariance = pipeline.predict_with_uncertainties(
                time, flux, transit_mask, batch_size=self.pipeline_params.get('batch_size', 20)
            )
            
            # Handle calibration based on mode
            calibration_mode = self.pipeline_params.get('calibration_mode', 'Training')
            calibration_file = self.pipeline_params.get('calibration_file')
            print(f"DEBUG: Calibration mode: {calibration_mode}")
            if calibration_mode == 'Disabled':
                self.progress_signal.emit("Calibration disabled - using uncalibrated predictions")
            elif calibration_mode == 'Training':
                # Training mode: train calibration models if ground truth available
                if self.observation.has_ground_truth():
                    ground_truth = self.observation.get_ground_truth()
                    self.progress_signal.emit("Training calibration models on training data...")
                    
                    # Train calibration models
                    pipeline.train_calibration_models(predictions, uncertainties, ground_truth)
                    
                    # Save calibration models
                    if calibration_file:
                        save_file = calibration_file
                    else:
                        save_file = f"calibration_models_{self.observation.planet_id}_{self.observation.instrument}.json"
                    
                    pipeline.save_calibration_models(save_file)
                    self.progress_signal.emit(f"Calibration models saved to {save_file}")
                    
                    # Apply calibration
                    predictions, uncertainties, covariance = pipeline._calibrate_models(
                        predictions, uncertainties, covariance, ground_truth
                    )
                else:
                    self.progress_signal.emit("No ground truth available for training - using uncalibrated predictions")
            elif calibration_mode == 'Inference':
                # Inference mode: load and apply pre-trained calibration models
                if calibration_file:
                    try:
                        pipeline.load_calibration_models(calibration_file)
                        self.progress_signal.emit(f"Loaded calibration models from {calibration_file}")
                    except FileNotFoundError:
                        self.progress_signal.emit(f"Calibration file {calibration_file} not found - using default values")
                        pipeline.set_default_calibration_values()
                else:
                    # Try to load auto-generated calibration file
                    auto_file = f"calibration_models_{self.observation.planet_id}_{self.observation.instrument}.json"
                    try:
                        pipeline.load_calibration_models(auto_file)
                        self.progress_signal.emit(f"Loaded calibration models from {auto_file}")
                    except FileNotFoundError:
                        self.progress_signal.emit("No calibration file found - using default values")
                        pipeline.set_default_calibration_values()
                print("DEBUG: Applying calibration...")
                # Apply calibration
                predictions, uncertainties, covariance = pipeline._calibrate_models(
                    predictions, uncertainties, covariance
                )
            
            # Store results (but don't store the pipeline object to avoid state persistence)
            results = {
                'predictions': predictions,
                'uncertainties': uncertainties,
                'covariance': covariance,
                'time': time,
                'flux': flux,
                'transit_mask': transit_mask,
                'pipeline_params': self.pipeline_params,  # Store params instead of pipeline object
                # Store fitted components for Component Analysis tab
                'fitted_components': pipeline.fitted_components.copy(),
                'mcmc_samples': pipeline.get_mcmc_samples(),
                'backend_info': pipeline.get_backend_info(),
                'calibration_info': pipeline.get_calibration_info(),
                'ground_truth_available': self.observation.has_ground_truth()
            }
            # Clean up pipeline object to prevent state persistence
            del pipeline
            
            # Force GPU cleanup after pipeline completion
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("DEBUG: Forced GPU cleanup after Bayesian pipeline completion")
            except ImportError:
                pass
            
            # Final check before emitting success
            self.check_stop()
            
            print("DEBUG: Bayesian pipeline completed successfully")
            self.finished_signal.emit(results)
            
        except InterruptedError:
            print("DEBUG: Bayesian pipeline interrupted by user")
            # Operation was stopped by user - don't emit error
            pass
        except Exception as e:
            print(f"DEBUG: Bayesian pipeline error: {e}")
            self.error_signal.emit(str(e))
        finally:
            # Ensure GPU cleanup even if there's an error
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("DEBUG: Forced GPU cleanup in Bayesian pipeline finally block")
            except ImportError:
                pass
    
    def check_stop(self):
        """Check if a stop has been requested."""
        if self._should_stop:
            raise InterruptedError("Bayesian pipeline stopped by user")
    
    def stop(self):
        """Request the worker to stop."""
        self._should_stop = True
        
        # Clean up GPU memory immediately when stopping
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("DEBUG: Cleared GPU cache when stopping Bayesian pipeline worker")
        except ImportError:
            pass

class PipelineWorker(QThread):
    """Worker thread for running the pipeline asynchronously."""
    
    progress_signal = pyqtSignal(str)  # Simple string messages for pipeline steps
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    def __init__(self, observation, detrender, mask_method="empirical", steps_to_run=None, n_bins=100):
        super().__init__()
        self.observation = observation
        self.detrender = detrender
        self.mask_method = mask_method
        self.steps_to_run = steps_to_run
        self.n_bins = n_bins
        self._should_stop = False
    
    def run(self):
        """Run the pipeline in a separate thread."""
        try:
            print("DEBUG: Starting pipeline in worker thread")
            
            # Run calibration
            print("DEBUG: Running calibration...")
            self.progress_signal.emit("Running calibration...")
            self.check_stop()
            self.observation.run_calibration_pipeline(self.steps_to_run)
            print("DEBUG: Calibration completed")
            
            # Run photometry
            print("DEBUG: Running photometry...")
            self.progress_signal.emit("Running photometry...")
            self.check_stop()
            self.observation.run_photometry()
            print("DEBUG: Photometry completed")
            
            # Run detrending if detrender is provided
            if self.detrender:
                print(f"DEBUG: Running detrending with {self.detrender.__class__.__name__}...")
                self.progress_signal.emit("Running detrending...")
                self.check_stop()
                
                # Connect the detrender's progress signals to our signal
                if hasattr(self.detrender, 'add_observer') and hasattr(self.detrender, '_observers'):
                    self.detrender.add_observer(self._on_detrending_progress)
                
                # Run the detrending
                self.observation.run_detrending(self.detrender, mask_method=self.mask_method)
                print("DEBUG: Detrending completed")
                
                # Clean up observer
                if hasattr(self.detrender, 'remove_observer') and hasattr(self.detrender, '_observers'):
                    self.detrender.remove_observer(self._on_detrending_progress)
                
                # Clean up GPU memory after detrending
                detrender_class = self.detrender.__class__.__name__
                if 'GPyTorch' in detrender_class or 'GPU' in detrender_class:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            print("DEBUG: Cleared GPU cache after detrending in worker")
                    except ImportError:
                        pass
            else:
                print("DEBUG: No detrender provided, skipping detrending")
            
            # Run phase folding
            print("DEBUG: Running phase folding...")
            self.progress_signal.emit("Running phase folding...")
            self.check_stop()
            self.observation.run_phase_folding(n_bins=self.n_bins)
            print("DEBUG: Phase folding completed")
            
            # Final check before emitting success
            self.check_stop()
            
            print("DEBUG: Pipeline completed successfully")
            # Emit finished signal
            self.finished_signal.emit()
            
        except InterruptedError:
            print("DEBUG: Pipeline interrupted by user")
            # Operation was stopped by user - don't emit error
            pass
        except Exception as e:
            print(f"DEBUG: Pipeline error: {e}")
            self.error_signal.emit(str(e))
    
    def _on_detrending_progress(self, progress):
        """Handle progress updates from the detrender."""
        if hasattr(progress, 'message'):
            self.progress_signal.emit(progress.message)
        else:
            self.progress_signal.emit("Detrending...")
    
    def check_stop(self):
        """Check if a stop has been requested."""
        if self._should_stop:
            raise InterruptedError("Operation stopped by user")
    
    def stop(self):
        """Request the worker to stop."""
        self._should_stop = True
        if hasattr(self.detrender, 'request_stop'):
            self.detrender.request_stop()
        
        # Clean up GPU memory immediately when stopping
        if hasattr(self, 'detrender') and self.detrender:
            detrender_class = self.detrender.__class__.__name__
            if 'GPyTorch' in detrender_class or 'GPU' in detrender_class:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("DEBUG: Cleared GPU cache when stopping pipeline worker")
                except ImportError:
                    pass

class DataInspector(QMainWindow):
    """
    A PyQt GUI application to inspect Ariel dataset files and visualize
    the full data reduction pipeline.
    """
    
    # Pre-coded tab-pipeline associations
    TAB_PIPELINE_MAPPING = {
        "Detector View": ["all"],  # Always visible
        "Photometry View": ["preprocessing", "cnn", "diffusion"],  # Pipelines that do photometry
        "Detrended View": ["preprocessing"],  # Only preprocessing does detrending
        "Phase-Folded View": ["preprocessing"],  # Only preprocessing does phase folding
        "Bayesian Results": ["bayesian"],  # Only Bayesian pipeline
        "Component Analysis": ["bayesian"],  # Only Bayesian has fitted components
        "Baseline Results": ["baseline"],  # Only Baseline pipeline
        "Performance Log": ["all"]  # Always visible
    }
    
    def __init__(self):
        super().__init__()
        
        # Enable keyboard focus and events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # --- Initialize Data ---
        self.dataset_dir = Path("dataset")
        self.axis_info_df = None
        self.star_info_df = None
        self.observation = None
        self.bayesian_results = None  # Store Bayesian pipeline results
        
        # Track current backend and instrument to prevent crashes
        self.current_backend = None  # Track which backend the data is loaded for
        self.current_instrument = None  # Track which instrument the data is loaded for
        self.navigation_locked = False  # Track which panel is locked
        
        # Tab indices for dynamic visibility management
        self.tab_indices = {}
        
        # Current planet tracking
        self.current_planet_id = None
        
        # Keyboard navigation settings
        self.keyboard_step_normal = 1  # Normal step size
        self.keyboard_step_fine = 1    # Fine step size (same as normal for combo boxes)
        
        self.setWindowTitle("Ariel Data Inspector")
        self.setGeometry(100, 100, 1600, 900)
        
        # Set application icon
        icon_path = Path(__file__).parent.parent / "arielml" / "assets" / "ESA_Ariel_official_mission_patch.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            print(f"Warning: Icon not found at {icon_path}")

        self.observation = None
        self.star_info_df = None
        self.worker = None
        self.loading_worker = None
        self.current_backend = None  # Track which backend the data is loaded on
        self.current_instrument = None  # Track which instrument the data is loaded for
        self.navigation_locked = False  # Track which panel is locked
        
        # Tab indices for dynamic visibility management
        self.tab_indices = {}
        
        # Current planet tracking
        self.current_planet_id = None

        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        visuals_panel = self._create_visuals_panel()
        controls_panel = self._create_controls_panel()

        main_layout.addWidget(visuals_panel, 3)
        main_layout.addWidget(controls_panel, 1)
        
        self.setStatusBar(QStatusBar())
        self.populate_planet_ids()
        
        # Set initial state: navigation unlocked, settings locked
        self._unlock_navigation_panel()
        self._lock_settings_panel()
        
        # Set initial pipeline mode state to ensure correct tab visibility
        # Since "Preprocessing Pipeline" is selected by default (index 0), 
        # we need to explicitly call the change handler to set the correct tab
        self.on_pipeline_mode_change("Preprocessing Pipeline")
        
        # Show keyboard shortcuts help in status bar
        self.statusBar().showMessage(
            "Keyboard Navigation: ←→ Planet | ↑↓ Observation | Shift+←→ Split | Shift+↑↓ Instrument | Alt+←→ Backend | Ctrl for fine control", 
            10000
        )

    # --- UI Creation Methods (Broken down for clarity) ---

    def _create_bayesian_settings_group(self):
        """Create the Bayesian pipeline settings group."""
        group = QGroupBox("Bayesian Pipeline Settings")
        layout = QVBoxLayout()
        
        # PCA Components
        pca_layout = QHBoxLayout()
        pca_layout.addWidget(QLabel("PCA Components:"))
        self.bayesian_pca_spinbox = QSpinBox()
        self.bayesian_pca_spinbox.setRange(0, 10)
        self.bayesian_pca_spinbox.setValue(1)
        pca_layout.addWidget(self.bayesian_pca_spinbox)
        layout.addLayout(pca_layout)
        
        # Iterations
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Iterations:"))
        self.bayesian_iter_spinbox = QSpinBox()
        self.bayesian_iter_spinbox.setRange(1, 50)
        self.bayesian_iter_spinbox.setValue(7)
        iter_layout.addWidget(self.bayesian_iter_spinbox)
        layout.addLayout(iter_layout)
        
        # Samples
        samples_layout = QHBoxLayout()
        samples_layout.addWidget(QLabel("MCMC Samples:"))
        self.bayesian_samples_spinbox = QSpinBox()
        self.bayesian_samples_spinbox.setRange(10, 1000)
        self.bayesian_samples_spinbox.setValue(100)
        samples_layout.addWidget(self.bayesian_samples_spinbox)
        layout.addLayout(samples_layout)
        
        # Batch Size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.bayesian_batch_spinbox = QSpinBox()
        self.bayesian_batch_spinbox.setRange(1, 100)
        self.bayesian_batch_spinbox.setValue(20)
        batch_layout.addWidget(self.bayesian_batch_spinbox)
        layout.addLayout(batch_layout)
        
        # Drift Batch Size
        drift_batch_layout = QHBoxLayout()
        drift_batch_layout.addWidget(QLabel("Drift Batch Size:"))
        self.drift_batch_spinbox = QSpinBox()
        self.drift_batch_spinbox.setRange(1, 20)
        self.drift_batch_spinbox.setValue(8)
        drift_batch_layout.addWidget(self.drift_batch_spinbox)
        layout.addLayout(drift_batch_layout)
        
        # Calibration Settings
        calibration_group = QGroupBox("Calibration Settings")
        calibration_layout = QVBoxLayout()
        
        # Calibration Mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Calibration Mode:"))
        self.calibration_mode_combo = QComboBox()
        self.calibration_mode_combo.addItems(["Training", "Inference", "Disabled"])
        self.calibration_mode_combo.setCurrentText("Inference")
        mode_layout.addWidget(self.calibration_mode_combo)
        calibration_layout.addLayout(mode_layout)
        
        # Calibration File
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Calibration File:"))
        self.calibration_file_edit = QLineEdit()
        self.calibration_file_edit.setPlaceholderText("Leave empty for auto-generated filename")
        file_layout.addWidget(self.calibration_file_edit)
        calibration_group.setLayout(calibration_layout)
        layout.addWidget(calibration_group)
        
        # Calibration Options
        self.apply_sigma_fudger = QCheckBox("Apply SigmaFudger")
        self.apply_sigma_fudger.setChecked(True)
        layout.addWidget(self.apply_sigma_fudger)
        
        self.apply_mean_bias_fitter = QCheckBox("Apply MeanBiasFitter")
        self.apply_mean_bias_fitter.setChecked(True)
        layout.addWidget(self.apply_mean_bias_fitter)
        
        group.setLayout(layout)
        return group

    def _create_controls_panel(self):
        """Creates the right-hand panel with all user controls."""
        main_layout = QVBoxLayout()
        # Static Navigation Group
        main_layout.addWidget(self._create_navigation_group())
        # Add current data label
        self.current_data_label = QLabel("Current Data: Instrument = None | Backend = None")
        main_layout.addWidget(self.current_data_label)
        # --- Pipeline Mode Selection ---
        pipeline_mode_group = QGroupBox("Pipeline Mode")
        pipeline_mode_layout = QVBoxLayout()
        self.pipeline_mode_combo = QComboBox()
        self.pipeline_mode_combo.addItems(["Preprocessing Pipeline", "Bayesian Pipeline"])
        self.pipeline_mode_combo.currentTextChanged.connect(self.on_pipeline_mode_change)
        pipeline_mode_layout.addWidget(self.pipeline_mode_combo)
        pipeline_mode_group.setLayout(pipeline_mode_layout)
        main_layout.addWidget(pipeline_mode_group)
        # --- Tabbed Interface for Settings ---
        self.settings_tabs = QTabWidget()
        self.settings_tabs.addTab(self._create_calibration_group(), "Calibration")
        self.detrending_group = self._create_detrending_group()
        self.detrending_tab_index = self.settings_tabs.addTab(self.detrending_group, "Detrending")
        self.settings_tabs.addTab(self._create_analysis_group(), "Analysis")
        self.bayesian_settings_group = self._create_bayesian_settings_group()
        self.bayesian_tab_index = self.settings_tabs.addTab(self.bayesian_settings_group, "Bayesian Pipeline")
        main_layout.addWidget(self.settings_tabs)
        # --- Pipeline Execution Button ---
        self.pipeline_button = QPushButton("Apply Changes & Rerun Pipeline")
        self.pipeline_button.clicked.connect(self.run_pipeline)
        self.pipeline_button.setEnabled(False)  # Disabled until data is loaded
        main_layout.addWidget(self.pipeline_button)
        # Progress tracking
        main_layout.addWidget(self._create_progress_group())
        main_layout.addStretch()
        container = QWidget()
        container.setLayout(main_layout)
        return container

    def _create_navigation_group(self):
        nav_group = QGroupBox("Navigation")
        nav_layout = QFormLayout()

        # Split selection
        self.split_combo = QComboBox()
        self.split_combo.setObjectName("Split")
        self.split_combo.addItems(["train", "test"])
        self.split_combo.currentTextChanged.connect(self.populate_planet_ids)
        nav_layout.addRow("Split:", self.split_combo)

        # Planet selection
        self.planet_combo = QComboBox()
        self.planet_combo.setObjectName("Planet")
        self.planet_combo.setEditable(True)
        self.planet_combo.currentTextChanged.connect(self.populate_obs_ids)
        nav_layout.addRow("Planet:", self.planet_combo)

        # Observation selection
        self.obs_combo = QComboBox()
        self.obs_combo.setObjectName("Observation")
        nav_layout.addRow("Observation:", self.obs_combo)

        # Instrument selection
        self.instrument_combo = QComboBox()
        self.instrument_combo.setObjectName("Instrument")
        self.instrument_combo.addItems(["AIRS-CH0", "FGS1", "Both"])
        self.instrument_combo.currentTextChanged.connect(self.populate_obs_ids)
        nav_layout.addRow("Instrument:", self.instrument_combo)

        # Backend selection
        self.backend_combo = QComboBox()
        self.backend_combo.setObjectName("Backend")
        self.backend_combo.addItems(["cpu", "gpu"])
        if not GPU_ENABLED:
            self.backend_combo.model().item(1).setEnabled(False)
        nav_layout.addRow("Backend:", self.backend_combo)

        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        nav_layout.addRow("", self.load_button)

        nav_group.setLayout(nav_layout)
        return nav_group

    def _create_calibration_group(self):
        calib_group = QGroupBox("Calibration Steps")
        layout = QVBoxLayout()
        self.calib_checkboxes = {
            "adc": QCheckBox("ADC Conversion"),
            "mask": QCheckBox("Mask Hot/Dead Pixels"),
            "linearity": QCheckBox("Linearity Correction"),
            "dark": QCheckBox("Dark Current Subtraction"),
            "cds": QCheckBox("Correlated Double Sampling (CDS)"),
            "flat": QCheckBox("Flat Field Correction"),
        }
        for checkbox in self.calib_checkboxes.values():
            checkbox.setChecked(True)
            layout.addWidget(checkbox)
        calib_group.setLayout(layout)
        return calib_group

    def _create_detrending_group(self):
        detrend_group = QGroupBox("Detrending Model")
        layout = QVBoxLayout()
        
        # --- Dropdown for model selection ---
        self.detrend_model_combo = QComboBox()
        self.detrend_model_combo.currentTextChanged.connect(self.on_detrend_model_change)
        layout.addWidget(self.detrend_model_combo)

        # --- Stacked widget for model parameters ---
        self.detrend_params_stack = QStackedWidget()
        self.model_widget_map = {}

        # Polynomial parameters
        poly_widget = QWidget()
        poly_layout = QHBoxLayout(poly_widget)
        poly_layout.addWidget(QLabel("Degree:"))
        self.poly_degree_spinbox = QSpinBox()
        self.poly_degree_spinbox.setRange(1, 10)
        self.poly_degree_spinbox.setValue(2)
        poly_layout.addWidget(self.poly_degree_spinbox)
        self.model_widget_map["Polynomial"] = self.detrend_params_stack.addWidget(poly_widget)
        
        # Savitzky-Golay parameters
        savgol_widget = QWidget()
        savgol_layout = QHBoxLayout(savgol_widget)
        savgol_layout.addWidget(QLabel("Window:"))
        self.savgol_window_spinbox = QSpinBox()
        self.savgol_window_spinbox.setRange(3, 999)
        self.savgol_window_spinbox.setSingleStep(2)
        self.savgol_window_spinbox.setValue(51)
        savgol_layout.addWidget(self.savgol_window_spinbox)
        savgol_layout.addWidget(QLabel("Order:"))
        self.savgol_order_spinbox = QSpinBox()
        self.savgol_order_spinbox.setRange(1, 10)
        self.savgol_order_spinbox.setValue(2)
        savgol_layout.addWidget(self.savgol_order_spinbox)
        self.model_widget_map["Savitzky-Golay"] = self.detrend_params_stack.addWidget(savgol_widget)
        
        # George GP parameters
        george_widget = QWidget()
        george_layout = QHBoxLayout(george_widget)
        george_layout.addWidget(QLabel("Kernel:"))
        self.george_kernel_combo = QComboBox()
        self.george_kernel_combo.addItems(['Matern32'])
        george_layout.addWidget(self.george_kernel_combo)
        self.model_widget_map["GP (george CPU)"] = self.detrend_params_stack.addWidget(george_widget)
        
        # GPyTorch GP parameters
        if GP_GPU_ENABLED:
            gpytorch_widget = QWidget()
            gpytorch_layout = QHBoxLayout(gpytorch_widget)
            gpytorch_layout.addWidget(QLabel("Iterations:"))
            self.gpytorch_iter_spinbox = QSpinBox()
            self.gpytorch_iter_spinbox.setRange(10, 500)
            self.gpytorch_iter_spinbox.setValue(50)
            gpytorch_layout.addWidget(self.gpytorch_iter_spinbox)
            self.model_widget_map["GP (GPyTorch GPU)"] = self.detrend_params_stack.addWidget(gpytorch_widget)
        
        # Hybrid GP parameters
        hybrid_widget = QWidget()
        hybrid_layout = QVBoxLayout(hybrid_widget)
        hybrid_poly_layout = QHBoxLayout()
        hybrid_poly_layout.addWidget(QLabel("Residual Degree:"))
        self.hybrid_poly_degree_spinbox = QSpinBox()
        self.hybrid_poly_degree_spinbox.setRange(1, 5)
        self.hybrid_poly_degree_spinbox.setValue(2)
        hybrid_poly_layout.addWidget(self.hybrid_poly_degree_spinbox)
        hybrid_layout.addLayout(hybrid_poly_layout)
        hybrid_iter_layout = QHBoxLayout()
        hybrid_iter_layout.addWidget(QLabel("GP Iterations:"))
        self.hybrid_iter_spinbox = QSpinBox()
        self.hybrid_iter_spinbox.setRange(10, 500)
        self.hybrid_iter_spinbox.setValue(50)
        hybrid_iter_layout.addWidget(self.hybrid_iter_spinbox)
        hybrid_layout.addLayout(hybrid_iter_layout)
        self.model_widget_map["Hybrid GP (CPU)"] = self.detrend_params_stack.addWidget(hybrid_widget)
        if GP_GPU_ENABLED:
             self.model_widget_map["Hybrid GP (GPU)"] = self.detrend_params_stack.indexOf(hybrid_widget)
        
        # Advanced GP methods are not yet implemented
        # Advanced GP (2-Step) parameters
        # Multi-Kernel GP parameters  
        # Transit Window GP parameters
        
        # AIRS Drift parameters
        airs_widget = QWidget()
        airs_layout = QVBoxLayout(airs_widget)
        
        # Average drift parameters
        avg_layout = QHBoxLayout()
        avg_layout.addWidget(QLabel("Avg Kernel:"))
        self.airs_avg_kernel_combo = QComboBox()
        self.airs_avg_kernel_combo.addItems(['Matern32', 'RBF', 'Matern52'])
        avg_layout.addWidget(self.airs_avg_kernel_combo)
        avg_layout.addWidget(QLabel("Length Scale:"))
        self.airs_avg_length_spinbox = QDoubleSpinBox()
        self.airs_avg_length_spinbox.setRange(0.1, 10.0)
        self.airs_avg_length_spinbox.setValue(1.0)
        self.airs_avg_length_spinbox.setSingleStep(0.1)
        avg_layout.addWidget(self.airs_avg_length_spinbox)
        airs_layout.addLayout(avg_layout)
        
        # Spectral drift parameters
        spec_layout = QHBoxLayout()
        spec_layout.addWidget(QLabel("Spectral Kernel:"))
        self.airs_spec_kernel_combo = QComboBox()
        self.airs_spec_kernel_combo.addItems(['Matern32', 'RBF', 'Matern52'])
        spec_layout.addWidget(self.airs_spec_kernel_combo)
        spec_layout.addWidget(QLabel("Time Scale:"))
        self.airs_time_scale_spinbox = QDoubleSpinBox()
        self.airs_time_scale_spinbox.setRange(0.1, 10.0)
        self.airs_time_scale_spinbox.setValue(0.4)
        self.airs_time_scale_spinbox.setSingleStep(0.1)
        spec_layout.addWidget(self.airs_time_scale_spinbox)
        spec_layout.addWidget(QLabel("Wavelength Scale:"))
        self.airs_wl_scale_spinbox = QDoubleSpinBox()
        self.airs_wl_scale_spinbox.setRange(0.01, 1.0)
        self.airs_wl_scale_spinbox.setValue(0.05)
        self.airs_wl_scale_spinbox.setSingleStep(0.01)
        spec_layout.addWidget(self.airs_wl_scale_spinbox)
        airs_layout.addLayout(spec_layout)
        
        # Sparse approximation
        sparse_layout = QHBoxLayout()
        self.airs_sparse_checkbox = QCheckBox("Use Sparse Approximation")
        self.airs_sparse_checkbox.setChecked(True)
        sparse_layout.addWidget(self.airs_sparse_checkbox)
        airs_layout.addLayout(sparse_layout)
        
        self.model_widget_map["AIRS Drift (CPU)"] = self.detrend_params_stack.addWidget(airs_widget)
        if GP_GPU_ENABLED:
            self.model_widget_map["AIRS Drift (GPU)"] = self.detrend_params_stack.indexOf(airs_widget)
        
        # FGS Drift parameters
        fgs_widget = QWidget()
        fgs_layout = QHBoxLayout(fgs_widget)
        fgs_layout.addWidget(QLabel("Kernel:"))
        self.fgs_kernel_combo = QComboBox()
        self.fgs_kernel_combo.addItems(['Matern32', 'RBF', 'Matern52'])
        fgs_layout.addWidget(self.fgs_kernel_combo)
        fgs_layout.addWidget(QLabel("Length Scale:"))
        self.fgs_length_spinbox = QDoubleSpinBox()
        self.fgs_length_spinbox.setRange(0.1, 10.0)
        self.fgs_length_spinbox.setValue(1.0)
        self.fgs_length_spinbox.setSingleStep(0.1)
        fgs_layout.addWidget(self.fgs_length_spinbox)
        
        self.model_widget_map["FGS Drift (CPU)"] = self.detrend_params_stack.addWidget(fgs_widget)
        if GP_GPU_ENABLED:
            self.model_widget_map["FGS Drift (GPU)"] = self.detrend_params_stack.indexOf(fgs_widget)
        
        # Bayesian Multi-Component parameters
        bayesian_widget = QWidget()
        bayesian_layout = QVBoxLayout(bayesian_widget)
        
        # PCA components
        pca_layout = QHBoxLayout()
        pca_layout.addWidget(QLabel("PCA Components:"))
        self.bayesian_pca_spinbox = QSpinBox()
        self.bayesian_pca_spinbox.setRange(0, 5)
        self.bayesian_pca_spinbox.setValue(1)
        pca_layout.addWidget(self.bayesian_pca_spinbox)
        bayesian_layout.addLayout(pca_layout)
        
        # Iterations
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Iterations:"))
        self.bayesian_iter_spinbox = QSpinBox()
        self.bayesian_iter_spinbox.setRange(1, 20)
        self.bayesian_iter_spinbox.setValue(7)
        iter_layout.addWidget(self.bayesian_iter_spinbox)
        bayesian_layout.addLayout(iter_layout)
        
        # Samples
        samples_layout = QHBoxLayout()
        samples_layout.addWidget(QLabel("Samples:"))
        self.bayesian_samples_spinbox = QSpinBox()
        self.bayesian_samples_spinbox.setRange(10, 1000)
        self.bayesian_samples_spinbox.setValue(100)
        samples_layout.addWidget(self.bayesian_samples_spinbox)
        samples_layout.addWidget(QLabel(""))
        bayesian_layout.addLayout(samples_layout)
        
        # Batch size for MCMC
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("MCMC Batch Size:"))
        self.bayesian_batch_spinbox = QSpinBox()
        self.bayesian_batch_spinbox.setRange(5, 50)
        self.bayesian_batch_spinbox.setValue(20)
        self.bayesian_batch_spinbox.setToolTip("Number of MCMC samples to process at once. Higher values are faster but use more GPU memory.")
        batch_layout.addWidget(self.bayesian_batch_spinbox)
        batch_layout.addWidget(QLabel(""))
        bayesian_layout.addLayout(batch_layout)
        
        # Add placeholder widget for preprocessing pipeline (no parameters needed)
        placeholder_widget = QWidget()
        placeholder_layout = QVBoxLayout(placeholder_widget)
        placeholder_layout.addWidget(QLabel("No additional parameters needed for preprocessing pipeline."))
        placeholder_layout.addStretch()
        
        # Map models to their parameter widgets
        self.model_widget_map["AIRS Drift (CPU)"] = self.detrend_params_stack.addWidget(airs_widget)
        self.model_widget_map["FGS Drift (CPU)"] = self.detrend_params_stack.addWidget(fgs_widget)
        self.model_widget_map["Bayesian Multi-Component (CPU)"] = self.detrend_params_stack.addWidget(placeholder_widget)
        
        if GP_GPU_ENABLED:
            self.model_widget_map["AIRS Drift (GPU)"] = self.detrend_params_stack.indexOf(airs_widget)
            self.model_widget_map["FGS Drift (GPU)"] = self.detrend_params_stack.indexOf(fgs_widget)
            self.model_widget_map["Bayesian Multi-Component (GPU)"] = self.detrend_params_stack.indexOf(placeholder_widget)

        layout.addWidget(self.detrend_params_stack)
        detrend_group.setLayout(layout)
        
        # Initialize with CPU options (after all parameter widgets are created)
        self.update_detrending_options()
        
        return detrend_group

    def _create_analysis_group(self):
        """Creates a group box for analysis settings like masking and folding."""
        analysis_group = QGroupBox("Analysis Settings")
        layout = QVBoxLayout()

        mask_group = QGroupBox("Transit Mask Method")
        mask_layout = QVBoxLayout(mask_group)
        self.mask_method_combo = QComboBox()
        self.mask_method_combo.addItems(["empirical", "physical"])
        mask_layout.addWidget(self.mask_method_combo)
        layout.addWidget(mask_group)

        fold_group = QGroupBox("Phase Folding")
        fold_layout = QVBoxLayout(fold_group)
        fold_layout.addWidget(QLabel("Number of Bins:"))
        self.bins_spinbox = QSpinBox()
        self.bins_spinbox.setRange(10, 500)
        self.bins_spinbox.setValue(100)
        fold_layout.addWidget(self.bins_spinbox)
        layout.addWidget(fold_group)

        analysis_group.setLayout(layout)
        return analysis_group
    
    def _create_progress_group(self):
        """Create the progress tracking UI group."""
        progress_group = QGroupBox("Progress")
        layout = QVBoxLayout()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.progress_label = QLabel("Ready")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)
        
        # Stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.stop_pipeline)
        layout.addWidget(self.stop_button)
        
        progress_group.setLayout(layout)
        return progress_group

    def _create_visuals_panel(self):
        main_layout = QVBoxLayout()
        tabs = QTabWidget()

        frame_group = QGroupBox("Frame & Wavelength Selector")
        frame_layout = QHBoxLayout(frame_group)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_spinbox = QSpinBox()
        self.wavelength_slider = QSlider(Qt.Orientation.Horizontal)
        self.wavelength_spinbox = QSpinBox()
        self.frame_slider.valueChanged.connect(self.update_image_plot)
        self.frame_spinbox.valueChanged.connect(self.frame_slider.setValue)
        self.frame_slider.valueChanged.connect(self.frame_spinbox.setValue)
        self.wavelength_slider.valueChanged.connect(self.update_light_curve_plots)
        self.wavelength_spinbox.valueChanged.connect(self.wavelength_slider.setValue)
        self.wavelength_slider.valueChanged.connect(self.wavelength_spinbox.setValue)
        frame_layout.addWidget(QLabel("Frame Index:")); frame_layout.addWidget(self.frame_slider); frame_layout.addWidget(self.frame_spinbox)
        frame_layout.addWidget(QLabel("Wavelength Column:")); frame_layout.addWidget(self.wavelength_slider); frame_layout.addWidget(self.wavelength_spinbox)
        main_layout.addWidget(frame_group)

        self.image_canvas = FigureCanvas(Figure(figsize=(10, 5), tight_layout=True))
        self.single_lc_canvas = FigureCanvas(Figure(figsize=(10, 3), tight_layout=True))
        self.detrended_lc_canvas = FigureCanvas(Figure(figsize=(10, 3), tight_layout=True))
        self.phase_folded_canvas = FigureCanvas(Figure(figsize=(10, 3), tight_layout=True))
        
        # Bayesian visualization canvases
        self.bayesian_canvas = FigureCanvas(Figure(figsize=(10, 6), tight_layout=True))
        self.components_canvas = FigureCanvas(Figure(figsize=(12, 8), tight_layout=True))
        
        # Baseline visualization canvas
        self.baseline_canvas = FigureCanvas(Figure(figsize=(10, 6), tight_layout=True))
        
        self.image_ax = self.image_canvas.figure.subplots()
        self.single_lc_ax = self.single_lc_canvas.figure.subplots()
        self.detrended_lc_ax = self.detrended_lc_canvas.figure.subplots()
        self.phase_folded_ax = self.phase_folded_canvas.figure.subplots()
        
        # Bayesian visualization axes
        self.bayesian_ax = self.bayesian_canvas.figure.subplots()
        self.components_ax = self.components_canvas.figure.subplots(2, 3)  # 2x3 grid for components
        self.components_ax = self.components_ax.flatten()  # Flatten for easier indexing
        
        # Baseline visualization axes
        self.baseline_ax = self.baseline_canvas.figure.subplots()
        
        # Initialize plotters
        self.bayesian_plotter = BayesianPlotter(self.bayesian_ax, self.bayesian_canvas)
        self.baseline_plotter = BaselinePlotter(self.baseline_ax, self.baseline_canvas)
        
        self.image_cbar = None

        # Store tab indices for dynamic visibility management
        self.tab_indices["Detector View"] = tabs.addTab(self._create_detector_tab(), "Detector View")
        self.tab_indices["Photometry View"] = tabs.addTab(self.single_lc_canvas, "Photometry View")
        self.tab_indices["Detrended View"] = tabs.addTab(self._create_detrended_tab(), "Detrended View")
        self.tab_indices["Phase-Folded View"] = tabs.addTab(self.phase_folded_canvas, "Phase-Folded View")
        self.tab_indices["Bayesian Results"] = tabs.addTab(self._create_bayesian_tab(), "Bayesian Results")
        self.tab_indices["Component Analysis"] = tabs.addTab(self._create_components_tab(), "Component Analysis")
        self.tab_indices["Baseline Results"] = tabs.addTab(self._create_baseline_tab(), "Baseline Results")
        self.log_display = QTextEdit(); self.log_display.setReadOnly(True)
        self.tab_indices["Performance Log"] = tabs.addTab(self.log_display, "Performance Log")
        
        # Store reference to tab widget for visibility management
        self.tab_widget = tabs
        
        main_layout.addWidget(tabs)
        container = QWidget(); container.setLayout(main_layout)
        return container

    def _create_bayesian_tab(self):
        """Create the Bayesian results tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls for Bayesian visualization
        controls_group = QGroupBox("Bayesian Visualization Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.show_uncertainties_checkbox = QCheckBox("Show Uncertainties")
        self.show_uncertainties_checkbox.setChecked(True)
        self.show_uncertainties_checkbox.toggled.connect(self.update_bayesian_plot)
        controls_layout.addWidget(self.show_uncertainties_checkbox)
        
        self.show_covariance_checkbox = QCheckBox("Show Covariance Matrix")
        self.show_covariance_checkbox.setChecked(False)
        self.show_covariance_checkbox.toggled.connect(self.update_bayesian_plot)
        controls_layout.addWidget(self.show_covariance_checkbox)
        
        self.show_ground_truth_checkbox = QCheckBox("Show Ground Truth")
        self.show_ground_truth_checkbox.setChecked(True)
        self.show_ground_truth_checkbox.setEnabled(False)  # Will be enabled when data is loaded
        self.show_ground_truth_checkbox.toggled.connect(self.update_bayesian_plot)
        
        # Add log scale option for better comparison
        self.log_scale_checkbox = QCheckBox("Log Scale Y-Axis")
        self.log_scale_checkbox.setChecked(False)
        self.log_scale_checkbox.toggled.connect(self.update_bayesian_plot)
        controls_layout.addWidget(self.log_scale_checkbox)
        controls_layout.addWidget(self.show_ground_truth_checkbox)
        
        layout.addWidget(controls_group)
        layout.addWidget(self.bayesian_canvas)
        
        return tab
    
    def _create_components_tab(self):
        """Create the component analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls for component visualization
        controls_group = QGroupBox("Component Analysis Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.show_stellar_checkbox = QCheckBox("Stellar Spectrum")
        self.show_stellar_checkbox.setChecked(True)
        self.show_stellar_checkbox.toggled.connect(self.update_components_plot)
        controls_layout.addWidget(self.show_stellar_checkbox)
        
        self.show_drift_checkbox = QCheckBox("Drift Model")
        self.show_drift_checkbox.setChecked(True)
        self.show_drift_checkbox.toggled.connect(self.update_components_plot)
        controls_layout.addWidget(self.show_drift_checkbox)
        
        self.show_transit_checkbox = QCheckBox("Transit Depth")
        self.show_transit_checkbox.setChecked(True)
        self.show_transit_checkbox.toggled.connect(self.update_components_plot)
        controls_layout.addWidget(self.show_transit_checkbox)
        
        self.show_window_checkbox = QCheckBox("Transit Window")
        self.show_window_checkbox.setChecked(True)
        self.show_window_checkbox.toggled.connect(self.update_components_plot)
        controls_layout.addWidget(self.show_window_checkbox)
        
        self.show_noise_checkbox = QCheckBox("Noise Model")
        self.show_noise_checkbox.setChecked(True)
        self.show_noise_checkbox.toggled.connect(self.update_components_plot)
        controls_layout.addWidget(self.show_noise_checkbox)
        
        layout.addWidget(controls_group)
        layout.addWidget(self.components_canvas)
        
        return tab

    def _create_detector_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); info_group = QGroupBox("Star & Planet Info"); info_layout = QVBoxLayout(info_group); self.info_display = QTextEdit(); self.info_display.setReadOnly(True); info_group.setFixedHeight(150); info_layout.addWidget(self.info_display); layout.addWidget(info_group); layout.addWidget(self.image_canvas); self.image_canvas.mpl_connect('motion_notify_event', self.on_mouse_move); return tab
    
    def _create_detrended_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); self.zoom_checkbox = QCheckBox("Zoom to Transit"); self.zoom_checkbox.toggled.connect(self.update_detrended_plot); layout.addWidget(self.zoom_checkbox); layout.addWidget(self.detrended_lc_canvas); return tab

    def _create_baseline_tab(self):
        """Create the baseline results tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls for baseline visualization
        controls_group = QGroupBox("Baseline Visualization Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.baseline_show_uncertainties_checkbox = QCheckBox("Show Uncertainties")
        self.baseline_show_uncertainties_checkbox.setChecked(True)
        self.baseline_show_uncertainties_checkbox.toggled.connect(self.update_baseline_plot)
        controls_layout.addWidget(self.baseline_show_uncertainties_checkbox)
        
        self.baseline_show_ground_truth_checkbox = QCheckBox("Show Ground Truth")
        self.baseline_show_ground_truth_checkbox.setChecked(True)
        self.baseline_show_ground_truth_checkbox.setEnabled(False)  # Will be enabled when data is loaded
        self.baseline_show_ground_truth_checkbox.toggled.connect(self.update_baseline_plot)
        controls_layout.addWidget(self.baseline_show_ground_truth_checkbox)
        
        # Add log scale option for better comparison
        self.baseline_log_scale_checkbox = QCheckBox("Log Scale Y-Axis")
        self.baseline_log_scale_checkbox.setChecked(False)
        self.baseline_log_scale_checkbox.toggled.connect(self.update_baseline_plot)
        controls_layout.addWidget(self.baseline_log_scale_checkbox)
        
        layout.addWidget(controls_group)
        layout.addWidget(self.baseline_canvas)
        
        return tab

    # --- Core Logic ---

    def run_pipeline(self):
        """Run the appropriate pipeline based on the selected mode."""
        if not self.observation:
            self.statusBar().showMessage("Please load data first.", 5000)
            return
        
        # Check if we have both instruments or single instrument
        if isinstance(self.observation, dict):
            # Both instruments loaded
            if not all(obs.is_loaded for obs in self.observation.values()):
                self.statusBar().showMessage("Please load data first.", 5000)
                return
        else:
            # Single instrument loaded
            if not self.observation.is_loaded:
                self.statusBar().showMessage("Please load data first.", 5000)
                return
        
        # Check backend compatibility
        selected_backend = self.backend_combo.currentText()
        if self.current_backend and self.current_backend != selected_backend:
            self.statusBar().showMessage(
                f"Backend mismatch! Data loaded on {self.current_backend}, but {selected_backend} selected. "
                "Please reload data with the correct backend.", 10000
            )
            return
        
        # Determine which pipeline to run based on mode
        mode = self.pipeline_mode_combo.currentText()
        
        if mode == "Preprocessing Pipeline":
            self.run_preprocessing_pipeline()
        elif mode == "Bayesian Pipeline":
            self.run_bayesian_pipeline()
        elif mode == "Baseline Pipeline":
            self.run_baseline_pipeline()
    
    def run_preprocessing_pipeline(self):
        """Run the preprocessing pipeline with detrending."""
        # Get detrender
        detrender = self._get_detrender(self.detrend_model_combo.currentText())
        if detrender is None:
            return
        
        # Get mask method
        mask_method = self.mask_method_combo.currentText().lower()
        
        # Get steps to run
        steps_to_run = {k: v.isChecked() for k, v in self.calib_checkboxes.items()}
        
        # Run pipeline
        self._run_pipeline_async(detrender, mask_method, steps_to_run)
    
    def _run_pipeline_async(self, detrender, mask_method, steps_to_run):
        """Run entire pipeline asynchronously with progress tracking."""
        # Create and start worker
        self.worker = PipelineWorker(
            self.observation, detrender, mask_method, steps_to_run, 
            n_bins=self.bins_spinbox.value()
        )
        self.worker.progress_signal.connect(self._on_pipeline_progress)
        self.worker.finished_signal.connect(self._on_pipeline_finished)
        self.worker.error_signal.connect(self._on_pipeline_error)
        self.worker.start()
    
    def _run_detrending_async(self, detrender, mask_method):
        """Run detrending asynchronously with progress tracking."""
        # Set UI to busy state
        self._set_ui_busy(True, "detrending")
        
        # Create and start worker
        self.worker = PipelineWorker(self.observation, detrender, mask_method)
        self.worker.progress_signal.connect(self._on_detrending_progress)
        self.worker.finished_signal.connect(self._on_detrending_finished)
        self.worker.error_signal.connect(self._on_detrending_error)
        self.worker.start()
    
    def _on_pipeline_progress(self, message: str):
        """Handle progress updates from the pipeline worker."""
        # Update progress label
        self.progress_label.setText(message)
        
        # Update status bar
        self.statusBar().showMessage(message, 1000)
        
        # Update progress bar based on message (starting from 25% where loading left off)
        if "calibration" in message.lower():
            self.progress_bar.setValue(30)
        elif "photometry" in message.lower():
            self.progress_bar.setValue(45)
        elif "detrending" in message.lower():
            self.progress_bar.setValue(70)
        elif "phase folding" in message.lower():
            self.progress_bar.setValue(85)
    
    def _on_pipeline_finished(self):
        """Handle completion of the pipeline."""
        try:
            # Update progress for plotting
            self.progress_bar.setValue(95)
            self.progress_label.setText("Updating plots...")
            
            # Update plots
            self._update_all_plots()
            
            # Complete
            self.progress_bar.setValue(100)
            self.progress_label.setText("Pipeline completed")
            
            # Set UI to idle state
            self._set_ui_busy(False)
            
            self.statusBar().showMessage("Pipeline finished.", 5000)
            
        except Exception as e:
            self._on_pipeline_error(str(e))
        finally:
            # Clean up worker
            if self.worker:
                self.worker.deleteLater()
                self.worker = None
    
    def _on_pipeline_error(self, error_message: str):
        """Handle errors during pipeline execution."""
        # Set UI to idle state
        self._set_ui_busy(False)
        
        # Show error (but not for user stops)
        if "stopped by user" not in error_message.lower():
            self.statusBar().showMessage(f"ERROR: {error_message}", 15000)
            print(f"ERROR: {error_message}")
        else:
            self.statusBar().showMessage("Pipeline stopped by user", 5000)
        
        # Clean up worker
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
    
    def _on_detrending_progress(self, progress: DetrendingProgress):
        """Handle progress updates from the detrender."""
        # Update progress bar (detrending is 70-85% of total progress)
        detrending_progress = 70 + int(progress.progress * 15)
        self.progress_bar.setValue(detrending_progress)
        
        # Update status message
        message = progress.message
        if progress.current_wavelength is not None and progress.total_wavelengths is not None:
            message += f" ({progress.current_wavelength + 1}/{progress.total_wavelengths})"
        self.progress_label.setText(message)
        
        # Update status bar
        self.statusBar().showMessage(message, 1000)
    
    def _on_detrending_finished(self):
        """Handle completion of detrending."""
        try:
            # Update progress for phase folding
            self.progress_bar.setValue(85)
            self.progress_label.setText("Running phase folding...")
            
            # Run phase folding
            self.observation.run_phase_folding(n_bins=self.bins_spinbox.value())
            
            # Update progress for plotting
            self.progress_bar.setValue(95)
            self.progress_label.setText("Updating plots...")
            
            # Update plots
            self._update_all_plots()
            
            # Complete
            self.progress_bar.setValue(100)
            self.progress_label.setText("Pipeline completed")
            
            # Set UI to idle state
            self._set_ui_busy(False)
            
            self.statusBar().showMessage("Pipeline finished.", 5000)
            
        except Exception as e:
            self._on_detrending_error(str(e))
        finally:
            # Clean up worker and GPU memory
            if self.worker:
                # Explicitly clean up GPU memory if using GPyTorch
                if hasattr(self.worker, 'detrender') and self.worker.detrender:
                    detrender_class = self.worker.detrender.__class__.__name__
                    if 'GPyTorch' in detrender_class or 'GPU' in detrender_class:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                print("DEBUG: Cleared GPU cache after detrending")
                        except ImportError:
                            pass
                
                self.worker.deleteLater()
                self.worker = None
    
    def _on_detrending_error(self, error_message: str):
        """Handle errors during detrending."""
        # Set UI to idle state
        self._set_ui_busy(False)
        
        # Show error (but not for user stops)
        if "stopped by user" not in error_message.lower():
            self.statusBar().showMessage(f"ERROR: {error_message}", 15000)
            print(f"ERROR: {error_message}")
        else:
            self.statusBar().showMessage("Pipeline stopped by user", 5000)
        
        # Clean up worker
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
    
    def _set_ui_busy(self, busy: bool, operation: str = "processing"):
        """Set UI to busy or idle state."""
        if busy:
            # Only disable buttons to prevent double-clicks, keep configuration accessible
            self.load_button.setEnabled(False)
            self.pipeline_button.setEnabled(False)
            
            # Show progress UI
            self.progress_bar.setVisible(True)
            self.progress_label.setVisible(True)
            self.stop_button.setVisible(True)
            
            # Don't reset progress - let the worker thread manage it
            self.progress_label.setText(f"Starting {operation}...")
            
        else:
            # Enable buttons
            self.load_button.setEnabled(True)
            self.pipeline_button.setEnabled(True)
            
            # Hide progress UI
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)
            self.stop_button.setVisible(False)
    
    def stop_pipeline(self):
        """Stop the currently running pipeline and safely abort all changes."""
        if self.worker and self.worker.isRunning():
            # Stop pipeline worker
            self.worker.stop()
            self.worker.wait()  # Wait for thread to finish
            
            # Safely abort pipeline changes
            self._abort_pipeline_changes()
            self._on_pipeline_error("Pipeline stopped by user")
            
        elif self.bayesian_worker and self.bayesian_worker.isRunning():
            # Stop Bayesian pipeline worker
            self.bayesian_worker.stop()
            self.bayesian_worker.wait()  # Wait for thread to finish
            
            # Safely abort Bayesian pipeline changes
            self._abort_pipeline_changes()
            self._on_bayesian_pipeline_error("Bayesian Pipeline stopped by user")
            
        elif hasattr(self, 'baseline_worker') and self.baseline_worker and self.baseline_worker.isRunning():
            # Stop baseline pipeline worker
            self.baseline_worker.stop()
            self.baseline_worker.wait()  # Wait for thread to finish
            
            # Safely abort baseline pipeline changes
            self._abort_pipeline_changes()
            self._on_baseline_pipeline_error("Baseline Pipeline stopped by user")
            
        elif self.loading_worker and self.loading_worker.isRunning():
            # Stop loading worker
            self.loading_worker.stop()
            self.loading_worker.wait()  # Wait for thread to finish
            
            # Safely abort loading changes
            self._abort_loading_changes()
            self._on_loading_error("Data loading stopped by user")
    
    def _abort_pipeline_changes(self):
        """Safely abort any partial pipeline changes."""
        # Clean up GPU memory if we were using GPyTorch
        if hasattr(self, 'worker') and self.worker and hasattr(self.worker, 'detrender'):
            detrender_class = self.worker.detrender.__class__.__name__
            if 'GPyTorch' in detrender_class or 'GPU' in detrender_class:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("DEBUG: Cleared GPU cache after pipeline abort")
                except ImportError:
                    pass
        
        if self.observation and self.observation.is_loaded:
            # Clear any partial pipeline results
            if hasattr(self.observation, 'detrended_light_curves'):
                self.observation.detrended_light_curves = None
            if hasattr(self.observation, 'noise_models'):
                self.observation.noise_models = None
            if hasattr(self.observation, 'phase_folded_lc'):
                self.observation.phase_folded_lc = None
            
            # Ensure we're back to a clean state
            self._ensure_clean_state()
            
            # Update plots to show original data
            self._update_all_plots()
        
        # Clear any stored results
        if hasattr(self, 'bayesian_results'):
            self.bayesian_results = None
        if hasattr(self, 'baseline_results'):
            self.baseline_results = None
    
    def _abort_detrending_changes(self):
        """Safely abort any partial detrending changes."""
        if self.observation and self.observation.is_loaded:
            # Clear any partial detrending results
            if hasattr(self.observation, 'detrended_light_curves'):
                self.observation.detrended_light_curves = None
            if hasattr(self.observation, 'noise_models'):
                self.observation.noise_models = None
            if hasattr(self.observation, 'phase_folded_lc'):
                self.observation.phase_folded_lc = None
            
            # Ensure we're back to a clean state
            self._ensure_clean_state()
            
            # Update plots to show original data
            self._update_all_plots()
    
    def _ensure_clean_state(self):
        """Ensure the observation is in a clean state after stopping."""
        if self.observation and self.observation.is_loaded:
            # Verify that only the original data remains
            # Detrending results should be None
            if hasattr(self.observation, 'detrended_light_curves') and self.observation.detrended_light_curves is not None:
                self.observation.detrended_light_curves = None
            
            if hasattr(self.observation, 'noise_models') and self.observation.noise_models is not None:
                self.observation.noise_models = None
            
            if hasattr(self.observation, 'phase_folded_lc') and self.observation.phase_folded_lc is not None:
                self.observation.phase_folded_lc = None
            
            # Ensure light curves are still available (from photometry)
            if not hasattr(self.observation, 'light_curves') or self.observation.light_curves is None:
                # If light curves are missing, we need to re-run photometry
                try:
                    self.observation.run_photometry()
                except Exception:
                    # If photometry fails, we're in a bad state
                    pass
    
    def _abort_loading_changes(self):
        """Safely abort any partial loading changes."""
        # Clear the observation object if it was partially created
        if hasattr(self, 'observation'):
            self.observation = None
        
        # Clear any UI elements that depend on loaded data
        self.info_display.setText("")
        
        # Reset sliders to safe defaults
        self.frame_slider.setRange(0, 0)
        self.frame_spinbox.setRange(0, 0)
        self.wavelength_slider.setRange(0, 0)
        self.wavelength_spinbox.setRange(0, 0)
        
        # Clear plots
        self._clear_all_plots()
    
    def _clear_all_plots(self):
        """Clear all plots to show empty state."""
        try:
            # Clear image plot
            if hasattr(self, 'image_ax') and self.image_ax is not None:
                try:
                    self.image_ax.clear()
                    self.image_ax.set_title("No data loaded")
                    if hasattr(self, 'image_canvas') and self.image_canvas is not None:
                        self.image_canvas.draw()
                except Exception as e:
                    print(f"Warning: Could not clear image plot: {e}")
            
            # Clear light curve plots
            if hasattr(self, 'single_lc_ax') and self.single_lc_ax is not None:
                try:
                    self.single_lc_ax.clear()
                    self.single_lc_ax.set_title("No data loaded")
                    if hasattr(self, 'single_lc_canvas') and self.single_lc_canvas is not None:
                        self.single_lc_canvas.draw()
                except Exception as e:
                    print(f"Warning: Could not clear single light curve plot: {e}")
            
            if hasattr(self, 'detrended_lc_ax') and self.detrended_lc_ax is not None:
                try:
                    self.detrended_lc_ax.clear()
                    self.detrended_lc_ax.set_title("No data loaded")
                    if hasattr(self, 'detrended_lc_canvas') and self.detrended_lc_canvas is not None:
                        self.detrended_lc_canvas.draw()
                except Exception as e:
                    print(f"Warning: Could not clear detrended light curve plot: {e}")
            
            if hasattr(self, 'phase_folded_ax') and self.phase_folded_ax is not None:
                try:
                    self.phase_folded_ax.clear()
                    self.phase_folded_ax.set_title("No data loaded")
                    if hasattr(self, 'phase_folded_canvas') and self.phase_folded_canvas is not None:
                        self.phase_folded_canvas.draw()
                except Exception as e:
                    print(f"Warning: Could not clear phase folded plot: {e}")
            
            # Clear Bayesian plots
            if hasattr(self, 'bayesian_ax') and self.bayesian_ax is not None:
                try:
                    self.bayesian_ax.clear()
                    self.bayesian_ax.set_title("No Bayesian Results")
                    if hasattr(self, 'bayesian_canvas') and self.bayesian_canvas is not None:
                        self.bayesian_canvas.draw()
                except Exception as e:
                    print(f"Warning: Could not clear Bayesian plot: {e}")
            
            # Clear baseline plots
            if hasattr(self, 'baseline_ax') and self.baseline_ax is not None:
                try:
                    self.baseline_ax.clear()
                    self.baseline_ax.set_title("No Baseline Results")
                    if hasattr(self, 'baseline_canvas') and self.baseline_canvas is not None:
                        self.baseline_canvas.draw()
                except Exception as e:
                    print(f"Warning: Could not clear baseline plot: {e}")
            
            # Clear component analysis plots
            if hasattr(self, 'components_ax') and self.components_ax is not None:
                try:
                    for ax in self.components_ax:
                        if ax is not None:
                            ax.clear()
                            ax.set_title("No Component Analysis")
                    if hasattr(self, 'components_canvas') and self.components_canvas is not None:
                        self.components_canvas.draw()
                except Exception as e:
                    print(f"Warning: Could not clear component analysis plots: {e}")
            
            # Clear log display
            if hasattr(self, 'log_display') and self.log_display is not None:
                try:
                    self.log_display.setText("")
                except Exception as e:
                    print(f"Warning: Could not clear log display: {e}")
                    
        except Exception as e:
            print(f"Warning: Error in _clear_all_plots: {e}")
            # Continue execution even if plot clearing fails

    def _get_detrender(self, model_name):
        # Check for not implemented methods
        if "❌ Not Implemented" in model_name:
            self.statusBar().showMessage("This method is not implemented yet.", 5000)
            return None
            
        # Basic methods
        if model_name == "Polynomial": 
            return detrending.PolynomialDetrender(degree=self.poly_degree_spinbox.value())
        elif model_name == "Savitzky-Golay": 
            return detrending.SavGolDetrender(window_length=self.savgol_window_spinbox.value(), polyorder=self.savgol_order_spinbox.value())
        elif model_name == "GP (george CPU) ⚠️ Very Slow": 
            return detrending.GPDetrender(kernel=self.george_kernel_combo.currentText())
        elif model_name == "GP (GPyTorch GPU)":
            if GP_GPU_ENABLED: 
                self.statusBar().showMessage("Running GPyTorch Detrending (GPU)...", 30000)
                QApplication.processEvents()
                return detrending.GPyTorchDetrender(training_iter=self.gpytorch_iter_spinbox.value())
        elif model_name == "Hybrid GP (CPU)": 
            return detrending.HybridDetrender(use_gpu=False, training_iter=self.hybrid_iter_spinbox.value(), poly_degree=self.hybrid_poly_degree_spinbox.value())
        elif model_name == "Hybrid GP (GPU)":
            if GP_GPU_ENABLED: 
                self.statusBar().showMessage("Running Hybrid GP Detrending (GPU)...", 30000)
                QApplication.processEvents()
                return detrending.HybridDetrender(use_gpu=True, training_iter=self.hybrid_iter_spinbox.value(), poly_degree=self.hybrid_poly_degree_spinbox.value())
        
        # Advanced sklearn methods are not yet implemented
        # elif model_name == "Advanced GP (2-Step)":
        # elif model_name == "Multi-Kernel GP":
        # elif model_name == "Transit Window GP":
        
        # ariel_gp-style methods with KISS-GP integration
        elif model_name == "AIRS Drift (CPU) 🚀 KISS-GP":
            self.statusBar().showMessage("Running AIRS Drift Detrending with KISS-GP (CPU)...", 30000)
            QApplication.processEvents()
            return detrending.create_airs_drift_detrender(
                instrument="AIRS-CH0",
                use_gpu=False,
                avg_kernel=self.airs_avg_kernel_combo.currentText(),
                avg_length_scale=self.airs_avg_length_spinbox.value(),
                spectral_kernel=self.airs_spec_kernel_combo.currentText(),
                time_scale=self.airs_time_scale_spinbox.value(),
                wavelength_scale=self.airs_wl_scale_spinbox.value(),
                use_sparse=self.airs_sparse_checkbox.isChecked()
            )
        
        elif model_name == "AIRS Drift (GPU) 🚀 KISS-GP ⚡ Fast":
            if GP_GPU_ENABLED:
                self.statusBar().showMessage("Running AIRS Drift Detrending with KISS-GP (GPU) - Fast Batch Processing...", 30000)
                QApplication.processEvents()
                return detrending.create_airs_drift_detrender(
                    instrument="AIRS-CH0",
                    use_gpu=True,
                    avg_kernel=self.airs_avg_kernel_combo.currentText(),
                    avg_length_scale=self.airs_avg_length_spinbox.value(),
                    spectral_kernel=self.airs_spec_kernel_combo.currentText(),
                    time_scale=self.airs_time_scale_spinbox.value(),
                    wavelength_scale=self.airs_wl_scale_spinbox.value(),
                    use_sparse=self.airs_sparse_checkbox.isChecked()
                )
        
        elif model_name == "FGS Drift (CPU)":
            self.statusBar().showMessage("Running FGS Drift Detrending (CPU)...", 30000)
            QApplication.processEvents()
            return detrending.create_fgs_drift_detrender(
                instrument="FGS1",
                use_gpu=False,
                kernel=self.fgs_kernel_combo.currentText(),
                length_scale=self.fgs_length_spinbox.value()
            )
        
        elif model_name == "FGS Drift (GPU) ⚡ Fast":
            if GP_GPU_ENABLED:
                self.statusBar().showMessage("Running FGS Drift Detrending (GPU) - Fast Batch Processing...", 30000)
                QApplication.processEvents()
                return detrending.create_fgs_drift_detrender(
                    instrument="FGS1",
                    use_gpu=True,
                    kernel=self.fgs_kernel_combo.currentText(),
                    length_scale=self.fgs_length_spinbox.value()
                )
        
        elif model_name == "Bayesian Multi-Component (CPU)":
            self.statusBar().showMessage("Running Bayesian Multi-Component Detrending (CPU)...", 60000)
            QApplication.processEvents()
            return detrending.BayesianMultiComponentDetrender(
                use_gpu=False,
                n_pca=self.bayesian_pca_spinbox.value(),
                n_iter=self.bayesian_iter_spinbox.value(),
                n_samples=self.bayesian_samples_spinbox.value()
            )
        
        elif model_name == "Bayesian Multi-Component (GPU)":
            if GP_GPU_ENABLED:
                self.statusBar().showMessage("Running Bayesian Multi-Component Detrending (GPU)...", 60000)
                QApplication.processEvents()
                return detrending.BayesianMultiComponentDetrender(
                    use_gpu=True,
                    n_pca=self.bayesian_pca_spinbox.value(),
                    n_iter=self.bayesian_iter_spinbox.value(),
                    n_samples=self.bayesian_samples_spinbox.value()
                )
        
        return None

    # --- Plotting and UI Updates ---

    def _update_all_plots(self):
        # Handle calibration log for both single and multiple observations
        if isinstance(self.observation, dict):
            # Both instruments loaded - use AIRS-CH0 for calibration log
            airs_obs = self.observation.get("AIRS-CH0")
            if airs_obs and hasattr(airs_obs, 'calibration_log'):
                self.log_display.setText("\n".join(airs_obs.calibration_log))
        else:
            # Single instrument loaded
            if hasattr(self.observation, 'calibration_log'):
                self.log_display.setText("\n".join(self.observation.calibration_log))
        
        self.update_image_plot()
        self.update_light_curve_plots()
        # Update Bayesian plots if results are available
        if hasattr(self, 'bayesian_results') and self.bayesian_results is not None:
            self.update_bayesian_plot()
            self.update_components_plot()
        # Update baseline plots if results are available
        if hasattr(self, 'baseline_results') and self.baseline_results is not None:
            self.update_baseline_plot()
    
    def update_image_plot(self):
        if not self.observation: 
            print("DEBUG: update_image_plot - observation not loaded")
            return
        
        # Handle both single and multiple observations
        if isinstance(self.observation, dict):
            # Both instruments loaded - use AIRS-CH0 for image display
            observation = self.observation.get("AIRS-CH0")
            if not observation or not observation.is_loaded:
                print("DEBUG: update_image_plot - AIRS-CH0 observation not loaded")
                return
        else:
            # Single instrument loaded
            observation = self.observation
            if not observation.is_loaded:
                print("DEBUG: update_image_plot - observation not loaded")
                return
        
        # Debug: Check observation state
        print(f"DEBUG: update_image_plot - observation.is_loaded: {observation.is_loaded}")
        print(f"DEBUG: update_image_plot - processed_signal is None: {observation.processed_signal is None}")
        if observation.processed_signal is not None:
            print(f"DEBUG: update_image_plot - processed_signal shape: {observation.processed_signal.shape}")
            print(f"DEBUG: update_image_plot - processed_signal type: {type(observation.processed_signal)}")
        
        # Safely remove the previous colorbar if it exists and is still in the figure
        if self.image_cbar is not None:
            try:
                if hasattr(self.image_cbar, 'ax') and self.image_cbar.ax in self.image_canvas.figure.axes:
                    self.image_cbar.remove()
            except Exception as e:
                print(f"Warning: Could not remove image colorbar: {e}")
            self.image_cbar = None
        
        self.image_ax.clear()
        
        # Get processed signal with explicit conversion
        processed_signal = observation.get_data(return_type='numpy')
        print(f"DEBUG: update_image_plot - after get_data, type: {type(processed_signal)}")
        print(f"DEBUG: update_image_plot - after get_data, shape: {processed_signal.shape if processed_signal is not None else 'None'}")
        
        if processed_signal is None:
            print("DEBUG: update_image_plot - processed_signal is None after get_data")
            self.image_ax.set_title("No Data Available")
            self.image_canvas.draw()
            return
        
        frame_idx = self.frame_slider.value()
        if self.calib_checkboxes["cds"].isChecked(): 
            frame_idx //= 2
        
        if frame_idx >= processed_signal.shape[0]: 
            frame_idx = 0
        
        img_data = processed_signal[frame_idx]
        print(f"DEBUG: update_image_plot - img_data shape: {img_data.shape}")
        print(f"DEBUG: update_image_plot - img_data min/max: {img_data.min()}/{img_data.max()}")
        
        try:
            im = self.image_ax.imshow(img_data, aspect='auto', cmap='viridis', origin='lower')
            self.image_cbar = self.image_canvas.figure.colorbar(im, ax=self.image_ax)
            self.image_ax.set_title(f"Detector Frame (Index: {frame_idx})")
            self.image_canvas.draw()
        except Exception as e:
            print(f"Warning: Could not create image colorbar: {e}")
            im = self.image_ax.imshow(img_data, aspect='auto', cmap='viridis', origin='lower')
            self.image_ax.set_title(f"Detector Frame (Index: {frame_idx})")
            self.image_canvas.draw()
    
    def update_light_curve_plots(self):
        if not self.observation: return
        
        # Handle both single and multiple observations
        if isinstance(self.observation, dict):
            # Both instruments loaded - use AIRS-CH0 for light curve display
            observation = self.observation.get("AIRS-CH0")
            if not observation or not observation.is_loaded:
                return
        else:
            # Single instrument loaded
            observation = self.observation
            if not observation.is_loaded:
                return
        
        self.update_photometry_plot(observation)
        self.update_detrended_plot(observation)
        self.update_phase_folded_plot(observation)
    
    def update_photometry_plot(self, observation):
        self.single_lc_ax.clear()
        if observation.light_curves is None: 
            self.single_lc_canvas.draw()
            return
        light_curves = observation.get_light_curves(return_type='numpy')
        config_wavelength_col = self.wavelength_slider.value()
        wavelength_col = self._convert_wavelength_index(config_wavelength_col, light_curves.shape[1])
        
        self.single_lc_ax.plot(observation.get_time_array(), light_curves[:, wavelength_col], '.-', alpha=0.8, color='dodgerblue')
        self.single_lc_ax.set_title(f"Raw Light Curve (Wavelength: {config_wavelength_col})")
        self.single_lc_ax.set_xlabel("Time (days)")
        self.single_lc_ax.set_ylabel("Flux")
        self.single_lc_ax.grid(True, alpha=0.3)
        self.single_lc_canvas.draw()
    
    def update_detrended_plot(self, observation):
        self.detrended_lc_ax.clear()
        if observation.detrended_light_curves is None: 
            self.detrended_lc_canvas.draw()
            return
        detrended_lcs = observation.get_detrended_light_curves(return_type='numpy')
        original_lcs = observation.get_light_curves(return_type='numpy')
        noise_models = observation.get_noise_models(return_type='numpy')
        config_wavelength_col = self.wavelength_slider.value()
        wavelength_col = self._convert_wavelength_index(config_wavelength_col, detrended_lcs.shape[1])
        
        time_arr = observation.get_time_array()
        if self.zoom_checkbox.isChecked():
            self.detrended_lc_ax.plot(time_arr, detrended_lcs[:, wavelength_col], '.', color='black')
        else:
            self.detrended_lc_ax.plot(time_arr, original_lcs[:, wavelength_col], '.', color='grey', alpha=0.2, label='Original')
            self.detrended_lc_ax.plot(time_arr, noise_models[:, wavelength_col], '-', color='red', label='Noise Model')
            self.detrended_lc_ax.plot(time_arr, detrended_lcs[:, wavelength_col], '.', color='black', alpha=0.8, label='Detrended')
            self.detrended_lc_ax.legend()
        
        self.detrended_lc_ax.set_title(f"Detrended Light Curve (Wavelength: {config_wavelength_col})")
        self.detrended_lc_ax.set_xlabel("Time (days)")
        self.detrended_lc_ax.set_ylabel("Normalized Flux")
        self.detrended_lc_ax.grid(True, alpha=0.3)
        self.detrended_lc_canvas.draw()
    
    def update_phase_folded_plot(self, observation):
        """Update the phase-folded plot with static scale based on overall data range."""
        self.phase_folded_ax.clear()
        
        if observation.phase_folded_lc is None:
            self.phase_folded_canvas.draw()
            return
            
        folded_data = observation.get_phase_folded_lc()
        config_wavelength_col = self.wavelength_slider.value()
        wavelength_col = self._convert_wavelength_index(config_wavelength_col, len(folded_data))
            
        bin_centers, binned_flux, binned_error = folded_data[wavelength_col]
        
        # Calculate overall data range from all wavelengths for static scale
        all_flux_values = []
        all_error_values = []
        
        for wl_idx in range(len(folded_data)):
            try:
                _, flux, error = folded_data[wl_idx]
                all_flux_values.extend(flux)
                all_error_values.extend(error)
            except (IndexError, ValueError):
                continue
        
        # Convert to numpy arrays for easier calculation
        all_flux_values = np.array(all_flux_values)
        all_error_values = np.array(all_error_values)
        
        # Calculate overall range with padding
        if len(all_flux_values) > 0:
            flux_min = np.min(all_flux_values)
            flux_max = np.max(all_flux_values)
            flux_range = flux_max - flux_min
            
            # Add padding to the range (10% on each side)
            padding = flux_range * 0.1
            y_min = flux_min - padding
            y_max = flux_max + padding
            
            # Ensure we don't go below 0 for normalized flux
            y_min = max(y_min, 0.0)
            
            # Set static y-axis limits
            self.phase_folded_ax.set_ylim(y_min, y_max)
        
        # Plot the current wavelength data
        self.phase_folded_ax.errorbar(bin_centers, binned_flux, yerr=binned_error, 
                                    fmt='o', color='black', ecolor='gray', 
                                    elinewidth=1, capsize=2)
        
        # Add reference line at 1.0
        self.phase_folded_ax.axhline(1.0, color='r', linestyle='--', alpha=0.7)
        
        # Format the plot
        self.phase_folded_ax.set_title(f"Phase-Folded Transit (Wavelength: {config_wavelength_col})")
        self.phase_folded_ax.set_xlabel("Orbital Phase")
        self.phase_folded_ax.set_ylabel("Normalized Flux")
        self.phase_folded_ax.grid(True, alpha=0.3)
        
        self.phase_folded_canvas.draw()
    
    def load_data(self):
        planet_id = self.planet_combo.currentText()
        obs_id = self.obs_combo.currentText()
        
        if not all([planet_id, obs_id]): 
            self.statusBar().showMessage("Error: Missing Planet or Observation ID.", 5000)
            return
        
        # Set UI to busy state
        self._set_ui_busy(True, "data loading")
        
        # Create and start loading worker
        self.loading_worker = DataLoadingWorker(
            planet_id=planet_id,
            instrument=self.instrument_combo.currentText(),
            obs_id=obs_id,
            split=self.split_combo.currentText(),
            backend=self.backend_combo.currentText()
        )
        
        # Connect signals
        self.loading_worker.progress_signal.connect(self._on_loading_progress)
        self.loading_worker.finished_signal.connect(self._on_loading_finished)
        self.loading_worker.error_signal.connect(self._on_loading_error)
        
        # Start loading
        self.loading_worker.start()
    
    def _on_loading_progress(self, message: str):
        """Handle progress updates from data loading."""
        self.progress_label.setText(message)
        self.statusBar().showMessage(message, 1000)
        # Use a more gradual progress for loading (0-25%)
        if "Loading" in message:
            self.progress_bar.setValue(10)
        elif "Calibrating" in message:
            self.progress_bar.setValue(15)
        elif "Photometry" in message:
            self.progress_bar.setValue(20)
        else:
            self.progress_bar.setValue(25)  # Final loading step
    
    def _on_loading_finished(self, observations):
        """Handle completion of data loading."""
        try:
            # Store observations based on type
            if isinstance(observations, dict):
                # Both instruments loaded
                self.observation = observations  # Store as dict
                self.current_instrument = "Both"
                # Use the backend from the first observation (they should be the same)
                first_obs = list(observations.values())[0]
                self.current_backend = self.backend_combo.currentText()
                
                # Check if both observations have ground truth
                airs_has_gt = observations["AIRS-CH0"].has_ground_truth()
                fgs_has_gt = observations["FGS1"].has_ground_truth()
                ground_truth_available = airs_has_gt or fgs_has_gt
                
                # Update UI controls - disable wavelength controls for "Both" mode
                self.wavelength_slider.setEnabled(False)
                self.wavelength_spinbox.setEnabled(False)
                
                # Update slider ranges based on AIRS-CH0 (more data points)
                airs_obs = observations["AIRS-CH0"]
                self.frame_slider.setRange(0, airs_obs.raw_signal.shape[0] - 1)
                self.frame_spinbox.setRange(0, airs_obs.raw_signal.shape[0] - 1)
                
                # Set wavelength slider range based on AIRS-CH0 config
                start_col, end_col = self._get_wavelength_range_for_instrument("AIRS-CH0")
                if start_col == end_col == 0:
                    max_wl = airs_obs.raw_signal.shape[2] - 1
                    self.wavelength_slider.setRange(0, max_wl)
                    self.wavelength_spinbox.setRange(0, max_wl)
                else:
                    self.wavelength_slider.setRange(start_col, end_col)
                    self.wavelength_spinbox.setRange(start_col, end_col)
                
            else:
                # Single instrument loaded
                self.observation = observations
                self.current_instrument = self.instrument_combo.currentText()
                self.current_backend = self.backend_combo.currentText()
                
                # Update ground truth checkbox state based on observation
                ground_truth_available = self.observation.has_ground_truth()
                
                # Update UI controls based on instrument
                is_fgs = (self.current_instrument == 'FGS1')
                self.wavelength_slider.setEnabled(not is_fgs)
                self.wavelength_spinbox.setEnabled(not is_fgs)
                
                # Update slider ranges
                self.frame_slider.setRange(0, self.observation.raw_signal.shape[0] - 1)
                self.frame_spinbox.setRange(0, self.observation.raw_signal.shape[0] - 1)
                
                # Set wavelength slider range based on config
                start_col, end_col = self._get_wavelength_range_for_instrument(self.current_instrument)
                if start_col == end_col == 0:
                    # Fallback to full range if not in config
                    max_wl = self.observation.raw_signal.shape[2] - 1
                    self.wavelength_slider.setRange(0, max_wl)
                    self.wavelength_spinbox.setRange(0, max_wl)
                else:
                    # Use config range
                    self.wavelength_slider.setRange(start_col, end_col)
                    self.wavelength_spinbox.setRange(start_col, end_col)
            
            # Update ground truth checkbox state
            if hasattr(self, 'show_ground_truth_checkbox'):
                self.show_ground_truth_checkbox.setEnabled(ground_truth_available)
                self.show_ground_truth_checkbox.setChecked(ground_truth_available)
            
            # Update baseline ground truth checkbox state
            if hasattr(self, 'baseline_show_ground_truth_checkbox'):
                self.baseline_show_ground_truth_checkbox.setEnabled(ground_truth_available)
                self.baseline_show_ground_truth_checkbox.setChecked(ground_truth_available)
            
            # Update star info display
            if self.star_info_df is not None and int(self.planet_combo.currentText()) in self.star_info_df.index:
                self.info_display.setText(self.star_info_df.loc[int(self.planet_combo.currentText())].to_string())
            
            # Set UI to idle state
            self._set_ui_busy(False)
            
            # Update detrending options based on loaded instrument and backend
            self.update_detrending_options()
            
            # Update pipeline mode options based on instrument selection
            self.update_pipeline_mode_options()
            
            # Lock navigation, unlock settings
            self._lock_navigation_panel()
            self._unlock_settings_panel()
            self.load_button.setText("Change Data")
            self.load_button.clicked.disconnect()
            self.load_button.clicked.connect(self._unlock_for_new_data)
            self.navigation_locked = True
            
            # Update current data label
            ground_truth_status = " | Ground Truth: Available" if ground_truth_available else " | Ground Truth: Not Available"
            self.current_data_label.setText(f"Current Data: Instrument = {self.current_instrument} | Backend = {self.current_backend}{ground_truth_status}")

            # Don't run pipeline automatically - wait for user to click "Apply"
            # Just update the raw plots
            self._update_all_plots()
            
        except Exception as e:
            self._on_loading_error(str(e))
        finally:
            # Clean up worker
            if self.loading_worker:
                self.loading_worker.deleteLater()
                self.loading_worker = None
    
    def _on_loading_error(self, error_message: str):
        """Handle errors during data loading."""
        # Set UI to idle state
        self._set_ui_busy(False)
        
        # Show error (but not for user stops)
        if "stopped by user" not in error_message.lower():
            self.statusBar().showMessage(f"Error loading data: {error_message}", 15000)
            print(f"ERROR: {error_message}")
        else:
            self.statusBar().showMessage("Data loading stopped by user", 5000)
        
        # Clean up worker
        if self.loading_worker:
            self.loading_worker.deleteLater()
            self.loading_worker = None
    
    def populate_planet_ids(self):
        current_planet = self.planet_combo.currentText(); self.planet_combo.clear(); split = self.split_combo.currentText(); self.star_info_df = loaders.load_star_info(split)
        if self.star_info_df is None: return
        path = DATASET_DIR / split
        if path.exists():
            planet_ids = sorted([p.name for p in path.iterdir() if p.is_dir()]); self.planet_combo.addItems(planet_ids)
            if current_planet in planet_ids: self.planet_combo.setCurrentText(current_planet)
        self.populate_obs_ids()
    
    def populate_obs_ids(self):
        self.obs_combo.clear(); planet_id = self.planet_combo.currentText()
        if not planet_id: return
        planet_dir = DATASET_DIR / self.split_combo.currentText() / planet_id
        if not planet_dir.exists(): return
        
        current_instrument = self.instrument_combo.currentText()
        if current_instrument == "Both":
            # For "Both", check for AIRS-CH0 signal files (more common)
            obs_ids = sorted([f.stem.split('_')[-1] for f in planet_dir.glob(f"AIRS-CH0_signal_*.parquet") if f.stem.split('_')[-1].isdigit()], key=int)
        else:
            # For single instrument, check for that instrument's signal files
            obs_ids = sorted([f.stem.split('_')[-1] for f in planet_dir.glob(f"{current_instrument}_signal_*.parquet") if f.stem.split('_')[-1].isdigit()], key=int)
        
        if obs_ids: self.obs_combo.addItems(obs_ids)
        
        # Update wavelength controls based on instrument type
        is_fgs = (current_instrument == 'FGS1')
        self.wavelength_slider.setEnabled(not is_fgs and current_instrument != "Both")
        self.wavelength_spinbox.setEnabled(not is_fgs and current_instrument != "Both")
        
        # Set wavelength slider values based on config
        if current_instrument == "Both":
            # For "Both", use AIRS-CH0 config
            start_col, end_col = self._get_wavelength_range_for_instrument("AIRS-CH0")
        else:
            start_col, end_col = self._get_wavelength_range_for_instrument(current_instrument)
        
        if is_fgs or start_col == end_col:
            # FGS1 or single wavelength: Set to start_col and disable
            self.wavelength_slider.setRange(start_col, end_col)
            self.wavelength_spinbox.setRange(start_col, end_col)
            self.wavelength_slider.setValue(start_col)
            self.wavelength_spinbox.setValue(start_col)
        else:
            # Multi-wavelength instrument: Enable and set to start_col
            self.wavelength_slider.setRange(start_col, end_col)
            self.wavelength_spinbox.setRange(start_col, end_col)
            self.wavelength_slider.setValue(start_col)
            self.wavelength_spinbox.setValue(start_col)
    
    def update_detrending_options(self):
        """Update the detrending dropdown based on the loaded backend and instrument."""
        # Use the actual loaded data, not the dropdown selections
        current_backend = self.current_backend if self.current_backend else self.backend_combo.currentText()
        current_instrument = self.current_instrument if self.current_instrument else self.instrument_combo.currentText()
        
        # Clear current options
        self.detrend_model_combo.clear()
        
        # Build detrending options based on loaded backend and instrument
        detrend_options = []
        
        if current_backend == "cpu":
            # CPU-only options (available for all instruments)
            detrend_options.extend([
                "Polynomial",
                "Savitzky-Golay", 
                "GP (george CPU) ⚠️ Very Slow",
                "Hybrid GP (CPU)"
            ])
            
            # Instrument-specific ariel_gp-style models (CPU) - Fallback versions
            if current_instrument == "AIRS-CH0":
                detrend_options.extend([
                    "AIRS Drift (CPU) 🚀 KISS-GP",
                    "Bayesian Multi-Component (CPU)"
                ])
            elif current_instrument == "FGS1":
                detrend_options.extend([
                    "FGS Drift (CPU)",
                    "Bayesian Multi-Component (CPU)"
                ])
            
        elif current_backend == "gpu":
            # GPU-only options (if available)
            if GP_GPU_ENABLED:
                detrend_options.extend([
                    "GP (GPyTorch GPU)",
                    "Hybrid GP (GPU)"
                ])
                
                # Instrument-specific ariel_gp-style models (GPU) - Prioritize fast GPU versions
                if current_instrument == "AIRS-CH0":
                    detrend_options.extend([
                        "AIRS Drift (GPU) 🚀 KISS-GP ⚡ Fast",  # Prioritize GPU version
                        "Bayesian Multi-Component (GPU)"
                    ])
                elif current_instrument == "FGS1":
                    detrend_options.extend([
                        "FGS Drift (GPU) ⚡ Fast",  # Prioritize GPU version
                        "Bayesian Multi-Component (GPU)"
                    ])
            else:
                # Show not implemented options when GPU is not available
                detrend_options.extend([
                    "GP (GPyTorch GPU) ❌ Not Implemented",
                    "Hybrid GP (GPU) ❌ Not Implemented"
                ])
                
                if current_instrument == "AIRS-CH0":
                    detrend_options.extend([
                        "AIRS Drift (GPU) ❌ Not Implemented",
                        "Bayesian Multi-Component (GPU) ❌ Not Implemented"
                    ])
                elif current_instrument == "FGS1":
                    detrend_options.extend([
                        "FGS Drift (GPU) ❌ Not Implemented",
                        "Bayesian Multi-Component (GPU) ❌ Not Implemented"
                    ])
        
        self.detrend_model_combo.addItems(detrend_options)
        
        # Set a default selection if available
        if detrend_options:
            self.detrend_model_combo.setCurrentIndex(0)

    def on_pipeline_mode_change(self, mode: str):
        """Handle pipeline mode changes."""
        print(f"DEBUG: Pipeline mode changed to: {mode}")
        
        # Update tab visibility based on selected pipeline
        self.update_tab_visibility(mode)
        
        if mode == "Preprocessing Pipeline":
            # Show detrending tab, hide Bayesian tab
            self.settings_tabs.setTabVisible(self.detrending_tab_index, True)
            self.settings_tabs.setTabVisible(self.bayesian_tab_index, False)
            self.pipeline_button.setText("Apply Changes & Rerun Pipeline")
            self.pipeline_button.setEnabled(self.observation is not None and self.observation.is_loaded)
            print(f"DEBUG: Set button text to: {self.pipeline_button.text()}")
            
            # Update preprocessing plots to ensure they're visible
            if self.observation and self.observation.is_loaded:
                self._update_all_plots()
                
                # Show ground truth info in status bar if available
                if self.observation.has_ground_truth():
                    ground_truth = self.observation.get_ground_truth()
                    self.statusBar().showMessage(
                        f"Ground Truth Available: {len(ground_truth)} wavelengths, "
                        f"depth range: {ground_truth.min():.4f} - {ground_truth.max():.4f}", 
                        5000
                    )
        else:  # Bayesian Pipeline
            # Hide detrending tab, show Bayesian tab
            self.settings_tabs.setTabVisible(self.detrending_tab_index, False)
            self.settings_tabs.setTabVisible(self.bayesian_tab_index, True)
            self.pipeline_button.setText("Run Bayesian Pipeline")
            self.pipeline_button.setEnabled(self.observation is not None and self.observation.is_loaded)
            print(f"DEBUG: Set button text to: {self.pipeline_button.text()}")
            
            # Show ground truth data immediately when Bayesian pipeline is selected
            if self.observation and self.observation.is_loaded:
                self.bayesian_plotter.plot_results(
                    bayesian_results=None,
                    observation=self.observation,
                    show_uncertainties=False,
                    show_ground_truth=True,
                    log_scale=False
                )
            
            # Update Bayesian plots if results are available
            if hasattr(self, 'bayesian_results') and self.bayesian_results is not None:
                self.update_bayesian_plot()
                self.update_components_plot()
        # Force a repaint to ensure the button text is updated
        self.pipeline_button.repaint()

    def run_bayesian_pipeline(self):
        """Run the Bayesian pipeline."""
        if not (self.observation and self.observation.is_loaded):
            self.statusBar().showMessage("Please load data first.", 5000)
            return
        
        # Check backend compatibility
        selected_backend = self.backend_combo.currentText()
        if self.current_backend and self.current_backend != selected_backend:
            self.statusBar().showMessage(
                f"Backend mismatch! Data loaded on {self.current_backend}, but {selected_backend} selected. "
                "Please reload data with the correct backend.", 10000
            )
            return
        
        # Clear previous Bayesian results to ensure clean state
        if hasattr(self, 'bayesian_results'):
            self.bayesian_results = None
        
        # Force GPU cleanup before starting
        force_gpu_cleanup()
        
        # Set UI to busy state
        self._set_ui_busy(True, "bayesian_pipeline")
        
        # Get pipeline parameters
        pipeline_params = {
            'instrument': self.current_instrument,
            'use_gpu': self.current_backend == 'gpu',
            'n_pca': self.bayesian_pca_spinbox.value(),
            'n_iter': self.bayesian_iter_spinbox.value(),
            'n_samples': self.bayesian_samples_spinbox.value(),
            'batch_size': self.bayesian_batch_spinbox.value(),
            'drift_batch_size': self.drift_batch_spinbox.value(),
            'apply_sigma_fudger': self.apply_sigma_fudger.isChecked(),
            'apply_mean_bias_fitter': self.apply_mean_bias_fitter.isChecked(),
            'calibration_mode': self.calibration_mode_combo.currentText(),
            'calibration_file': self.calibration_file_edit.text().strip() if self.calibration_file_edit.text().strip() else None
        }
        
        # Automatically reduce batch sizes for FGS to prevent OOM
        if self.current_instrument == "FGS1":
            pipeline_params['drift_batch_size'] = min(pipeline_params['drift_batch_size'], 2)
            pipeline_params['batch_size'] = min(pipeline_params['batch_size'], 10)
            print(f"DEBUG: Reduced batch sizes for FGS - Drift: {pipeline_params['drift_batch_size']}, MCMC: {pipeline_params['batch_size']}")
        
        # Run Bayesian pipeline asynchronously
        self._run_bayesian_pipeline_async(pipeline_params)
    
    def _run_bayesian_pipeline_async(self, pipeline_params):
        """Run Bayesian pipeline asynchronously with progress tracking."""
        # Create and start worker
        self.bayesian_worker = BayesianPipelineWorker(self.observation, pipeline_params)
        self.bayesian_worker.progress_signal.connect(self._on_bayesian_pipeline_progress)
        self.bayesian_worker.finished_signal.connect(self._on_bayesian_pipeline_finished)
        self.bayesian_worker.error_signal.connect(self._on_bayesian_pipeline_error)
        self.bayesian_worker.start()
    
    def _on_bayesian_pipeline_progress(self, message: str):
        """Handle progress updates from the Bayesian pipeline worker."""
        # Update progress label
        self.progress_label.setText(message)
        
        # Update status bar
        self.statusBar().showMessage(message, 1000)
        
        # Update progress bar
        if "creating" in message.lower():
            self.progress_bar.setValue(20)
        elif "fitting" in message.lower():
            self.progress_bar.setValue(50)
        elif "mcmc" in message.lower():
            self.progress_bar.setValue(80)
    
    def _on_bayesian_pipeline_finished(self, results):
        """Handle completion of the Bayesian pipeline."""
        try:
            # Update progress for plotting
            self.progress_bar.setValue(95)
            self.progress_label.setText("Updating plots...")
            
            # Store results
            self.bayesian_results = results
            
            # Ensure we're in Bayesian mode and the correct tab is visible
            current_mode = self.pipeline_mode_combo.currentText()
            if current_mode != "Bayesian Pipeline":
                # Force the mode to Bayesian Pipeline
                self.pipeline_mode_combo.setCurrentText("Bayesian Pipeline")
                self.on_pipeline_mode_change("Bayesian Pipeline")
            
            # Update only Bayesian plots, not traditional plots
            if hasattr(self, 'bayesian_results') and self.bayesian_results is not None:
                self.update_bayesian_plot()
                self.update_components_plot()
            
            # Complete
            self.progress_bar.setValue(100)
            self.progress_label.setText("Bayesian Pipeline completed")
            
            # Set UI to idle state
            self._set_ui_busy(False)
            
            self.statusBar().showMessage("Bayesian Pipeline finished.", 5000)
            
        except Exception as e:
            self._on_bayesian_pipeline_error(str(e))
        finally:
            # Clean up GPU memory after Bayesian pipeline
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("DEBUG: Cleared GPU cache after Bayesian pipeline completion")
            except ImportError:
                pass
            
            # Clean up worker
            if hasattr(self, 'bayesian_worker') and self.bayesian_worker:
                self.bayesian_worker.deleteLater()
                self.bayesian_worker = None
    
    def _on_bayesian_pipeline_error(self, error_message: str):
        """Handle errors during Bayesian pipeline execution."""
        # Set UI to idle state
        self._set_ui_busy(False)
        
        # Show error (but not for user stops)
        if "stopped by user" not in error_message.lower():
            self.statusBar().showMessage(f"ERROR: {error_message}", 15000)
            print(f"ERROR: {error_message}")
        else:
            self.statusBar().showMessage("Bayesian Pipeline stopped by user", 5000)
        
        # Clean up worker
        if hasattr(self, 'bayesian_worker') and self.bayesian_worker:
            self.bayesian_worker.deleteLater()
            self.bayesian_worker = None

    def on_detrend_model_change(self, model_name):
        index = self.model_widget_map.get(model_name, 0)
        self.detrend_params_stack.setCurrentIndex(index)
    
    def on_mouse_move(self, event):
        if not (event.inaxes and event.inaxes == self.image_ax and self.observation and self.observation.processed_signal is not None):
            return
            
        x, y = int(event.xdata or 0), int(event.ydata or 0)
        data = self.observation.get_data('numpy')
        
        # Check if the mouse cursor is within the valid bounds of the image data array.
        # This prevents IndexError if the mouse moves off the edge of the plot.
        if 0 <= y < data.shape[1] and 0 <= x < data.shape[2]:
            # Get the value of the pixel under the cursor for the current frame.
            pixel_value = data[self.frame_slider.value(), y, x]
            
            # Display the coordinates and value in the status bar.
            self.statusBar().showMessage(f"Pixel (x={x}, y={y}) | Value: {pixel_value:.2f}")

    def _unlock_for_new_data(self):
        """Unlock navigation panel to allow loading new data."""
        self._unlock_navigation_panel()
        self._lock_settings_panel()
        self.load_button.setText("Load Data")
        self.load_button.clicked.disconnect()
        self.load_button.clicked.connect(self.load_data)
        self.navigation_locked = False
        
        # Clear current data tracking
        self.current_backend = None
        self.current_instrument = None
        self.current_planet_id = None
        self.current_data_label.setText("Current Data: Instrument = None | Backend = None | Ground Truth: Not Available")
        
        # Clear observation and results
        self.observation = None
        if hasattr(self, 'bayesian_results'):
            self.bayesian_results = None
        if hasattr(self, 'baseline_results'):
            self.baseline_results = None
        
        # Clear all plots
        self._clear_all_plots()

    def _lock_navigation_panel(self):
        """Lock the navigation panel (disable all controls)."""
        self.split_combo.setEnabled(False)
        self.planet_combo.setEnabled(False)
        self.obs_combo.setEnabled(False)
        self.instrument_combo.setEnabled(False)
        self.backend_combo.setEnabled(False)
        # Don't disable the load button as it becomes the change button
    
    def _unlock_navigation_panel(self):
        """Unlock the navigation panel (enable all controls)."""
        self.split_combo.setEnabled(True)
        self.planet_combo.setEnabled(True)
        self.obs_combo.setEnabled(True)
        self.instrument_combo.setEnabled(True)
        self.backend_combo.setEnabled(True)
    
    def _lock_settings_panel(self):
        """Lock the settings panel (disable all controls)."""
        # Disable all settings controls
        for checkbox in self.calib_checkboxes.values():
            checkbox.setEnabled(False)
        self.detrend_model_combo.setEnabled(False)
        self.detrend_params_stack.setEnabled(False)
        self.mask_method_combo.setEnabled(False)
        self.bins_spinbox.setEnabled(False)
        self.pipeline_button.setEnabled(False)
        # Disable pipeline mode selection
        self.pipeline_mode_combo.setEnabled(False)
    
    def _unlock_settings_panel(self):
        """Unlock the settings panel (enable all controls)."""
        # Enable all settings controls
        for checkbox in self.calib_checkboxes.values():
            checkbox.setEnabled(True)
        self.detrend_model_combo.setEnabled(True)
        self.detrend_params_stack.setEnabled(True)
        self.mask_method_combo.setEnabled(True)
        self.bins_spinbox.setEnabled(True)
        self.pipeline_button.setEnabled(True)
        # Enable pipeline mode selection
        self.pipeline_mode_combo.setEnabled(True)
    
    def update_tab_visibility(self, selected_pipeline):
        """Show/hide tabs based on selected pipeline BEFORE execution."""
        # Convert pipeline name to mapping key
        pipeline_key = self._get_pipeline_key(selected_pipeline)
        
        for tab_name, supported_pipelines in self.TAB_PIPELINE_MAPPING.items():
            if tab_name in self.tab_indices:
                tab_index = self.tab_indices[tab_name]
                
                if "all" in supported_pipelines or pipeline_key in supported_pipelines:
                    self.tab_widget.setTabVisible(tab_index, True)
                else:
                    self.tab_widget.setTabVisible(tab_index, False)
    
    def _get_pipeline_key(self, pipeline_name):
        """Convert pipeline display name to mapping key."""
        if "Preprocessing" in pipeline_name:
            return "preprocessing"
        elif "Bayesian" in pipeline_name:
            return "bayesian"
        elif "CNN" in pipeline_name:
            return "cnn"
        elif "Diffusion" in pipeline_name:
            return "diffusion"
        elif "Baseline" in pipeline_name:
            return "baseline"
        else:
            return "unknown"
    
    def _get_wavelength_range_for_instrument(self, instrument):
        """Get the wavelength range for the given instrument from config."""
        if instrument in WAVELENGTH_RANGES:
            start_col, end_col = WAVELENGTH_RANGES[instrument]
            return start_col, end_col
        else:
            # Fallback: return full range if instrument not found in config
            return 0, 0  # Will be updated with actual data range
    
    def _convert_wavelength_index(self, config_index, data_shape):
        """Convert config-based wavelength index to actual data index."""
        if not hasattr(self, 'current_instrument'):
            return config_index
            
        start_col, end_col = self._get_wavelength_range_for_instrument(self.current_instrument)
        
        if self.current_instrument == 'AIRS-CH0' and start_col != end_col:
            # For AIRS-CH0, convert from config range (39-321) to data index (0-based)
            relative_index = config_index - start_col
            if relative_index < 0 or relative_index >= data_shape:
                relative_index = 0
            return relative_index
        else:
            # For other instruments or single wavelength, use the index directly
            if config_index >= data_shape:
                return 0
            return config_index
    


    def update_bayesian_plot(self):
        """Update the Bayesian results plot using the new plotter."""
        # Get checkbox states
        show_uncertainties = self.show_uncertainties_checkbox.isChecked()
        show_ground_truth = self.show_ground_truth_checkbox.isChecked()
        log_scale = self.log_scale_checkbox.isChecked()
        
        # Use the Bayesian plotter to handle the visualization
        self.bayesian_plotter.plot_results(
            bayesian_results=getattr(self, 'bayesian_results', None),
            observation=self.observation,
            show_uncertainties=show_uncertainties,
            show_ground_truth=show_ground_truth,
            log_scale=log_scale
        )
        
        # Handle covariance matrix separately if requested
        if (hasattr(self, 'bayesian_results') and self.bayesian_results and 
            self.show_covariance_checkbox.isChecked()):
            covariance = self.bayesian_results.get('covariance')
            if covariance is not None:
                CovariancePlotter.plot_covariance(covariance)
    
    def update_components_plot(self):
        """Update the component analysis plot."""
        if not hasattr(self, 'bayesian_results') or self.bayesian_results is None:
            # Clear all subplots
            for ax in self.components_ax:
                ax.clear()
                ax.set_title("No Data")
            self.components_canvas.draw()
            return
        
        # Check if we have fitted components
        if 'fitted_components' not in self.bayesian_results:
            # Clear all subplots and show message
            for ax in self.components_ax:
                ax.clear()
                ax.set_title("No Component Data")
            self.components_canvas.draw()
            return
        
        fitted_components = self.bayesian_results['fitted_components']
        
        # Plot each component in a separate subplot
        component_names = list(fitted_components.keys())
        for i, ax in enumerate(self.components_ax):
            ax.clear()
            
            if i < len(component_names):
                component_name = component_names[i]
                component_data = fitted_components[component_name]
                
                # Plot the component data
                if component_data is not None and len(component_data) > 0:
                    if component_name == 'stellar':
                        # Stellar spectrum: plot as wavelength-dependent
                        wavelengths = np.arange(len(component_data))
                        ax.plot(wavelengths, component_data, 'r-', linewidth=2)
                        ax.set_title(f"Stellar Spectrum")
                        ax.set_xlabel("Wavelength Index")
                        ax.set_ylabel("Flux")
                    elif component_name == 'drift':
                        # Drift model: plot as time series for first wavelength
                        if len(component_data.shape) > 1:
                            # Multi-wavelength drift, show first wavelength
                            time_points = np.arange(component_data.shape[0])
                            ax.plot(time_points, component_data[:, 0], 'g-', linewidth=2)
                            ax.set_title(f"Drift Model (Wavelength 0)")
                            ax.set_xlabel("Time Index")
                            ax.set_ylabel("Drift Factor")
                        else:
                            # Single wavelength drift
                            time_points = np.arange(len(component_data))
                            ax.plot(time_points, component_data, 'g-', linewidth=2)
                            ax.set_title(f"Drift Model")
                            ax.set_xlabel("Time Index")
                            ax.set_ylabel("Drift Factor")
                    elif component_name == 'transit_depth':
                        # Transit depths: plot as wavelength-dependent
                        wavelengths = np.arange(len(component_data))
                        ax.plot(wavelengths, component_data, 'b-', linewidth=2)
                        ax.set_title(f"Transit Depth Variation")
                        ax.set_xlabel("Wavelength Index")
                        ax.set_ylabel("Transit Depth")
                    elif component_name == 'transit_window':
                        # Transit window: plot as time series
                        time_points = np.arange(len(component_data))
                        ax.plot(time_points, component_data, 'purple', linewidth=2)
                        ax.set_title(f"Transit Window")
                        ax.set_xlabel("Time Index")
                        ax.set_ylabel("Transit Window")
                    elif component_name == 'noise':
                        # Noise levels: plot as wavelength-dependent
                        wavelengths = np.arange(len(component_data))
                        ax.plot(wavelengths, component_data, 'orange', linewidth=2)
                        ax.set_title(f"Noise Levels")
                        ax.set_xlabel("Wavelength Index")
                        ax.set_ylabel("Noise Level")
                    else:
                        # Generic component plotting
                        data_points = np.arange(len(component_data))
                        ax.plot(data_points, component_data, 'k-', linewidth=2)
                        ax.set_title(f"{component_name.replace('_', ' ').title()}")
                        ax.set_xlabel("Index")
                        ax.set_ylabel("Value")
                    
                    ax.grid(True, alpha=0.3)
                else:
                    ax.set_title(f"{component_name.replace('_', ' ').title()}")
                    ax.text(0.5, 0.5, "No data available", 
                           transform=ax.transAxes, ha='center', va='center', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            else:
                ax.set_title("No Component")
                ax.text(0.5, 0.5, "No component data", 
                       transform=ax.transAxes, ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        self.components_canvas.figure.tight_layout()
        self.components_canvas.draw()

    def keyPressEvent(self, event):
        """Handle keyboard navigation events."""
        # Only handle arrow keys
        if event.key() not in [Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down]:
            super().keyPressEvent(event)
            return
        
        # Check if navigation panel is locked - if so, don't allow keyboard navigation
        if self.navigation_locked:
            self.statusBar().showMessage("Navigation locked - click 'Change Data' to unlock", 2000)
            super().keyPressEvent(event)
            return
        
        # Determine step size based on Ctrl modifier
        step_size = self.keyboard_step_fine if event.modifiers() & Qt.KeyboardModifier.ControlModifier else self.keyboard_step_normal
        
        # Check for modifier keys to navigate different combo boxes
        shift_pressed = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        alt_pressed = event.modifiers() & Qt.KeyboardModifier.AltModifier
        
        # Map arrow keys to combo box navigation
        if event.key() == Qt.Key.Key_Left:
            if alt_pressed:
                self._navigate_combo_box(self.backend_combo, -step_size)
            elif shift_pressed:
                self._navigate_combo_box(self.split_combo, -step_size)
            else:
                self._navigate_combo_box(self.planet_combo, -step_size)
        elif event.key() == Qt.Key.Key_Right:
            if alt_pressed:
                self._navigate_combo_box(self.backend_combo, step_size)
            elif shift_pressed:
                self._navigate_combo_box(self.split_combo, step_size)
            else:
                self._navigate_combo_box(self.planet_combo, step_size)
        elif event.key() == Qt.Key.Key_Up:
            if shift_pressed:
                self._navigate_combo_box(self.instrument_combo, -step_size)
            else:
                self._navigate_combo_box(self.obs_combo, -step_size)
        elif event.key() == Qt.Key.Key_Down:
            if shift_pressed:
                self._navigate_combo_box(self.instrument_combo, step_size)
            else:
                self._navigate_combo_box(self.obs_combo, step_size)
        
        # Don't call super() for arrow keys as we've handled them
        
    def _navigate_combo_box(self, combo_box, step):
        """Navigate a combo box by the specified step amount."""
        if not combo_box.isEnabled() or combo_box.count() == 0:
            combo_name = combo_box.objectName() or 'combo'
            self.statusBar().showMessage(f"{combo_name} navigation not available (disabled or empty)", 2000)
            return
        
        current_index = combo_box.currentIndex()
        new_index = current_index + step
        
        # Clamp to valid range
        new_index = max(0, min(new_index, combo_box.count() - 1))
        
        if new_index != current_index:
            combo_box.setCurrentIndex(new_index)
            # Show feedback in status bar
            ctrl_text = " (fine)" if step == self.keyboard_step_fine else ""
            combo_name = combo_box.objectName() or 'combo'
            self.statusBar().showMessage(
                f"Keyboard navigation{ctrl_text}: {combo_name} → {combo_box.currentText()}", 
                3000
            )
        else:
            # At boundary, show helpful message
            boundary = "start" if new_index == 0 else "end"
            combo_name = combo_box.objectName() or 'combo'
            self.statusBar().showMessage(f"{combo_name} at {boundary} of list", 1500)

    def update_pipeline_mode_options(self):
        """Update pipeline mode options based on instrument selection."""
        current_instrument = self.current_instrument if self.current_instrument else self.instrument_combo.currentText()
        
        # Store current selection if any
        current_selection = self.pipeline_mode_combo.currentText()
        
        # Clear current options
        self.pipeline_mode_combo.clear()
        
        if current_instrument == "Both":
            # Only show Baseline Pipeline for both instruments
            self.pipeline_mode_combo.addItems(["Baseline Pipeline"])
            # Set to Baseline Pipeline
            self.pipeline_mode_combo.setCurrentText("Baseline Pipeline")
        else:
            # Show all pipeline options for single instrument
            self.pipeline_mode_combo.addItems(["Preprocessing Pipeline", "Bayesian Pipeline"])
            # Restore previous selection if it was valid
            if current_selection in ["Preprocessing Pipeline", "Bayesian Pipeline"]:
                self.pipeline_mode_combo.setCurrentText(current_selection)
            else:
                self.pipeline_mode_combo.setCurrentText("Preprocessing Pipeline")
        
        # Trigger the change handler to update tab visibility
        self.on_pipeline_mode_change(self.pipeline_mode_combo.currentText())

class BaselinePlotter:
    """Plotter for baseline pipeline results."""
    
    def __init__(self, ax, canvas):
        self.ax = ax
        self.canvas = canvas
    
    def plot_results(self, baseline_results, observation, show_uncertainties=True, 
                    show_ground_truth=True, log_scale=False):
        """Plot baseline pipeline results."""
        if not baseline_results:
            self.ax.clear()
            self.ax.set_title("No Baseline Results Available")
            self.ax.set_xlabel("Wavelength (μm)")
            self.ax.set_ylabel("Transit Depth")
            self.canvas.draw()
            return
        
        self.ax.clear()
        
        # Extract results
        predictions = baseline_results['predictions']
        uncertainties = baseline_results['uncertainties']
        
        # Get wavelength array
        if observation and observation.has_wavelengths():
            wavelengths = observation.get_wavelengths()
            print(f"DEBUG: Using observation wavelengths: {len(wavelengths)} points, range {wavelengths.min():.3f}-{wavelengths.max():.3f} μm")
        else:
            wavelengths = np.arange(len(predictions))
            print(f"DEBUG: Using index-based wavelengths: {len(wavelengths)} points")
        
        # Plot predictions
        print(f"DEBUG: Baseline predictions: {len(predictions)} points, range {predictions.min():.6f}-{predictions.max():.6f}")
        self.ax.plot(wavelengths, predictions, 'b-', linewidth=2, alpha=0.8, label='Baseline Predictions')
        
        # Show uncertainties if requested
        if show_uncertainties and uncertainties is not None:
            self.ax.fill_between(wavelengths, 
                               predictions - uncertainties, 
                               predictions + uncertainties, 
                               alpha=0.3, color='blue', label='±1σ Uncertainty')
        
        # Show ground truth if available and requested
        if observation and observation.has_ground_truth() and show_ground_truth:
            ground_truth = observation.get_ground_truth()
            print(f"DEBUG: Ground truth: {len(ground_truth)} points, range {ground_truth.min():.6f}-{ground_truth.max():.6f}")
            
            # Use the same wavelength array for consistency
            if len(wavelengths) >= len(ground_truth):
                gt_wavelengths = wavelengths[:len(ground_truth)]
            else:
                ground_truth = ground_truth[:len(wavelengths)]
                gt_wavelengths = wavelengths
            
            print(f"DEBUG: Ground truth wavelengths: {len(gt_wavelengths)} points, range {gt_wavelengths.min():.3f}-{gt_wavelengths.max():.3f} μm")
            self.ax.plot(gt_wavelengths, ground_truth, 'r--', linewidth=3, alpha=0.9, label='Ground Truth')
            
            # Calculate and display metrics
            metrics = self._calculate_metrics(predictions[:len(ground_truth)], ground_truth)
            self._add_metrics_text(metrics)
        
        # Apply log scale if requested
        if log_scale:
            self.ax.set_yscale('log')
            # Set reasonable log scale limits
            all_values = np.concatenate([predictions, ground_truth if observation and observation.has_ground_truth() else []])
            min_val = np.min(all_values[all_values > 0])  # Avoid log(0)
            max_val = np.max(all_values)
            self.ax.set_ylim(min_val * 0.5, max_val * 2)
        
        # Format the plot
        self.ax.set_title("Baseline Pipeline Results")
        if observation and observation.has_wavelengths():
            self.ax.set_xlabel("Wavelength (μm)")
        else:
            self.ax.set_xlabel("Wavelength Index")
        self.ax.set_ylabel("Transit Depth")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        # Add pipeline info
        if 'pipeline_params' in baseline_results:
            pipeline_params = baseline_results['pipeline_params']
            info_text = "Pipeline Info:\n"
            info_text += f"  Instrument: {pipeline_params.get('instrument', 'Unknown')}\n"
            info_text += f"  Scale: {pipeline_params.get('scale', 0.95):.3f}\n"
            info_text += f"  Sigma: {pipeline_params.get('sigma', 0.0009):.6f}\n"
            info_text += f"  Optimized Depth: {baseline_results.get('optimized_transit_depth', 0):.6f}"
            self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                        verticalalignment='top', fontsize=8, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.canvas.draw()
    
    def _calculate_metrics(self, predictions, ground_truth):
        """Calculate metrics between predictions and ground truth."""
        if len(predictions) != len(ground_truth):
            return {}
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Calculate metrics
        mse = np.mean((predictions - ground_truth) ** 2)
        rms_error = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - ground_truth))
        
        # Calculate correlation coefficient
        if len(predictions) > 1:
            correlation = np.corrcoef(predictions, ground_truth)[0, 1]
        else:
            correlation = 1.0 if predictions[0] == ground_truth[0] else 0.0
        
        return {
            'rms_error': rms_error,
            'mae': mae,
            'correlation': correlation
        }
    
    def _add_metrics_text(self, metrics):
        """Add metrics text to the plot."""
        if not metrics:
            return
        
        metrics_text = f"Metrics:\n"
        metrics_text += f"RMSE: {metrics['rms_error']:.4f}\n"
        metrics_text += f"MAE: {metrics['mae']:.4f}\n"
        metrics_text += f"R²: {metrics['correlation']:.4f}"
        
        self.ax.text(0.02, 0.02, metrics_text, transform=self.ax.transAxes, 
                    verticalalignment='bottom', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    def run_baseline_pipeline(self):
        """Run the baseline pipeline."""
        if not self.observation:
            self.statusBar().showMessage("Please load data first.", 5000)
            return
        
        # Check backend compatibility
        selected_backend = self.backend_combo.currentText()
        if self.current_backend and self.current_backend != selected_backend:
            self.statusBar().showMessage(
                f"Backend mismatch! Data loaded on {self.current_backend}, but {selected_backend} selected. "
                "Please reload data with the correct backend.", 10000
            )
            return
        
        # Set UI to busy state
        self._set_ui_busy(True, "baseline_pipeline")
        
        # Get pipeline parameters
        pipeline_params = {
            'instrument': self.current_instrument,
            'use_gpu': self.current_backend == 'gpu',
            'scale': 0.95,  # Default baseline parameters
            'sigma': 0.0009,
            'cut_inf': 39,
            'cut_sup': 321,
            'binning': 30,
            'phase_detection_slice': (30, 140),  # Tuple instead of slice for serialization
            'optimization_delta': 7,
            'polynomial_degree': 3
        }
        
        # Run baseline pipeline asynchronously
        self._run_baseline_pipeline_async(pipeline_params)
    
    def _run_baseline_pipeline_async(self, pipeline_params):
        """Run baseline pipeline asynchronously with progress tracking."""
        # Create and start worker
        self.baseline_worker = BaselinePipelineWorker(self.observation, pipeline_params)
        self.baseline_worker.progress_signal.connect(self._on_baseline_pipeline_progress)
        self.baseline_worker.finished_signal.connect(self._on_baseline_pipeline_finished)
        self.baseline_worker.error_signal.connect(self._on_baseline_pipeline_error)
        self.baseline_worker.start()
    
    def _on_baseline_pipeline_progress(self, message: str):
        """Handle progress updates from the baseline pipeline worker."""
        # Update progress label
        self.progress_label.setText(message)
        
        # Update status bar
        self.statusBar().showMessage(message, 1000)
        
        # Update progress bar
        if "preprocessing" in message.lower():
            self.progress_bar.setValue(30)
        elif "phase detection" in message.lower():
            self.progress_bar.setValue(60)
        elif "optimization" in message.lower():
            self.progress_bar.setValue(80)
    
    def _on_baseline_pipeline_finished(self, results):
        """Handle completion of the baseline pipeline."""
        try:
            # Update progress for plotting
            self.progress_bar.setValue(95)
            self.progress_label.setText("Updating plots...")
            
            # Store results
            self.baseline_results = results
            
            # Update baseline plots
            self.update_baseline_plot()
            
            # Complete
            self.progress_bar.setValue(100)
            self.progress_label.setText("Baseline Pipeline completed")
            
            # Set UI to idle state
            self._set_ui_busy(False)
            
            self.statusBar().showMessage("Baseline Pipeline finished.", 5000)
            
        except Exception as e:
            self._on_baseline_pipeline_error(str(e))
        finally:
            # Clean up worker
            if hasattr(self, 'baseline_worker') and self.baseline_worker:
                self.baseline_worker.deleteLater()
                self.baseline_worker = None
    
    def _on_baseline_pipeline_error(self, error_message: str):
        """Handle errors during baseline pipeline execution."""
        # Set UI to idle state
        self._set_ui_busy(False)
        
        # Show error (but not for user stops)
        if "stopped by user" not in error_message.lower():
            self.statusBar().showMessage(f"ERROR: {error_message}", 15000)
            print(f"ERROR: {error_message}")
        else:
            self.statusBar().showMessage("Baseline Pipeline stopped by user", 5000)
        
        # Clean up worker
        if hasattr(self, 'baseline_worker') and self.baseline_worker:
            self.baseline_worker.deleteLater()
            self.baseline_worker = None

    def update_baseline_plot(self):
        """Update the baseline results plot using the baseline plotter."""
        # Get checkbox states
        show_uncertainties = self.baseline_show_uncertainties_checkbox.isChecked()
        show_ground_truth = self.baseline_show_ground_truth_checkbox.isChecked()
        log_scale = self.baseline_log_scale_checkbox.isChecked()
        
        # Get the appropriate observation for ground truth
        observation = None
        if isinstance(self.observation, dict):
            # Both instruments loaded - use AIRS-CH0 for ground truth (more wavelengths)
            observation = self.observation.get("AIRS-CH0")
        else:
            # Single instrument loaded
            observation = self.observation
        
        # Use the baseline plotter to handle the visualization
        self.baseline_plotter.plot_results(
            baseline_results=getattr(self, 'baseline_results', None),
            observation=observation,
            show_uncertainties=show_uncertainties,
            show_ground_truth=show_ground_truth,
            log_scale=log_scale
        )

    def update_components_plot(self):
        """Update the component analysis plot."""
        if not hasattr(self, 'bayesian_results') or self.bayesian_results is None:
            # Clear all subplots
            for ax in self.components_ax:
                ax.clear()
                ax.set_title("No Data")
            self.components_canvas.draw()
            return
        
        # Check if we have fitted components
        if 'fitted_components' not in self.bayesian_results:
            # Clear all subplots and show message
            for ax in self.components_ax:
                ax.clear()
                ax.set_title("No Component Data")
            self.components_canvas.draw()
            return
        
        fitted_components = self.bayesian_results['fitted_components']
        
        # Plot each component in a separate subplot
        component_names = list(fitted_components.keys())
        for i, ax in enumerate(self.components_ax):
            ax.clear()
            
            if i < len(component_names):
                component_name = component_names[i]
                component_data = fitted_components[component_name]
                
                # Plot the component data
                if component_data is not None and len(component_data) > 0:
                    if component_name == 'stellar':
                        # Stellar spectrum: plot as wavelength-dependent
                        wavelengths = np.arange(len(component_data))
                        ax.plot(wavelengths, component_data, 'r-', linewidth=2)
                        ax.set_title(f"Stellar Spectrum")
                        ax.set_xlabel("Wavelength Index")
                        ax.set_ylabel("Flux")
                    elif component_name == 'drift':
                        # Drift model: plot as time series for first wavelength
                        if len(component_data.shape) > 1:
                            # Multi-wavelength drift, show first wavelength
                            time_points = np.arange(component_data.shape[0])
                            ax.plot(time_points, component_data[:, 0], 'g-', linewidth=2)
                            ax.set_title(f"Drift Model (Wavelength 0)")
                            ax.set_xlabel("Time Index")
                            ax.set_ylabel("Drift Factor")
                        else:
                            # Single wavelength drift
                            time_points = np.arange(len(component_data))
                            ax.plot(time_points, component_data, 'g-', linewidth=2)
                            ax.set_title(f"Drift Model")
                            ax.set_xlabel("Time Index")
                            ax.set_ylabel("Drift Factor")
                    elif component_name == 'transit_depth':
                        # Transit depths: plot as wavelength-dependent
                        wavelengths = np.arange(len(component_data))
                        ax.plot(wavelengths, component_data, 'b-', linewidth=2)
                        ax.set_title(f"Transit Depth Variation")
                        ax.set_xlabel("Wavelength Index")
                        ax.set_ylabel("Transit Depth")
                    elif component_name == 'transit_window':
                        # Transit window: plot as time series
                        time_points = np.arange(len(component_data))
                        ax.plot(time_points, component_data, 'purple', linewidth=2)
                        ax.set_title(f"Transit Window")
                        ax.set_xlabel("Time Index")
                        ax.set_ylabel("Transit Window")
                    elif component_name == 'noise':
                        # Noise levels: plot as wavelength-dependent
                        wavelengths = np.arange(len(component_data))
                        ax.plot(wavelengths, component_data, 'orange', linewidth=2)
                        ax.set_title(f"Noise Levels")
                        ax.set_xlabel("Wavelength Index")
                        ax.set_ylabel("Noise Level")
                    else:
                        # Generic component plotting
                        data_points = np.arange(len(component_data))
                        ax.plot(data_points, component_data, 'k-', linewidth=2)
                        ax.set_title(f"{component_name.replace('_', ' ').title()}")
                        ax.set_xlabel("Index")
                        ax.set_ylabel("Value")
                    
                    ax.grid(True, alpha=0.3)
                else:
                    ax.set_title(f"{component_name.replace('_', ' ').title()}")
                    ax.text(0.5, 0.5, "No data available", 
                           transform=ax.transAxes, ha='center', va='center', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            else:
                ax.set_title("No Component")
                ax.text(0.5, 0.5, "No component data", 
                       transform=ax.transAxes, ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        self.components_canvas.figure.tight_layout()
        self.components_canvas.draw()

class BaselinePipelineWorker(QThread):
    """Worker thread for running the baseline pipeline asynchronously."""
    
    progress_signal = pyqtSignal(str)  # Simple string messages for pipeline steps
    finished_signal = pyqtSignal(object)  # Emit the pipeline results
    error_signal = pyqtSignal(str)
    
    def __init__(self, observation, pipeline_params):
        super().__init__()
        self.observation = observation
        self.pipeline_params = pipeline_params
        self._should_stop = False
    
    def run(self):
        """Run the baseline pipeline in a separate thread."""
        try:
            print("DEBUG: Starting baseline pipeline in worker thread")
            
            # Import baseline pipeline
            from arielml.pipelines.baseline_pipeline import BaselinePipeline
            
            # Create baseline pipeline
            self.progress_signal.emit("Creating baseline pipeline...")
            self.check_stop()
            
            pipeline = BaselinePipeline(**self.pipeline_params)
            
            # Handle both single and multiple observations
            if isinstance(self.observation, dict):
                # Both instruments loaded
                self.progress_signal.emit("Processing both instruments...")
                self.check_stop()
                
                # Fit the pipeline with both observations
                pipeline.fit(self.observation)
                
                # Make predictions
                self.progress_signal.emit("Making predictions...")
                self.check_stop()
                
                predictions, uncertainties = pipeline.predict(self.observation)
                
            else:
                # Single instrument loaded
                self.progress_signal.emit("Processing single instrument...")
                self.check_stop()
                
                # Fit the pipeline
                pipeline.fit(self.observation)
                
                # Make predictions
                self.progress_signal.emit("Making predictions...")
                self.check_stop()
                
                predictions, uncertainties = pipeline.predict(self.observation)
            
            # Store results
            results = {
                'predictions': predictions,
                'uncertainties': uncertainties,
                'pipeline_params': self.pipeline_params,
                'optimized_transit_depth': getattr(pipeline, 'optimized_transit_depth', None)
            }
            
            # Final check before emitting success
            self.check_stop()
            
            print("DEBUG: Baseline pipeline completed successfully")
            self.finished_signal.emit(results)
            
        except InterruptedError:
            print("DEBUG: Baseline pipeline interrupted by user")
            # Operation was stopped by user - don't emit error
            pass
        except Exception as e:
            print(f"DEBUG: Baseline pipeline error: {e}")
            self.error_signal.emit(str(e))
    
    def check_stop(self):
        """Check if a stop has been requested."""
        if self._should_stop:
            raise InterruptedError("Baseline pipeline stopped by user")
    
    def stop(self):
        """Request the worker to stop."""
        self._should_stop = True

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application icon
    icon_path = Path(__file__).parent.parent / "arielml" / "assets" / "ESA_Ariel_official_mission_patch.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    else:
        print(f"Warning: Application icon not found at {icon_path}")
    
    inspector = DataInspector()
    inspector.show()
    sys.exit(app.exec())