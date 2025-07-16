# arielml/data/detrending.py

import numpy as np
from abc import ABC, abstractmethod
from scipy import signal as scipy_signal
from scipy.optimize import minimize
from typing import Tuple
import pickle
from pathlib import Path

# Standard GP library (CPU-based)
import george
from george import kernels

# GPU-accelerated GP libraries (if available)
try:
    import torch
    import gpytorch
    GP_GPU_ENABLED = True
except ImportError:
    GP_GPU_ENABLED = False

# Progress tracking
from ..utils.observable import Observable
from ..utils.signals import DetrendingStep, DetrendingProgress



# KISS-GP implementation for 2D spectral drift
class KISSGP2D:
    """
    KISS-GP implementation for 2D Gaussian Processes.
    
    This class implements the KISS-GP approximation for 2D GPs by:
    1. Creating a regular 2D grid of inducing points
    2. Using bilinear interpolation to map between data points and grid
    3. Delegating the actual GP computation to an internal model
    
    This makes 2D GP computation much faster than dense methods.
    """
    
    def __init__(self, 
                 h: Tuple[float, float] = (0.4, 0.05),
                 features: Tuple[str, str] = ('time', 'wavelength'),
                 use_sparse: bool = True):
        """
        Initialize KISS-GP 2D approximation.
        
        Args:
            h: Grid resolution [hx, hy] for the 2D grid
            features: Names of the two features to use for 2D GP
            use_sparse: Whether to use sparse matrix operations
        """
        self.h = np.array(h)
        self.features = features
        self.use_sparse = use_sparse
        
        # Grid and interpolation matrices (set during initialization)
        self.grid_x = None
        self.grid_y = None
        self.transform_matrix = None
        self.n_grid_points = None
        
        # Internal model (set externally)
        self.internal_model = None
        
    def create_grid(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a regular 2D grid based on the data range.
        
        Args:
            x: First feature values (e.g., time)
            y: Second feature values (e.g., wavelength)
            
        Returns:
            grid_x, grid_y: Regular grid coordinates
        """
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        
        # Extend grid slightly beyond data range
        xt = np.arange(min_x - self.h[0], max_x + 2*self.h[0], self.h[0])
        yt = np.arange(min_y - self.h[1], max_y + 2*self.h[1], self.h[1])
        
        return xt, yt
    
    def compute_interpolation_matrix(self, 
                                   x: np.ndarray, 
                                   y: np.ndarray,
                                   grid_x: np.ndarray,
                                   grid_y: np.ndarray):
        """
        Compute the bilinear interpolation matrix.
        
        This matrix maps from the regular grid to the actual data points
        using bilinear interpolation.
        
        Args:
            x: Data x-coordinates
            y: Data y-coordinates  
            grid_x: Grid x-coordinates
            grid_y: Grid y-coordinates
            
        Returns:
            Sparse interpolation matrix
        """
        n_data = len(x)
        n_grid_x = len(grid_x)
        n_grid_y = len(grid_y)
        n_grid = n_grid_x * n_grid_y
        
        # Grid spacing
        hx = grid_x[1] - grid_x[0]
        hy = grid_y[1] - grid_y[0]
        
        # Find grid indices for each data point
        ind_x_left = np.floor((x - grid_x[0]) / hx).astype(int)
        ind_y_top = np.floor((y - grid_y[0]) / hy).astype(int)
        
        # Clamp indices to valid range
        ind_x_left = np.clip(ind_x_left, 0, n_grid_x - 2)
        ind_y_top = np.clip(ind_y_top, 0, n_grid_y - 2)
        
        # Interpolation weights
        scale_x = 1 - (x - grid_x[ind_x_left]) / hx
        scale_y = 1 - (y - grid_y[ind_y_top]) / hy
        
        # Grid indices for the 4 corners
        ind_top_left = ind_y_top + n_grid_y * ind_x_left
        
        # Create sparse matrix entries
        rows = np.arange(n_data)
        
        # Top-left corner
        cols1 = ind_top_left
        vals1 = scale_x * scale_y
        
        # Top-right corner  
        cols2 = ind_top_left + n_grid_y
        vals2 = (1 - scale_x) * scale_y
        
        # Bottom-left corner
        cols3 = ind_top_left + 1
        vals3 = scale_x * (1 - scale_y)
        
        # Bottom-right corner
        cols4 = ind_top_left + n_grid_y + 1
        vals4 = (1 - scale_x) * (1 - scale_y)
        
        # Combine all entries
        all_rows = np.concatenate([rows, rows, rows, rows])
        all_cols = np.concatenate([cols1, cols2, cols3, cols4])
        all_vals = np.concatenate([vals1, vals2, vals3, vals4])
        
        # Create sparse matrix
        from scipy.sparse import csr_matrix
        transform_matrix = csr_matrix(
            (all_vals, (all_rows, all_cols)), 
            shape=(n_data, n_grid)
        )
        
        return transform_matrix
    
    def initialize(self, x: np.ndarray, y: np.ndarray):
        """
        Initialize the KISS-GP grid and interpolation matrix.
        
        Args:
            x: First feature values
            y: Second feature values
        """
        # Create grid
        self.grid_x, self.grid_y = self.create_grid(x, y)
        self.n_grid_points = len(self.grid_x) * len(self.grid_y)
        
        # Compute interpolation matrix
        self.transform_matrix = self.compute_interpolation_matrix(
            x, y, self.grid_x, self.grid_y
        )
    
    def get_grid_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the full grid coordinates as 2D arrays.
        
        Returns:
            grid_x_2d, grid_y_2d: 2D coordinate arrays
        """
        if self.grid_x is None or self.grid_y is None:
            raise ValueError("KISS-GP not initialized. Call initialize() first.")
        
        grid_x_2d, grid_y_2d = np.meshgrid(self.grid_x, self.grid_y, indexing='ij')
        return grid_x_2d, grid_y_2d
    
    def get_grid_points_1d(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get grid points as 1D arrays for GP fitting.
        
        Returns:
            grid_x_1d, grid_y_1d: 1D coordinate arrays
        """
        if self.grid_x is None or self.grid_y is None:
            raise ValueError("KISS-GP not initialized. Call initialize() first.")
        
        grid_x_2d, grid_y_2d = self.get_grid_coordinates()
        grid_x_1d = grid_x_2d.flatten()
        grid_y_1d = grid_y_2d.flatten()
        
        return grid_x_1d, grid_y_1d
    
    def interpolate_predictions(self, grid_predictions: np.ndarray) -> np.ndarray:
        """
        Interpolate predictions from grid to data points.
        
        Args:
            grid_predictions: Predictions on the grid points
            
        Returns:
            Interpolated predictions at data points
        """
        if self.transform_matrix is None:
            raise ValueError("KISS-GP not initialized. Call initialize() first.")
        
        # Apply interpolation matrix
        data_predictions = self.transform_matrix @ grid_predictions
        
        return data_predictions


class BaseDetrender(Observable, ABC):
    """Abstract base class for all detrending models with progress tracking."""
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def detrend(self, time, flux, transit_mask, xp) -> Tuple["xp.ndarray", "xp.ndarray"]:
        pass

class PolynomialDetrender(BaseDetrender):
    """Fits and removes a polynomial trend."""
    def __init__(self, degree: int = 2):
        super().__init__()
        self.degree = degree

    def detrend(self, time, flux, transit_mask, xp):
        noise_model = xp.full_like(flux, xp.nan)
        for i in range(flux.shape[1]):
            lc, finite_mask = flux[:, i], xp.isfinite(flux[:, i])
            oot_mask = ~transit_mask & finite_mask
            if xp.sum(oot_mask) < self.degree + 1:
                raise ValueError(f"Not enough OOT points for Polynomial fit on channel {i}.")
            poly_coeffs = xp.polyfit(time[oot_mask], lc[oot_mask], self.degree)
            noise_model[:, i] = xp.polyval(poly_coeffs, time)
        
        return flux / noise_model, noise_model

class SavGolDetrender(BaseDetrender):
    """Uses a Savitzky-Golay filter to smooth the data."""
    def __init__(self, window_length: int, polyorder: int):
        super().__init__()
        self.window_length, self.polyorder = window_length, polyorder

    def detrend(self, time, flux, transit_mask, xp):
        is_gpu = (xp.__name__ == 'cupy')
        time_np, flux_np, transit_mask_np = (d.get() if is_gpu else d for d in (time, flux, transit_mask))
        noise_model_np = np.full_like(flux_np, np.nan)
        for i in range(flux_np.shape[1]):
            lc, finite_mask = flux_np[:, i], np.isfinite(flux_np[:, i])
            oot_mask = ~transit_mask_np & finite_mask
            if np.sum(oot_mask) < self.window_length:
                raise ValueError(f"Not enough OOT points for SavGol fit on channel {i}.")
            oot_time, oot_flux = time_np[oot_mask], lc[oot_mask]
            sort_indices = np.argsort(oot_time)
            smoothed_oot_flux = scipy_signal.savgol_filter(oot_flux[sort_indices], self.window_length, self.polyorder)
            noise_model_np[:, i] = np.interp(time_np, oot_time[sort_indices], smoothed_oot_flux)
        noise_model = xp.asarray(noise_model_np) if is_gpu else noise_model_np
        return flux / noise_model, noise_model

class GPDetrender(BaseDetrender):
    """CPU-based Gaussian Process detrender using the 'george' library."""
    def __init__(self, kernel: str = 'Matern32'):
        super().__init__()
        self.kernel_name = kernel

    def _get_kernel(self):
        if self.kernel_name == 'Matern32': return kernels.Matern32Kernel(metric=1.0**2)
        raise ValueError(f"Unknown kernel: {self.kernel_name}")

    def detrend(self, time, flux, transit_mask, xp):
        # Reset stop flag for new operation
        self.reset_stop_flag()
        
        # Notify initialization
        self.notify_observers(DetrendingProgress(
            step=DetrendingStep.INITIALIZING,
            progress=0.0,
            message="Initializing GP detrending...",
            total_wavelengths=flux.shape[1]
        ))
        
        is_gpu = (xp.__name__ == 'cupy')
        time_np, flux_np, transit_mask_np = (d.get() if is_gpu else d for d in (time, flux, transit_mask))
        noise_model_np = np.full_like(flux_np, np.nan)
        
        total_wavelengths = flux_np.shape[1]
        
        for i in range(total_wavelengths):
            # Check for stop request
            self.check_stop_request()
            
            # Update progress
            progress = (i + 1) / total_wavelengths
            self.notify_observers(DetrendingProgress(
                step=DetrendingStep.FITTING_PER_WAVELENGTH,
                progress=progress,
                message=f"Fitting GP for wavelength {i+1}/{total_wavelengths}",
                current_wavelength=i,
                total_wavelengths=total_wavelengths
            ))
            
            lc, median_val = flux_np[:, i], np.nanmedian(flux_np[:, i])
            if np.isnan(median_val) or median_val == 0:
                noise_model_np[:, i] = 1.0
                continue
                
            lc_norm = lc / median_val
            finite_mask = np.isfinite(lc_norm)
            oot_mask = ~transit_mask_np & finite_mask
            if np.sum(oot_mask) < 20:
                raise ValueError(f"Not enough OOT points for GP fit on channel {i}.")
                
            oot_time, oot_flux = time_np[oot_mask], lc_norm[oot_mask]
            try:
                kernel = self._get_kernel()
                gp = george.GP(kernel, mean=np.nanmean(oot_flux), fit_mean=True, 
                             white_noise=np.log(np.nanstd(oot_flux)**2), fit_white_noise=True)
                gp.compute(oot_time)
                p0 = gp.get_parameter_vector()
                results = minimize(lambda p: -gp.log_likelihood(oot_flux, p), p0, 
                                 jac=lambda p: -gp.grad_log_likelihood(oot_flux, p))
                gp.set_parameter_vector(results.x)
                residuals = oot_flux - gp.get_parameter('mean:value')
                pred_noise = gp.predict(residuals, time_np, return_cov=False)
                noise_model_np[:, i] = (pred_noise + gp.get_parameter('mean:value')) * median_val
            except Exception as e:
                raise RuntimeError(f"George GP failed for channel {i}: {e}")
        
        # Notify completion
        self.notify_observers(DetrendingProgress(
            step=DetrendingStep.FINALIZING,
            progress=1.0,
            message="GP detrending completed"
        ))
        
        noise_model = xp.asarray(noise_model_np) if is_gpu else noise_model_np
        return flux / noise_model, noise_model

if GP_GPU_ENABLED:
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, kernel_type='Matern32', length_scale=1.0):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            
            # Create kernel based on type
            if kernel_type == 'Matern32':
                base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, lengthscale=length_scale)
            elif kernel_type == 'RBF':
                base_kernel = gpytorch.kernels.RBFKernel(lengthscale=length_scale)
            else:
                base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, lengthscale=length_scale)
            
            self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    # Simplified approach: Use individual GPs in parallel instead of complex multi-output model

    class GPyTorchDetrender(BaseDetrender):
        """GPU-accelerated Gaussian Process detrender using GPyTorch."""
        def __init__(self, training_iter=50):
            super().__init__()
            self.training_iter = training_iter

        def detrend(self, time, flux, transit_mask, xp):
            # Reset stop flag for new operation
            self.reset_stop_flag()
            
            # Notify initialization
            self.notify_observers(DetrendingProgress(
                step=DetrendingStep.INITIALIZING,
                progress=0.0,
                message="Initializing GPyTorch detrending...",
                total_wavelengths=flux.shape[1]
            ))
            
            is_gpu = (xp.__name__ == 'cupy')
            time_np, flux_np, transit_mask_np = (d.get() if is_gpu else d for d in (time, flux, transit_mask))
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            noise_model_np = np.full_like(flux_np, np.nan)
            total_wavelengths = flux_np.shape[1]

            for i in range(total_wavelengths):
                # Check for stop request
                self.check_stop_request()
                
                # Update progress
                progress = (i + 1) / total_wavelengths
                self.notify_observers(DetrendingProgress(
                    step=DetrendingStep.FITTING_PER_WAVELENGTH,
                    progress=progress,
                    message=f"Fitting GPyTorch GP for wavelength {i+1}/{total_wavelengths}",
                    current_wavelength=i,
                    total_wavelengths=total_wavelengths
                ))
                
                lc, median_val = flux_np[:, i], np.nanmedian(flux_np[:, i])
                if np.isnan(median_val) or median_val == 0:
                    noise_model_np[:, i] = 1.0
                    continue
                
                lc_norm = lc / median_val
                finite_mask = np.isfinite(lc_norm)
                oot_mask = ~transit_mask_np & finite_mask

                if np.sum(oot_mask) < 20:
                    raise ValueError(f"Not enough OOT points for GPyTorch fit on channel {i}.")

                train_x = torch.from_numpy(time_np[oot_mask]).to(device, dtype=torch.float32)
                train_y = torch.from_numpy(lc_norm[oot_mask]).to(device, dtype=torch.float32)
                
                likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
                model = ExactGPModel(train_x, train_y, likelihood).to(device)

                model.train(); likelihood.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

                # Training with iteration-level progress tracking
                for iter_idx in range(self.training_iter):
                    # Check for stop request
                    self.check_stop_request()
                    
                    # Update iteration progress
                    iter_progress = iter_idx / self.training_iter
                    wavelength_progress = i / total_wavelengths
                    total_progress = wavelength_progress + (iter_progress / total_wavelengths)
                    
                    self.notify_observers(DetrendingProgress(
                        step=DetrendingStep.FITTING_PER_WAVELENGTH,
                        progress=total_progress,
                        message=f"Fitting GPyTorch GP for wavelength {i+1}/{total_wavelengths} - Iteration {iter_idx+1}/{self.training_iter}",
                        current_wavelength=i,
                        total_wavelengths=total_wavelengths
                    ))
                    
                    optimizer.zero_grad()
                    output = model(train_x)
                    loss = -mll(output, train_y)
                    loss.backward()
                    optimizer.step()
                
                # Clean up training tensors immediately
                del optimizer, mll, output, loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                
                model.eval(); likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    test_x = torch.from_numpy(time_np).to(device, dtype=torch.float32)
                    observed_pred = likelihood(model(test_x))
                    pred_mean = observed_pred.mean.cpu().numpy()

                noise_model_np[:, i] = pred_mean * median_val
                
                # Clean up model and tensors after each wavelength
                del model, likelihood, train_x, train_y, test_x, observed_pred, pred_mean
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # Notify completion
            self.notify_observers(DetrendingProgress(
                step=DetrendingStep.FINALIZING,
                progress=1.0,
                message="GPyTorch detrending completed"
            ))

            noise_model = xp.asarray(noise_model_np) if is_gpu else noise_model_np
            return flux / noise_model, noise_model

class HybridDetrender(BaseDetrender):
    """
    Implements the hybrid detrending approach:
    1. Models the common-mode noise across all wavelengths with a GP.
    2. Models the remaining per-wavelength residuals with a simple Polynomial.
    """
    def __init__(self, use_gpu=False, training_iter=50, poly_degree=2):
        super().__init__()
        self.use_gpu = use_gpu
        self.training_iter = training_iter
        self.poly_degree = poly_degree
        if self.use_gpu and not GP_GPU_ENABLED:
            raise RuntimeError("HybridDetrender(use_gpu=True) requires a CUDA-enabled PyTorch installation.")

    def _get_common_mode_noise(self, time, flux, transit_mask, xp):
        """Step 1: Model the common 1D drift with a GP."""
        common_mode_lc = xp.nanmedian(flux, axis=1)
        median_val = xp.nanmedian(common_mode_lc)
        if xp.isnan(median_val) or median_val == 0:
            return xp.ones_like(time)
        
        common_mode_norm = common_mode_lc / median_val
        finite_mask = xp.isfinite(common_mode_norm)
        oot_mask = ~transit_mask & finite_mask
        
        if xp.sum(oot_mask) < 20:
            raise ValueError("Not enough OOT points to fit the common-mode GP.")
        
        if self.use_gpu:
            common_mode_model_norm = self._train_gpytorch(time, common_mode_norm, oot_mask, xp)
        else:
            common_mode_model_norm = self._train_george(time, common_mode_norm, oot_mask, xp)
            
        return common_mode_model_norm * median_val

    def _train_george(self, time, lc, oot_mask, xp):
        # FIX: Ensure all arrays are on the CPU (NumPy) before mixing them.
        time_np, lc_np, oot_mask_np = (d.get() if hasattr(d, 'get') else d for d in (time, lc, oot_mask))
        
        oot_time, oot_flux = time_np[oot_mask_np], lc_np[oot_mask_np]
        
        kernel = kernels.Matern32Kernel(metric=1.0**2)
        gp = george.GP(kernel, mean=np.nanmean(oot_flux), fit_mean=True)
        gp.compute(oot_time, np.nanstd(oot_flux))
        
        p0 = gp.get_parameter_vector()
        results = minimize(lambda p: -gp.log_likelihood(oot_flux, p), p0, jac=lambda p: -gp.grad_log_likelihood(oot_flux, p))
        gp.set_parameter_vector(results.x)
        
        residuals = oot_flux - gp.get_parameter('mean:value')
        pred_noise = gp.predict(residuals, time_np, return_cov=False)
        return xp.asarray(pred_noise + gp.get_parameter('mean:value'))

    def _train_gpytorch(self, time, lc, oot_mask, xp):
        # FIX: Ensure all arrays are on the CPU (NumPy) before passing to PyTorch.
        time_np, lc_np, oot_mask_np = (d.get() if hasattr(d, 'get') else d for d in (time, lc, oot_mask))
        device = torch.device("cuda")
        
        train_x = torch.from_numpy(time_np[oot_mask_np]).to(device, dtype=torch.float32)
        train_y = torch.from_numpy(lc_np[oot_mask_np]).to(device, dtype=torch.float32)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = ExactGPModel(train_x, train_y, likelihood).to(device)

        model.train(); likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(self.training_iter):
            optimizer.zero_grad(); output = model(train_x); loss = -mll(output, train_y); loss.backward(); optimizer.step()

        model.eval(); likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.from_numpy(time_np).to(device, dtype=torch.float32)
            pred_mean = likelihood(model(test_x)).mean.cpu().numpy()
        
        return xp.asarray(pred_mean)

    def detrend(self, time, flux, transit_mask, xp):
        # Reset stop flag for new operation
        self.reset_stop_flag()
        
        # Notify initialization
        self.notify_observers(DetrendingProgress(
            step=DetrendingStep.INITIALIZING,
            progress=0.0,
            message="Initializing hybrid detrending...",
            total_wavelengths=flux.shape[1]
        ))
        
        # Step 1: Get common-mode noise model
        self.notify_observers(DetrendingProgress(
            step=DetrendingStep.FITTING_COMMON_MODE,
            progress=0.2,
            message="Fitting common-mode GP..."
        ))
        self.check_stop_request()
        
        common_noise_model = self._get_common_mode_noise(time, flux, transit_mask, xp)
        
        # Step 2: Remove common-mode and fit per-wavelength polynomials
        self.notify_observers(DetrendingProgress(
            step=DetrendingStep.FITTING_PER_WAVELENGTH,
            progress=0.5,
            message="Fitting per-wavelength polynomials..."
        ))
        self.check_stop_request()
        
        flux_residuals = flux / common_noise_model[:, xp.newaxis]

        poly_detrender = PolynomialDetrender(degree=self.poly_degree)
        detrended_flux, residual_noise_model = poly_detrender.detrend(time, flux_residuals, transit_mask, xp)
        
        total_noise_model = common_noise_model[:, xp.newaxis] * residual_noise_model
        
        # Notify completion
        self.notify_observers(DetrendingProgress(
            step=DetrendingStep.FINALIZING,
            progress=1.0,
            message="Hybrid detrending completed"
        ))
        
        return detrended_flux, total_noise_model


# --- Advanced Detrending Classes (ariel_gp-style) ---

class AIRSDriftDetrender(BaseDetrender):
    """
    Advanced detrending for AIRS instrument based on ariel_gp approach.
    
    Models both average drift (1D GP over time) and spectral drift (2D GP over time-wavelength)
    using KISS-GP approximation for efficiency.
    """
    
    def __init__(self, 
                 avg_kernel: str = 'Matern32',
                 avg_length_scale: float = 1.0,
                 spectral_kernel: str = 'Matern32', 
                 time_scale: float = 0.4,
                 wavelength_scale: float = 0.05,
                 use_sparse: bool = True,
                 use_gpu: bool = False,
                 hyperparams: dict = None,
                 batch_size: int = None,
                 **kwargs):
        super().__init__()
        self.avg_kernel = avg_kernel
        self.avg_length_scale = avg_length_scale
        self.spectral_kernel = spectral_kernel
        self.time_scale = time_scale
        self.wavelength_scale = wavelength_scale
        self.use_sparse = use_sparse
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        # Store hyperparameters
        self.hyperparams = hyperparams or {}
        if self.use_gpu and not GP_GPU_ENABLED:
            raise RuntimeError("GPU support requires PyTorch and GPyTorch")
    
    def detrend(self, time, flux, transit_mask, xp) -> Tuple["xp.ndarray", "xp.ndarray"]:
        """Main detrending method."""
        if self.use_gpu:
            return self._detrend_gpu(time, flux, transit_mask, xp)
        else:
            return self._detrend_cpu(time, flux, transit_mask, xp)
    
    def _detrend_cpu(self, time, flux, transit_mask, xp):
        """CPU implementation using george library."""
        # Reset stop flag for new operation
        self.reset_stop_flag()
        
        # Notify initialization
        self.notify_observers(DetrendingProgress(
            step=DetrendingStep.INITIALIZING,
            progress=0.0,
            message="Initializing AIRS drift detrending...",
            total_wavelengths=flux.shape[1]
        ))
        
        # Convert to numpy if needed
        time_np = time.get() if hasattr(time, 'get') else time
        flux_np = flux.get() if hasattr(flux, 'get') else flux
        transit_mask_np = transit_mask.get() if hasattr(transit_mask, 'get') else transit_mask
        
        noise_model = np.full_like(flux_np, np.nan)
        total_wavelengths = flux_np.shape[1]
        
        for i in range(total_wavelengths):
            # Check for stop request
            self.check_stop_request()
            
            # Update progress
            progress = (i + 1) / total_wavelengths
            self.notify_observers(DetrendingProgress(
                step=DetrendingStep.FITTING_DRIFT,
                progress=progress,
                message=f"Fitting AIRS drift for wavelength {i+1}/{total_wavelengths}",
                current_wavelength=i,
                total_wavelengths=total_wavelengths
            ))
            
            lc = flux_np[:, i]
            median_val = np.nanmedian(lc)
            if np.isnan(median_val) or median_val == 0:
                noise_model[:, i] = 1.0
                continue
                
            lc_norm = lc / median_val
            finite_mask = np.isfinite(lc_norm)
            oot_mask = ~transit_mask_np & finite_mask
            
            if np.sum(oot_mask) < 20:
                raise ValueError(f"Not enough OOT points for AIRS drift fit on channel {i}.")
            
            # Fit average drift (1D GP over time)
            avg_drift = self._fit_average_drift(time_np, lc_norm, oot_mask)
            
            # Fit spectral drift (2D GP over time-wavelength)
            if self.use_sparse:
                spectral_drift = self._fit_spectral_drift_sparse(time_np, lc_norm, oot_mask, i)
            else:
                spectral_drift = self._fit_spectral_drift_dense(time_np, lc_norm, oot_mask, i)
            
            # Combine drifts
            total_drift = avg_drift + spectral_drift
            noise_model[:, i] = total_drift * median_val
        
        # Notify completion
        self.notify_observers(DetrendingProgress(
            step=DetrendingStep.FINALIZING,
            progress=1.0,
            message="AIRS drift detrending completed"
        ))
        
        noise_model_xp = xp.asarray(noise_model) if hasattr(xp, 'asarray') else noise_model
        return flux / noise_model_xp, noise_model_xp
    
    def _detrend_gpu(self, time, flux, transit_mask, xp):
        """GPU implementation using GPyTorch for fast parallel processing."""
        # Reset stop flag for new operation
        self.reset_stop_flag()
        
        # Notify initialization
        self.notify_observers(DetrendingProgress(
            step=DetrendingStep.INITIALIZING,
            progress=0.0,
            message="Initializing GPU AIRS drift detrending...",
            total_wavelengths=flux.shape[1]
        ))
        
        # Convert to numpy if needed
        time_np = time.get() if hasattr(time, 'get') else time
        flux_np = flux.get() if hasattr(flux, 'get') else flux
        transit_mask_np = transit_mask.get() if hasattr(transit_mask, 'get') else transit_mask
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Clear GPU cache before starting to prevent OOM
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        noise_model = np.full_like(flux_np, np.nan)
        total_wavelengths = flux_np.shape[1]
        
        # Use user batch size if provided
        batch_size = self.batch_size if self.batch_size is not None else min(8, total_wavelengths)
        
        # Process wavelengths in batches for better GPU utilization
        # Reduce batch size to prevent OOM errors
        for batch_start in range(0, total_wavelengths, batch_size):
            # Check for stop request
            self.check_stop_request()
            
            batch_end = min(batch_start + batch_size, total_wavelengths)
            batch_wavelengths = list(range(batch_start, batch_end))
            
            # Update progress
            progress = (batch_end) / total_wavelengths
            self.notify_observers(DetrendingProgress(
                step=DetrendingStep.FITTING_DRIFT,
                progress=progress,
                message=f"Fitting GPU AIRS drift for wavelengths {batch_start+1}-{batch_end}/{total_wavelengths}",
                current_wavelength=batch_start,
                total_wavelengths=total_wavelengths
            ))
            
            # Process batch
            batch_noise_model = self._fit_batch_gpu(
                time_np, flux_np, transit_mask_np, batch_wavelengths, device
            )
            
            # Store results
            for i, wavelength_idx in enumerate(batch_wavelengths):
                noise_model[:, wavelength_idx] = batch_noise_model[:, i]
        
        # Notify completion
        self.notify_observers(DetrendingProgress(
            step=DetrendingStep.FINALIZING,
            progress=1.0,
            message="GPU AIRS drift detrending completed"
        ))
        
        noise_model_xp = xp.asarray(noise_model) if hasattr(xp, 'asarray') else noise_model
        
        # Ensure we return NumPy arrays to avoid CuPy conversion issues
        if hasattr(noise_model_xp, 'get'):  # CuPy array
            noise_model_np = noise_model_xp.get()
        elif hasattr(noise_model_xp, 'cpu'):  # PyTorch tensor
            noise_model_np = noise_model_xp.cpu().numpy()
        else:
            noise_model_np = noise_model_xp
            
        if hasattr(flux, 'get'):  # CuPy array
            flux_np = flux.get()
        elif hasattr(flux, 'cpu'):  # PyTorch tensor
            flux_np = flux.cpu().numpy()
        else:
            flux_np = flux
            
        return flux_np / noise_model_np, noise_model_np
    
    def _fit_batch_gpu(self, time, flux, transit_mask, wavelength_indices, device):
        """Fit GP for a batch of wavelengths using GPU."""
        batch_size = len(wavelength_indices)
        
        # Prepare batch data
        batch_oot_masks = []
        batch_oot_times = []
        batch_oot_fluxes = []
        batch_median_vals = []
        
        for i in wavelength_indices:
            lc = flux[:, i]
            median_val = np.nanmedian(lc)
            if np.isnan(median_val) or median_val == 0:
                # Handle invalid data
                batch_oot_masks.append(np.zeros_like(time, dtype=bool))
                batch_oot_times.append(time[:10])  # Dummy data
                batch_oot_fluxes.append(np.ones(10))
                batch_median_vals.append(1.0)
                continue
            
            lc_norm = lc / median_val
            finite_mask = np.isfinite(lc_norm)
            oot_mask = ~transit_mask & finite_mask
            
            if np.sum(oot_mask) < 20:
                # Handle insufficient data
                batch_oot_masks.append(np.zeros_like(time, dtype=bool))
                batch_oot_times.append(time[:10])
                batch_oot_fluxes.append(np.ones(10))
                batch_median_vals.append(1.0)
                continue
            
            batch_oot_masks.append(oot_mask)
            batch_oot_times.append(time[oot_mask])
            batch_oot_fluxes.append(lc_norm[oot_mask])
            batch_median_vals.append(median_val)
        
        # Find common OOT time points (intersection of all valid masks)
        common_oot_mask = np.ones_like(time, dtype=bool)
        for oot_mask in batch_oot_masks:
            if np.any(oot_mask):
                common_oot_mask &= oot_mask
        
        if np.sum(common_oot_mask) < 20:
            # Fall back to individual processing if no common points
            return self._fit_individual_batch_gpu(time, flux, transit_mask, wavelength_indices, device)
        
        # Use common OOT points for batch processing
        common_oot_time = time[common_oot_mask]
        common_oot_flux = np.column_stack([
            flux[common_oot_mask, i] / batch_median_vals[j] 
            for j, i in enumerate(wavelength_indices)
        ])
        
        # Use individual GPs in parallel (simpler and more reliable)
        predictions = np.zeros((len(time), batch_size))
        
        for j, i in enumerate(wavelength_indices):
            lc = flux[:, i]
            median_val = np.nanmedian(lc)
            if np.isnan(median_val) or median_val == 0:
                predictions[:, j] = 1.0
                continue
            
            lc_norm = lc / median_val
            finite_mask = np.isfinite(lc_norm)
            oot_mask = ~transit_mask & finite_mask
            
            if np.sum(oot_mask) < 20:
                predictions[:, j] = 1.0
                continue
            
            # Fit individual GP
            oot_time = time[oot_mask]
            oot_flux = lc_norm[oot_mask]
            
            train_x = torch.from_numpy(oot_time).to(device, dtype=torch.float32)
            train_y = torch.from_numpy(oot_flux).to(device, dtype=torch.float32)
            
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            model = ExactGPModel(
                train_x, train_y, likelihood,
                kernel_type=self.spectral_kernel,
                length_scale=self.time_scale * (1.0 + 0.1 * i)
            ).to(device)
            
            # Quick training
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            
            for _ in range(15):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
            
            # Predict
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.from_numpy(time).to(device, dtype=torch.float32)
                pred_mean = likelihood(model(test_x)).mean.cpu().numpy()
                predictions[:, j] = pred_mean * median_val
        
        return predictions
    
    def _fit_individual_batch_gpu(self, time, flux, transit_mask, wavelength_indices, device):
        """Fallback: fit individual GPs for each wavelength."""
        batch_size = len(wavelength_indices)
        predictions = np.zeros((len(time), batch_size))
        
        for j, i in enumerate(wavelength_indices):
            lc = flux[:, i]
            median_val = np.nanmedian(lc)
            if np.isnan(median_val) or median_val == 0:
                predictions[:, j] = 1.0
                continue
            
            lc_norm = lc / median_val
            finite_mask = np.isfinite(lc_norm)
            oot_mask = ~transit_mask & finite_mask
            
            if np.sum(oot_mask) < 20:
                predictions[:, j] = 1.0
                continue
            
            # Fit individual GP
            oot_time = time[oot_mask]
            oot_flux = lc_norm[oot_mask]
            
            train_x = torch.from_numpy(oot_time).to(device, dtype=torch.float32)
            train_y = torch.from_numpy(oot_flux).to(device, dtype=torch.float32)
            
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            model = ExactGPModel(
                train_x, train_y, likelihood,
                kernel_type=self.spectral_kernel,
                length_scale=self.time_scale * (1.0 + 0.1 * i)
            ).to(device)
            
            # Quick training
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            
            for _ in range(15):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
            
            # Predict
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.from_numpy(time).to(device, dtype=torch.float32)
                pred_mean = likelihood(model(test_x)).mean.cpu().numpy()
                predictions[:, j] = pred_mean * median_val
        
        return predictions
    
    def _fit_average_drift(self, time, lc, oot_mask):
        """Fit 1D GP for average drift over time."""
        oot_time = time[oot_mask]
        oot_flux = lc[oot_mask]
        
        # Create kernel
        if self.avg_kernel == 'Matern32':
            kernel = kernels.Matern32Kernel(metric=self.avg_length_scale**2)
        elif self.avg_kernel == 'RBF':
            kernel = kernels.ExpSquaredKernel(metric=self.avg_length_scale**2)
        else:
            kernel = kernels.Matern32Kernel(metric=self.avg_length_scale**2)
        
        # Fit GP
        gp = george.GP(kernel, mean=np.nanmean(oot_flux), fit_mean=True)
        gp.compute(oot_time)
        
        p0 = gp.get_parameter_vector()
        results = minimize(
            lambda p: -gp.log_likelihood(oot_flux, p), 
            p0, 
            jac=lambda p: -gp.grad_log_likelihood(oot_flux, p)
        )
        gp.set_parameter_vector(results.x)
        
        # Predict on full time range
        residuals = oot_flux - gp.get_parameter('mean:value')
        pred_drift = gp.predict(residuals, time, return_cov=False)
        return pred_drift + gp.get_parameter('mean:value')
    
    def _fit_spectral_drift_sparse(self, time, lc, oot_mask, wavelength_idx):
        """Fit spectral drift using KISS-GP 2D approximation."""
        # True 2D GP over (time, wavelength) using KISS-GP
        oot_time = time[oot_mask]
        oot_flux = lc[oot_mask]
        
        # Create 2D coordinates for OOT data: (time, wavelength)
        oot_x = oot_time
        oot_y = np.full_like(oot_time, wavelength_idx)
        
        # Create 2D coordinates for full time range
        full_x = time
        full_y = np.full_like(time, wavelength_idx)
        
        
        
        # Initialize KISS-GP with full time range to ensure grid covers all data
        kiss_gp = KISSGP2D(h=(self.time_scale, self.wavelength_scale))
        kiss_gp.initialize(full_x, full_y)
        
        # Get grid coordinates for GP fitting
        grid_x_1d, grid_y_1d = kiss_gp.get_grid_points_1d()
        grid_coords = np.column_stack([grid_x_1d, grid_y_1d])
        
        # Use a simpler approach: 1D kernel with wavelength-dependent parameters
        # This avoids the 2D kernel complexity while still capturing spectral variations
        adjusted_time_scale = self.time_scale * (1.0 + 0.1 * wavelength_idx)
        
        if self.spectral_kernel == 'Matern32':
            kernel = kernels.Matern32Kernel(metric=adjusted_time_scale**2)
        elif self.spectral_kernel == 'RBF':
            kernel = kernels.ExpSquaredKernel(metric=adjusted_time_scale**2)
        else:
            kernel = kernels.Matern32Kernel(metric=adjusted_time_scale**2)
        
        # Fit GP on OOT data points (using 1D time data)
        gp = george.GP(kernel, mean=np.nanmean(oot_flux), fit_mean=True)
        gp.compute(oot_time)
        
        p0 = gp.get_parameter_vector()
        results = minimize(
            lambda p: -gp.log_likelihood(oot_flux, p), 
            p0, 
            jac=lambda p: -gp.grad_log_likelihood(oot_flux, p)
        )
        gp.set_parameter_vector(results.x)
        
        # Predict on full time range (1D)
        residuals = oot_flux - gp.get_parameter('mean:value')
        full_predictions = gp.predict(residuals, time, return_cov=False)
        full_predictions += gp.get_parameter('mean:value')
        
        # Ensure predictions have the right shape
        if len(full_predictions) != len(time):
            raise ValueError(f"Predictions length {len(full_predictions)} doesn't match time length {len(time)}")
        
        return full_predictions
    
    def _fit_spectral_drift_dense(self, time, lc, oot_mask, wavelength_idx):
        """Fit 2D GP for spectral drift using dense approach."""
        # Similar to sparse but without KISS-GP approximation
        return self._fit_spectral_drift_sparse(time, lc, oot_mask, wavelength_idx)


class FGSDriftDetrender(BaseDetrender):
    """
    Advanced detrending for FGS instrument based on ariel_gp approach.
    
    Models only average drift (1D GP over time) since FGS is a photometer.
    """
    
    def __init__(self, 
                 kernel: str = 'Matern32',
                 length_scale: float = 1.0,
                 use_gpu: bool = False,
                 batch_size: int = None,
                 **kwargs):
        super().__init__()
        self.kernel = kernel
        self.length_scale = length_scale
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        
        if self.use_gpu and not GP_GPU_ENABLED:
            raise RuntimeError("GPU support requires PyTorch and GPyTorch")
    
    def detrend(self, time, flux, transit_mask, xp) -> Tuple["xp.ndarray", "xp.ndarray"]:
        """Main detrending method."""
        if self.use_gpu:
            return self._detrend_gpu(time, flux, transit_mask, xp)
        else:
            return self._detrend_cpu(time, flux, transit_mask, xp)
    
    def _detrend_cpu(self, time, flux, transit_mask, xp):
        """CPU implementation using george library."""
        # Reset stop flag for new operation
        self.reset_stop_flag()
        
        # Notify initialization
        self.notify_observers(DetrendingProgress(
            step=DetrendingStep.INITIALIZING,
            progress=0.0,
            message="Initializing FGS drift detrending...",
            total_wavelengths=flux.shape[1]
        ))
        
        # Convert to numpy if needed
        time_np = time.get() if hasattr(time, 'get') else time
        flux_np = flux.get() if hasattr(flux, 'get') else flux
        transit_mask_np = transit_mask.get() if hasattr(transit_mask, 'get') else transit_mask
        
        noise_model = np.full_like(flux_np, np.nan)
        total_wavelengths = flux_np.shape[1]
        
        for i in range(total_wavelengths):
            # Check for stop request
            self.check_stop_request()
            
            # Update progress
            progress = (i + 1) / total_wavelengths
            self.notify_observers(DetrendingProgress(
                step=DetrendingStep.FITTING_DRIFT,
                progress=progress,
                message=f"Fitting FGS drift for wavelength {i+1}/{total_wavelengths}",
                current_wavelength=i,
                total_wavelengths=total_wavelengths
            ))
            
            lc = flux_np[:, i]
            median_val = np.nanmedian(lc)
            if np.isnan(median_val) or median_val == 0:
                noise_model[:, i] = 1.0
                continue
                
            lc_norm = lc / median_val
            finite_mask = np.isfinite(lc_norm)
            oot_mask = ~transit_mask_np & finite_mask
            
            if np.sum(oot_mask) < 20:
                raise ValueError(f"Not enough OOT points for FGS drift fit on channel {i}.")
            
            # Fit average drift (1D GP over time)
            drift = self._fit_average_drift(time_np, lc_norm, oot_mask)
            noise_model[:, i] = drift * median_val
        
        # Notify completion
        self.notify_observers(DetrendingProgress(
            step=DetrendingStep.FINALIZING,
            progress=1.0,
            message="FGS drift detrending completed"
        ))
        
        noise_model_xp = xp.asarray(noise_model) if hasattr(xp, 'asarray') else noise_model
        
        # Ensure we return NumPy arrays to avoid CuPy conversion issues
        if hasattr(noise_model_xp, 'get'):  # CuPy array
            noise_model_np = noise_model_xp.get()
        elif hasattr(noise_model_xp, 'cpu'):  # PyTorch tensor
            noise_model_np = noise_model_xp.cpu().numpy()
        else:
            noise_model_np = noise_model_xp
            
        if hasattr(flux, 'get'):  # CuPy array
            flux_np = flux.get()
        elif hasattr(flux, 'cpu'):  # PyTorch tensor
            flux_np = flux.cpu().numpy()
        else:
            flux_np = flux
            
        return flux_np / noise_model_np, noise_model_np
    
    def _detrend_gpu(self, time, flux, transit_mask, xp):
        """GPU implementation using GPyTorch for fast processing."""
        # Reset stop flag for new operation
        self.reset_stop_flag()
        
        # Notify initialization
        self.notify_observers(DetrendingProgress(
            step=DetrendingStep.INITIALIZING,
            progress=0.0,
            message="Initializing GPU FGS drift detrending...",
            total_wavelengths=flux.shape[1]
        ))
        
        # Convert to numpy if needed
        time_np = time.get() if hasattr(time, 'get') else time
        flux_np = flux.get() if hasattr(flux, 'get') else flux
        transit_mask_np = transit_mask.get() if hasattr(transit_mask, 'get') else transit_mask
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Clear GPU cache before starting to prevent OOM
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        noise_model = np.full_like(flux_np, np.nan)
        total_wavelengths = flux_np.shape[1]
        
        # Use user batch size if provided
        batch_size = self.batch_size if self.batch_size is not None else min(8, total_wavelengths)
        
        # Process wavelengths in very small batches to prevent OOM
        # FGS has more wavelengths, so use smaller batches
        for batch_start in range(0, total_wavelengths, batch_size):
            # Check for stop request
            self.check_stop_request()
            
            batch_end = min(batch_start + batch_size, total_wavelengths)
            batch_wavelengths = list(range(batch_start, batch_end))
            
            # Update progress
            progress = (batch_end) / total_wavelengths
            self.notify_observers(DetrendingProgress(
                step=DetrendingStep.FITTING_DRIFT,
                progress=progress,
                message=f"Fitting GPU FGS drift for wavelengths {batch_start+1}-{batch_end}/{total_wavelengths}",
                current_wavelength=batch_start,
                total_wavelengths=total_wavelengths
            ))
            
            # Process batch
            batch_noise_model = self._fit_fgs_batch_gpu(
                time_np, flux_np, transit_mask_np, batch_wavelengths, device
            )
            
            # Store results
            for i, wavelength_idx in enumerate(batch_wavelengths):
                noise_model[:, wavelength_idx] = batch_noise_model[:, i]
            
            # Clear GPU cache after each batch to prevent memory buildup
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        # Notify completion
        self.notify_observers(DetrendingProgress(
            step=DetrendingStep.FINALIZING,
            progress=1.0,
            message="GPU FGS drift detrending completed"
        ))
        
        noise_model_xp = xp.asarray(noise_model) if hasattr(xp, 'asarray') else noise_model
        
        # Ensure we return NumPy arrays to avoid CuPy conversion issues
        if hasattr(noise_model_xp, 'get'):  # CuPy array
            noise_model_np = noise_model_xp.get()
        elif hasattr(noise_model_xp, 'cpu'):  # PyTorch tensor
            noise_model_np = noise_model_xp.cpu().numpy()
        else:
            noise_model_np = noise_model_xp
            
        if hasattr(flux, 'get'):  # CuPy array
            flux_np = flux.get()
        elif hasattr(flux, 'cpu'):  # PyTorch tensor
            flux_np = flux.cpu().numpy()
        else:
            flux_np = flux
            
        return flux_np / noise_model_np, noise_model_np
    
    def _fit_fgs_batch_gpu(self, time, flux, transit_mask, wavelength_indices, device):
        """Fit GP for a batch of FGS wavelengths using GPU with memory optimization."""
        batch_size = len(wavelength_indices)
        predictions = np.zeros((len(time), batch_size))
        
        # FGS has many more time points than AIRS, so we need to subsample for GPU memory
        # Use every 10th point for training to reduce memory usage
        subsample_factor = 10 if len(time) > 50000 else 1
        
        # Use individual GPs in parallel (simpler and more reliable)
        for j, i in enumerate(wavelength_indices):
            lc = flux[:, i]
            median_val = np.nanmedian(lc)
            if np.isnan(median_val) or median_val == 0:
                predictions[:, j] = 1.0
                continue
            
            lc_norm = lc / median_val
            finite_mask = np.isfinite(lc_norm)
            oot_mask = ~transit_mask & finite_mask
            
            if np.sum(oot_mask) < 20:
                predictions[:, j] = 1.0
                continue
            
            # Fit individual GP with memory cleanup
            oot_time = time[oot_mask]
            oot_flux = lc_norm[oot_mask]
            
            # Subsample for large datasets to prevent OOM
            if len(oot_time) > 50000:
                # Take every nth point for training
                train_indices = np.arange(0, len(oot_time), subsample_factor)
                oot_time_train = oot_time[train_indices]
                oot_flux_train = oot_flux[train_indices]
            else:
                oot_time_train = oot_time
                oot_flux_train = oot_flux
            
            train_x = torch.from_numpy(oot_time_train).to(device, dtype=torch.float32)
            train_y = torch.from_numpy(oot_flux_train).to(device, dtype=torch.float32)
            
            # Use simpler kernel to reduce memory usage
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            model = ExactGPModel(
                train_x, train_y, likelihood,
                kernel_type='RBF',  # Use RBF instead of Matern32 for less memory
                length_scale=self.length_scale
            ).to(device)
            
            # Quick training with fewer iterations and progress tracking
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            
            for iter_idx in range(5):  # Reduced from 10 to 5 iterations
                # Check for stop request
                self.check_stop_request()
                
                # Update iteration progress for FGS batch
                iter_progress = iter_idx / 5
                wavelength_progress = j / batch_size
                total_progress = wavelength_progress + (iter_progress / batch_size)
                
                self.notify_observers(DetrendingProgress(
                    step=DetrendingStep.FITTING_DRIFT,
                    progress=total_progress,
                    message=f"Fitting FGS GPU GP for wavelength {wavelength_indices[j]+1} - Iteration {iter_idx+1}/5",
                    current_wavelength=wavelength_indices[j],
                    total_wavelengths=len(wavelength_indices)
                ))
                
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
            
            # Predict
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.from_numpy(time).to(device, dtype=torch.float32)
                pred_mean = likelihood(model(test_x)).mean.cpu().numpy()
                predictions[:, j] = pred_mean * median_val
            
            # Explicit cleanup to free GPU memory
            del model, likelihood, optimizer, mll, train_x, train_y, test_x, output, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        return predictions
    
    def _fit_fgs_individual_batch_gpu(self, time, flux, transit_mask, wavelength_indices, device):
        """Fallback: fit individual GPs for each FGS wavelength with memory optimization."""
        batch_size = len(wavelength_indices)
        predictions = np.zeros((len(time), batch_size))
        
        for j, i in enumerate(wavelength_indices):
            lc = flux[:, i]
            median_val = np.nanmedian(lc)
            if np.isnan(median_val) or median_val == 0:
                predictions[:, j] = 1.0
                continue
            
            lc_norm = lc / median_val
            finite_mask = np.isfinite(lc_norm)
            oot_mask = ~transit_mask & finite_mask
            
            if np.sum(oot_mask) < 20:
                predictions[:, j] = 1.0
                continue
            
            # Fit individual GP with memory cleanup
            oot_time = time[oot_mask]
            oot_flux = lc_norm[oot_mask]
            
            train_x = torch.from_numpy(oot_time).to(device, dtype=torch.float32)
            train_y = torch.from_numpy(oot_flux).to(device, dtype=torch.float32)
            
            # Use simpler kernel to reduce memory usage
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            model = ExactGPModel(
                train_x, train_y, likelihood,
                kernel_type='RBF',  # Use RBF instead of Matern32 for less memory
                length_scale=self.length_scale
            ).to(device)
            
            # Quick training with fewer iterations
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            
            for _ in range(5):  # Reduced from 10 to 5 iterations
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
            
            # Predict
            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.from_numpy(time).to(device, dtype=torch.float32)
                pred_mean = likelihood(model(test_x)).mean.cpu().numpy()
                predictions[:, j] = pred_mean * median_val
            
            # Explicit cleanup to free GPU memory
            del model, likelihood, optimizer, mll, train_x, train_y, test_x, output, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        return predictions
    
    def _fit_average_drift(self, time, lc, oot_mask):
        """Fit 1D GP for average drift over time."""
        oot_time = time[oot_mask]
        oot_flux = lc[oot_mask]
        
        # Create kernel
        if self.kernel == 'Matern32':
            kernel = kernels.Matern32Kernel(metric=self.length_scale**2)
        elif self.kernel == 'RBF':
            kernel = kernels.ExpSquaredKernel(metric=self.length_scale**2)
        else:
            kernel = kernels.Matern32Kernel(metric=self.length_scale**2)
        
        # Fit GP
        gp = george.GP(kernel, mean=np.nanmean(oot_flux), fit_mean=True)
        gp.compute(oot_time)
        
        p0 = gp.get_parameter_vector()
        results = minimize(
            lambda p: -gp.log_likelihood(oot_flux, p), 
            p0, 
            jac=lambda p: -gp.grad_log_likelihood(oot_flux, p)
        )
        gp.set_parameter_vector(results.x)
        
        # Predict on full time range
        residuals = oot_flux - gp.get_parameter('mean:value')
        pred_drift = gp.predict(residuals, time, return_cov=False)
        return pred_drift + gp.get_parameter('mean:value')


# Convenience functions for easy access
def create_airs_drift_detrender(instrument: str = "AIRS-CH0", use_gpu: bool = True, **kwargs):
    """Create an AIRS drift detrender with default settings from config."""
    from ..config import DEFAULT_DETRENDING_PARAMS
    
    # Get default params for the instrument
    default_params = DEFAULT_DETRENDING_PARAMS.get(instrument, {})
    
    # Enable GPU by default if available
    if use_gpu and not GP_GPU_ENABLED:
        print("Warning: GPU requested but PyTorch/GPyTorch not available. Falling back to CPU.")
        use_gpu = False
    
    # Merge with provided kwargs (kwargs take precedence)
    params = {**default_params, "use_gpu": use_gpu, **kwargs}
    
    return AIRSDriftDetrender(**params)

def create_fgs_drift_detrender(instrument: str = "FGS1", use_gpu: bool = True, **kwargs):
    """Create an FGS drift detrender with default settings from config."""
    from ..config import DEFAULT_DETRENDING_PARAMS
    
    # Get default params for the instrument
    default_params = DEFAULT_DETRENDING_PARAMS.get(instrument, {})
    
    # Enable GPU by default if available
    if use_gpu and not GP_GPU_ENABLED:
        print("Warning: GPU requested but PyTorch/GPyTorch not available. Falling back to CPU.")
        use_gpu = False
    
    # Merge with provided kwargs (kwargs take precedence)
    params = {**default_params, "use_gpu": use_gpu, **kwargs}
    
    return FGSDriftDetrender(**params)
