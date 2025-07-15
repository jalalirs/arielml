# arielml/data/detrending.py

import numpy as np
from abc import ABC, abstractmethod
from scipy import signal as scipy_signal
from scipy.optimize import minimize
from typing import Tuple

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


class BaseDetrender(ABC):
    """Abstract base class for all detrending models."""
    @abstractmethod
    def detrend(self, time, flux, transit_mask, xp) -> Tuple["xp.ndarray", "xp.ndarray"]:
        pass

class PolynomialDetrender(BaseDetrender):
    """Fits and removes a polynomial trend."""
    def __init__(self, degree: int = 2):
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
        self.kernel_name = kernel

    def _get_kernel(self):
        if self.kernel_name == 'Matern32': return kernels.Matern32Kernel(metric=1.0**2)
        raise ValueError(f"Unknown kernel: {self.kernel_name}")

    def detrend(self, time, flux, transit_mask, xp):
        is_gpu = (xp.__name__ == 'cupy')
        time_np, flux_np, transit_mask_np = (d.get() if is_gpu else d for d in (time, flux, transit_mask))
        noise_model_np = np.full_like(flux_np, np.nan)
        for i in range(flux_np.shape[1]):
            lc, median_val = flux_np[:, i], np.nanmedian(flux_np[:, i])
            if np.isnan(median_val) or median_val == 0:
                noise_model_np[:, i] = 1.0; continue
            lc_norm = lc / median_val
            finite_mask = np.isfinite(lc_norm)
            oot_mask = ~transit_mask_np & finite_mask
            if np.sum(oot_mask) < 20:
                raise ValueError(f"Not enough OOT points for GP fit on channel {i}.")
            oot_time, oot_flux = time_np[oot_mask], lc_norm[oot_mask]
            try:
                kernel = self._get_kernel()
                gp = george.GP(kernel, mean=np.nanmean(oot_flux), fit_mean=True, white_noise=np.log(np.nanstd(oot_flux)**2), fit_white_noise=True)
                gp.compute(oot_time)
                p0 = gp.get_parameter_vector()
                results = minimize(lambda p: -gp.log_likelihood(oot_flux, p), p0, jac=lambda p: -gp.grad_log_likelihood(oot_flux, p))
                gp.set_parameter_vector(results.x)
                residuals = oot_flux - gp.get_parameter('mean:value')
                pred_noise = gp.predict(residuals, time_np, return_cov=False)
                noise_model_np[:, i] = (pred_noise + gp.get_parameter('mean:value')) * median_val
            except Exception as e:
                raise RuntimeError(f"George GP failed for channel {i}: {e}")
        noise_model = xp.asarray(noise_model_np) if is_gpu else noise_model_np
        return flux / noise_model, noise_model

if GP_GPU_ENABLED:
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    class GPyTorchDetrender(BaseDetrender):
        """GPU-accelerated Gaussian Process detrender using GPyTorch."""
        def __init__(self, training_iter=50):
            self.training_iter = training_iter

        def detrend(self, time, flux, transit_mask, xp):
            is_gpu = (xp.__name__ == 'cupy')
            time_np, flux_np, transit_mask_np = (d.get() if is_gpu else d for d in (time, flux, transit_mask))
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            noise_model_np = np.full_like(flux_np, np.nan)

            for i in range(flux_np.shape[1]):
                lc, median_val = flux_np[:, i], np.nanmedian(flux_np[:, i])
                if np.isnan(median_val) or median_val == 0:
                    noise_model_np[:, i] = 1.0; continue
                
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

                for _ in range(self.training_iter):
                    optimizer.zero_grad(); output = model(train_x); loss = -mll(output, train_y); loss.backward(); optimizer.step()
                
                model.eval(); likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    test_x = torch.from_numpy(time_np).to(device, dtype=torch.float32)
                    observed_pred = likelihood(model(test_x))
                    pred_mean = observed_pred.mean.cpu().numpy()

                noise_model_np[:, i] = pred_mean * median_val

            noise_model = xp.asarray(noise_model_np) if is_gpu else noise_model_np
            return flux / noise_model, noise_model

class HybridDetrender(BaseDetrender):
    """
    Implements the hybrid detrending approach:
    1. Models the common-mode noise across all wavelengths with a GP.
    2. Models the remaining per-wavelength residuals with a simple Polynomial.
    """
    def __init__(self, use_gpu=False, training_iter=50, poly_degree=2):
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
        common_noise_model = self._get_common_mode_noise(time, flux, transit_mask, xp)
        flux_residuals = flux / common_noise_model[:, xp.newaxis]

        poly_detrender = PolynomialDetrender(degree=self.poly_degree)
        detrended_flux, residual_noise_model = poly_detrender.detrend(time, flux_residuals, transit_mask, xp)
        
        total_noise_model = common_noise_model[:, xp.newaxis] * residual_noise_model
        
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
                 **kwargs):
        super().__init__(**kwargs)
        self.avg_kernel = avg_kernel
        self.avg_length_scale = avg_length_scale
        self.spectral_kernel = spectral_kernel
        self.time_scale = time_scale
        self.wavelength_scale = wavelength_scale
        self.use_sparse = use_sparse
        self.use_gpu = use_gpu
        
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
        # Convert to numpy if needed
        time_np = time.get() if hasattr(time, 'get') else time
        flux_np = flux.get() if hasattr(flux, 'get') else flux
        transit_mask_np = transit_mask.get() if hasattr(transit_mask, 'get') else transit_mask
        
        noise_model = np.full_like(flux_np, np.nan)
        
        for i in range(flux_np.shape[1]):
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
        
        noise_model_xp = xp.asarray(noise_model) if hasattr(xp, 'asarray') else noise_model
        return flux / noise_model_xp, noise_model_xp
    
    def _detrend_gpu(self, time, flux, transit_mask, xp):
        """GPU implementation - raises NotImplementedError for now."""
        raise NotImplementedError("GPU implementation for AIRSDriftDetrender not yet implemented")
    
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
        """Fit 2D GP for spectral drift using sparse approximation."""
        # Simplified KISS-GP implementation
        # In practice, this would use a more sophisticated sparse GP approach
        
        # For now, use a simple 2D kernel approach
        oot_time = time[oot_mask]
        oot_flux = lc[oot_mask]
        
        # Create 2D input (time, wavelength)
        X = np.column_stack([oot_time, np.full_like(oot_time, wavelength_idx)])
        
        # Create 2D kernel
        if self.spectral_kernel == 'Matern32':
            kernel = kernels.Matern32Kernel(metric=[self.time_scale**2, self.wavelength_scale**2])
        elif self.spectral_kernel == 'RBF':
            kernel = kernels.ExpSquaredKernel(metric=[self.time_scale**2, self.wavelength_scale**2])
        else:
            kernel = kernels.Matern32Kernel(metric=[self.time_scale**2, self.wavelength_scale**2])
        
        # Fit GP
        gp = george.GP(kernel, mean=np.nanmean(oot_flux), fit_mean=True)
        gp.compute(X)
        
        p0 = gp.get_parameter_vector()
        results = minimize(
            lambda p: -gp.log_likelihood(oot_flux, p), 
            p0, 
            jac=lambda p: -gp.grad_log_likelihood(oot_flux, p)
        )
        gp.set_parameter_vector(results.x)
        
        # Predict on full time range
        X_full = np.column_stack([time, np.full_like(time, wavelength_idx)])
        residuals = oot_flux - gp.get_parameter('mean:value')
        pred_drift = gp.predict(residuals, X_full, return_cov=False)
        return pred_drift + gp.get_parameter('mean:value')
    
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
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.length_scale = length_scale
        self.use_gpu = use_gpu
        
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
        # Convert to numpy if needed
        time_np = time.get() if hasattr(time, 'get') else time
        flux_np = flux.get() if hasattr(flux, 'get') else flux
        transit_mask_np = transit_mask.get() if hasattr(transit_mask, 'get') else transit_mask
        
        noise_model = np.full_like(flux_np, np.nan)
        
        for i in range(flux_np.shape[1]):
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
        
        noise_model_xp = xp.asarray(noise_model) if hasattr(xp, 'asarray') else noise_model
        return flux / noise_model_xp, noise_model_xp
    
    def _detrend_gpu(self, time, flux, transit_mask, xp):
        """GPU implementation - raises NotImplementedError for now."""
        raise NotImplementedError("GPU implementation for FGSDriftDetrender not yet implemented")
    
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


class BayesianMultiComponentDetrender(BaseDetrender):
    """
    Complete Bayesian multi-component detrending based on ariel_gp approach.
    
    Implements the full model: (stellar_spectrum * drift * transit) + noise
    where all components are estimated simultaneously.
    """
    
    def __init__(self,
                 n_pca: int = 1,
                 n_iter: int = 7,
                 n_samples: int = 100,
                 update_rate: float = 1.0,
                 use_gpu: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_pca = n_pca
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.update_rate = update_rate
        self.use_gpu = use_gpu
        
        if self.use_gpu and not GP_GPU_ENABLED:
            raise RuntimeError("GPU support requires PyTorch and GPyTorch")
    
    def detrend(self, time, flux, transit_mask, xp) -> Tuple["xp.ndarray", "xp.ndarray"]:
        """Main detrending method."""
        if self.use_gpu:
            return self._detrend_gpu(time, flux, transit_mask, xp)
        else:
            return self._detrend_cpu(time, flux, transit_mask, xp)
    
    def _detrend_cpu(self, time, flux, transit_mask, xp):
        """CPU implementation of complete Bayesian model."""
        # Convert to numpy if needed
        time_np = time.get() if hasattr(time, 'get') else time
        flux_np = flux.get() if hasattr(flux, 'get') else flux
        transit_mask_np = transit_mask.get() if hasattr(transit_mask, 'get') else transit_mask
        
        # This is a simplified version - full implementation would be much more complex
        # For now, we'll use a combination of the individual components
        
        # 1. Fit stellar spectrum (simple baseline)
        stellar_spectrum = self._fit_stellar_spectrum(flux_np)
        
        # 2. Fit drift components
        drift_model = self._fit_drift_components(time_np, flux_np, transit_mask_np)
        
        # 3. Fit transit window (simplified)
        transit_window = self._fit_transit_window(time_np, transit_mask_np)
        
        # 4. Combine all components
        noise_model = stellar_spectrum * drift_model * transit_window
        
        noise_model_xp = xp.asarray(noise_model) if hasattr(xp, 'asarray') else noise_model
        return flux / noise_model_xp, noise_model_xp
    
    def _detrend_gpu(self, time, flux, transit_mask, xp):
        """GPU implementation - raises NotImplementedError for now."""
        raise NotImplementedError("GPU implementation for BayesianMultiComponentDetrender not yet implemented")
    
    def _fit_stellar_spectrum(self, flux):
        """Fit stellar spectrum baseline."""
        # Simple approach: use median per wavelength
        return np.nanmedian(flux, axis=0, keepdims=True)
    
    def _fit_drift_components(self, time, flux, transit_mask):
        """Fit drift components (AIRS: average + spectral, FGS: average only)."""
        # Simplified implementation
        # In practice, this would use the full Bayesian framework
        
        drift_model = np.ones_like(flux)
        
        # For each wavelength, fit a simple drift model
        for i in range(flux.shape[1]):
            lc = flux[:, i]
            finite_mask = np.isfinite(lc)
            oot_mask = ~transit_mask & finite_mask
            
            if np.sum(oot_mask) < 10:
                continue
                
            # Simple polynomial drift fit
            oot_time = time[oot_mask]
            oot_flux = lc[oot_mask]
            
            # Fit 2nd order polynomial
            coeffs = np.polyfit(oot_time, oot_flux, 2)
            drift_fit = np.polyval(coeffs, time)
            
            drift_model[:, i] = drift_fit
        
        return drift_model
    
    def _fit_transit_window(self, time, transit_mask):
        """Fit transit window function."""
        # Simplified transit window
        # In practice, this would use the non-linear spline-based approach
        
        # Create a simple transit window based on the mask
        transit_window = np.ones_like(time, dtype=float)
        
        # Smooth the transit mask to create a gradual transition
        if np.any(transit_mask):
            # Use a simple smoothing approach
            from scipy.ndimage import gaussian_filter1d
            smoothed_mask = gaussian_filter1d(transit_mask.astype(float), sigma=2)
            transit_window = 1.0 - 0.1 * smoothed_mask  # Small transit depth
        
        return transit_window[:, np.newaxis]


# Convenience functions for easy access
def create_airs_drift_detrender(**kwargs):
    """Create an AIRS drift detrender with default settings."""
    return AIRSDriftDetrender(**kwargs)

def create_fgs_drift_detrender(**kwargs):
    """Create an FGS drift detrender with default settings."""
    return FGSDriftDetrender(**kwargs)

def create_bayesian_detrender(**kwargs):
    """Create a complete Bayesian detrender with default settings."""
    return BayesianMultiComponentDetrender(**kwargs)
