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
