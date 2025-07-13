# arielml/data/detrending.py

import numpy as np
from abc import ABC, abstractmethod
from scipy import signal as scipy_signal
from typing import Tuple

class BaseDetrender(ABC):
    """
    Abstract base class for all detrending models.
    """
    @abstractmethod
    def detrend(
        self,
        time,
        flux,
        transit_mask,
        xp
    ) -> Tuple["xp.ndarray", "xp.ndarray"]:
        """
        Detrends a light curve by modeling and removing systematic noise.

        Returns:
            A tuple containing:
            - detrended_flux (xp.ndarray): The flattened, normalized light curve.
            - noise_model (xp.ndarray): The trend model that was divided out.
        """
        pass

class PolynomialDetrender(BaseDetrender):
    """
    A detrending model that fits and removes a polynomial trend from the
    out-of-transit portion of a light curve.
    """
    def __init__(self, degree: int = 2):
        if not isinstance(degree, int) or degree < 0:
            raise ValueError("Degree must be a non-negative integer.")
        self.degree = degree

    def detrend(self, time, flux, transit_mask, xp):
        """
        Fits a polynomial to out-of-transit data and divides it out.
        """
        finite_mask = xp.all(xp.isfinite(flux), axis=1)
        oot_mask = ~transit_mask & finite_mask
        
        if not xp.any(oot_mask):
            median_flux = xp.nanmedian(flux, axis=0)
            return flux / median_flux, xp.full_like(flux, median_flux)
            
        poly_coeffs = xp.polyfit(time[oot_mask], flux[oot_mask], self.degree)

        # --- FIX: Ensure coefficients are 1D for single light curves (like FGS1) ---
        # cupy.polyval requires a 1D coefficient array.
        if poly_coeffs.ndim > 1 and poly_coeffs.shape[1] == 1:
            poly_coeffs = poly_coeffs.squeeze(axis=1)

        noise_model = xp.polyval(poly_coeffs, time)
        
        # For a single light curve, noise_model might be 1D. We need to match flux shape.
        if noise_model.ndim == 1 and flux.ndim == 2:
            noise_model = noise_model[:, xp.newaxis]

        detrended_flux = flux / noise_model
        return detrended_flux, noise_model

class SavGolDetrender(BaseDetrender):
    """
    A detrending model that uses a Savitzky-Golay filter to smooth the
    out-of-transit data and create a noise model.
    """
    def __init__(self, window_length: int, polyorder: int):
        if window_length % 2 == 0:
            raise ValueError("window_length must be an odd integer.")
        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length.")
        self.window_length = window_length
        self.polyorder = polyorder

    def detrend(self, time, flux, transit_mask, xp):
        """
        Applies a Savitzky-Golay filter to the out-of-transit data and
        interpolates the result to create a full noise model.
        """
        is_gpu = (xp.__name__ == 'cupy')
        if is_gpu:
            time_np, flux_np, transit_mask_np = time.get(), flux.get(), transit_mask.get()
        else:
            time_np, flux_np, transit_mask_np = time, flux, transit_mask

        noise_model_np = np.full_like(flux_np, np.nan)
        
        for i in range(flux_np.shape[1]):
            lc = flux_np[:, i]
            finite_mask = np.isfinite(lc)
            oot_mask = ~transit_mask_np & finite_mask
            
            if np.sum(oot_mask) < self.window_length:
                if np.any(finite_mask):
                    median_val = np.nanmedian(lc)
                else:
                    median_val = 1.0
                if np.isnan(median_val): median_val = 1.0
                noise_model_np[:, i] = median_val
                continue

            oot_time = time_np[oot_mask]
            oot_flux = lc[oot_mask]
            
            smoothed_oot_flux = scipy_signal.savgol_filter(
                oot_flux,
                window_length=self.window_length,
                polyorder=self.polyorder,
                axis=0
            )
            noise_model_np[:, i] = np.interp(time_np, oot_time, smoothed_oot_flux)

        if is_gpu:
            noise_model = xp.asarray(noise_model_np)
        else:
            noise_model = noise_model_np
            
        return flux / noise_model, noise_model
