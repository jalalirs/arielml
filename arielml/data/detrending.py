# arielml/data/detrending.py

from abc import ABC, abstractmethod

class BaseDetrender(ABC):
    """
    Abstract base class for all detrending models.
    Ensures that any new detrender we create has a consistent interface.
    """
    @abstractmethod
    def detrend(
        self,
        time,
        flux,
        transit_mask,
        xp
    ) -> "xp.ndarray":
        """
        Detrends a light curve by modeling and removing systematic noise.

        Args:
            time (xp.ndarray): The 1D time array for the light curve.
            flux (xp.ndarray): The 2D flux array of shape (time, wavelengths).
            transit_mask (xp.ndarray): A 1D boolean mask where True indicates
                                       an in-transit data point.
            xp (module): The numerical backend (numpy or cupy).

        Returns:
            xp.ndarray: The flattened, detrended light curve array.
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

    def detrend(self, time, flux, transit_mask, xp) -> "xp.ndarray":
        """
        Fits a polynomial to out-of-transit data and divides it out.
        This operation is vectorized to handle multiple light curves at once.
        """
        # Create a mask for finite, out-of-transit data points
        finite_mask = xp.all(xp.isfinite(flux), axis=1)
        oot_mask = ~transit_mask & finite_mask
        
        if not xp.any(oot_mask):
            return flux / xp.nanmedian(flux, axis=0)
            
        # Fit polynomials to all light curves (columns) at once
        poly_coeffs = xp.polyfit(time[oot_mask], flux[oot_mask], self.degree)
        
        # Evaluate the polynomials over the entire time range
        noise_model = xp.polyval(poly_coeffs, time)
        
        detrended_flux = flux / noise_model
        return detrended_flux

class SavGolDetrender(BaseDetrender):
    """
    A detrending model that uses a Savitzky-Golay filter to smooth the
    out-of-transit data and create a noise model.
    """
    def __init__(self, window_length: int, polyorder: int):
        if window_length % 2 == 0:
            raise ValueError("window_length must be an odd integer.")
        self.window_length = window_length
        self.polyorder = polyorder

    def detrend(self, time, flux, transit_mask, xp) -> "xp.ndarray":
        """
        Applies a Savitzky-Golay filter to the out-of-transit data.
        """
        # Import the correct signal processing library based on the backend
        if xp.__name__ == 'cupy':
            from cupyx.scipy import signal
        else:
            from scipy import signal

        # Create a copy of the flux to modify
        trend_flux = xp.copy(flux)
        
        # In-transit data points are not used to build the trend model,
        # so we replace them with NaN before filtering.
        trend_flux[transit_mask] = xp.nan
        
        # The Sav-Gol filter doesn't handle NaNs, so we must interpolate over them.
        # This is a simple linear interpolation.
        for i in range(trend_flux.shape[1]): # Loop over each wavelength
            lc = trend_flux[:, i]
            nans = xp.isnan(lc)
            if xp.any(nans):
                x = lambda z: z.nonzero()[0]
                lc[nans] = xp.interp(x(nans), x(~nans), lc[~nans])
            trend_flux[:, i] = lc

        # Apply the filter to the interpolated out-of-transit data
        noise_model = signal.savgol_filter(
            trend_flux,
            window_length=self.window_length,
            polyorder=self.polyorder,
            axis=0 # Apply along the time axis
        )
        
        return flux / noise_model
