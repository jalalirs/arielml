# arielml/analysis.py

import numpy as np
from scipy.stats import binned_statistic

def phase_fold_and_bin(time, flux, period, n_bins=100):
    """
    Phase-folds a light curve and bins the data to increase signal-to-noise.

    This function requires numpy arrays as it uses scipy for binning.

    Args:
        time (np.ndarray): The time array of the observation.
        flux (np.ndarray): The flux array (light curve).
        period (float): The orbital period of the planet in days.
        n_bins (int): The number of bins to group the data into.

    Returns:
        A tuple containing:
        - bin_centers (np.ndarray): The center phase of each bin.
        - binned_flux (np.ndarray): The median flux in each bin.
        - binned_error (np.ndarray): The standard error on the mean for each bin.
    """
    # Calculate the phase of each data point
    phase = (time % period) / period
    
    # Use scipy's binned_statistic to group the data by phase
    # We calculate the median flux in each bin, which is robust to outliers.
    binned_flux, bin_edges, _ = binned_statistic(
        phase, flux, statistic='median', bins=n_bins
    )
    
    # We also calculate the standard deviation in each bin and divide by the
    # square root of the number of points in the bin to get the standard error.
    binned_std, _, _ = binned_statistic(
        phase, flux, statistic='std', bins=n_bins
    )
    binned_count, _, _ = binned_statistic(
        phase, flux, statistic='count', bins=n_bins
    )
    
    # Avoid division by zero for empty bins
    binned_error = np.divide(
        binned_std, np.sqrt(binned_count), 
        out=np.full_like(binned_std, np.nan), 
        where=binned_count > 0
    )

    # Calculate the center of each bin for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return bin_centers, binned_flux, binned_error