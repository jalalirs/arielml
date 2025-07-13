# arielml/data/calibration.py

import itertools
from astropy.stats import sigma_clip

def apply_adc_conversion(signal, gain, offset, xp):
    """Converts raw signal from ADU to electrons using the specified backend."""
    return (signal.astype(xp.float64) / gain) + offset

def mask_hot_dead_pixels(signal, dead_map, dark_map, xp, sigma=5):
    """Masks dead pixels and hot pixels using the specified backend."""
    if hasattr(dark_map, 'get'): # Check if it's a cupy array
        dark_map_np = dark_map.get()
    else:
        dark_map_np = dark_map

    hot_pixel_mask_np = sigma_clip(dark_map_np, sigma=sigma, maxiters=5).mask
    hot_pixel_mask = xp.asarray(hot_pixel_mask_np)

    tiled_dead_mask = xp.tile(dead_map, (signal.shape[0], 1, 1))
    tiled_hot_mask = xp.tile(hot_pixel_mask, (signal.shape[0], 1, 1))
    
    combined_mask = xp.logical_or(tiled_dead_mask, tiled_hot_mask)
    return signal, combined_mask

def apply_linearity_correction(signal, linear_corr_coeffs, xp):
    """
    Corrects for detector non-linearity using a vectorized polynomial evaluation.
    """
    n_coeffs = linear_corr_coeffs.shape[0]
    corrected_signal = xp.full_like(signal, linear_corr_coeffs[n_coeffs - 1, :, :])
    
    for i in range(n_coeffs - 2, -1, -1):
        corrected_signal = linear_corr_coeffs[i, :, :] + signal * corrected_signal
        
    return corrected_signal

def subtract_dark_current(signal, dark_map, integration_times, xp):
    """Subtracts scaled dark current from the signal using the specified backend."""
    tiled_dark = xp.tile(dark_map, (signal.shape[0], 1, 1))
    scaled_dark = tiled_dark * integration_times[:, xp.newaxis, xp.newaxis]
    return signal - scaled_dark

def perform_cds(signal, xp):
    """Performs Correlated Double Sampling using the specified backend."""
    num_frames = signal.shape[0]
    if num_frames % 2 != 0:
        signal = signal[:-1, :, :]
    return signal[1::2, :, :] - signal[::2, :, :]

def apply_flat_field(signal, flat_map, dead_map, xp, epsilon=1e-6):
    """
    Corrects for pixel-to-pixel sensitivity variations using the specified backend.
    Zeros, small values, and dead pixels in the flat_map are clipped.
    """
    safe_flat = xp.copy(flat_map)

    # FIX: Use the dead_map to mask the flat field, in addition to checking
    # for zeros or non-finite values. This is the key insight from the notebook.
    invalid_mask = (safe_flat < epsilon) | ~xp.isfinite(safe_flat) | dead_map.astype(bool)
    
    # Replace these invalid values with NaN so they can be ignored by nanmedian.
    safe_flat[invalid_mask] = xp.nan
    
    # Calculate the median of the remaining, valid flat field values.
    median_flat = xp.nanmedian(safe_flat)
    
    # If the entire flat field was invalid, the median will be NaN.
    # In this case, use 1.0 as a neutral fallback value.
    if xp.isnan(median_flat) or median_flat < epsilon:
        median_flat = 1.0
        
    # Replace the NaNs in our safe_flat array with the calculated median.
    # This ensures that bad pixels in the flat field don't corrupt the signal.
    safe_flat = xp.nan_to_num(safe_flat, nan=median_flat)
    
    # Clip any remaining near-zero values to prevent division issues.
    safe_flat[safe_flat < epsilon] = median_flat

    # Now it's safe to divide.
    return signal / safe_flat
