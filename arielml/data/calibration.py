# arielml/data/calibration.py

import itertools
from astropy.stats import sigma_clip

def apply_adc_conversion(signal, gain, offset, xp):
    """Converts raw signal from ADU to electrons using the specified backend."""
    return (signal.astype(xp.float64) / gain) + offset

def mask_hot_dead_pixels(signal, dead_map, dark_map, xp, sigma=5):
    """Masks dead pixels and hot pixels using the specified backend."""
    # astropy's sigma_clip works on numpy arrays, so we may need to transfer data
    if hasattr(dark_map, 'get'): # Check if it's a cupy array
        dark_map_np = dark_map.get()
    else:
        dark_map_np = dark_map

    hot_pixel_mask_np = sigma_clip(dark_map_np, sigma=sigma, maxiters=5).mask
    hot_pixel_mask = xp.asarray(hot_pixel_mask_np)

    tiled_dead_mask = xp.tile(dead_map, (signal.shape[0], 1, 1))
    tiled_hot_mask = xp.tile(hot_pixel_mask, (signal.shape[0], 1, 1))
    
    combined_mask = xp.logical_or(tiled_dead_mask, tiled_hot_mask)
    # The calling function will be responsible for applying the mask.
    return signal, combined_mask

def apply_linearity_correction(signal, linear_corr_coeffs, xp):
    """
    Corrects for detector non-linearity using a vectorized polynomial evaluation
    (Horner's method) that is efficient on both CPU (Numpy) and GPU (Cupy).

    Args:
        signal (xp.ndarray): The input signal array of shape (F, H, W).
        linear_corr_coeffs (xp.ndarray): Polynomial coefficients of shape (C, H, W),
                                         ordered from c0, c1, ..., c_n.
        xp (module): The numerical backend (numpy or cupy).

    Returns:
        xp.ndarray: The corrected signal array.
    """
    # This function implements Horner's method for polynomial evaluation:
    # y = c0 + x*(c1 + x*(c2 + ... + x*cn))
    # It is fully vectorized and requires no explicit Python loops over pixels.
    
    n_coeffs = linear_corr_coeffs.shape[0]
    
    # Start with the highest-order coefficient, c_n
    # The result needs to be broadcastable to the shape of the signal.
    corrected_signal = xp.full_like(signal, linear_corr_coeffs[n_coeffs - 1, :, :])
    
    # Iteratively apply Horner's method from c_{n-1} down to c0
    for i in range(n_coeffs - 2, -1, -1):
        # corrected_signal becomes the evaluation of the inner part of the parenthesis
        # The next step is: new_signal = c_i + x * old_signal
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

def apply_flat_field(signal, flat_map, xp):
    """Corrects for pixel-to-pixel sensitivity variations using the specified backend."""
    return signal / flat_map
