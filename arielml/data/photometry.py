# arielml/data/photometry.py

from typing import Dict, List
from astropy.stats import sigma_clip

def extract_aperture_photometry(
    signal_cube,
    signal_aperture: Dict,
    background_apertures: List[Dict],
    instrument: str,
    xp
):
    """
    Performs aperture photometry on a calibrated 2D signal cube.
    It uses different logic based on the instrument type.

    Args:
        signal_cube (xp.ndarray): The 3D calibrated data cube of shape (F, H, W).
        signal_aperture (dict): A dictionary defining the signal region.
        background_apertures (list): A list of dictionaries defining the background regions.
        instrument (str): The name of the instrument ('AIRS-CH0' or 'FGS1').
        xp (module): The numerical backend (numpy or cupy).

    Returns:
        xp.ndarray: A 2D array of light curves. Shape is (F, W) for spectrometers
                    and (F, 1) for photometers.
    """
    # --- 1. Extract Background ---
    bg_slices = []
    for bg_ap in background_apertures:
        y_start, y_end = bg_ap['y_start'], bg_ap['y_end']
        x_start, x_end = bg_ap['x_start'], bg_ap['x_end']
        bg_slices.append(signal_cube[:, y_start:y_end+1, x_start:x_end+1])
    
    background_pixels = xp.concatenate(bg_slices, axis=1)
    bg_axis = (1, 2) if instrument == 'FGS1' else 1
    median_background_per_pixel = xp.nanmedian(background_pixels, axis=bg_axis, keepdims=True)

    # --- 2. Extract Signal ---
    sig_y_start, sig_y_end = signal_aperture['y_start'], signal_aperture['y_end']
    sig_x_start, sig_x_end = signal_aperture['x_start'], signal_aperture['x_end']
    signal_pixels = signal_cube[:, sig_y_start:sig_y_end+1, sig_x_start:sig_x_end+1]
    
    # --- 3. Sum Flux and Subtract Background (with instrument-specific logic) ---
    if instrument == 'FGS1':
        # OPTIMIZATION: For FGS1, sum first, then clip temporally. This is much faster.
        raw_flux = xp.nansum(signal_pixels, axis=(1, 2))
        num_pixels_in_aperture = (sig_y_end - sig_y_start + 1) * (sig_x_end - sig_x_start + 1)
        background_contribution = median_background_per_pixel.squeeze() * num_pixels_in_aperture
        light_curves_with_outliers = raw_flux - background_contribution
        light_curves_with_outliers = light_curves_with_outliers[:, xp.newaxis]
    else: # For AIRS-CH0 (spectrometer)
        # For AIRS, we must clip spatially first to remove cosmic rays from each column.
        if hasattr(signal_pixels, 'get'):
            signal_pixels_np = signal_pixels.get()
        else:
            signal_pixels_np = signal_pixels
        clipped_signal_pixels_np = sigma_clip(signal_pixels_np, sigma=5, axis=1)
        clipped_signal_pixels = xp.asarray(clipped_signal_pixels_np)
        raw_flux = xp.nansum(clipped_signal_pixels, axis=1)
        num_pixels_in_aperture = sig_y_end - sig_y_start + 1
        background_contribution = median_background_per_pixel.squeeze(axis=1) * num_pixels_in_aperture
        light_curves_with_outliers = raw_flux - background_contribution

    # --- 4. Final Temporal Outlier Rejection ---
    if hasattr(light_curves_with_outliers, 'get'):
        light_curves_np = light_curves_with_outliers.get()
    else:
        light_curves_np = light_curves_with_outliers
        
    final_light_curves_np = sigma_clip(light_curves_np, sigma=5, axis=0)
    
    final_light_curves_clean = xp.asarray(light_curves_with_outliers)
    final_light_curves_clean[xp.asarray(final_light_curves_np.mask)] = xp.nan

    return final_light_curves_clean
