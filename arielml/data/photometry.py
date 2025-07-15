# arielml/data/photometry.py

from typing import Dict, List
from astropy.stats import sigma_clip
import numpy as np # Import numpy for fill_value

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
    """
    # --- DEBUG: Check the state of the input data ---
    print(f"\n--- Entering extract_aperture_photometry for {instrument} ---")
    print(f"Input signal_cube shape: {signal_cube.shape}")
    print(f"Input NaNs in signal_cube: {xp.sum(xp.isnan(signal_cube))}")

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
    
    # --- 3. Sum Flux and Subtract Background ---
    if instrument == 'FGS1':
        all_nan_mask = xp.all(xp.isnan(signal_pixels), axis=(1, 2))
        raw_flux = xp.nansum(signal_pixels, axis=(1, 2))
        raw_flux[all_nan_mask] = xp.nan

        num_pixels_in_aperture = (sig_y_end - sig_y_start + 1) * (sig_x_end - sig_x_start + 1)
        background_contribution = median_background_per_pixel.squeeze() * num_pixels_in_aperture
        light_curves_with_outliers = raw_flux - background_contribution
        light_curves_with_outliers = light_curves_with_outliers[:, xp.newaxis]
    
    else: # For AIRS-CH0
        if hasattr(signal_pixels, 'get'):
            signal_pixels_np = signal_pixels.get()
        else:
            signal_pixels_np = signal_pixels
        
        clipped_signal_pixels_np = sigma_clip(signal_pixels_np, sigma=5, axis=1, masked=False)
        clipped_signal_pixels = xp.asarray(clipped_signal_pixels_np)
        
        all_nan_mask = xp.all(xp.isnan(clipped_signal_pixels), axis=1)

        # --- DEBUG: Check raw flux before and after NaN correction ---
        raw_flux_before_fix = xp.nansum(clipped_signal_pixels, axis=1)
        print(f"NaNs in raw_flux for channel 0 (before fix): {xp.sum(xp.isnan(raw_flux_before_fix[:, 0]))}")
        
        raw_flux = xp.nansum(clipped_signal_pixels, axis=1)
        raw_flux[all_nan_mask] = xp.nan
        print(f"NaNs in raw_flux for channel 0 (after fix): {xp.sum(xp.isnan(raw_flux[:, 0]))}")

        num_pixels_in_aperture = sig_y_end - sig_y_start + 1
        
        # FIX: Properly handle background contribution dimensions for AIRS-CH0
        # median_background_per_pixel has shape (n_frames, 1, 1) for AIRS-CH0
        # We need to broadcast it to match raw_flux shape (n_frames, n_wavelengths)
        background_per_pixel = median_background_per_pixel.squeeze()  # Shape: (n_frames,)
        background_contribution = background_per_pixel[:, xp.newaxis] * num_pixels_in_aperture  # Shape: (n_frames, 1)
        
        light_curves_with_outliers = raw_flux - background_contribution

    # --- 4. Final Temporal Outlier Rejection ---
    if hasattr(light_curves_with_outliers, 'get'):
        light_curves_np = light_curves_with_outliers.get()
    else:
        light_curves_np = light_curves_with_outliers
        
    clipped_lc_masked_array = sigma_clip(light_curves_np, sigma=5, axis=0)
    final_light_curves_np = clipped_lc_masked_array.filled(fill_value=np.nan)
    
    final_light_curves_clean = xp.asarray(final_light_curves_np)
    
    # --- DEBUG: Check the final output ---
    print(f"Final NaNs in light_curves for channel 0: {xp.sum(xp.isnan(final_light_curves_clean[:, 0]))}")
    print(f"Total frames for channel 0: {final_light_curves_clean.shape[0]}")
    print(f"--- Exiting extract_aperture_photometry ---\n")

    return final_light_curves_clean