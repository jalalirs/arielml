# arielml/utils/transit_masks.py
import numpy as np

def calculate_transit_mask_physical(time, period, semi_major_axis, stellar_radius, inclination, xp):
    """
    Calculates a boolean mask based on the physical, geometric model of the transit.
    """
    # Convert stellar radius from Solar radii to Astronomical Units (AU) for consistency.
    R_sun_to_AU = 0.00465047
    stellar_radius_au = stellar_radius * R_sun_to_AU

    # Convert inclination from degrees to radians for trigonometric functions.
    inc_rad = xp.deg2rad(inclination)

    # Calculate the full duration of the transit (from first to last contact).
    transit_duration_days = (period / np.pi) * xp.arcsin(
        (stellar_radius_au / semi_major_axis) * (1 / xp.sin(inc_rad))
    )
    
    transit_duration_phase = transit_duration_days / period

    # Calculate the orbital phase, centered on 0.
    phase = (time / period) % 1.0
    phase = xp.where(phase > 0.5, phase - 1.0, phase)

    half_duration_phase = transit_duration_phase / 2.0
    
    in_transit_mask = xp.abs(phase) < half_duration_phase
    
    return in_transit_mask


def find_transit_mask_empirical(time, flux, Navg=250, Noffset=150, fit_order=5, xp=np):
    """
    Finds the transit window by looking for the steepest drop (ingress) and
    rise (egress) in the light curve data itself. This is a data-driven approach.
    
    Args:
        time (xp.ndarray): The time array of the observation.
        flux (xp.ndarray): The light curve flux (1D array).
        Navg (int): The window size for the moving average smoothing.
        Noffset (int): The offset for calculating the numerical derivative.
        fit_order (int): The polynomial order for detrending the smoothed curve.
        xp (module): The numerical backend (numpy or cupy).

    Returns:
        xp.ndarray: A boolean array where True indicates an in-transit data point.
    """
    print("DEBUG: find_transit_mask_empirical() called")
    print(f"DEBUG: Input shapes - time: {time.shape}, flux: {flux.shape}")
    
    # This function must run on the CPU with NumPy due to polyfit and cumsum.
    is_gpu = (xp.__name__ == 'cupy')
    if is_gpu:
        print("DEBUG: Converting GPU arrays to CPU")
        flux_np = flux.get()
        time_np = time.get()
    else:
        flux_np = flux
        time_np = time

    print("DEBUG: Starting moving average smoothing...")
    # --- 1. Smooth the light curve with a moving average ---
    # The cumsum trick is a fast way to implement a moving average.
    ret = np.cumsum(flux_np, dtype=float)
    ret[Navg:] = ret[Navg:] - ret[:-Navg]
    data_smoothed = ret[Navg - 1:] / Navg
    
    print("DEBUG: Starting polynomial detrending...")
    # --- 2. Detrend the smoothed curve with a polynomial ---
    x_smooth = np.arange(len(data_smoothed))
    try:
        poly_coeffs = np.polyfit(x_smooth, data_smoothed, fit_order)
    except np.linalg.LinAlgError:
        # If the polynomial fit fails, fall back to a lower order.
        print("DEBUG: Polynomial fit failed, falling back to order 1")
        poly_coeffs = np.polyfit(x_smooth, data_smoothed, 1)
        
    poly_fit = np.poly1d(poly_coeffs)
    data_detrended = data_smoothed - poly_fit(x_smooth)
    
    print("DEBUG: Finding derivative...")
    # --- 3. Find the derivative by shifting and subtracting ---
    # This finds where the slope changes most dramatically.
    diff = data_detrended[Noffset:] - data_detrended[:-Noffset]
    
    print("DEBUG: Finding ingress/egress...")
    # --- 4. Find ingress/egress and create the mask ---
    # Find the index of the minimum (steepest drop) and maximum (steepest rise).
    # Add offsets to account for the smoothing and differencing windows.
    idx_ingress = int(np.argmin(diff) + (Navg / 2) + (Noffset / 2))
    idx_egress = int(np.argmax(diff) + (Navg / 2) + (Noffset / 2))

    # Ensure ingress comes before egress
    if idx_ingress > idx_egress:
        idx_ingress, idx_egress = idx_egress, idx_ingress

    print(f"DEBUG: Ingress at {idx_ingress}, egress at {idx_egress}")
    
    # Create a boolean mask that is True between ingress and egress
    mask_np = np.zeros_like(time_np, dtype=bool)
    mask_np[idx_ingress:idx_egress] = True
    
    print("DEBUG: Converting back to original backend...")
    # Convert back to a GPU array if necessary
    result = xp.asarray(mask_np) if is_gpu else mask_np
    print("DEBUG: find_transit_mask_empirical() completed")
    return result 