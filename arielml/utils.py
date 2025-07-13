# arielml/utils.py
import numpy as np

def calculate_transit_mask(time, period, semi_major_axis, stellar_radius, inclination, xp):
    """
    Calculates a boolean mask identifying the in-transit portions of a light curve.
    
    This function uses a standard geometric model of the transit.
    
    Args:
        time (xp.ndarray): Array of observation times in days.
        period (float): Orbital period of the planet in days.
        semi_major_axis (float): Semi-major axis of the orbit in AU.
        stellar_radius (float): Radius of the star in solar radii.
        inclination (float): Orbital inclination in degrees.
        xp (module): The numerical backend (numpy or cupy).

    Returns:
        xp.ndarray: A boolean array where True indicates an in-transit data point.
    """
    # Convert stellar radius from Solar radii to Astronomical Units (AU) for consistency.
    R_sun_to_AU = 0.00465047
    stellar_radius_au = stellar_radius * R_sun_to_AU

    # Convert inclination from degrees to radians for trigonometric functions.
    inc_rad = xp.deg2rad(inclination)

    # Calculate the full duration of the transit (from first to last contact).
    # This formula is derived from the geometry of the star-planet system.
    # See, e.g., Winn (2010), "Transits and Occultations", Eq. 14.
    # We assume a circular orbit (e=0).
    transit_duration_days = (period / np.pi) * xp.arcsin(
        (stellar_radius_au / semi_major_axis) * (1 / xp.sin(inc_rad))
    )
    
    # Calculate the duration in terms of orbital phase.
    transit_duration_phase = transit_duration_days / period

    # Calculate the orbital phase. We assume t0 (mid-transit time) is 0.
    # The phase is calculated from -0.5 to 0.5 for easier masking around zero.
    phase = (time / period) % 1.0
    # Center the phase on 0 by wrapping values > 0.5 to the negative side.
    phase = xp.where(phase > 0.5, phase - 1.0, phase)

    # The transit is centered at phase 0. We need half the duration for masking.
    half_duration_phase = transit_duration_phase / 2.0
    
    # The mask is True for any point whose absolute phase is within half the transit duration.
    in_transit_mask = xp.abs(phase) < half_duration_phase
    
    return in_transit_mask
