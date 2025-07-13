
def calculate_transit_mask(
    time_array,
    period: float,
    semi_major_axis: float,
    stellar_radius: float,
    inclination: float,
    xp
) -> "xp.ndarray":
    """
    Calculates a boolean mask to identify in-transit data points using the
    specified backend.

    This is a simplified model assuming a circular orbit and a central transit time at t=0
    for the first transit. A full implementation would use the epoch of transit.

    Args:
        time_array (xp.ndarray): The array of observation times in days.
        period (float): The orbital period of the planet in days.
        semi_major_axis (float): The semi-major axis in units of stellar radii.
        stellar_radius (float): The stellar radius (used for unit consistency).
        inclination (float): The orbital inclination in degrees.
        xp (module): The numerical backend (numpy or cupy).

    Returns:
        xp.ndarray: A boolean array, where True indicates an in-transit point.
    """
    # Convert inclination to radians for trigonometric functions
    inc_rad = xp.deg2rad(inclination)

    # Calculate transit duration (from Winn, 2010, Eq. 14)
    transit_duration_days = (period / xp.pi) * xp.arcsin(
        (1 / semi_major_axis) * xp.sqrt(
            (1 + 0)**2 - (semi_major_axis * xp.cos(inc_rad))**2
        ) / xp.sin(inc_rad)
    )
    
    # Assume the transit center is at t=0 and repeats every `period`.
    phase = xp.mod(time_array + period / 2, period) - period / 2

    # A point is in-transit if its phase is within half the duration of the center.
    in_transit_mask = xp.abs(phase) < (transit_duration_days / 2)

    return in_transit_mask