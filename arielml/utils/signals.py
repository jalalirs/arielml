# arielml/utils/signals.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any

class DetrendingStep(Enum):
    """Specific steps within the detrending process."""
    INITIALIZING = "initializing"
    FITTING_COMMON_MODE = "fitting_common_mode"
    FITTING_PER_WAVELENGTH = "fitting_per_wavelength"
    FITTING_DRIFT = "fitting_drift"
    FITTING_TRANSIT = "fitting_transit"
    SAMPLING = "sampling"
    FINALIZING = "finalizing"

@dataclass
class DetrendingProgress:
    """Progress signal for detrending operations."""
    step: DetrendingStep
    progress: float  # 0.0 to 1.0
    message: str
    current_wavelength: Optional[int] = None
    total_wavelengths: Optional[int] = None
    current_iteration: Optional[int] = None
    total_iterations: Optional[int] = None
    error: Optional[str] = None
    data: Optional[Any] = None 