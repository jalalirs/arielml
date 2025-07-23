"""
Noise Model - Models instrument noise.

This model handles the noise component in the Bayesian framework.
"""

import numpy as np
from typing import Tuple, Optional

# Import backend management
try:
    from ..backend import get_backend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False


class NoiseModel:
    """Models instrument noise."""
    
    def __init__(self, backend: str = 'cpu'):
        self.noise_levels = None
        self.backend = backend
        
        # Get appropriate backend
        if BACKEND_AVAILABLE:
            self.xp, self.backend_name = get_backend(backend)
        else:
            self.xp = np
            self.backend_name = 'cpu'
    
    def fit(self, flux: np.ndarray, points_per_bin: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit noise levels per wavelength."""
        # Convert to backend arrays
        flux = self.xp.asarray(flux)
        
        # For GPU, move to CPU for optimization
        if self.backend == 'gpu':
            flux_cpu = flux.get() if hasattr(flux, 'get') else flux
            points_per_bin_cpu = points_per_bin.get() if hasattr(points_per_bin, 'get') else points_per_bin if points_per_bin is not None else None
        else:
            flux_cpu = flux
            points_per_bin_cpu = points_per_bin
        
        # Estimate noise from residuals
        if points_per_bin_cpu is not None:
            # Scale noise by number of points per bin
            self.noise_levels = np.nanstd(flux_cpu, axis=0) / np.sqrt(points_per_bin_cpu)
        else:
            self.noise_levels = np.nanstd(flux_cpu, axis=0)
        
        # Convert to backend if needed
        if self.backend_name == 'gpu':
            self.noise_levels = self.xp.asarray(self.noise_levels)
        
        # Ensure we return a NumPy array
        if hasattr(self.noise_levels, 'get'):  # CuPy array
            return self.noise_levels.get()
        elif hasattr(self.noise_levels, 'cpu'):  # PyTorch tensor
            return self.noise_levels.get() if hasattr(self.noise_levels, 'get') else self.noise_levels.cpu().numpy()
        else:
            return self.noise_levels
    
    def predict(self, shape: Tuple[int, int]) -> np.ndarray:
        """Predict noise levels for given shape."""
        if self.noise_levels is None:
            raise ValueError("Model not fitted yet.")
        
        # Use backend-appropriate tile operation
        if self.backend_name == 'gpu':
            # For GPU, we need to handle this carefully
            if hasattr(self.noise_levels, 'device'):
                # Noise levels are already on GPU
                return self.xp.tile(self.noise_levels, (shape[0], 1))
            else:
                # Noise levels are on CPU, convert to GPU
                noise_gpu = self.xp.asarray(self.noise_levels)
                return self.xp.tile(noise_gpu, (shape[0], 1))
        else:
            return self.xp.tile(self.noise_levels, (shape[0], 1))
    
    def get_backend_info(self) -> str:
        """Get information about which backend is being used."""
        return f"NoiseModel using {self.backend_name} backend" 