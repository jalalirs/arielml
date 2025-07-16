"""
Stellar Spectrum Model - Models the stellar spectrum component.

This model handles the stellar spectrum component in the Bayesian framework.
"""

import numpy as np
from typing import Tuple, Optional

# Import backend management
try:
    from ..backend import get_backend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False


class StellarSpectrumModel:
    """Models the stellar spectrum (uncorrelated per wavelength)."""
    
    def __init__(self, sigma: float = 1e20, backend: str = 'cpu'):
        self.sigma = sigma  # 'infinite' prior variance
        self.spectrum = None
        self.backend = backend
        
        # Get appropriate backend
        if BACKEND_AVAILABLE:
            self.xp, self.backend_name = get_backend(backend)
        else:
            self.xp = np
            self.backend_name = 'cpu'
    
    def fit(self, flux: np.ndarray) -> np.ndarray:
        """Fit stellar spectrum using median per wavelength."""
        # Convert to backend array if needed
        if self.backend_name == 'gpu' and not hasattr(flux, 'device'):
            # If we're on GPU but flux is numpy, we need to handle this
            # For now, keep it simple and use CPU for this operation
            flux_backend = flux
        else:
            flux_backend = flux
        
        # Simple approach: use median per wavelength
        # Note: GPU median operations can be complex, so we'll use CPU for now
        # In a full implementation, this could use GPU-accelerated statistics
        if self.backend_name == 'gpu':
            # For GPU, we'll compute on CPU for now (simple operation)
            if hasattr(flux, 'get'):  # CuPy array
                flux_cpu = flux.get()
            elif hasattr(flux, 'cpu'):  # PyTorch tensor
                flux_cpu = flux.cpu().numpy()
            else:
                flux_cpu = flux
            self.spectrum = np.nanmedian(flux_cpu, axis=0)
            # Always return NumPy array to avoid CuPy conversion issues
        else:
            self.spectrum = self.xp.nanmedian(flux_backend, axis=0)
        
        # Ensure we return a NumPy array
        if hasattr(self.spectrum, 'get'):  # CuPy array
            return self.spectrum.get()
        elif hasattr(self.spectrum, 'cpu'):  # PyTorch tensor
            return self.spectrum.cpu().numpy()
        else:
            return self.spectrum
    
    def predict(self, shape: Tuple[int, int]) -> np.ndarray:
        """Predict stellar spectrum for given shape."""
        if self.spectrum is None:
            raise ValueError("Model not fitted yet.")
        
        # Use backend-appropriate tile operation
        if self.backend_name == 'gpu':
            # For GPU, we need to handle this carefully
            if hasattr(self.spectrum, 'device'):
                # Spectrum is already on GPU
                return self.xp.tile(self.spectrum, (shape[0], 1))
            else:
                # Spectrum is on CPU, convert to GPU
                spectrum_gpu = self.xp.asarray(self.spectrum)
                return self.xp.tile(spectrum_gpu, (shape[0], 1))
        else:
            return self.xp.tile(self.spectrum, (shape[0], 1))
    
    def get_backend_info(self) -> str:
        """Get information about which backend is being used."""
        return f"StellarSpectrumModel using {self.backend_name} backend" 