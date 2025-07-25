"""
Transit Window Model - Models the transit window function (ingress/egress timing).

This model handles the transit window component in the Bayesian framework.
"""

import numpy as np
from typing import Optional




class TransitWindowModel:
    """Models the transit window function (ingress/egress timing)."""
    
    def __init__(self):
        self.ingress_time: Optional[float] = None
        self.egress_time: Optional[float] = None
        self.width: Optional[float] = None
        
        
        self.transit_splines = None
    
    def fit(self, time: np.ndarray, transit_mask: np.ndarray) -> np.ndarray:
        """Fit transit window parameters."""
        # Simple approach: estimate from transit mask
        transit_times = time[transit_mask]
        
        if len(transit_times) > 0:
            self.ingress_time = float(np.min(transit_times))
            self.egress_time = float(np.max(transit_times))
            self.width = float((self.egress_time - self.ingress_time) / 2.0)
        else:
            # Default values
            self.ingress_time = float(np.median(time))
            self.egress_time = float(np.median(time))
            self.width = 0.1
        
        return self._predict_window(time)
    
    def _predict_window(self, time: np.ndarray) -> np.ndarray:
        """Predict transit window function."""
        if self.ingress_time is None:
            raise ValueError("Model not fitted yet.")
        
        # Simple transit window function
        window = np.ones_like(time)
        
        # Create smooth transition
        center = (self.ingress_time + self.egress_time) / 2.0
        
        # Ingress
        ingress_mask = (time < center) & (time >= self.ingress_time - self.width)
        if np.any(ingress_mask):
            ingress_progress = (time[ingress_mask] - (self.ingress_time - self.width)) / (2 * self.width)
            window[ingress_mask] = 1.0 - 0.1 * ingress_progress
        
        # Transit
        transit_mask = (time >= center - self.width) & (time <= center + self.width)
        window[transit_mask] = 0.9
        
        # Egress
        egress_mask = (time > center) & (time <= self.egress_time + self.width)
        if np.any(egress_mask):
            egress_progress = ((self.egress_time + self.width) - time[egress_mask]) / (2 * self.width)
            window[egress_mask] = 1.0 - 0.1 * egress_progress
        
        return window 