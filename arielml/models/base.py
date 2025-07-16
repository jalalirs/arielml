from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np

class BaseModel(ABC):
    """
    Abstract base class for all models in arielml.
    
    Models can be either:
    1. Detrending models (pre-processing)
    2. Transit modeling models (extract transit depths)
    3. Complete models (do both detrending and transit modeling)
    """
    
    def __init__(self, **kwargs):
        self.model_params = kwargs
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, time: np.ndarray, flux: np.ndarray, transit_mask: np.ndarray, **kwargs) -> 'BaseModel':
        """Fit the model to the data."""
        pass
    
    @abstractmethod
    def predict(self, time: np.ndarray, flux: np.ndarray, transit_mask: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on the data.
        
        Returns:
            predictions: Predicted values (e.g., transit depths)
            additional_output: Additional model output (varies by model)
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model_params.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        self.model_params.update(params)
        return self 