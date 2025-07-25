"""
Calibration Models - Optimize uncertainty calibration and mean bias correction.

These models implement the SigmaFudger and MeanBiasFitter  to improve
prediction quality by calibrating uncertainties and correcting mean bias.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.optimize import minimize_scalar
from .base import BaseModel
from ..backend import get_backend


class SigmaFudger(BaseModel):
    """
    Multiplies all sigma values to optimize training score.
    
    This model finds the optimal scaling factor for uncertainties by minimizing
    the training score on available ground truth data.
    """
    
    def __init__(self, backend: str = 'cpu'):
        super().__init__(backend=backend)
        self.fudge_value = 1.0
        self.is_fitted = False
        self.backend = backend
        self.xp, _ = get_backend(backend)
    
    def fit(self, predictions: np.ndarray, uncertainties: np.ndarray, 
            ground_truth: np.ndarray, **kwargs) -> 'SigmaFudger':
        """
        Fit the sigma fudger by finding optimal scaling factor.
        
        Args:
            predictions: Model predictions
            uncertainties: Model uncertainties
            ground_truth: Ground truth values
            
        Returns:
            Self for chaining
        """
        # Convert to backend arrays
        predictions = self.xp.asarray(predictions)
        uncertainties = self.xp.asarray(uncertainties)
        ground_truth = self.xp.asarray(ground_truth)
        
        # Optimize using scipy (scipy doesn't support CuPy, so we use numpy for optimization)
        if self.backend == 'gpu':
            # For GPU, we need to move data to CPU for scipy optimization
            predictions_cpu = predictions.get() if hasattr(predictions, 'get') else predictions
            uncertainties_cpu = uncertainties.get() if hasattr(uncertainties, 'get') else uncertainties
            ground_truth_cpu = ground_truth.get() if hasattr(ground_truth, 'get') else ground_truth
            
            def score_fudged_cpu(log_fudge_value):
                fudge_value = np.exp(log_fudge_value)
                fudged_uncertainties = uncertainties_cpu * fudge_value
                return self._calculate_score(predictions_cpu, fudged_uncertainties, ground_truth_cpu)
            
            result = minimize_scalar(score_fudged_cpu, method='brent')
        else:
            # For CPU, we can use the arrays directly
            def score_fudged(log_fudge_value):
                fudge_value = np.exp(log_fudge_value)
                fudged_uncertainties = uncertainties * fudge_value
                return self._calculate_score(predictions, fudged_uncertainties, ground_truth)
            
            result = minimize_scalar(score_fudged, method='brent')
        
        self.fudge_value = np.exp(result.x)
        self.is_fitted = True
        
        return self
    
    def predict(self, uncertainties: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply sigma fudging to uncertainties.
        
        Args:
            uncertainties: Raw uncertainties
            
        Returns:
            Fudged uncertainties
        """
        if not self.is_fitted:
            raise ValueError("SigmaFudger must be fitted before prediction")
        
        uncertainties = self.xp.asarray(uncertainties)
        result = uncertainties * self.fudge_value
        
        # Convert back to NumPy array before returning
        if hasattr(result, 'get'):
            return result.get()
        else:
            return result
    
    def predict_covariance(self, covariance: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply sigma fudging to covariance matrix.
        
        Args:
            covariance: Raw covariance matrix
            
        Returns:
            Fudged covariance matrix
        """
        if not self.is_fitted:
            raise ValueError("SigmaFudger must be fitted before prediction")
        
        covariance = self.xp.asarray(covariance)
        result = covariance * (self.fudge_value ** 2)
        
        # Convert back to NumPy array before returning
        if hasattr(result, 'get'):
            return result.get()
        else:
            return result
    
    def _calculate_score(self, predictions: np.ndarray, uncertainties: np.ndarray, 
                        ground_truth: np.ndarray) -> float:
        """
        Calculate the score to minimize (negative log-likelihood).
        
        Args:
            predictions: Model predictions
            uncertainties: Model uncertainties
            ground_truth: Ground truth values
            
        Returns:
            Score value (lower is better)
        """
        # Convert to numpy for scipy compatibility
        if self.backend == 'gpu':
            predictions = predictions.get() if hasattr(predictions, 'get') else predictions
            uncertainties = uncertainties.get() if hasattr(uncertainties, 'get') else uncertainties
            ground_truth = ground_truth.get() if hasattr(ground_truth, 'get') else ground_truth
        
        # Calculate negative log-likelihood
        residuals = predictions - ground_truth
        log_likelihood = -0.5 * np.sum((residuals / uncertainties) ** 2 + np.log(2 * np.pi * uncertainties ** 2))
        return -log_likelihood
    
    def get_backend_info(self) -> str:
        """Get backend information."""
        return f"SigmaFudger using {self.backend} backend"


class MeanBiasFitter(BaseModel):
    """
    Applies scaling to mean predictions to correct bias.
    
    This model finds the optimal bias correction by minimizing the mean squared error
    between predictions and ground truth.
    """
    
    def __init__(self, backend: str = 'cpu'):
        super().__init__(backend=backend)
        self.bias = 0.0
        self.scale = 1.0
        self.is_fitted = False
        self.backend = backend
        self.xp, _ = get_backend(backend)
    
    def fit(self, predictions: np.ndarray, ground_truth: np.ndarray, **kwargs) -> 'MeanBiasFitter':
        """
        Fit the mean bias fitter by finding optimal bias and scale.
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth values
            
        Returns:
            Self for chaining
        """
        # Convert to backend arrays
        predictions = self.xp.asarray(predictions)
        ground_truth = self.xp.asarray(ground_truth)
        
        # For GPU, move to CPU for optimization
        if self.backend == 'gpu':
            predictions_cpu = predictions.get() if hasattr(predictions, 'get') else predictions
            ground_truth_cpu = ground_truth.get() if hasattr(ground_truth, 'get') else ground_truth
        else:
            predictions_cpu = predictions
            ground_truth_cpu = ground_truth
        
        # Ensure we have numpy arrays for the optimization (but don't use np.asarray on CuPy arrays)
        if hasattr(predictions_cpu, 'get'):
            predictions_cpu = predictions_cpu.get()
        if hasattr(ground_truth_cpu, 'get'):
            ground_truth_cpu = ground_truth_cpu.get()
        
        # Fit linear model: ground_truth = scale * predictions + bias
        # Using least squares
        A = np.column_stack([predictions_cpu.flatten(), np.ones_like(predictions_cpu.flatten())])
        b = ground_truth_cpu.flatten()
        
        try:
            # Use numpy's least squares solver
            solution = np.linalg.lstsq(A, b, rcond=None)[0]
            self.scale = float(solution[0])  # Ensure scalar values
            self.bias = float(solution[1])
        except np.linalg.LinAlgError:
            # Fallback to simple bias correction if matrix is singular
            self.scale = 1.0
            self.bias = float(np.mean(ground_truth_cpu - predictions_cpu))
        
        self.is_fitted = True
        return self
    
    def predict(self, predictions: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply bias correction to predictions.
        
        Args:
            predictions: Raw predictions
            
        Returns:
            Bias-corrected predictions
        """
        if not self.is_fitted:
            raise ValueError("MeanBiasFitter must be fitted before prediction")
        
        predictions = self.xp.asarray(predictions)
        result = self.scale * predictions + self.bias
        
        # Convert back to NumPy array before returning
        if hasattr(result, 'get'):
            return result.get()
        else:
            return result
    
    def get_params(self) -> dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            'bias': self.bias,
            'scale': self.scale,
            'is_fitted': self.is_fitted
        })
        return params
    
    def set_params(self, **params) -> 'MeanBiasFitter':
        """Set model parameters."""
        super().set_params(**params)
        if 'bias' in params:
            self.bias = params['bias']
        if 'scale' in params:
            self.scale = params['scale']
        if 'is_fitted' in params:
            self.is_fitted = params['is_fitted']
        return self
    
    def get_backend_info(self) -> str:
        """Get backend information."""
        return f"MeanBiasFitter using {self.backend} backend" 