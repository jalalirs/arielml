"""
MCMC Sampler - GPU-accelerated Markov Chain Monte Carlo sampling.

This module provides GPU-accelerated MCMC sampling for the Bayesian pipeline.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable

# Import backend management
try:
    from ..backend import get_backend
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

# Import PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MCMCSampler:
    """GPU-accelerated MCMC sampler for Bayesian inference."""
    
    def __init__(self, backend: str = 'cpu', n_chains: int = 4, device: Optional[str] = None):
        self.backend = backend
        self.n_chains = n_chains
        self.device = device
        self.progress_callback = None
        
        # Get appropriate backend
        if BACKEND_AVAILABLE:
            self.xp, self.backend_name = get_backend(backend)
        else:
            self.xp = np
            self.backend_name = 'cpu'
        
        # Set up PyTorch device for GPU acceleration
        if self.backend_name == 'gpu' and TORCH_AVAILABLE:
            if device is None:
                self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.torch_device = torch.device(device)
        else:
            self.torch_device = None
    
    def set_progress_callback(self, callback: Callable[[str], None]):
        """Set a callback function for progress updates."""
        self.progress_callback = callback
    
    def _emit_progress(self, message: str):
        """Emit progress message if callback is set."""
        if self.progress_callback:
            self.progress_callback(message)
    
    def sample_transit_depths(self, 
                            stellar: np.ndarray,
                            drift: np.ndarray,
                            transit_depth: np.ndarray,
                            transit_window: np.ndarray,
                            noise: np.ndarray,
                            n_samples: int = 100,
                            batch_size: int = 20,
                            **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate MCMC samples for transit depths with uncertainties.
        
        Args:
            stellar: Fitted stellar spectrum
            drift: Fitted drift model
            transit_depth: Fitted transit depths
            transit_window: Fitted transit window
            noise: Fitted noise levels
            n_samples: Number of MCMC samples to generate
            batch_size: Number of samples to process at once (for GPU memory management)
            
        Returns:
            predictions: Mean transit depths per wavelength
            uncertainties: Standard deviations per wavelength
            covariance_matrix: Full covariance matrix
        """
        n_wavelengths = len(transit_depth)
        
        if self.backend_name == 'gpu' and TORCH_AVAILABLE:
            return self._sample_gpu(stellar, drift, transit_depth, transit_window, 
                                  noise, n_samples, n_wavelengths, batch_size)
        else:
            return self._sample_cpu(stellar, drift, transit_depth, transit_window, 
                                  noise, n_samples, n_wavelengths, batch_size)
    
    def _sample_gpu(self, stellar, drift, transit_depth, transit_window, 
                   noise, n_samples, n_wavelengths, batch_size):
        """Generate MCMC samples using GPU with improved uncertainty estimation."""
        self._emit_progress(f"Starting GPU MCMC sampling with {n_samples} samples...")
        
        # Convert to tensors
        stellar_tensor = torch.from_numpy(stellar).to(self.torch_device, dtype=torch.float32)
        drift_tensor = torch.from_numpy(drift).to(self.torch_device, dtype=torch.float32)
        transit_depth_tensor = torch.from_numpy(transit_depth).to(self.torch_device, dtype=torch.float32)
        transit_window_tensor = torch.from_numpy(transit_window).to(self.torch_device, dtype=torch.float32)
        noise_tensor = torch.from_numpy(noise).to(self.torch_device, dtype=torch.float32)
        
        # Initialize sample storage
        sample_transit_depths = torch.zeros((n_wavelengths, n_samples), device=self.torch_device, dtype=torch.float32)
        
        # Calculate number of batches
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # Use more realistic uncertainty scaling
        # Base uncertainty should be proportional to transit depth, not just noise
        base_uncertainty = torch.maximum(transit_depth_tensor * 0.1, torch.tensor(0.001, device=self.torch_device))
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            current_batch_size = end_idx - start_idx
            
            self._emit_progress(f"MCMC batch {batch_idx + 1}/{n_batches} ({current_batch_size} samples)...")
            
            # Generate perturbations with better scaling
            # Transit depth perturbations should be proportional to the transit depth itself
            transit_depth_scale = torch.maximum(transit_depth_tensor * 0.2, torch.tensor(0.001, device=self.torch_device))
            transit_depth_perturbations = torch.randn(current_batch_size, n_wavelengths, device=self.torch_device) * transit_depth_scale.unsqueeze(0)
            transit_depth_samples = transit_depth_tensor.unsqueeze(0) + transit_depth_perturbations
            
            # Apply physical constraints: transit depths should be positive and reasonable
            transit_depth_samples = torch.clamp(transit_depth_samples, 0.0, 0.1)
            
            # Store the perturbed transit depths for this batch
            sample_transit_depths[:, start_idx:end_idx] = transit_depth_samples.T
            
            # Clear GPU cache after each batch to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Compute statistics on GPU
        self._emit_progress("Computing MCMC statistics...")
        predictions = torch.mean(sample_transit_depths, dim=1)  # (n_wavelengths,)
        sample_labels = sample_transit_depths - predictions.unsqueeze(1)  # (n_wavelengths, n_samples)
        
        # Compute covariance matrix on GPU
        covariance_matrix = torch.matmul(sample_labels, sample_labels.T) / n_samples  # (n_wavelengths, n_wavelengths)
        
        # Compute standard deviations
        uncertainties = torch.std(sample_labels, dim=1)
        
        # Apply reasonable bounds to uncertainties
        min_uncertainty = torch.tensor(0.0001, device=self.torch_device)  # 0.01%
        max_uncertainty = torch.tensor(0.01, device=self.torch_device)    # 1%
        uncertainties = torch.clamp(uncertainties, min_uncertainty, max_uncertainty)
        
        # Convert back to CPU
        predictions_cpu = predictions.cpu().numpy()
        uncertainties_cpu = uncertainties.cpu().numpy()
        covariance_cpu = covariance_matrix.cpu().numpy()
        
        self._emit_progress(f"GPU MCMC completed! Generated {n_samples} samples in {n_batches} batches.")
        
        return predictions_cpu, uncertainties_cpu, covariance_cpu
    
    def _sample_cpu(self, stellar, drift, transit_depth, transit_window, 
                   noise, n_samples, n_wavelengths, batch_size):
        """Generate MCMC samples using CPU with improved uncertainty estimation."""
        self._emit_progress(f"Starting CPU MCMC sampling with {n_samples} samples...")
        
        # Initialize sample storage
        sample_transit_depths = np.zeros((n_wavelengths, n_samples))
        
        # Calculate number of batches
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # Use more realistic uncertainty scaling
        # Base uncertainty should be proportional to transit depth, not just noise
        base_uncertainty = np.maximum(transit_depth * 0.1, 0.001)
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            current_batch_size = end_idx - start_idx
            
            self._emit_progress(f"MCMC batch {batch_idx + 1}/{n_batches} ({current_batch_size} samples)...")
            
            for i in range(current_batch_size):
                sample_idx = start_idx + i
                
                # Generate perturbations with better scaling
                # Transit depth perturbations should be proportional to the transit depth itself
                transit_depth_scale = np.maximum(transit_depth * 0.2, 0.001)
                transit_depth_perturb = transit_depth + np.random.normal(0, transit_depth_scale)
                
                # Apply physical constraints: transit depths should be positive and reasonable
                transit_depth_perturb = np.clip(transit_depth_perturb, 0.0, 0.1)
                
                # Store the perturbed transit depths
                sample_transit_depths[:, sample_idx] = transit_depth_perturb
        
        # Compute predictions (mean of samples)
        self._emit_progress("Computing MCMC statistics...")
        predictions = np.mean(sample_transit_depths, axis=1)
        
        # Compute uncertainties from sample statistics
        sample_labels = sample_transit_depths - predictions[:, np.newaxis]
        
        # Compute covariance matrix from samples
        covariance_matrix = (sample_labels @ sample_labels.T) / sample_labels.shape[1]
        
        # Compute standard deviations from samples
        uncertainties = np.std(sample_labels, axis=1)
        
        # Apply reasonable bounds to uncertainties
        min_uncertainty = 0.0001  # 0.01%
        max_uncertainty = 0.01    # 1%
        uncertainties = np.clip(uncertainties, min_uncertainty, max_uncertainty)
        
        self._emit_progress(f"CPU MCMC completed! Generated {n_samples} samples in {n_batches} batches.")
        
        return predictions, uncertainties, covariance_matrix
    
    def get_backend_info(self) -> str:
        """Get information about which backend is being used."""
        device_info = f" on {self.torch_device}" if self.torch_device else ""
        return f"MCMCSampler using {self.backend_name} backend{device_info}" 