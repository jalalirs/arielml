# arielml/backend.py

import numpy
import importlib

# --- NumPy/CuPy Backend Management ---
cupy_spec = importlib.util.find_spec("cupy")
if cupy_spec:
    try:
        import cupy
        if cupy.is_available():
            GPU_ENABLED = True
            print("CuPy GPU backend available")
        else:
            GPU_ENABLED = False
    except ImportError:
        GPU_ENABLED = False
else:
    GPU_ENABLED = False

# --- GPyTorch Backend Management ---
torch_spec = importlib.util.find_spec("torch")
gpytorch_spec = importlib.util.find_spec("gpytorch")
if torch_spec and gpytorch_spec:
    try:
        import torch
        if torch.cuda.is_available():
            GP_GPU_ENABLED = True
            print("GPyTorch GPU backend available")
        else:
            GP_GPU_ENABLED = False
            print("GPyTorch found, but no CUDA device. GP-GPU will run on CPU.")
    except ImportError:
        GP_GPU_ENABLED = False
else:
    GP_GPU_ENABLED = False


def get_backend(backend_name: str = 'cpu'):
    """
    Returns the appropriate numerical library (backend) and its name for CuPy/NumPy.
    """
    if backend_name == 'gpu' and GPU_ENABLED:
        import cupy
        return cupy, 'gpu'
    
    return numpy, 'cpu'
