# arielml/backend.py

import numpy
import importlib

# --- Backend Management ---
# This module dynamically selects the numerical backend (cupy for GPU, numpy for CPU).

# Check if cupy is installed and a GPU is available
cupy_spec = importlib.util.find_spec("cupy")
if cupy_spec:
    try:
        import cupy
        # Check if a GPU device is actually available
        if cupy.is_available():
            GPU_ENABLED = True
            print("GPU backend available")
        else:
            GPU_ENABLED = False
    except ImportError:
        GPU_ENABLED = False
else:
    GPU_ENABLED = False

def get_backend(backend_name: str = 'cpu'):
    """
    Returns the appropriate numerical library (backend) and its name.

    Args:
        backend_name (str): 'gpu' or 'cpu'. If 'gpu' is requested but not
                            available, it falls back to 'cpu'.

    Returns:
        tuple: A tuple containing the module (numpy or cupy) and its name ('cpu' or 'gpu').
    """
    if backend_name == 'gpu' and GPU_ENABLED:
        import cupy
        return cupy, 'gpu'
    
    return numpy, 'cpu'