import os
from pathlib import Path

# --- Core Paths ---
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output"

# --- Detector Constants ---
ADC_GAIN = 0.4369
ADC_OFFSET = -1000

# --- Image Dimensions ---
# (frames, height, width)
INSTRUMENT_SHAPES = {
    "AIRS-CH0": (11250, 32, 356),
    "FGS1": (135000, 32, 32),
}

# --- Calibration Defaults ---
CALIBRATION_OPTIONS = {
    "apply_adc": True,
    "mask_pixels": True,
    "apply_linearity": False,
    "subtract_dark": True,
    "perform_cds": True,
    "apply_flat": True,
}

# --- NEW: Photometry Settings ---
# Defines the rectangular regions for signal and background extraction.
# Coordinates are inclusive pixel indices [start, end].
PHOTOMETRY_APERTURES = {
    "AIRS-CH0": {
        "signal": {"y_start": 10, "y_end": 22, "x_start": 0, "x_end": 355},
        "background": [
            {"y_start": 2, "y_end": 8, "x_start": 0, "x_end": 355},
            {"y_start": 24, "y_end": 30, "x_start": 0, "x_end": 355},
        ]
    },
    "FGS1": {
        "signal": {"y_start": 8, "y_end": 24, "x_start": 0, "x_end": 31},
        "background": [
            {"y_start": 2, "y_end": 6, "x_start": 0, "x_end": 31},
            {"y_start": 26, "y_end": 30, "x_start": 0, "x_end": 31},
        ]
    }
}

# --- Detrending Hyperparameters ---
# Legacy hyperparameters extracted from arielgp pickle files

# AIRS-CH0 drift hyperparameters
AIRS_DRIFT_HYPERPARAMS = {
    "average_sigmas": [
        2.19103084e-05, 8.33059374e-05, 1.15768805e-03, 2.55943982e-03,
        5.42496576e-04, 2.55943982e-06
    ],
    "spectral_sigmas": [
        3.09492878e-05, 7.50746445e-06, 1.28501279e-04, 1.64698433e-05,
        1.48351827e-05, 4.61709836e-06, 1.41905849e-05, 1.13769278e-04,
        1.88183894e-03, 2.64610783e-04, 4.46907421e-05, 2.83404837e-05,
        1.19528786e-03, 1.79849635e-04, 1.17774865e-04, 1.88999464e-06
    ]
}

# FGS1 drift hyperparameters  
FGS_DRIFT_HYPERPARAMS = {
    "average_sigmas": [
        2.62636798e-05, 5.46354934e-06, 1.37964537e-05, 1.92727785e-04,
        1.28351865e-03, 2.10931505e-03, 1.16662041e-03, 2.12891411e-06
    ]
}

# Default detrending parameters
DEFAULT_DETRENDING_PARAMS = {
    "AIRS-CH0": {
        "avg_kernel": "Matern32",
        "avg_length_scale": 1.0,
        "spectral_kernel": "Matern32",
        "time_scale": 0.4,
        "wavelength_scale": 0.05,
        "use_sparse": True,
        "use_gpu": False,
        "hyperparams": AIRS_DRIFT_HYPERPARAMS
    },
    "FGS1": {
        "kernel": "Matern32", 
        "length_scale": 1.0,
        "use_gpu": False,
        "hyperparams": FGS_DRIFT_HYPERPARAMS
    }
}