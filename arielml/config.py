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