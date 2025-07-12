import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from arielml.config import DATASET_DIR, INSTRUMENT_SHAPES

def load_signal_file(
    planet_id: str,
    instrument: str,
    obs_id: int = 0,
    split: str = "train"
) -> np.ndarray:
    """Loads a single signal file and reshapes it to its image dimensions."""
    file_path = DATASET_DIR / split / str(planet_id) / f"{instrument}_signal_{obs_id}.parquet"
    df = pd.read_parquet(file_path)
    shape = INSTRUMENT_SHAPES[instrument]
    # The number of frames can vary, so we use -1 to infer it.
    return df.values.astype(np.float64).reshape((-1, shape[1], shape[2]))

def load_calibration_files(
    planet_id: str,
    instrument: str,
    obs_id: int = 0,
    split: str = "train"
) -> Dict[str, Any]:
    """Loads all relevant calibration files for a given observation."""
    calib_dir = DATASET_DIR / split / str(planet_id) / f"{instrument}_calibration_{obs_id}"
    shape = INSTRUMENT_SHAPES[instrument]
    
    calib_data = {}
    for calib_type in ["dark", "dead", "flat", "linear_corr", "read"]:
        file_path = calib_dir / f"{calib_type}.parquet"
        if not file_path.exists():
            continue
        
        df = pd.read_parquet(file_path)
        
        if calib_type == "linear_corr":
            # Reshape linearity correction coeffs
            calib_data[calib_type] = df.values.astype(np.float64).reshape((-1, shape[1], shape[2]))
        else:
            # Reshape other 2D calibration frames
            calib_data[calib_type] = df.values.astype(np.float64).reshape((shape[1], shape[2]))
            
    return calib_data

def load_axis_info() -> pd.DataFrame:
    """Loads the axis_info metadata file."""
    return pd.read_parquet(DATASET_DIR / "axis_info.parquet")

def load_star_info(split: str = "train") -> pd.DataFrame:
    """
    Loads the star information metadata file for a given split and sets
    planet_id as the index for easy lookup.
    """
    file_path = DATASET_DIR / f"{split}_star_info.csv"
    if not file_path.exists():
        return None
    df = pd.read_csv(file_path)
    df = df.set_index('planet_id')
    return df
