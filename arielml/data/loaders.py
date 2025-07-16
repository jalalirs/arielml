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

def load_wavelengths() -> np.ndarray:
    """
    Loads the wavelength information for the spectral channels.
    
    Returns:
        Wavelength values in microns as numpy array, or None if not available
    """
    try:
        # Look for wavelength files in the dataset directory
        possible_files = [
            DATASET_DIR / "wavelengths.csv",
            DATASET_DIR.parent / "dataset" / "wavelengths.csv"
        ]
        
        wavelength_path = None
        for file_path in possible_files:
            if file_path.exists():
                wavelength_path = file_path
                break
        
        if wavelength_path is None:
            print("Wavelength file not found. Expected: wavelengths.csv")
            return None
            
        print(f"Loading wavelengths from: {wavelength_path}")
        wavelength_df = pd.read_csv(wavelength_path)
        
        # Extract wavelength columns (wl_1, wl_2, etc.)
        wl_columns = [col for col in wavelength_df.columns if col.startswith('wl_')]
        
        if not wl_columns:
            print(f"No wavelength columns found in wavelength file")
            return None
            
        # Sort columns by wavelength number (wl_1, wl_2, ..., wl_283)
        try:
            wl_columns.sort(key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
        except:
            # If sorting fails, use original order
            pass
        
        # For AIRS-CH0, we need to use the correct wavelength range (columns 39-321)
        # This corresponds to wavelengths 40-322 in the 1-indexed column names
        # But since we're using 0-indexed column numbers, we need columns 39-321
        # The ground truth has 283 wavelengths, so we need to extract the correct range
        
        # Check if we have enough columns for the expected range
        if len(wl_columns) >= 322:  # We need at least 322 columns (0-321)
            # Extract the correct range: columns 40-322 (1-indexed) = indices 39-321 (0-indexed)
            # But ground truth has 283 values, so we need to map correctly
            # The original ariel_gp uses columns 39-321, which gives 283 wavelengths
            wl_columns_subset = wl_columns[39:322]  # 283 wavelengths (39 to 321 inclusive)
        else:
            # Fallback: use all available columns
            wl_columns_subset = wl_columns
        
        wavelength_values = wavelength_df[wl_columns_subset].values.flatten()
        print(f"Loaded {len(wavelength_values)} wavelengths from columns: {wl_columns_subset[:5]}...")
        print(f"Wavelength range: {wavelength_values.min():.2f}-{wavelength_values.max():.2f} μm")
        
        return wavelength_values.astype(np.float64)
        
    except Exception as e:
        print(f"Error loading wavelengths: {e}")
        return None

def load_ground_truth(planet_id: str, split: str = "train") -> np.ndarray:
    """
    Loads ground truth transit depths for a given planet (training set only).
    
    Args:
        planet_id: The planet ID to load ground truth for
        split: Dataset split ('train' or 'test')
        
    Returns:
        Ground truth transit depths as numpy array, or None if not available
    """
    if split != "train":
        return None  # No ground truth for test set
        
    try:
        # Look for ground truth files in the dataset directory
        # Try different possible file names and structures
        possible_files = [
            DATASET_DIR / "train_labels.csv",
            DATASET_DIR / "train.csv",
            DATASET_DIR.parent / "dataset" / "train_labels.csv",
            DATASET_DIR.parent / "dataset" / "train.csv"
        ]
        
        ground_truth_path = None
        for file_path in possible_files:
            if file_path.exists():
                ground_truth_path = file_path
                break
        
        if ground_truth_path is None:
            print("Ground truth file not found. Expected one of: train_labels.csv, train.csv")
            return None
            
        print(f"Loading ground truth from: {ground_truth_path}")
        ground_truth_df = pd.read_csv(ground_truth_path)
        
        # Find the row for this planet
        planet_row = ground_truth_df[ground_truth_df['planet_id'] == int(planet_id)]
        
        if planet_row.empty:
            print(f"No ground truth found for planet {planet_id}")
            return None
            
        # Extract transit depth values from wavelength columns (wl_1, wl_2, etc.)
        wl_columns = [col for col in planet_row.columns if col.startswith('wl_')]
        
        if not wl_columns:
            # Fallback: try to find other possible ground truth columns
            numeric_columns = planet_row.select_dtypes(include=[np.number]).columns
            # Exclude planet_id and any error columns
            wl_columns = [col for col in numeric_columns 
                         if col != 'planet_id' and not col.endswith('_err') and not col.endswith('_error')]
        
        if not wl_columns:
            print(f"No transit depth columns found in ground truth file")
            return None
            
        # Sort columns by wavelength number (wl_1, wl_2, ..., wl_283)
        try:
            wl_columns.sort(key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
        except:
            # If sorting fails, use original order
            pass
        
        # For AIRS-CH0, we need to extract the correct wavelength range
        # The ground truth should have 283 wavelengths corresponding to columns 39-321
        # But the ground truth file might have more columns, so we need to map correctly
        
        # Check if we have enough columns for the expected range
        if len(wl_columns) >= 322:  # We need at least 322 columns (0-321)
            # Extract the correct range: columns 40-322 (1-indexed) = indices 39-321 (0-indexed)
            # This gives us 283 wavelengths matching the photometry range
            wl_columns_subset = wl_columns[39:322]  # 283 wavelengths (39 to 321 inclusive)
        else:
            # Fallback: use all available columns
            wl_columns_subset = wl_columns
        
        ground_truth_values = planet_row[wl_columns_subset].values.flatten()
        print(f"Loaded ground truth for planet {planet_id}: {len(ground_truth_values)} wavelengths from columns: {wl_columns_subset[:5]}...")
        print(f"Ground truth range: {ground_truth_values.min():.4f}-{ground_truth_values.max():.4f}")
        
        return ground_truth_values.astype(np.float64)
        
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return None
