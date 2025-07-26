#!/usr/bin/env python3
"""
Ariel Data Challenge Submission Runner

This script reads YAML configuration files and executes the specified pipeline
to generate submissions for the Ariel Data Challenge.
"""

import yaml
import argparse
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path to import arielml
sys.path.append(str(Path(__file__).parent.parent))

from arielml.data.observation import DataObservation
from arielml.pipelines.baseline_pipeline import BaselinePipeline
from arielml.pipelines.bayesian_pipeline import BayesianPipeline


class SubmissionRunner:
    """Main class for running pipeline submissions."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration file."""
        self.config_path = config_path
        self.config = self._load_config()
        self.results_dir = Path("submission/results")
        self.results_dir.mkdir(exist_ok=True)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _get_pipeline(self) -> Any:
        """Create pipeline instance based on config."""
        pipeline_type = self.config['pipeline']['type']
        pipeline_params = self.config['pipeline']['parameters']
        
        if pipeline_type == 'baseline':
            return BaselinePipeline(**pipeline_params)
        elif pipeline_type == 'bayesian':
            return BayesianPipeline(**pipeline_params)
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    def _get_planet_ids(self) -> List[str]:
        """Get list of planet IDs to process."""
        data_config = self.config['data']
        
        # Get planet IDs to process
        if 'planet_ids' in data_config:
            planet_ids = data_config['planet_ids']
        elif 'planet_id_file' in data_config:
            # Load from CSV file
            df = pd.read_csv(data_config['planet_id_file'])
            planet_ids = df['planet_id'].tolist()
        else:
            raise ValueError("Must specify either 'planet_ids' or 'planet_id_file' in config")
        
        print(f"Found {len(planet_ids)} planets to process")
        return [str(pid) for pid in planet_ids]
    
    def _load_single_planet_data(self, planet_id: str) -> Any:
        """Load data for a single planet."""
        data_config = self.config['data']
        instruments = data_config.get('instruments', ['AIRS-CH0', 'FGS1'])
        split = data_config.get('split', 'test')
        obs_id = data_config.get('obs_id', 0)
        backend = data_config.get('backend', 'numpy')
        
        try:
            if len(instruments) == 2 and 'AIRS-CH0' in instruments and 'FGS1' in instruments:
                # Load both instruments
                airs_obs = DataObservation(planet_id, "AIRS-CH0", obs_id, split)
                fgs_obs = DataObservation(planet_id, "FGS1", obs_id, split)
                
                # Load the data with the specified backend
                airs_obs.load(backend=backend)
                fgs_obs.load(backend=backend)
                
                # Return as dictionary
                return {
                    "AIRS-CH0": airs_obs,
                    "FGS1": fgs_obs
                }
            else:
                # Load single instrument
                instrument = instruments[0] if instruments else "AIRS-CH0"
                obs = DataObservation(planet_id, instrument, obs_id, split)
                obs.load(backend=backend)
                return obs
                
        except Exception as e:
            print(f"Warning: Failed to load data for planet {planet_id}: {e}")
            return None
    
    def _process_planets_individually(self, planet_ids: List[str], pipeline: Any) -> Dict[str, Dict]:
        """Process planets one by one to avoid memory issues."""
        results = {}
        
        print(f"Processing {len(planet_ids)} planets individually with {pipeline.__class__.__name__}...")
        
        for i, planet_id in enumerate(planet_ids):
            print(f"  Progress: {i+1}/{len(planet_ids)} - Planet {planet_id}")
            
            # Load data for this planet only
            observation = self._load_single_planet_data(planet_id)
            
            if observation is None:
                # Use fallback values
                n_wavelengths = 283
                results[planet_id] = {
                    'predictions': np.full(n_wavelengths, 0.001),
                    'uncertainties': np.full(n_wavelengths, 0.01),
                    'covariance_matrix': np.eye(n_wavelengths) * 0.01**2
                }
                continue
            
            try:
                # Fit pipeline for this planet
                pipeline.fit(observation)
                
                # Make predictions
                predictions, uncertainties, covariance_matrix = pipeline.predict(observation)
                
                results[planet_id] = {
                    'predictions': predictions,
                    'uncertainties': uncertainties,
                    'covariance_matrix': covariance_matrix
                }
                
                # Clear memory after processing this planet
                del observation
                
                # Force garbage collection and GPU memory cleanup if using GPU
                import gc
                gc.collect()
                
                # GPU memory cleanup
                try:
                    backend_name = self.config['data'].get('backend', 'numpy')
                    if backend_name in ['gpu', 'cupy']:
                        import cupy as cp
                        if hasattr(cp, 'get_default_memory_pool'):
                            cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass
                
            except Exception as e:
                print(f"Warning: Failed to process planet {planet_id}: {e}")
                # Use fallback values
                n_wavelengths = 283
                results[planet_id] = {
                    'predictions': np.full(n_wavelengths, 0.001),
                    'uncertainties': np.full(n_wavelengths, 0.01),
                    'covariance_matrix': np.eye(n_wavelengths) * 0.01**2
                }
        
        print(f"Successfully processed {len(results)} planets")
        return results


    
    def _create_submission(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Create submission DataFrame in the required format."""
        submission_data = []
        
        for planet_id, result in results.items():
            predictions = result['predictions']
            uncertainties = result['uncertainties']
            
            # Ensure we have exactly 283 values
            if len(predictions) != 283:
                print(f"Warning: Planet {planet_id} has {len(predictions)} predictions, expected 283")
                # Pad or truncate to 283
                predictions = np.pad(predictions, (0, max(0, 283 - len(predictions))), 'constant', constant_values=0.001)[:283]
                uncertainties = np.pad(uncertainties, (0, max(0, 283 - len(uncertainties))), 'constant', constant_values=0.01)[:283]
            
            # Create row: [planet_id, predictions..., uncertainties...]
            row = [planet_id] + predictions.tolist() + uncertainties.tolist()
            submission_data.append(row)
        
        # Create column names
        pred_cols = [f'wl_{i+1}' for i in range(283)]
        uncert_cols = [f'wl_{i+1}_sigma' for i in range(283)]
        columns = ['planet_id'] + pred_cols + uncert_cols
        
        submission_df = pd.DataFrame(submission_data, columns=columns)
        return submission_df
    
    def run(self) -> str:
        """Run the complete submission pipeline."""
        print(f"Starting submission run with config: {self.config_path}")
        print(f"Pipeline: {self.config['pipeline']['type']}")
        
        # Get planet IDs to process
        planet_ids = self._get_planet_ids()
        
        # Create and configure pipeline
        pipeline = self._get_pipeline()
        
        # Process planets one by one
        results = self._process_planets_individually(planet_ids, pipeline)
        
        # Create submission
        submission_df = self._create_submission(results)
        
        # Save submission
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_name = self.config['pipeline']['type']
        filename = f"submission_{pipeline_name}_{timestamp}.csv"
        output_path = self.results_dir / filename
        
        submission_df.to_csv(output_path, index=False)
        
        print(f"Submission saved to: {output_path}")
        print(f"Submission shape: {submission_df.shape}")
        print(f"Processed {len(results)} planets")
        
        return str(output_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Ariel Data Challenge submission")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--validate", action="store_true", help="Validate submission format")
    
    args = parser.parse_args()
    
    # Run submission
    runner = SubmissionRunner(args.config)
    output_path = runner.run()
    
    # Validate if requested
    if args.validate:
        print("\nValidating submission format...")
        df = pd.read_csv(output_path)
        
        # Check shape
        expected_cols = 1 + 283 + 283  # planet_id + predictions + uncertainties
        if df.shape[1] != expected_cols:
            print(f"ERROR: Expected {expected_cols} columns, got {df.shape[1]}")
        else:
            print(f"✓ Correct number of columns: {df.shape[1]}")
        
        # Check for negative values
        numeric_cols = df.columns[1:]  # Skip planet_id
        if (df[numeric_cols] < 0).any().any():
            print("ERROR: Found negative values in submission")
        else:
            print("✓ No negative values found")
        
        # Check for NaN values
        if df[numeric_cols].isna().any().any():
            print("ERROR: Found NaN values in submission")
        else:
            print("✓ No NaN values found")
        
        print(f"✓ Validation complete for {len(df)} planets")


if __name__ == "__main__":
    main() 