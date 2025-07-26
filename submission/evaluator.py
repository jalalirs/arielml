#!/usr/bin/env python3
"""
Ariel Data Challenge Submission Evaluator

This script implements the competition evaluation metric (Gaussian Log-Likelihood)
and provides tools for evaluating submissions locally.
"""

import numpy as np
import pandas as pd
import pandas.api.types
import scipy.stats
from typing import Tuple, Optional
import argparse
from pathlib import Path


class ParticipantVisibleError(Exception):
    """Exception for validation errors visible to participants."""
    pass


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    naive_mean: float,
    naive_sigma: float,
    fsg_sigma_true: float = 1e-6,
    airs_sigma_true: float = 1e-5,
    fgs_weight: float = 57.846,
) -> float:
    """
    Calculate the Gaussian Log Likelihood based metric for the Ariel Data Challenge.
    
    This is a Gaussian Log Likelihood based metric. For a submission, which contains 
    the predicted mean (x_hat) and variance (x_hat_std), we calculate the Gaussian 
    Log-likelihood (GLL) value to the provided ground truth (x). We treat each pair 
    of x_hat, x_hat_std as a 1D gaussian, meaning there will be 283 1D gaussian 
    distributions, hence 283 values for each test spectrum, the GLL value for one 
    spectrum is the sum of all of them.

    Args:
        solution: Ground Truth spectra (from test set), shape: (nsamples, n_wavelengths)
        submission: Predicted spectra and errors (from participants), 
                   shape: (nsamples, n_wavelengths*2)
        row_id_column_name: Name of the row ID column (usually 'planet_id')
        naive_mean: Mean from the train set
        naive_sigma: Standard deviation from the train set
        fsg_sigma_true: Standard deviation from the FSG1 instrument for the test set
        airs_sigma_true: Standard deviation from the AIRS instrument for the test set
        fgs_weight: Relative weight of the fgs channel

    Returns:
        float: Score in the interval [0, 1], with higher scores being better
    """
    
    # Remove ID columns
    solution = solution.copy()
    submission = submission.copy()
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    # Validation checks
    if submission.min().min() < 0:
        raise ParticipantVisibleError('Negative values in the submission')
    
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')

    n_wavelengths = len(solution.columns)
    if len(submission.columns) != n_wavelengths * 2:
        raise ParticipantVisibleError('Wrong number of columns in the submission')

    # Extract predictions and uncertainties
    y_pred = submission.iloc[:, :n_wavelengths].values
    # Set a non-zero minimum sigma pred to prevent division by zero errors
    sigma_pred = np.clip(submission.iloc[:, n_wavelengths:].values, a_min=10**-15, a_max=None)
    
    # True sigma values (FGS1 has different sigma than AIRS-CH0)
    sigma_true = np.append(
        np.array([fsg_sigma_true]),
        np.ones(n_wavelengths - 1) * airs_sigma_true,
    )
    y_true = solution.values

    # Calculate Gaussian Log-Likelihoods
    GLL_pred = scipy.stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred)
    GLL_true = scipy.stats.norm.logpdf(y_true, loc=y_true, scale=sigma_true * np.ones_like(y_true))
    GLL_mean = scipy.stats.norm.logpdf(y_true, loc=naive_mean * np.ones_like(y_true), 
                                      scale=naive_sigma * np.ones_like(y_true))

    # Normalize the score
    ind_scores = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)

    # Apply weights (FGS1 gets higher weight)
    weights = np.append(np.array([fgs_weight]), np.ones(len(solution.columns) - 1))
    weights = weights * np.ones_like(ind_scores)
    
    # Calculate weighted average score
    submit_score = np.average(ind_scores, weights=weights)
    
    # Clip to [0, 1] range
    return float(np.clip(submit_score, 0.0, 1.0))


class SubmissionEvaluator:
    """Class for evaluating submissions locally."""
    
    def __init__(self, ground_truth_path: str, train_stats_path: Optional[str] = None):
        """
        Initialize evaluator with ground truth data.
        
        Args:
            ground_truth_path: Path to ground truth CSV file
            train_stats_path: Path to training statistics (for naive baseline)
        """
        self.ground_truth = pd.read_csv(ground_truth_path)
        
        # Load or calculate training statistics
        if train_stats_path:
            train_stats = pd.read_csv(train_stats_path)
            self.naive_mean = train_stats['mean'].iloc[0]
            self.naive_sigma = train_stats['std'].iloc[0]
        else:
            # Calculate from ground truth (not ideal, but fallback)
            numeric_cols = [col for col in self.ground_truth.columns if col != 'planet_id']
            self.naive_mean = self.ground_truth[numeric_cols].mean().mean()
            self.naive_sigma = self.ground_truth[numeric_cols].std().mean()
            print(f"Warning: Using ground truth for naive baseline (mean={self.naive_mean:.6f}, std={self.naive_sigma:.6f})")
    
    def evaluate(self, submission_path: str, verbose: bool = True) -> Tuple[float, dict]:
        """
        Evaluate a submission file.
        
        Args:
            submission_path: Path to submission CSV file
            verbose: Whether to print detailed results
            
        Returns:
            Tuple of (score, detailed_metrics)
        """
        submission = pd.read_csv(submission_path)
        
        if verbose:
            print(f"Evaluating submission: {submission_path}")
            print(f"Submission shape: {submission.shape}")
            print(f"Ground truth shape: {self.ground_truth.shape}")
        
        # Validate format
        self._validate_submission(submission, verbose)
        
        # Calculate score
        try:
            final_score = score(
                solution=self.ground_truth,
                submission=submission,
                row_id_column_name='planet_id',
                naive_mean=self.naive_mean,
                naive_sigma=self.naive_sigma
            )
            
            # Calculate additional metrics
            detailed_metrics = self._calculate_detailed_metrics(submission)
            detailed_metrics['final_score'] = final_score
            
            if verbose:
                print(f"\n=== EVALUATION RESULTS ===")
                print(f"Final Score: {final_score:.6f}")
                print(f"Mean Prediction: {detailed_metrics['mean_prediction']:.6f}")
                print(f"Mean Uncertainty: {detailed_metrics['mean_uncertainty']:.6f}")
                print(f"Prediction Range: [{detailed_metrics['min_prediction']:.6f}, {detailed_metrics['max_prediction']:.6f}]")
                print(f"Uncertainty Range: [{detailed_metrics['min_uncertainty']:.6f}, {detailed_metrics['max_uncertainty']:.6f}]")
            
            return final_score, detailed_metrics
            
        except ParticipantVisibleError as e:
            print(f"ERROR: {e}")
            return 0.0, {'error': str(e)}
    
    def _validate_submission(self, submission: pd.DataFrame, verbose: bool = True):
        """Validate submission format."""
        errors = []
        
        # Check columns
        expected_cols = 1 + 283 + 283  # planet_id + predictions + uncertainties
        if submission.shape[1] != expected_cols:
            errors.append(f"Expected {expected_cols} columns, got {submission.shape[1]}")
        
        # Check planet IDs match
        if not set(submission['planet_id']).issubset(set(self.ground_truth['planet_id'])):
            errors.append("Some planet IDs in submission not found in ground truth")
        
        # Check for negative values
        numeric_cols = [col for col in submission.columns if col != 'planet_id']
        if (submission[numeric_cols] < 0).any().any():
            errors.append("Found negative values in submission")
        
        # Check for NaN values
        if submission[numeric_cols].isna().any().any():
            errors.append("Found NaN values in submission")
        
        if errors:
            error_msg = "Validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ParticipantVisibleError(error_msg)
        
        if verbose:
            print("✓ Submission format validation passed")
    
    def _calculate_detailed_metrics(self, submission: pd.DataFrame) -> dict:
        """Calculate detailed metrics for analysis."""
        # Get prediction and uncertainty columns
        n_wavelengths = 283
        pred_cols = submission.columns[1:n_wavelengths+1]
        uncert_cols = submission.columns[n_wavelengths+1:2*n_wavelengths+1]
        
        predictions = submission[pred_cols].values
        uncertainties = submission[uncert_cols].values
        
        return {
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions),
            'min_prediction': np.min(predictions),
            'max_prediction': np.max(predictions),
            'mean_uncertainty': np.mean(uncertainties),
            'std_uncertainty': np.std(uncertainties),
            'min_uncertainty': np.min(uncertainties),
            'max_uncertainty': np.max(uncertainties),
            'n_planets': len(submission),
            'n_wavelengths': n_wavelengths
        }


def main():
    """Main entry point for command-line evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Ariel Data Challenge submission")
    parser.add_argument("submission", help="Path to submission CSV file")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth CSV file")
    parser.add_argument("--train-stats", help="Path to training statistics CSV file")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = SubmissionEvaluator(
        ground_truth_path=args.ground_truth,
        train_stats_path=args.train_stats
    )
    
    # Evaluate submission
    score_value, metrics = evaluator.evaluate(
        submission_path=args.submission,
        verbose=not args.quiet
    )
    
    # Output result
    if args.quiet:
        print(f"{score_value:.6f}")
    else:
        print(f"\nFinal evaluation score: {score_value:.6f}")


if __name__ == "__main__":
    main() 