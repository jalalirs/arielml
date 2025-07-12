# arielml/data/observation.py

import numpy as np
import time
from . import loaders, calibration, photometry
from ..backend import get_backend
from ..config import ADC_GAIN, ADC_OFFSET, PHOTOMETRY_APERTURES

class DataObservation:
    """
    A class to represent and process a single observation (one instrument, one visit).
    This class handles data loading, state management, and calibration on a
    specified backend (CPU or GPU), with performance logging and memory management.
    """
    def __init__(self, planet_id: str, instrument: str, obs_id: int, split: str = "train"):
        self.planet_id = planet_id
        self.instrument = instrument
        self.obs_id = obs_id
        self.split = split

        # Data attributes
        self.raw_signal = None
        self.calib_data = {}
        self.processed_signal = None
        self.light_curves = None # To store the final 1D light curves
        self.mask = None

        # State attributes
        self.is_loaded = False
        self.backend_name = 'cpu'
        self.xp = np
        self.calibration_log = []

    def load(self, backend: str = 'cpu'):
        """
        Loads data from disk and moves it to the specified backend device (CPU/GPU).
        Includes GPU memory management.
        """
        if self.backend_name == 'gpu':
            print("Clearing GPU memory pool...")
            self.xp.get_default_memory_pool().free_all_blocks()
        
        self.raw_signal = None
        self.processed_signal = None
        self.light_curves = None
        self.calib_data = {}

        self.xp, self.backend_name = get_backend(backend)
        
        print(f"Loading data for {self.planet_id} (Obs {self.obs_id}) onto {self.backend_name.upper()}...")
        
        raw_signal_np = loaders.load_signal_file(self.planet_id, self.instrument, self.obs_id, self.split)
        calib_data_np = loaders.load_calibration_files(self.planet_id, self.instrument, self.obs_id, self.split)
        
        self.raw_signal = self.xp.asarray(raw_signal_np)
        for key, value in calib_data_np.items():
            self.calib_data[key] = self.xp.asarray(value)
            
        self.processed_signal = self.xp.copy(self.raw_signal)
        self.is_loaded = True
        self.calibration_log.append("Data loaded.")
        print("Loading complete.")

    def _apply_timed_step(self, step_func, *args, step_name: str, **kwargs):
        """A helper to apply a function, time it, and log it."""
        if not self.is_loaded:
            raise RuntimeError("Data must be loaded before applying calibration.")
        
        if self.backend_name == 'gpu': self.xp.cuda.Stream.null.synchronize()
        start_time = time.perf_counter()
        
        result = step_func(*args, **kwargs)
        
        if self.backend_name == 'gpu': self.xp.cuda.Stream.null.synchronize()
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        log_message = f"{step_name}: {duration:.4f}s"
        print(log_message)
        self.calibration_log.append(log_message)
        return result

    def run_calibration_pipeline(self, steps_to_run: dict = None):
        """
        Runs a full calibration pipeline based on a configuration dictionary and
        returns the performance log.
        """
        if not self.is_loaded:
            raise RuntimeError("Data must be loaded before running the pipeline.")
        
        self.processed_signal = self.xp.copy(self.raw_signal)
        self.calibration_log = [f"Pipeline started on {self.backend_name.upper()}."]

        if steps_to_run is None:
            steps_to_run = {k: True for k in ['adc', 'mask', 'linearity', 'dark', 'cds', 'flat']}

        if steps_to_run.get('adc'):
            self.processed_signal = self._apply_timed_step(
                calibration.apply_adc_conversion, self.processed_signal, ADC_GAIN, ADC_OFFSET,
                step_name="ADC Conversion", xp=self.xp
            )

        if steps_to_run.get('mask'):
            signal_before_mask = self.processed_signal
            _, self.mask = calibration.mask_hot_dead_pixels(
                signal_before_mask, self.calib_data["dead"], self.calib_data["dark"], xp=self.xp
            )
            self.processed_signal = signal_before_mask.astype(self.xp.float64)
            self.processed_signal[self.mask] = self.xp.nan
            self.calibration_log.append("Mask Hot/Dead Pixels (not timed)")

        if steps_to_run.get('linearity'):
            self.processed_signal = self._apply_timed_step(
                calibration.apply_linearity_correction, self.processed_signal, self.calib_data["linear_corr"],
                step_name="Linearity Correction", xp=self.xp
            )

        if steps_to_run.get('dark'):
            axis_info = loaders.load_axis_info()
            if self.instrument == 'AIRS-CH0':
                dt = axis_info['AIRS-CH0-integration_time'].dropna().values
            else:
                dt = np.ones(self.raw_signal.shape[0]) * 0.1
            dt[1::2] += 0.1
            dt_xp = self.xp.asarray(dt)
            self.processed_signal = self._apply_timed_step(
                calibration.subtract_dark_current, self.processed_signal, self.calib_data["dark"], dt_xp,
                step_name="Dark Current Subtraction", xp=self.xp
            )

        if steps_to_run.get('cds'):
            self.processed_signal = self._apply_timed_step(
                calibration.perform_cds, self.processed_signal,
                step_name="Correlated Double Sampling", xp=self.xp
            )
        
        if steps_to_run.get('flat'):
            if 'Correlated Double Sampling' in " ".join(self.calibration_log):
                self.processed_signal = self._apply_timed_step(
                    calibration.apply_flat_field, self.processed_signal, self.calib_data["flat"],
                    step_name="Flat Field Correction", xp=self.xp
                )
            else:
                self.calibration_log.append("Skipping Flat Field: CDS must be applied first.")
        
        self.calibration_log.append("Calibration pipeline finished.")
        return self.calibration_log

    def run_photometry(self):
        """
        Runs aperture photometry on the processed signal to extract light curves.
        """
        if self.processed_signal is None:
            raise RuntimeError("Calibration must be run before photometry.")
        
        aperture_settings = PHOTOMETRY_APERTURES[self.instrument]
        
        self.light_curves = self._apply_timed_step(
            photometry.extract_aperture_photometry,
            self.processed_signal,
            aperture_settings['signal'],
            aperture_settings['background'],
            self.instrument, # FIX: Pass the instrument name
            step_name="Aperture Photometry",
            xp=self.xp
        )
        return self.calibration_log

    def get_data(self, return_type='numpy'):
        """Returns the processed 2D data cube."""
        if return_type == 'numpy':
            if hasattr(self.processed_signal, 'get'):
                return self.processed_signal.get()
            return self.processed_signal
        return self.processed_signal

    def get_light_curves(self, return_type='numpy'):
        """Returns the extracted 1D light curves."""
        if self.light_curves is None:
            return None
        if return_type == 'numpy':
            if hasattr(self.light_curves, 'get'):
                return self.light_curves.get()
            return self.light_curves
        return self.light_curves
