# arielml/data/observation.py

import numpy as np
import time
import pandas as pd
from . import loaders, calibration, photometry, detrending, analysis
from ..backend import get_backend
from ..config import ADC_GAIN, ADC_OFFSET, PHOTOMETRY_APERTURES
from ..utils import calculate_transit_mask_physical, find_transit_mask_empirical

class DataObservation:
    """
    The main class for handling all data and processing for a single observation.
    """
    def __init__(self, planet_id: str, instrument: str, obs_id: int, split: str = "train"):
        self.planet_id = planet_id
        self.instrument = instrument
        self.obs_id = obs_id
        self.split = split
        self.raw_signal = None
        self.calib_data = {}
        self.processed_signal = None
        self.light_curves = None
        self.detrended_light_curves = None
        self.noise_models = None
        self.phase_folded_lc = None
        self.mask = None
        self.star_info = None
        self.is_loaded = False
        self.backend_name = 'cpu'
        self.xp = np
        self.calibration_log = []

    def load(self, backend: str = 'cpu'):
        if self.backend_name == 'gpu' and hasattr(self.xp, 'get_default_memory_pool'):
            self.xp.get_default_memory_pool().free_all_blocks()
        
        self.raw_signal = None
        self.processed_signal = None
        self.light_curves = None
        self.calib_data = {}
        self.detrended_light_curves = None
        self.noise_models = None
        self.phase_folded_lc = None
        
        self.xp, self.backend_name = get_backend(backend)
        print(f"Loading data for {self.planet_id} (Obs {self.obs_id}) onto {self.backend_name.upper()}...")
        
        raw_signal_np = loaders.load_signal_file(self.planet_id, self.instrument, self.obs_id, self.split)
        calib_data_np = loaders.load_calibration_files(self.planet_id, self.instrument, self.obs_id, self.split)
        all_star_info = loaders.load_star_info(self.split)
        
        if all_star_info is not None and int(self.planet_id) in all_star_info.index:
            self.star_info = all_star_info.loc[int(self.planet_id)]
            
        self.raw_signal = self.xp.asarray(raw_signal_np)
        for key, value in calib_data_np.items():
            self.calib_data[key] = self.xp.asarray(value)
            
        self.processed_signal = self.xp.copy(self.raw_signal)
        self.is_loaded = True
        self.calibration_log.append("Data loaded.")
        print("Loading complete.")

    def run_calibration_pipeline(self, steps_to_run: dict = None):
        if not self.is_loaded:
            raise RuntimeError("Data must be loaded before running the pipeline.")
            
        self.processed_signal = self.xp.copy(self.raw_signal)
        self.calibration_log = [f"Pipeline started on {self.backend_name.upper()}."]
        if steps_to_run is None:
            steps_to_run = {k: True for k in ['adc', 'mask', 'linearity', 'dark', 'cds', 'flat']}
        
        if steps_to_run.get('adc'):
            self.processed_signal = self._apply_timed_step(calibration.apply_adc_conversion, self.processed_signal, ADC_GAIN, ADC_OFFSET, step_name="ADC Conversion", xp=self.xp)
        
        if steps_to_run.get('mask'):
            _, self.mask = calibration.mask_hot_dead_pixels(self.processed_signal, self.calib_data["dead"], self.calib_data["dark"], xp=self.xp)
            self.processed_signal[self.mask] = self.xp.nan
            self.calibration_log.append("Mask Hot/Dead Pixels")
            
        if steps_to_run.get('linearity'):
            self.processed_signal = self._apply_timed_step(calibration.apply_linearity_correction, self.processed_signal, self.calib_data["linear_corr"], step_name="Linearity Correction", xp=self.xp)
            
        if steps_to_run.get('dark'):
            axis_info = loaders.load_axis_info()
            dt = axis_info.get(f'{self.instrument}-integration_time', pd.Series(np.ones(self.raw_signal.shape[0]) * 0.1)).dropna().values
            if self.instrument == 'AIRS-CH0':
                 dt[1::2] += 0.1
            self.processed_signal = self._apply_timed_step(calibration.subtract_dark_current, self.processed_signal, self.calib_data["dark"], self.xp.asarray(dt), step_name="Dark Current Subtraction", xp=self.xp)
            
        if steps_to_run.get('cds'):
            self.processed_signal = self._apply_timed_step(calibration.perform_cds, self.processed_signal, step_name="Correlated Double Sampling", xp=self.xp)
            
        if steps_to_run.get('flat') and 'Correlated Double Sampling' in " ".join(self.calibration_log):
            self.processed_signal = self._apply_timed_step(calibration.apply_flat_field, self.processed_signal, self.calib_data["flat"], self.calib_data["dead"], step_name="Flat Field Correction", xp=self.xp)
        
        self.calibration_log.append("Calibration pipeline finished.")
        return self.calibration_log

    def run_photometry(self):
        if self.processed_signal is None:
            raise RuntimeError("Calibration must be run before photometry.")
        aperture_settings = PHOTOMETRY_APERTURES[self.instrument]
        self.light_curves = self._apply_timed_step(photometry.extract_aperture_photometry, self.processed_signal, aperture_settings['signal'], aperture_settings['background'], self.instrument, step_name="Aperture Photometry", xp=self.xp)
        return self.calibration_log

    def run_detrending(self, detrender: detrending.BaseDetrender, mask_method: str = "empirical"):
        """
        Runs the selected detrending model on the light curves.

        Args:
            detrender (BaseDetrender): An instance of a detrender class.
            mask_method (str): The method to use for generating the transit mask 
                               ('empirical' or 'physical').
        """
        if self.light_curves is None:
            raise RuntimeError("Photometry must be run before detrending.")
        
        time_array_xp = self.xp.asarray(self.get_time_array())
        
        # FIX: Use the mask_method passed from the UI
        transit_mask = self.get_transit_mask(method=mask_method)
        
        detrended_lcs, noise_models = self._apply_timed_step(
            detrender.detrend, time_array_xp, self.light_curves, transit_mask, 
            step_name=f"Detrending ({detrender.__class__.__name__})", xp=self.xp
        )
        
        self.detrended_light_curves = detrended_lcs
        self.noise_models = noise_models
        return self.calibration_log

    def run_phase_folding(self, n_bins: int = 100):
        if self.detrended_light_curves is None:
            raise RuntimeError("Detrending must be run before phase-folding.")
        time_np, detrended_lcs_np = self.get_time_array(), self.get_detrended_light_curves(return_type='numpy')
        binned_results = []
        for i in range(detrended_lcs_np.shape[1]):
            lc, finite_mask = detrended_lcs_np[:, i], np.isfinite(detrended_lcs_np[:, i])
            if not np.any(finite_mask):
                binned_results.append((np.full(n_bins, np.nan), np.full(n_bins, np.nan), np.full(n_bins, np.nan)))
                continue
            bin_centers, binned_flux, binned_error = analysis.phase_fold_and_bin(time_np[finite_mask], lc[finite_mask], self.star_info['P'], n_bins)
            binned_results.append((bin_centers, binned_flux, binned_error))
        self.phase_folded_lc = binned_results
        self.calibration_log.append(f"Phase-folding ({n_bins} bins).")
        return self.calibration_log

    def get_transit_mask(self, method: str = "empirical") -> "xp.ndarray":
        """
        Gets the transit mask using either the physical model or an empirical, data-driven method.
        """
        time_array = self.get_time_array()
        
        if method == "physical":
            if self.star_info is None:
                raise ValueError("Physical mask requires star info.")
            return calculate_transit_mask_physical(
                self.xp.asarray(time_array), self.star_info['P'], self.star_info['sma'], 
                self.star_info['Rs'], self.star_info['i'], self.xp
            )
        elif method == "empirical":
            if self.light_curves is None:
                raise ValueError("Empirical mask requires photometry to be run first.")
            summed_flux = self.xp.nanmedian(self.light_curves, axis=1)
            return find_transit_mask_empirical(self.xp.asarray(time_array), summed_flux, xp=self.xp)
        else:
            raise ValueError(f"Unknown transit mask method: {method}")

    def get_time_array(self) -> np.ndarray:
        if self.processed_signal is None: return np.array([])
        num_frames = self.processed_signal.shape[0]
        timestep_days = (0.1 / (24 * 3600)) * 2 
        return np.arange(num_frames) * timestep_days

    def _apply_timed_step(self, step_func, *args, step_name: str, **kwargs):
        if not self.is_loaded:
            raise RuntimeError("Data must be loaded before applying calibration.")
        if self.backend_name == 'gpu':
            self.xp.cuda.Stream.null.synchronize()
        start_time = time.perf_counter()
        result = step_func(*args, **kwargs)
        if self.backend_name == 'gpu':
            self.xp.cuda.Stream.null.synchronize()
        duration = time.perf_counter() - start_time
        log_message = f"{step_name}: {duration:.4f}s"
        print(log_message)
        self.calibration_log.append(log_message)
        return result

    def get_data(self, return_type='numpy'):
        data = self.processed_signal
        if return_type == 'numpy' and hasattr(data, 'get'):
            return data.get()
        return data

    def get_light_curves(self, return_type='numpy'):
        data = self.light_curves
        if data is None: return None
        if return_type == 'numpy' and hasattr(data, 'get'):
            return data.get()
        return data

    def get_detrended_light_curves(self, return_type='numpy'):
        data = self.detrended_light_curves
        if data is None: return None
        if return_type == 'numpy' and hasattr(data, 'get'):
            return data.get()
        return data

    def get_noise_models(self, return_type='numpy'):
        data = self.noise_models
        if data is None: return None
        if return_type == 'numpy' and hasattr(data, 'get'):
            return data.get()
        return data
        
    def get_phase_folded_lc(self):
        return self.phase_folded_lc
