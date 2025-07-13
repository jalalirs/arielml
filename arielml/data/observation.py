# arielml/data/observation.py

import numpy as np
import time
from . import loaders, calibration, photometry, detrending, analysis
from ..backend import get_backend
from ..config import ADC_GAIN, ADC_OFFSET, PHOTOMETRY_APERTURES
from .. import utils

class DataObservation:
    """
    The main class for handling all data and processing for a single observation.
    """
    def __init__(self, planet_id: str, instrument: str, obs_id: int, split: str = "train"):
        self.planet_id = planet_id
        self.instrument = instrument
        self.obs_id = obs_id
        self.split = split
        # Data attributes
        self.raw_signal = None; self.calib_data = {}; self.processed_signal = None
        self.light_curves = None; self.detrended_light_curves = None; self.noise_models = None
        self.phase_folded_lc = None # For phase-folded data
        self.mask = None; self.star_info = None
        # State attributes
        self.is_loaded = False; self.backend_name = 'cpu'; self.xp = np; self.calibration_log = []

    def load(self, backend: str = 'cpu'):
        if self.backend_name == 'gpu':
            if hasattr(self.xp, 'get_default_memory_pool'):
                self.xp.get_default_memory_pool().free_all_blocks()
        
        self.raw_signal = None; self.processed_signal = None; self.light_curves = None; self.calib_data = {}; self.detrended_light_curves = None; self.noise_models = None; self.phase_folded_lc = None
        self.xp, self.backend_name = get_backend(backend)
        print(f"Loading data for {self.planet_id} (Obs {self.obs_id}) onto {self.backend_name.upper()}...")
        raw_signal_np = loaders.load_signal_file(self.planet_id, self.instrument, self.obs_id, self.split)
        calib_data_np = loaders.load_calibration_files(self.planet_id, self.instrument, self.obs_id, self.split)
        all_star_info = loaders.load_star_info(self.split)
        if all_star_info is not None and int(self.planet_id) in all_star_info.index:
            self.star_info = all_star_info.loc[int(self.planet_id)]
        self.raw_signal = self.xp.asarray(raw_signal_np)
        for key, value in calib_data_np.items(): self.calib_data[key] = self.xp.asarray(value)
        self.processed_signal = self.xp.copy(self.raw_signal)
        self.is_loaded = True; self.calibration_log.append("Data loaded."); print("Loading complete.")

    def run_calibration_pipeline(self, steps_to_run: dict = None):
        if not self.is_loaded: raise RuntimeError("Data must be loaded before running the pipeline.")
        self.processed_signal = self.xp.copy(self.raw_signal)
        self.calibration_log = [f"Pipeline started on {self.backend_name.upper()}."]
        if steps_to_run is None: steps_to_run = {k: True for k in ['adc', 'mask', 'linearity', 'dark', 'cds', 'flat']}
        
        if steps_to_run.get('adc'): self.processed_signal = self._apply_timed_step(calibration.apply_adc_conversion, self.processed_signal, ADC_GAIN, ADC_OFFSET, step_name="ADC Conversion", xp=self.xp)
        
        if steps_to_run.get('mask'):
            signal_before_mask = self.processed_signal; _, self.mask = calibration.mask_hot_dead_pixels(signal_before_mask, self.calib_data["dead"], self.calib_data["dark"], xp=self.xp)
            self.processed_signal = signal_before_mask.astype(self.xp.float64); self.processed_signal[self.mask] = self.xp.nan; self.calibration_log.append("Mask Hot/Dead Pixels (not timed)")
        
        if steps_to_run.get('linearity'): self.processed_signal = self._apply_timed_step(calibration.apply_linearity_correction, self.processed_signal, self.calib_data["linear_corr"], step_name="Linearity Correction", xp=self.xp)
        
        if steps_to_run.get('dark'):
            axis_info = loaders.load_axis_info()
            if self.instrument == 'AIRS-CH0': dt = axis_info['AIRS-CH0-integration_time'].dropna().values
            else: dt = np.ones(self.raw_signal.shape[0]) * 0.1
            dt[1::2] += 0.1; dt_xp = self.xp.asarray(dt)
            self.processed_signal = self._apply_timed_step(calibration.subtract_dark_current, self.processed_signal, self.calib_data["dark"], dt_xp, step_name="Dark Current Subtraction", xp=self.xp)
        
        if steps_to_run.get('cds'): self.processed_signal = self._apply_timed_step(calibration.perform_cds, self.processed_signal, step_name="Correlated Double Sampling", xp=self.xp)
        
        if steps_to_run.get('flat'):
            if 'Correlated Double Sampling' in " ".join(self.calibration_log):
                self.processed_signal = self._apply_timed_step(
                    calibration.apply_flat_field, 
                    self.processed_signal, 
                    self.calib_data["flat"], 
                    self.calib_data["dead"],
                    step_name="Flat Field Correction", 
                    xp=self.xp
                )
            else:
                self.calibration_log.append("Skipping Flat Field: CDS must be applied first.")
        
        self.calibration_log.append("Calibration pipeline finished."); return self.calibration_log

    def run_photometry(self):
        if self.processed_signal is None: raise RuntimeError("Calibration must be run before photometry.")
        aperture_settings = PHOTOMETRY_APERTURES[self.instrument]
        self.light_curves = self._apply_timed_step(photometry.extract_aperture_photometry, self.processed_signal, aperture_settings['signal'], aperture_settings['background'], self.instrument, step_name="Aperture Photometry", xp=self.xp)
        return self.calibration_log

    def run_detrending(self, detrender: detrending.BaseDetrender):
        if self.light_curves is None: raise RuntimeError("Photometry must be run before detrending.")
        
        time_array = self.get_time_array()
        time_array_xp = self.xp.asarray(time_array)
        transit_mask = self.get_transit_mask()

        detrended_lcs, noise_models = self._apply_timed_step(detrender.detrend, time_array_xp, self.light_curves, transit_mask, step_name=f"Detrending ({detrender.__class__.__name__})", xp=self.xp)
        self.detrended_light_curves = detrended_lcs; self.noise_models = noise_models
        return self.calibration_log

    def run_phase_folding(self, n_bins: int = 100):
        if self.detrended_light_curves is None:
            raise RuntimeError("Detrending must be run before phase-folding.")
        
        time_np = self.get_time_array()
        detrended_lcs_np = self.get_detrended_light_curves(return_type='numpy')
        
        binned_results = []
        for i in range(detrended_lcs_np.shape[1]):
            lc = detrended_lcs_np[:, i]
            finite_mask = np.isfinite(lc)
            if not np.any(finite_mask):
                binned_results.append((np.full(n_bins, np.nan), np.full(n_bins, np.nan), np.full(n_bins, np.nan)))
                continue

            bin_centers, binned_flux, binned_error = analysis.phase_fold_and_bin(
                time_np[finite_mask], lc[finite_mask], self.star_info['P'], n_bins
            )
            binned_results.append((bin_centers, binned_flux, binned_error))
        
        self.phase_folded_lc = binned_results
        self.calibration_log.append(f"Phase-folding ({n_bins} bins).")
        return self.calibration_log

    def get_phase_folded_lc(self):
        return self.phase_folded_lc

    def get_time_array(self) -> np.ndarray:
        if self.processed_signal is None: return np.array([])
        num_frames = self.processed_signal.shape[0]
        # Timestep between frames after CDS is applied.
        timestep_days = (0.1 / (24 * 3600)) * 2 
        return np.arange(num_frames) * timestep_days

    def get_transit_mask(self) -> "xp.ndarray":
        if self.star_info is None: raise ValueError("Star info not loaded, cannot calculate transit mask.")
        time_array = self.get_time_array(); time_array_xp = self.xp.asarray(time_array)
        # Call the new, corrected utility function
        return utils.calculate_transit_mask(
            time_array_xp, 
            period=self.star_info['P'], 
            semi_major_axis=self.star_info['sma'], 
            stellar_radius=self.star_info['Rs'], 
            inclination=self.star_info['i'], 
            xp=self.xp
        )

    def _apply_timed_step(self, step_func, *args, step_name: str, **kwargs):
        if not self.is_loaded: raise RuntimeError("Data must be loaded before applying calibration.")
        if self.backend_name == 'gpu': self.xp.cuda.Stream.null.synchronize()
        start_time = time.perf_counter()
        result = step_func(*args, **kwargs)
        if self.backend_name == 'gpu': self.xp.cuda.Stream.null.synchronize()
        end_time = time.perf_counter()
        duration = end_time - start_time
        log_message = f"{step_name}: {duration:.4f}s"; print(log_message); self.calibration_log.append(log_message)
        return result

    def get_data(self, return_type='numpy'):
        if return_type == 'numpy':
            if hasattr(self.processed_signal, 'get'): return self.processed_signal.get()
            return self.processed_signal
        return self.processed_signal

    def get_light_curves(self, return_type='numpy'):
        if self.light_curves is None: return None
        if return_type == 'numpy':
            if hasattr(self.light_curves, 'get'): return self.light_curves.get()
            return self.light_curves
        return self.light_curves

    def get_detrended_light_curves(self, return_type='numpy'):
        if self.detrended_light_curves is None: return None
        if return_type == 'numpy':
            if hasattr(self.detrended_light_curves, 'get'): return self.detrended_light_curves.get()
            return self.detrended_light_curves
        return self.detrended_light_curves

    def get_noise_models(self, return_type='numpy'):
        if self.noise_models is None: return None
        if return_type == 'numpy':
            if hasattr(self.noise_models, 'get'): return self.noise_models.get()
            return self.noise_models
        return self.noise_models
