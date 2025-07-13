import sys
import os
from pathlib import Path

# Add the project root to the Python path to allow importing 'arielml'
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QPushButton, QSlider, QLabel, QGroupBox, QCheckBox, QSpinBox, QStatusBar,
    QTextEdit, QTabWidget, QStackedWidget
)
from PyQt6.QtCore import Qt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# Import all necessary components from our library
from arielml.data.observation import DataObservation
from arielml.data import loaders, detrending
from arielml.config import DATASET_DIR, INSTRUMENT_SHAPES, PHOTOMETRY_APERTURES
from arielml.backend import GPU_ENABLED

class DataInspector(QMainWindow):
    """
    A PyQt GUI application to inspect Ariel dataset files and visualize
    the full data reduction pipeline, with a final, clean two-panel layout.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ariel Data Inspector (Final)")
        self.setGeometry(100, 100, 1400, 800)

        self.observation = None
        self.star_info_df = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        visuals_panel = self.create_visuals_panel()
        main_layout.addWidget(visuals_panel, 3)

        controls_panel = self.create_controls_panel()
        main_layout.addWidget(controls_panel, 1)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.populate_planet_ids()

    def create_controls_panel(self):
        """Creates the right-hand panel with all user controls."""
        main_layout = QVBoxLayout()
        
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout()
        self.split_combo = QComboBox(); self.split_combo.addItems(["train", "test"]); self.split_combo.currentTextChanged.connect(self.populate_planet_ids)
        nav_layout.addWidget(QLabel("Dataset Split:")); nav_layout.addWidget(self.split_combo)
        self.planet_combo = QComboBox(); self.planet_combo.setEditable(True); self.planet_combo.currentTextChanged.connect(self.populate_obs_ids)
        nav_layout.addWidget(QLabel("Planet ID:")); nav_layout.addWidget(self.planet_combo)
        self.obs_combo = QComboBox()
        nav_layout.addWidget(QLabel("Observation ID:")); nav_layout.addWidget(self.obs_combo)
        self.instrument_combo = QComboBox(); self.instrument_combo.addItems(["AIRS-CH0", "FGS1"]); self.instrument_combo.currentTextChanged.connect(self.populate_obs_ids)
        nav_layout.addWidget(QLabel("Instrument:")); nav_layout.addWidget(self.instrument_combo)
        self.backend_combo = QComboBox(); self.backend_combo.addItems(["cpu", "gpu"])
        if not GPU_ENABLED: self.backend_combo.model().item(1).setEnabled(False); self.backend_combo.setToolTip("GPU / CuPy not available on this system.")
        nav_layout.addWidget(QLabel("Processing Backend:")); nav_layout.addWidget(self.backend_combo)
        self.load_button = QPushButton("Load Planet Data"); self.load_button.clicked.connect(self.load_data)
        nav_layout.addWidget(self.load_button)
        nav_group.setLayout(nav_layout)
        main_layout.addWidget(nav_group)

        calib_group = QGroupBox("Calibration Steps")
        calib_layout = QVBoxLayout()
        self.calib_checkboxes = {
            "adc": QCheckBox("ADC Conversion"), "mask": QCheckBox("Mask Hot/Dead Pixels"),
            "linearity": QCheckBox("Linearity Correction"), "dark": QCheckBox("Dark Current Subtraction"),
            "cds": QCheckBox("Correlated Double Sampling (CDS)"), "flat": QCheckBox("Flat Field Correction"),
        }
        for checkbox in self.calib_checkboxes.values():
            checkbox.setChecked(True)
            calib_layout.addWidget(checkbox)
        calib_group.setLayout(calib_layout)
        main_layout.addWidget(calib_group)
        
        detrend_group = QGroupBox("Detrending Model")
        detrend_layout = QVBoxLayout()
        self.detrend_model_combo = QComboBox()
        self.detrend_model_combo.addItems(["Polynomial", "Savitzky-Golay"])
        self.detrend_model_combo.currentTextChanged.connect(self.on_detrend_model_change)
        detrend_layout.addWidget(self.detrend_model_combo)
        self.detrend_params_stack = QStackedWidget()
        poly_widget = QWidget(); poly_layout = QHBoxLayout(poly_widget)
        poly_layout.addWidget(QLabel("Degree:")); self.poly_degree_spinbox = QSpinBox()
        self.poly_degree_spinbox.setRange(1, 10); self.poly_degree_spinbox.setValue(2)
        poly_layout.addWidget(self.poly_degree_spinbox); self.detrend_params_stack.addWidget(poly_widget)
        savgol_widget = QWidget(); savgol_layout = QHBoxLayout(savgol_widget)
        savgol_layout.addWidget(QLabel("Window:")); self.savgol_window_spinbox = QSpinBox()
        self.savgol_window_spinbox.setRange(3, 999); self.savgol_window_spinbox.setSingleStep(2); self.savgol_window_spinbox.setValue(51)
        savgol_layout.addWidget(self.savgol_window_spinbox)
        savgol_layout.addWidget(QLabel("Order:")); self.savgol_order_spinbox = QSpinBox()
        self.savgol_order_spinbox.setRange(1, 10); self.savgol_order_spinbox.setValue(2)
        savgol_layout.addWidget(self.savgol_order_spinbox); self.detrend_params_stack.addWidget(savgol_widget)
        detrend_layout.addWidget(self.detrend_params_stack)
        detrend_group.setLayout(detrend_layout)
        main_layout.addWidget(detrend_group)

        fold_group = QGroupBox("Phase Folding")
        fold_layout = QVBoxLayout()
        fold_layout.addWidget(QLabel("Number of Bins:"))
        self.bins_spinbox = QSpinBox()
        self.bins_spinbox.setRange(10, 500)
        self.bins_spinbox.setValue(100)
        fold_layout.addWidget(self.bins_spinbox)
        fold_group.setLayout(fold_layout)
        main_layout.addWidget(fold_group)
        
        self.apply_button = QPushButton("Apply Changes & Rerun Pipeline")
        self.apply_button.clicked.connect(self.run_pipeline_and_update_plots)
        main_layout.addWidget(self.apply_button)

        main_layout.addStretch()
        
        container = QWidget()
        container.setLayout(main_layout)
        return container

    def create_visuals_panel(self):
        """Creates the left-hand panel with all the plots and logs."""
        tabs = QTabWidget()
        
        frame_group = QGroupBox("Frame & Wavelength Selector")
        frame_layout = QHBoxLayout(frame_group)
        frame_layout.addWidget(QLabel("Frame Index:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal); self.frame_slider.valueChanged.connect(self.update_image_plot)
        self.frame_spinbox = QSpinBox(); self.frame_spinbox.valueChanged.connect(self.frame_slider.setValue); self.frame_slider.valueChanged.connect(self.frame_spinbox.setValue)
        frame_layout.addWidget(self.frame_slider); frame_layout.addWidget(self.frame_spinbox)
        frame_layout.addWidget(QLabel("Wavelength Column:"))
        self.wavelength_slider = QSlider(Qt.Orientation.Horizontal)
        self.wavelength_slider.valueChanged.connect(self.update_all_light_curve_plots)
        self.wavelength_spinbox = QSpinBox()
        self.wavelength_spinbox.valueChanged.connect(self.wavelength_slider.setValue)
        self.wavelength_slider.valueChanged.connect(self.wavelength_spinbox.setValue)
        frame_layout.addWidget(self.wavelength_slider); frame_layout.addWidget(self.wavelength_spinbox)

        detector_tab = QWidget(); detector_layout = QVBoxLayout(detector_tab)
        info_group = QGroupBox("Star & Planet Info"); info_layout_inner = QVBoxLayout(); self.info_display = QTextEdit(); self.info_display.setReadOnly(True); info_group.setFixedHeight(150); info_layout_inner.addWidget(self.info_display); info_group.setLayout(info_layout_inner); detector_layout.addWidget(info_group)
        self.image_canvas = FigureCanvas(Figure(figsize=(10, 5))); self.image_ax = self.image_canvas.figure.subplots(); self.image_cbar = None; detector_layout.addWidget(self.image_canvas); self.image_canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        photometry_tab = QWidget(); photometry_layout = QVBoxLayout(photometry_tab)
        self.single_lc_canvas = FigureCanvas(Figure(figsize=(10, 3))); self.single_lc_ax = self.single_lc_canvas.figure.subplots(); photometry_layout.addWidget(self.single_lc_canvas)
        
        detrended_tab = QWidget(); detrended_layout = QVBoxLayout(detrended_tab)
        self.zoom_checkbox = QCheckBox("Zoom to Transit"); self.zoom_checkbox.toggled.connect(self.update_detrended_plot); detrended_layout.addWidget(self.zoom_checkbox)
        self.detrended_lc_canvas = FigureCanvas(Figure(figsize=(10, 3))); self.detrended_lc_ax = self.detrended_lc_canvas.figure.subplots(); detrended_layout.addWidget(self.detrended_lc_canvas)
        
        phase_folded_tab = QWidget(); phase_folded_layout = QVBoxLayout(phase_folded_tab)
        self.phase_folded_canvas = FigureCanvas(Figure(figsize=(10, 3))); self.phase_folded_ax = self.phase_folded_canvas.figure.subplots(); phase_folded_layout.addWidget(self.phase_folded_canvas)
        
        log_tab = QWidget(); log_layout = QVBoxLayout(log_tab)
        self.log_display = QTextEdit(); self.log_display.setReadOnly(True); log_layout.addWidget(self.log_display)

        tabs.addTab(detector_tab, "Detector View"); tabs.addTab(photometry_tab, "Photometry View"); tabs.addTab(detrended_tab, "Detrended View"); tabs.addTab(phase_folded_tab, "Phase-Folded View"); tabs.addTab(log_tab, "Performance")
        
        main_visuals_layout = QVBoxLayout()
        main_visuals_layout.addWidget(frame_group)
        main_visuals_layout.addWidget(tabs)
        
        container = QWidget()
        container.setLayout(main_visuals_layout)
        return container

    def on_detrend_model_change(self, model_name):
        if model_name == "Polynomial": self.detrend_params_stack.setCurrentIndex(0)
        elif model_name == "Savitzky-Golay": self.detrend_params_stack.setCurrentIndex(1)
    
    def run_pipeline_and_update_plots(self):
        if not (self.observation and self.observation.is_loaded): return
        self.status_bar.showMessage("Running pipeline...", 10000); QApplication.processEvents()
        steps_to_run = {name: checkbox.isChecked() for name, checkbox in self.calib_checkboxes.items()}
        log_messages = self.observation.run_calibration_pipeline(steps_to_run)
        log_messages = self.observation.run_photometry()
        model_name = self.detrend_model_combo.currentText()
        if model_name == "Polynomial": detrender = detrending.PolynomialDetrender(degree=self.poly_degree_spinbox.value())
        elif model_name == "Savitzky-Golay": detrender = detrending.SavGolDetrender(window_length=self.savgol_window_spinbox.value(), polyorder=self.savgol_order_spinbox.value())
        else: return
        log_messages = self.observation.run_detrending(detrender)
        log_messages = self.observation.run_phase_folding(n_bins=self.bins_spinbox.value())
        self.log_display.setText("\n".join(log_messages))
        self.update_image_plot(); self.update_all_light_curve_plots()
        self.status_bar.showMessage("Pipeline finished.", 5000)

    def update_all_light_curve_plots(self):
        self.update_light_curve_plot(); self.update_detrended_plot(); self.update_phase_folded_plot()

    def update_image_plot(self):
        if not (self.observation and self.observation.is_loaded): return
        processed_signal = self.observation.get_data(return_type='numpy'); frame_idx = self.frame_slider.value(); is_cds_applied = self.calib_checkboxes["cds"].isChecked()
        if is_cds_applied: frame_idx //= 2
        if frame_idx >= processed_signal.shape[0]: frame_idx = processed_signal.shape[0] - 1
        img_data = processed_signal[frame_idx]
        if self.image_cbar: self.image_cbar.remove()
        self.image_ax.clear()
        im = self.image_ax.imshow(img_data, aspect='auto', cmap='viridis', origin='lower'); self.image_cbar = self.image_canvas.figure.colorbar(im, ax=self.image_ax)
        self.image_ax.set_title(f"Detector Frame (Index: {frame_idx}) with Apertures"); self.image_ax.set_xlabel("Pixel Column (Wavelength)"); self.image_ax.set_ylabel("Pixel Row (Spatial)")
        aperture_settings = PHOTOMETRY_APERTURES[self.observation.instrument]; sig = aperture_settings['signal']
        sig_rect = Rectangle((sig['x_start']-0.5, sig['y_start']-0.5), sig['x_end']-sig['x_start']+1, sig['y_end']-sig['y_start']+1, linewidth=1.5, edgecolor='lime', facecolor='none', linestyle='--'); self.image_ax.add_patch(sig_rect)
        for bg in aperture_settings['background']:
            bg_rect = Rectangle((bg['x_start']-0.5, bg['y_start']-0.5), bg['x_end']-bg['x_start']+1, bg['y_end']-bg['y_start']+1, linewidth=1, edgecolor='r', facecolor='none', linestyle=':'); self.image_ax.add_patch(bg_rect)
        self.image_canvas.draw()

    def update_light_curve_plot(self):
        if not (self.observation and self.observation.is_loaded): return
        light_curves = self.observation.get_light_curves(return_type='numpy'); self.single_lc_ax.clear()
        if light_curves is not None:
            wavelength_col = self.wavelength_slider.value()
            if wavelength_col >= light_curves.shape[1]: wavelength_col = light_curves.shape[1] - 1; self.wavelength_slider.setValue(wavelength_col)
            single_lc = light_curves[:, wavelength_col]
            self.single_lc_ax.plot(single_lc, '.-', color='dodgerblue', alpha=0.8); self.single_lc_ax.set_title(f"Light Curve for Wavelength Column: {wavelength_col}"); self.single_lc_ax.set_ylabel("Background-Subtracted Flux"); self.single_lc_ax.grid(True, alpha=0.3)
        self.single_lc_canvas.draw()

    def update_detrended_plot(self):
        if not (self.observation and self.observation.is_loaded): return
        detrended_lcs = self.observation.get_detrended_light_curves(return_type='numpy'); original_lcs = self.observation.get_light_curves(return_type='numpy'); noise_models = self.observation.get_noise_models(return_type='numpy'); self.detrended_lc_ax.clear()
        if detrended_lcs is not None and original_lcs is not None and noise_models is not None:
            wavelength_col = self.wavelength_slider.value()
            if wavelength_col >= detrended_lcs.shape[1]: wavelength_col = detrended_lcs.shape[1] - 1
            detrended_lc = detrended_lcs[:, wavelength_col]; original_lc = original_lcs[:, wavelength_col]; noise_model = noise_models[:, wavelength_col]
            if self.zoom_checkbox.isChecked():
                self.detrended_lc_ax.plot(detrended_lc, '.', color='black', alpha=0.8, label='Detrended Data')
                valid_data = detrended_lc[np.isfinite(detrended_lc)]
                if len(valid_data) > 0:
                    data_min, data_max = np.min(valid_data), np.max(valid_data); margin = (data_max - data_min) * 0.1
                    self.detrended_lc_ax.set_ylim(data_min - margin, data_max + margin)
            else:
                self.detrended_lc_ax.plot(original_lc, '.', color='grey', alpha=0.2, label='Original Data'); self.detrended_lc_ax.plot(noise_model, '-', color='red', alpha=0.6, label='Noise Model'); self.detrended_lc_ax.plot(detrended_lc, '.', color='black', alpha=0.8, label='Detrended Data')
            self.detrended_lc_ax.axhline(1.0, color='k', linestyle='--', alpha=0.5); self.detrended_lc_ax.set_title(f"Detrended Light Curve for Wavelength Column: {wavelength_col}"); self.detrended_lc_ax.set_ylabel("Normalized Flux"); self.detrended_lc_ax.legend(); self.detrended_lc_ax.grid(True, alpha=0.3)
        self.detrended_lc_canvas.draw()

    def update_phase_folded_plot(self):
        if not (self.observation and self.observation.is_loaded): return
        folded_data = self.observation.get_phase_folded_lc(); self.phase_folded_ax.clear()
        if folded_data is not None:
            wavelength_col = self.wavelength_slider.value()
            if wavelength_col >= len(folded_data): wavelength_col = len(folded_data) - 1
            bin_centers, binned_flux, binned_error = folded_data[wavelength_col]
            self.phase_folded_ax.errorbar(bin_centers, binned_flux, yerr=binned_error, fmt='o', color='black', ecolor='gray', elinewidth=1, capsize=2)
            self.phase_folded_ax.axhline(1.0, color='r', linestyle='--', alpha=0.7); self.phase_folded_ax.set_title(f"Phase-Folded Transit for Wavelength: {wavelength_col}"); self.phase_folded_ax.set_xlabel("Orbital Phase"); self.phase_folded_ax.set_ylabel("Normalized Flux"); self.phase_folded_ax.grid(True, alpha=0.3)
            valid_data = binned_flux[np.isfinite(binned_flux)]
            if len(valid_data) > 0:
                data_min, data_max = np.min(valid_data), np.max(valid_data); margin = (data_max - data_min) * 0.2
                self.phase_folded_ax.set_ylim(data_min - margin, data_max + margin)
        self.phase_folded_canvas.draw()

    def load_data(self):
        planet_id_str = self.planet_combo.currentText(); instrument = self.instrument_combo.currentText(); split = self.split_combo.currentText(); obs_id_str = self.obs_combo.currentText(); backend = self.backend_combo.currentText()
        if not all([planet_id_str, obs_id_str]): self.status_bar.showMessage("Error: Missing Planet or Observation ID.", 5000); return
        try:
            self.status_bar.showMessage(f"Loading data for {planet_id_str} on {backend.upper()}..."); QApplication.processEvents()
            self.info_display.clear()
            if self.star_info_df is not None:
                try:
                    planet_id_int = int(planet_id_str)
                    if planet_id_int in self.star_info_df.index: self.info_display.setText(self.star_info_df.loc[planet_id_int].to_string())
                    else: self.info_display.setText(f"Info not found for planet_id: {planet_id_str}")
                except (ValueError, KeyError): self.info_display.setText(f"Info not found for planet_id: {planet_id_str}")
            self.observation = DataObservation(planet_id_str, instrument, int(obs_id_str), split)
            self.observation.load(backend=backend)
            
            # --- FIX: Enable/disable wavelength slider based on instrument ---
            is_fgs = (instrument == 'FGS1')
            self.wavelength_slider.setEnabled(not is_fgs)
            self.wavelength_spinbox.setEnabled(not is_fgs)

            self.frame_slider.setRange(0, self.observation.raw_signal.shape[0] - 1); self.frame_spinbox.setRange(0, self.observation.raw_signal.shape[0] - 1)
            self.wavelength_slider.setRange(0, self.observation.raw_signal.shape[2] - 1); self.wavelength_spinbox.setRange(0, self.observation.raw_signal.shape[2] - 1)
            self.status_bar.showMessage("Data loaded. Running initial pipeline...", 5000)
            self.run_pipeline_and_update_plots()
        except Exception as e:
            self.status_bar.showMessage(f"Error loading data: {e}", 5000); self.observation = None
    
    def populate_planet_ids(self):
        current_planet = self.planet_combo.currentText(); self.planet_combo.clear(); split = self.split_combo.currentText()
        self.star_info_df = loaders.load_star_info(split)
        path = DATASET_DIR / split
        if path.exists():
            planet_ids = sorted([p.name for p in path.iterdir() if p.is_dir()]); self.planet_combo.addItems(planet_ids)
            if current_planet in planet_ids: self.planet_combo.setCurrentText(current_planet)
        self.populate_obs_ids()
    def populate_obs_ids(self):
        self.obs_combo.clear(); planet_id = self.planet_combo.currentText(); instrument = self.instrument_combo.currentText(); split = self.split_combo.currentText()
        if not planet_id: return
        planet_dir = DATASET_DIR / split / planet_id
        if not planet_dir.exists(): return
        obs_ids = []
        for f in planet_dir.glob(f"{instrument}_signal_*.parquet"):
            try:
                obs_id = f.stem.split('_')[-1]
                if obs_id.isdigit(): obs_ids.append(obs_id)
            except (IndexError, ValueError): continue
        if obs_ids: self.obs_combo.addItems(sorted(obs_ids))
    def on_mouse_move(self, event):
        if event.inaxes and event.inaxes == self.image_ax:
            x, y = int(event.xdata or 0), int(event.ydata or 0)
            if self.observation and self.observation.processed_signal is not None:
                processed_signal = self.observation.get_data(return_type='numpy'); frame_idx = self.frame_slider.value()
                if self.calib_checkboxes["cds"].isChecked(): frame_idx //= 2
                if 0 <= y < processed_signal.shape[1] and 0 <= x < processed_signal.shape[2]:
                    value = processed_signal[frame_idx, y, x]
                    self.status_bar.showMessage(f"Pixel (x={x}, y={y}) | Value: {value:.2f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    inspector = DataInspector()
    inspector.show()
    sys.exit(app.exec())

    