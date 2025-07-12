import sys
import os
from pathlib import Path

# Add the project root to the Python path to allow importing 'arielml'
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QPushButton, QSlider, QLabel, QGroupBox, QCheckBox, QSpinBox, QStatusBar,
    QTextEdit, QTabWidget
)
from PyQt6.QtCore import Qt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# Import the new object-oriented components
from arielml.data.observation import DataObservation
from arielml.data import loaders
from arielml.config import DATASET_DIR, INSTRUMENT_SHAPES, PHOTOMETRY_APERTURES
from arielml.backend import GPU_ENABLED

class DataInspector(QMainWindow):
    """
    A PyQt GUI application to inspect Ariel dataset files and visualize
    the full calibration and photometry pipeline using a tabbed interface.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ariel Data Inspector (v3.4 Optimized)")
        self.setGeometry(100, 100, 1366, 768)

        self.observation = None
        self.star_info_df = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        controls_layout = self.create_controls_panel()
        main_layout.addLayout(controls_layout, 1)

        visuals_layout = self.create_visuals_panel()
        main_layout.addLayout(visuals_layout, 3)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.populate_planet_ids()

    def create_controls_panel(self):
        """Creates the left panel with all user controls."""
        layout = QVBoxLayout()
        layout.setSpacing(10)

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
        layout.addWidget(nav_group)

        frame_group = QGroupBox("Frame Selector")
        frame_layout = QVBoxLayout()
        # --- FIX: Connect frame slider to image plot update ---
        self.frame_slider = QSlider(Qt.Orientation.Horizontal); self.frame_slider.valueChanged.connect(self.update_image_plot)
        self.frame_spinbox = QSpinBox(); self.frame_spinbox.valueChanged.connect(self.frame_slider.setValue); self.frame_slider.valueChanged.connect(self.frame_spinbox.setValue)
        frame_layout.addWidget(self.frame_slider); frame_layout.addWidget(self.frame_spinbox)
        frame_group.setLayout(frame_layout)
        layout.addWidget(frame_group)

        phot_group = QGroupBox("Photometry")
        phot_layout = QVBoxLayout()
        # --- FIX: Connect wavelength slider to light curve plot update ---
        self.wavelength_slider = QSlider(Qt.Orientation.Horizontal)
        self.wavelength_slider.valueChanged.connect(self.update_light_curve_plot)
        self.wavelength_spinbox = QSpinBox()
        self.wavelength_spinbox.valueChanged.connect(self.wavelength_slider.setValue)
        self.wavelength_slider.valueChanged.connect(self.wavelength_spinbox.setValue)
        phot_layout.addWidget(QLabel("Wavelength Column:"))
        phot_layout.addWidget(self.wavelength_slider)
        phot_layout.addWidget(self.wavelength_spinbox)
        phot_group.setLayout(phot_layout)
        layout.addWidget(phot_group)

        calib_group = QGroupBox("Calibration Steps")
        calib_layout = QVBoxLayout()
        self.calib_checkboxes = {
            "adc": QCheckBox("ADC Conversion"), "mask": QCheckBox("Mask Hot/Dead Pixels"),
            "linearity": QCheckBox("Linearity Correction"), "dark": QCheckBox("Dark Current Subtraction"),
            "cds": QCheckBox("Correlated Double Sampling (CDS)"), "flat": QCheckBox("Flat Field Correction"),
        }
        for checkbox in self.calib_checkboxes.values():
            checkbox.setChecked(True)
            checkbox.toggled.connect(self.run_pipeline_and_update_plots)
            calib_layout.addWidget(checkbox)
        calib_group.setLayout(calib_layout)
        layout.addWidget(calib_group)
        
        layout.addStretch()
        return layout

    def create_visuals_panel(self):
        """Creates the right panel with a tabbed interface for all visuals."""
        tabs = QTabWidget()
        detector_tab = QWidget()
        detector_layout = QVBoxLayout(detector_tab)
        info_group = QGroupBox("Star & Planet Info")
        info_layout_inner = QVBoxLayout()
        self.info_display = QTextEdit(); self.info_display.setReadOnly(True)
        info_group.setFixedHeight(150)
        info_layout_inner.addWidget(self.info_display)
        info_group.setLayout(info_layout_inner)
        detector_layout.addWidget(info_group)
        self.image_canvas = FigureCanvas(Figure(figsize=(10, 5)))
        self.image_ax = self.image_canvas.figure.subplots()
        self.image_cbar = None
        detector_layout.addWidget(self.image_canvas)
        self.image_canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        photometry_tab = QWidget()
        photometry_layout = QVBoxLayout(photometry_tab)
        self.single_lc_canvas = FigureCanvas(Figure(figsize=(10, 3)))
        self.single_lc_ax = self.single_lc_canvas.figure.subplots()
        photometry_layout.addWidget(self.single_lc_canvas)
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log_display = QTextEdit(); self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)
        tabs.addTab(detector_tab, "Detector View")
        tabs.addTab(photometry_tab, "Photometry View")
        tabs.addTab(log_tab, "Performance")
        main_visuals_layout = QVBoxLayout()
        main_visuals_layout.addWidget(tabs)
        return main_visuals_layout

    def load_data(self):
        planet_id_str = self.planet_combo.currentText()
        instrument = self.instrument_combo.currentText()
        split = self.split_combo.currentText(); obs_id_str = self.obs_combo.currentText(); backend = self.backend_combo.currentText()
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
            self.frame_slider.setRange(0, self.observation.raw_signal.shape[0] - 1)
            self.frame_spinbox.setRange(0, self.observation.raw_signal.shape[0] - 1)
            self.wavelength_slider.setRange(0, self.observation.raw_signal.shape[2] - 1)
            self.wavelength_spinbox.setRange(0, self.observation.raw_signal.shape[2] - 1)
            self.status_bar.showMessage("Data loaded. Running initial calibration...", 5000)
            self.run_pipeline_and_update_plots()
        except Exception as e:
            self.status_bar.showMessage(f"Error loading data: {e}", 5000)
            self.observation = None

    def run_pipeline_and_update_plots(self):
        if not (self.observation and self.observation.is_loaded): return
        steps_to_run = {name: checkbox.isChecked() for name, checkbox in self.calib_checkboxes.items()}
        log_messages = self.observation.run_calibration_pipeline(steps_to_run)
        log_messages = self.observation.run_photometry()
        self.log_display.setText("\n".join(log_messages))
        self.update_image_plot()
        self.update_light_curve_plot()

    def update_image_plot(self):
        if not (self.observation and self.observation.is_loaded): return
        processed_signal = self.observation.get_data(return_type='numpy')
        frame_idx = self.frame_slider.value()
        is_cds_applied = self.calib_checkboxes["cds"].isChecked()
        if is_cds_applied: frame_idx //= 2
        if frame_idx >= processed_signal.shape[0]: frame_idx = processed_signal.shape[0] - 1
        img_data = processed_signal[frame_idx]
        if self.image_cbar: self.image_cbar.remove()
        self.image_ax.clear()
        im = self.image_ax.imshow(img_data, aspect='auto', cmap='viridis', origin='lower')
        self.image_cbar = self.image_canvas.figure.colorbar(im, ax=self.image_ax)
        self.image_ax.set_title(f"Detector Frame (Index: {frame_idx}) with Apertures")
        self.image_ax.set_xlabel("Pixel Column (Wavelength)")
        self.image_ax.set_ylabel("Pixel Row (Spatial)")
        aperture_settings = PHOTOMETRY_APERTURES[self.observation.instrument]
        sig = aperture_settings['signal']
        sig_rect = Rectangle((sig['x_start']-0.5, sig['y_start']-0.5), sig['x_end']-sig['x_start']+1, sig['y_end']-sig['y_start']+1, linewidth=1.5, edgecolor='lime', facecolor='none', linestyle='--')
        self.image_ax.add_patch(sig_rect)
        for bg in aperture_settings['background']:
            bg_rect = Rectangle((bg['x_start']-0.5, bg['y_start']-0.5), bg['x_end']-bg['x_start']+1, bg['y_end']-bg['y_start']+1, linewidth=1, edgecolor='r', facecolor='none', linestyle=':')
            self.image_ax.add_patch(bg_rect)
        self.image_canvas.draw()

    def update_light_curve_plot(self):
        if not (self.observation and self.observation.is_loaded): return
        light_curves = self.observation.get_light_curves(return_type='numpy')
        self.single_lc_ax.clear()
        if light_curves is not None:
            wavelength_col = self.wavelength_slider.value()
            if wavelength_col >= light_curves.shape[1]:
                wavelength_col = light_curves.shape[1] - 1
                self.wavelength_slider.setValue(wavelength_col)
            single_lc = light_curves[:, wavelength_col]
            self.single_lc_ax.plot(single_lc, '.-', color='dodgerblue', alpha=0.8)
            self.single_lc_ax.set_title(f"Light Curve for Wavelength Column: {wavelength_col}")
            self.single_lc_ax.set_ylabel("Background-Subtracted Flux")
            self.single_lc_ax.grid(True, alpha=0.3)
        self.single_lc_canvas.draw()
        
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
