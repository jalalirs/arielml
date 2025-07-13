import sys
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
from arielml.config import DATASET_DIR, PHOTOMETRY_APERTURES
from arielml.backend import GPU_ENABLED, GP_GPU_ENABLED

class DataInspector(QMainWindow):
    """
    A PyQt GUI application to inspect Ariel dataset files and visualize
    the full data reduction pipeline.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ariel Data Inspector")
        self.setGeometry(100, 100, 1600, 900)

        self.observation = None
        self.star_info_df = None

        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        visuals_panel = self._create_visuals_panel()
        controls_panel = self._create_controls_panel()

        main_layout.addWidget(visuals_panel, 3)
        main_layout.addWidget(controls_panel, 1)
        
        self.setStatusBar(QStatusBar())
        self.populate_planet_ids()

    # --- UI Creation Methods (Broken down for clarity) ---

    def _create_controls_panel(self):
        """Creates the right-hand panel with all user controls."""
        main_layout = QVBoxLayout()
        
        main_layout.addWidget(self._create_navigation_group())
        main_layout.addWidget(self._create_calibration_group())
        main_layout.addWidget(self._create_detrending_group())
        main_layout.addWidget(self._create_folding_group())
        
        self.apply_button = QPushButton("Apply Changes & Rerun Pipeline")
        self.apply_button.clicked.connect(self.run_pipeline_and_update_plots)
        main_layout.addWidget(self.apply_button)
        
        main_layout.addStretch()
        
        container = QWidget()
        container.setLayout(main_layout)
        return container

    def _create_navigation_group(self):
        nav_group = QGroupBox("Navigation")
        layout = QVBoxLayout()

        self.split_combo = QComboBox()
        self.split_combo.addItems(["train", "test"])
        self.split_combo.currentTextChanged.connect(self.populate_planet_ids)

        self.planet_combo = QComboBox()
        self.planet_combo.setEditable(True)
        self.planet_combo.currentTextChanged.connect(self.populate_obs_ids)

        self.obs_combo = QComboBox()

        self.instrument_combo = QComboBox()
        self.instrument_combo.addItems(["AIRS-CH0", "FGS1"])
        self.instrument_combo.currentTextChanged.connect(self.populate_obs_ids)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["cpu", "gpu"])
        if not GPU_ENABLED:
            self.backend_combo.model().item(1).setEnabled(False)

        self.load_button = QPushButton("Load Planet Data")
        self.load_button.clicked.connect(self.load_data)

        for label, widget in [
            ("Dataset Split:", self.split_combo), ("Planet ID:", self.planet_combo),
            ("Observation ID:", self.obs_combo), ("Instrument:", self.instrument_combo),
            ("Processing Backend:", self.backend_combo)
        ]:
            layout.addWidget(QLabel(label))
            layout.addWidget(widget)
        layout.addWidget(self.load_button)

        nav_group.setLayout(layout)
        return nav_group

    def _create_calibration_group(self):
        calib_group = QGroupBox("Calibration Steps")
        layout = QVBoxLayout()
        self.calib_checkboxes = {
            "adc": QCheckBox("ADC Conversion"),
            "mask": QCheckBox("Mask Hot/Dead Pixels"),
            "linearity": QCheckBox("Linearity Correction"),
            "dark": QCheckBox("Dark Current Subtraction"),
            "cds": QCheckBox("Correlated Double Sampling (CDS)"),
            "flat": QCheckBox("Flat Field Correction"),
        }
        for checkbox in self.calib_checkboxes.values():
            checkbox.setChecked(True)
            layout.addWidget(checkbox)
        calib_group.setLayout(layout)
        return calib_group

    def _create_detrending_group(self):
        detrend_group = QGroupBox("Detrending Model")
        layout = QVBoxLayout()
        
        self.detrend_model_combo = QComboBox()
        detrend_options = [
            "Polynomial", "Savitzky-Golay", "GP (george CPU)", "Hybrid GP (CPU)"
        ]
        if GP_GPU_ENABLED: 
            detrend_options.extend(["GP (GPyTorch GPU)", "Hybrid GP (GPU)"])
        self.detrend_model_combo.addItems(sorted(detrend_options))
        self.detrend_model_combo.currentTextChanged.connect(self.on_detrend_model_change)
        layout.addWidget(self.detrend_model_combo)

        self.detrend_params_stack = QStackedWidget()
        self.model_widget_map = {}

        # Polynomial parameters
        poly_widget = QWidget(); poly_layout = QHBoxLayout(poly_widget); poly_layout.addWidget(QLabel("Degree:")); self.poly_degree_spinbox = QSpinBox(); self.poly_degree_spinbox.setRange(1, 10); self.poly_degree_spinbox.setValue(2); poly_layout.addWidget(self.poly_degree_spinbox); self.model_widget_map["Polynomial"] = self.detrend_params_stack.addWidget(poly_widget)
        
        # Savitzky-Golay parameters
        savgol_widget = QWidget(); savgol_layout = QHBoxLayout(savgol_widget); savgol_layout.addWidget(QLabel("Window:")); self.savgol_window_spinbox = QSpinBox(); self.savgol_window_spinbox.setRange(3, 999); self.savgol_window_spinbox.setSingleStep(2); self.savgol_window_spinbox.setValue(51); savgol_layout.addWidget(self.savgol_window_spinbox); savgol_layout.addWidget(QLabel("Order:")); self.savgol_order_spinbox = QSpinBox(); self.savgol_order_spinbox.setRange(1, 10); self.savgol_order_spinbox.setValue(2); savgol_layout.addWidget(self.savgol_order_spinbox); self.model_widget_map["Savitzky-Golay"] = self.detrend_params_stack.addWidget(savgol_widget)
        
        # George GP parameters
        george_widget = QWidget(); george_layout = QHBoxLayout(george_widget); george_layout.addWidget(QLabel("Kernel:")); self.george_kernel_combo = QComboBox(); self.george_kernel_combo.addItems(['Matern32']); george_layout.addWidget(self.george_kernel_combo); self.model_widget_map["GP (george CPU)"] = self.detrend_params_stack.addWidget(george_widget)
        
        # GPyTorch GP parameters
        if GP_GPU_ENABLED:
            gpytorch_widget = QWidget(); gpytorch_layout = QHBoxLayout(gpytorch_widget); gpytorch_layout.addWidget(QLabel("Iterations:")); self.gpytorch_iter_spinbox = QSpinBox(); self.gpytorch_iter_spinbox.setRange(10, 500); self.gpytorch_iter_spinbox.setValue(50); gpytorch_layout.addWidget(self.gpytorch_iter_spinbox); self.model_widget_map["GP (GPyTorch GPU)"] = self.detrend_params_stack.addWidget(gpytorch_widget)
        
        # Hybrid GP parameters
        hybrid_widget = QWidget(); hybrid_layout = QVBoxLayout(hybrid_widget); hybrid_poly_layout = QHBoxLayout(); hybrid_poly_layout.addWidget(QLabel("Residual Degree:")); self.hybrid_poly_degree_spinbox = QSpinBox(); self.hybrid_poly_degree_spinbox.setRange(1, 5); self.hybrid_poly_degree_spinbox.setValue(2); hybrid_poly_layout.addWidget(self.hybrid_poly_degree_spinbox); hybrid_layout.addLayout(hybrid_poly_layout); hybrid_iter_layout = QHBoxLayout(); hybrid_iter_layout.addWidget(QLabel("GP Iterations:")); self.hybrid_iter_spinbox = QSpinBox(); self.hybrid_iter_spinbox.setRange(10, 500); self.hybrid_iter_spinbox.setValue(50); hybrid_iter_layout.addWidget(self.hybrid_iter_spinbox); hybrid_layout.addLayout(hybrid_iter_layout); 
        self.model_widget_map["Hybrid GP (CPU)"] = self.detrend_params_stack.addWidget(hybrid_widget)
        if GP_GPU_ENABLED:
             self.model_widget_map["Hybrid GP (GPU)"] = self.detrend_params_stack.indexOf(hybrid_widget)

        layout.addWidget(self.detrend_params_stack)
        detrend_group.setLayout(layout)
        return detrend_group

    def _create_folding_group(self):
        fold_group = QGroupBox("Phase Folding")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Number of Bins:"))
        self.bins_spinbox = QSpinBox()
        self.bins_spinbox.setRange(10, 500)
        self.bins_spinbox.setValue(100)
        layout.addWidget(self.bins_spinbox)
        fold_group.setLayout(layout)
        return fold_group

    def _create_visuals_panel(self):
        main_layout = QVBoxLayout()
        tabs = QTabWidget()

        frame_group = QGroupBox("Frame & Wavelength Selector")
        frame_layout = QHBoxLayout(frame_group)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_spinbox = QSpinBox()
        self.wavelength_slider = QSlider(Qt.Orientation.Horizontal)
        self.wavelength_spinbox = QSpinBox()
        self.frame_slider.valueChanged.connect(self.update_image_plot)
        self.frame_spinbox.valueChanged.connect(self.frame_slider.setValue)
        self.frame_slider.valueChanged.connect(self.frame_spinbox.setValue)
        self.wavelength_slider.valueChanged.connect(self.update_light_curve_plots)
        self.wavelength_spinbox.valueChanged.connect(self.wavelength_slider.setValue)
        self.wavelength_slider.valueChanged.connect(self.wavelength_spinbox.setValue)
        frame_layout.addWidget(QLabel("Frame Index:")); frame_layout.addWidget(self.frame_slider); frame_layout.addWidget(self.frame_spinbox)
        frame_layout.addWidget(QLabel("Wavelength Column:")); frame_layout.addWidget(self.wavelength_slider); frame_layout.addWidget(self.wavelength_spinbox)
        main_layout.addWidget(frame_group)

        self.image_canvas = FigureCanvas(Figure(figsize=(10, 5), tight_layout=True))
        self.single_lc_canvas = FigureCanvas(Figure(figsize=(10, 3), tight_layout=True))
        self.detrended_lc_canvas = FigureCanvas(Figure(figsize=(10, 3), tight_layout=True))
        self.phase_folded_canvas = FigureCanvas(Figure(figsize=(10, 3), tight_layout=True))
        self.image_ax = self.image_canvas.figure.subplots()
        self.single_lc_ax = self.single_lc_canvas.figure.subplots()
        self.detrended_lc_ax = self.detrended_lc_canvas.figure.subplots()
        self.phase_folded_ax = self.phase_folded_canvas.figure.subplots()
        self.image_cbar = None

        tabs.addTab(self._create_detector_tab(), "Detector View")
        tabs.addTab(self.single_lc_canvas, "Photometry View")
        tabs.addTab(self._create_detrended_tab(), "Detrended View")
        tabs.addTab(self.phase_folded_canvas, "Phase-Folded View")
        self.log_display = QTextEdit(); self.log_display.setReadOnly(True)
        tabs.addTab(self.log_display, "Performance Log")
        
        main_layout.addWidget(tabs)
        container = QWidget(); container.setLayout(main_layout)
        return container

    def _create_detector_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); info_group = QGroupBox("Star & Planet Info"); info_layout = QVBoxLayout(info_group); self.info_display = QTextEdit(); self.info_display.setReadOnly(True); info_group.setFixedHeight(150); info_layout.addWidget(self.info_display); layout.addWidget(info_group); layout.addWidget(self.image_canvas); self.image_canvas.mpl_connect('motion_notify_event', self.on_mouse_move); return tab
    def _create_detrended_tab(self):
        tab = QWidget(); layout = QVBoxLayout(tab); self.zoom_checkbox = QCheckBox("Zoom to Transit"); self.zoom_checkbox.toggled.connect(self.update_detrended_plot); layout.addWidget(self.zoom_checkbox); layout.addWidget(self.detrended_lc_canvas); return tab

    # --- Core Logic ---

    def run_pipeline_and_update_plots(self):
        if not (self.observation and self.observation.is_loaded):
            self.statusBar().showMessage("Please load data first.", 5000)
            return
            
        self.statusBar().showMessage("Running pipeline...", 10000)
        QApplication.processEvents()
        
        try:
            steps_to_run = {name: cb.isChecked() for name, cb in self.calib_checkboxes.items()}
            self.observation.run_calibration_pipeline(steps_to_run)
            self.observation.run_photometry()
            detrender = self._get_detrender(self.detrend_model_combo.currentText())
            if detrender:
                self.observation.run_detrending(detrender)
            self.observation.run_phase_folding(n_bins=self.bins_spinbox.value())
        except (ValueError, RuntimeError) as e:
            self.statusBar().showMessage(f"ERROR: {e}", 15000)
            print(f"ERROR: {e}")
            return
        
        self._update_all_plots()
        self.statusBar().showMessage("Pipeline finished.", 5000)

    def _get_detrender(self, model_name):
        if model_name == "Polynomial":
            return detrending.PolynomialDetrender(degree=self.poly_degree_spinbox.value())
        elif model_name == "Savitzky-Golay":
            return detrending.SavGolDetrender(window_length=self.savgol_window_spinbox.value(), polyorder=self.savgol_order_spinbox.value())
        elif model_name == "GP (george CPU)":
            return detrending.GPDetrender(kernel=self.george_kernel_combo.currentText())
        elif model_name == "GP (GPyTorch GPU)":
            if GP_GPU_ENABLED:
                self.statusBar().showMessage("Running GPyTorch Detrending (GPU)...", 30000)
                QApplication.processEvents()
                return detrending.GPyTorchDetrender(training_iter=self.gpytorch_iter_spinbox.value())
        elif model_name == "Hybrid GP (CPU)":
            return detrending.HybridDetrender(use_gpu=False, training_iter=self.hybrid_iter_spinbox.value(), poly_degree=self.hybrid_poly_degree_spinbox.value())
        elif model_name == "Hybrid GP (GPU)":
            if GP_GPU_ENABLED:
                self.statusBar().showMessage("Running Hybrid GP Detrending (GPU)...", 30000)
                QApplication.processEvents()
                return detrending.HybridDetrender(use_gpu=True, training_iter=self.hybrid_iter_spinbox.value(), poly_degree=self.hybrid_poly_degree_spinbox.value())
        return None

    # --- Plotting and UI Updates ---

    def _update_all_plots(self):
        self.log_display.setText("\n".join(self.observation.calibration_log))
        self.update_image_plot()
        self.update_light_curve_plots()

    def update_image_plot(self):
        if not (self.observation and self.observation.is_loaded): return
        
        if self.image_cbar:
            self.image_cbar.remove()
            self.image_cbar = None
        self.image_ax.clear()
        
        processed_signal = self.observation.get_data(return_type='numpy')
        frame_idx = self.frame_slider.value()
        if self.calib_checkboxes["cds"].isChecked():
            frame_idx //= 2
        if frame_idx >= processed_signal.shape[0]:
            frame_idx = 0
        
        img_data = processed_signal[frame_idx]
        im = self.image_ax.imshow(img_data, aspect='auto', cmap='viridis', origin='lower')
        self.image_cbar = self.image_canvas.figure.colorbar(im, ax=self.image_ax)
        self.image_ax.set_title(f"Detector Frame (Index: {frame_idx})")
        self.image_canvas.draw()

    def update_light_curve_plots(self):
        if not (self.observation and self.observation.is_loaded): return
        self.update_photometry_plot()
        self.update_detrended_plot()
        self.update_phase_folded_plot()

    def update_photometry_plot(self):
        self.single_lc_ax.clear()
        if self.observation.light_curves is None:
            self.single_lc_canvas.draw()
            return
        
        light_curves = self.observation.get_light_curves(return_type='numpy')
        wavelength_col = self.wavelength_slider.value()
        if wavelength_col >= light_curves.shape[1]:
            wavelength_col = 0
        
        self.single_lc_ax.plot(self.observation.get_time_array(), light_curves[:, wavelength_col], '.-', alpha=0.8, color='dodgerblue')
        self.single_lc_ax.set_title(f"Raw Light Curve (Wavelength: {wavelength_col})")
        self.single_lc_ax.set_xlabel("Time (days)")
        self.single_lc_ax.set_ylabel("Flux")
        self.single_lc_ax.grid(True, alpha=0.3)
        self.single_lc_canvas.draw()

    def update_detrended_plot(self):
        self.detrended_lc_ax.clear()
        if self.observation.detrended_light_curves is None:
            self.detrended_lc_canvas.draw()
            return

        detrended_lcs = self.observation.get_detrended_light_curves(return_type='numpy')
        original_lcs = self.observation.get_light_curves(return_type='numpy')
        noise_models = self.observation.get_noise_models(return_type='numpy')
        wavelength_col = self.wavelength_slider.value()
        if wavelength_col >= detrended_lcs.shape[1]:
            wavelength_col = 0
        time_arr = self.observation.get_time_array()

        if self.zoom_checkbox.isChecked():
            self.detrended_lc_ax.plot(time_arr, detrended_lcs[:, wavelength_col], '.', color='black')
        else:
            self.detrended_lc_ax.plot(time_arr, original_lcs[:, wavelength_col], '.', color='grey', alpha=0.2, label='Original')
            self.detrended_lc_ax.plot(time_arr, noise_models[:, wavelength_col], '-', color='red', label='Noise Model')
            self.detrended_lc_ax.plot(time_arr, detrended_lcs[:, wavelength_col], '.', color='black', alpha=0.8, label='Detrended')
            self.detrended_lc_ax.legend()
        
        self.detrended_lc_ax.set_title(f"Detrended Light Curve (Wavelength: {wavelength_col})")
        self.detrended_lc_ax.set_xlabel("Time (days)")
        self.detrended_lc_ax.set_ylabel("Normalized Flux")
        self.detrended_lc_ax.grid(True, alpha=0.3)
        self.detrended_lc_canvas.draw()

    def update_phase_folded_plot(self):
        self.phase_folded_ax.clear()
        if self.observation.phase_folded_lc is None:
            self.phase_folded_canvas.draw()
            return

        folded_data = self.observation.get_phase_folded_lc()
        wavelength_col = self.wavelength_slider.value()
        if wavelength_col >= len(folded_data):
            wavelength_col = 0
        
        bin_centers, binned_flux, binned_error = folded_data[wavelength_col]
        self.phase_folded_ax.errorbar(bin_centers, binned_flux, yerr=binned_error, fmt='o', color='black', ecolor='gray', elinewidth=1, capsize=2)
        self.phase_folded_ax.axhline(1.0, color='r', linestyle='--', alpha=0.7)
        self.phase_folded_ax.set_title(f"Phase-Folded Transit (Wavelength: {wavelength_col})")
        self.phase_folded_ax.set_xlabel("Orbital Phase")
        self.phase_folded_ax.set_ylabel("Normalized Flux")
        self.phase_folded_ax.grid(True, alpha=0.3)
        self.phase_folded_canvas.draw()

    # --- Data Loading and UI Callbacks ---

    def load_data(self):
        planet_id = self.planet_combo.currentText()
        obs_id = self.obs_combo.currentText()
        if not all([planet_id, obs_id]):
            self.statusBar().showMessage("Error: Missing Planet or Observation ID.", 5000)
            return
        try:
            self.statusBar().showMessage(f"Loading data for {planet_id}...")
            QApplication.processEvents()
            self.observation = DataObservation(
                planet_id, self.instrument_combo.currentText(), int(obs_id), self.split_combo.currentText()
            )
            self.observation.load(backend=self.backend_combo.currentText())

            if self.star_info_df is not None and int(planet_id) in self.star_info_df.index:
                self.info_display.setText(self.star_info_df.loc[int(planet_id)].to_string())

            is_fgs = (self.instrument_combo.currentText() == 'FGS1')
            self.wavelength_slider.setEnabled(not is_fgs)
            self.wavelength_spinbox.setEnabled(not is_fgs)
            
            # FIX: Set the range for both the slider and the spinbox
            max_wl = self.observation.raw_signal.shape[2] - 1
            self.frame_slider.setRange(0, self.observation.raw_signal.shape[0] - 1)
            self.frame_spinbox.setRange(0, self.observation.raw_signal.shape[0] - 1)
            self.wavelength_slider.setRange(0, max_wl)
            self.wavelength_spinbox.setRange(0, max_wl)
            
            self.run_pipeline_and_update_plots()
        except Exception as e:
            self.statusBar().showMessage(f"Error loading data: {e}", 15000)
            print(f"ERROR: {e}")

    def populate_planet_ids(self):
        current_planet = self.planet_combo.currentText()
        self.planet_combo.clear()
        split = self.split_combo.currentText()
        self.star_info_df = loaders.load_star_info(split)
        if self.star_info_df is None: return
        path = DATASET_DIR / split
        if path.exists():
            planet_ids = sorted([p.name for p in path.iterdir() if p.is_dir()])
            self.planet_combo.addItems(planet_ids)
            if current_planet in planet_ids:
                self.planet_combo.setCurrentText(current_planet)
        self.populate_obs_ids()

    def populate_obs_ids(self):
        self.obs_combo.clear()
        planet_id = self.planet_combo.currentText()
        if not planet_id: return
        planet_dir = DATASET_DIR / self.split_combo.currentText() / planet_id
        if not planet_dir.exists(): return
        obs_ids = sorted([
            f.stem.split('_')[-1] for f in planet_dir.glob(f"{self.instrument_combo.currentText()}_signal_*.parquet")
            if f.stem.split('_')[-1].isdigit()
        ], key=int)
        if obs_ids:
            self.obs_combo.addItems(obs_ids)

    def on_detrend_model_change(self, model_name):
        index = self.model_widget_map.get(model_name, 0)
        self.detrend_params_stack.setCurrentIndex(index)

    def on_mouse_move(self, event):
        if not (event.inaxes and event.inaxes == self.image_ax and self.observation): return
        x, y = int(event.xdata or 0), int(event.ydata or 0)
        if self.observation.processed_signal is not None:
            data = self.observation.get_data('numpy')
            if 0 <= y < data.shape[1] and 0 <= x < data.shape[2]:
                self.statusBar().showMessage(f"Pixel (x={x}, y={y}) | Value: {data[self.frame_slider.value(), y, x]:.2f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    inspector = DataInspector()
    inspector.show()
    sys.exit(app.exec())
