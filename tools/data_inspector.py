import sys
from pathlib import Path

# Add the project root to the Python path to allow importing 'arielml'
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QPushButton, QSlider, QLabel, QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox, QStatusBar,
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

# Check if sklearn GP is available
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    SKLEARN_GP_ENABLED = True
except ImportError:
    SKLEARN_GP_ENABLED = False

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
        
        # Static Navigation Group
        main_layout.addWidget(self._create_navigation_group())

        # --- Tabbed Interface for Settings ---
        settings_tabs = QTabWidget()
        settings_tabs.addTab(self._create_calibration_group(), "Calibration")
        settings_tabs.addTab(self._create_detrending_group(), "Detrending")
        settings_tabs.addTab(self._create_analysis_group(), "Analysis")
        
        main_layout.addWidget(settings_tabs)
        
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

        # Add widgets with labels
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
        
        # --- Dropdown for model selection ---
        self.detrend_model_combo = QComboBox()
        
        # Build detrending options based on available backends
        detrend_options = []
        
        # Always available (CPU)
        detrend_options.extend([
            "Polynomial",
            "Savitzky-Golay", 
            "GP (george CPU) ⚠️ Very Slow",
            "Hybrid GP (CPU)"
        ])
        
        # GPU options (if available)
        if GP_GPU_ENABLED: 
            detrend_options.extend([
                "GP (GPyTorch GPU)",
                "Hybrid GP (GPU)"
            ])
        else:
            detrend_options.extend([
                "GP (GPyTorch GPU) ❌ Not Implemented",
                "Hybrid GP (GPU) ❌ Not Implemented"
            ])
        
        # Advanced models (if sklearn available)
        if SKLEARN_GP_ENABLED:
            detrend_options.extend([
                "Advanced GP (2-Step)",
                "Multi-Kernel GP", 
                "Transit Window GP"
            ])
        
        # ariel_gp-style models
        detrend_options.extend([
            "AIRS Drift (CPU)",
            "FGS Drift (CPU)", 
            "Bayesian Multi-Component (CPU)"
        ])
        
        if GP_GPU_ENABLED:
            detrend_options.extend([
                "AIRS Drift (GPU)",
                "FGS Drift (GPU)",
                "Bayesian Multi-Component (GPU)"
            ])
        else:
            detrend_options.extend([
                "AIRS Drift (GPU) ❌ Not Implemented",
                "FGS Drift (GPU) ❌ Not Implemented", 
                "Bayesian Multi-Component (GPU) ❌ Not Implemented"
            ])
        
        self.detrend_model_combo.addItems(detrend_options)
        self.detrend_model_combo.currentTextChanged.connect(self.on_detrend_model_change)
        layout.addWidget(self.detrend_model_combo)

        # --- Stacked widget for model parameters ---
        self.detrend_params_stack = QStackedWidget()
        self.model_widget_map = {}

        # Polynomial parameters
        poly_widget = QWidget()
        poly_layout = QHBoxLayout(poly_widget)
        poly_layout.addWidget(QLabel("Degree:"))
        self.poly_degree_spinbox = QSpinBox()
        self.poly_degree_spinbox.setRange(1, 10)
        self.poly_degree_spinbox.setValue(2)
        poly_layout.addWidget(self.poly_degree_spinbox)
        self.model_widget_map["Polynomial"] = self.detrend_params_stack.addWidget(poly_widget)
        
        # Savitzky-Golay parameters
        savgol_widget = QWidget()
        savgol_layout = QHBoxLayout(savgol_widget)
        savgol_layout.addWidget(QLabel("Window:"))
        self.savgol_window_spinbox = QSpinBox()
        self.savgol_window_spinbox.setRange(3, 999)
        self.savgol_window_spinbox.setSingleStep(2)
        self.savgol_window_spinbox.setValue(51)
        savgol_layout.addWidget(self.savgol_window_spinbox)
        savgol_layout.addWidget(QLabel("Order:"))
        self.savgol_order_spinbox = QSpinBox()
        self.savgol_order_spinbox.setRange(1, 10)
        self.savgol_order_spinbox.setValue(2)
        savgol_layout.addWidget(self.savgol_order_spinbox)
        self.model_widget_map["Savitzky-Golay"] = self.detrend_params_stack.addWidget(savgol_widget)
        
        # George GP parameters
        george_widget = QWidget()
        george_layout = QHBoxLayout(george_widget)
        george_layout.addWidget(QLabel("Kernel:"))
        self.george_kernel_combo = QComboBox()
        self.george_kernel_combo.addItems(['Matern32'])
        george_layout.addWidget(self.george_kernel_combo)
        self.model_widget_map["GP (george CPU)"] = self.detrend_params_stack.addWidget(george_widget)
        
        # GPyTorch GP parameters
        if GP_GPU_ENABLED:
            gpytorch_widget = QWidget()
            gpytorch_layout = QHBoxLayout(gpytorch_widget)
            gpytorch_layout.addWidget(QLabel("Iterations:"))
            self.gpytorch_iter_spinbox = QSpinBox()
            self.gpytorch_iter_spinbox.setRange(10, 500)
            self.gpytorch_iter_spinbox.setValue(50)
            gpytorch_layout.addWidget(self.gpytorch_iter_spinbox)
            self.model_widget_map["GP (GPyTorch GPU)"] = self.detrend_params_stack.addWidget(gpytorch_widget)
        
        # Hybrid GP parameters
        hybrid_widget = QWidget()
        hybrid_layout = QVBoxLayout(hybrid_widget)
        hybrid_poly_layout = QHBoxLayout()
        hybrid_poly_layout.addWidget(QLabel("Residual Degree:"))
        self.hybrid_poly_degree_spinbox = QSpinBox()
        self.hybrid_poly_degree_spinbox.setRange(1, 5)
        self.hybrid_poly_degree_spinbox.setValue(2)
        hybrid_poly_layout.addWidget(self.hybrid_poly_degree_spinbox)
        hybrid_layout.addLayout(hybrid_poly_layout)
        hybrid_iter_layout = QHBoxLayout()
        hybrid_iter_layout.addWidget(QLabel("GP Iterations:"))
        self.hybrid_iter_spinbox = QSpinBox()
        self.hybrid_iter_spinbox.setRange(10, 500)
        self.hybrid_iter_spinbox.setValue(50)
        hybrid_iter_layout.addWidget(self.hybrid_iter_spinbox)
        hybrid_layout.addLayout(hybrid_iter_layout)
        self.model_widget_map["Hybrid GP (CPU)"] = self.detrend_params_stack.addWidget(hybrid_widget)
        if GP_GPU_ENABLED:
             self.model_widget_map["Hybrid GP (GPU)"] = self.detrend_params_stack.indexOf(hybrid_widget)
        
        # Advanced GP (2-Step) parameters
        if SKLEARN_GP_ENABLED:
            advanced_widget = QWidget()
            advanced_layout = QVBoxLayout(advanced_widget)
            
            # Drift kernel
            drift_kernel_layout = QHBoxLayout()
            drift_kernel_layout.addWidget(QLabel("Drift Kernel:"))
            self.advanced_drift_kernel_combo = QComboBox()
            self.advanced_drift_kernel_combo.addItems(['RBF', 'Matern32', 'Matern52'])
            drift_kernel_layout.addWidget(self.advanced_drift_kernel_combo)
            advanced_layout.addLayout(drift_kernel_layout)
            
            # Depth kernel
            depth_kernel_layout = QHBoxLayout()
            depth_kernel_layout.addWidget(QLabel("Depth Kernel:"))
            self.advanced_depth_kernel_combo = QComboBox()
            self.advanced_depth_kernel_combo.addItems(['RBF', 'Matern32', 'Matern52'])
            self.advanced_depth_kernel_combo.setCurrentText('Matern32')
            depth_kernel_layout.addWidget(self.advanced_depth_kernel_combo)
            advanced_layout.addLayout(depth_kernel_layout)
            
            # Noise level
            noise_layout = QHBoxLayout()
            noise_layout.addWidget(QLabel("Noise Level:"))
            self.advanced_noise_spinbox = QDoubleSpinBox()
            self.advanced_noise_spinbox.setRange(0.01, 1.0)
            self.advanced_noise_spinbox.setValue(0.1)
            self.advanced_noise_spinbox.setSingleStep(0.01)
            noise_layout.addWidget(self.advanced_noise_spinbox)
            advanced_layout.addLayout(noise_layout)
            
            self.model_widget_map["Advanced GP (2-Step)"] = self.detrend_params_stack.addWidget(advanced_widget)
            
            # Multi-Kernel GP parameters
            multikernel_widget = QWidget()
            multikernel_layout = QVBoxLayout(multikernel_widget)
            
            # Average kernel
            avg_kernel_layout = QHBoxLayout()
            avg_kernel_layout.addWidget(QLabel("Average Kernel:"))
            self.multikernel_avg_kernel_combo = QComboBox()
            self.multikernel_avg_kernel_combo.addItems(['RBF', 'Matern32', 'Matern52'])
            avg_kernel_layout.addWidget(self.multikernel_avg_kernel_combo)
            multikernel_layout.addLayout(avg_kernel_layout)
            
            # Spectral kernel
            spec_kernel_layout = QHBoxLayout()
            spec_kernel_layout.addWidget(QLabel("Spectral Kernel:"))
            self.multikernel_spec_kernel_combo = QComboBox()
            self.multikernel_spec_kernel_combo.addItems(['RBF', 'Matern32', 'Matern52'])
            self.multikernel_spec_kernel_combo.setCurrentText('Matern32')
            spec_kernel_layout.addWidget(self.multikernel_spec_kernel_combo)
            multikernel_layout.addLayout(spec_kernel_layout)
            
            # Sparse approximation
            sparse_layout = QHBoxLayout()
            self.multikernel_sparse_checkbox = QCheckBox("Use Sparse Approximation")
            self.multikernel_sparse_checkbox.setChecked(True)
            sparse_layout.addWidget(self.multikernel_sparse_checkbox)
            multikernel_layout.addLayout(sparse_layout)
            
            self.model_widget_map["Multi-Kernel GP"] = self.detrend_params_stack.addWidget(multikernel_widget)
            
            # Transit Window GP parameters
            transit_window_widget = QWidget()
            transit_window_layout = QVBoxLayout(transit_window_widget)
            
            # Transit kernel
            transit_kernel_layout = QHBoxLayout()
            transit_kernel_layout.addWidget(QLabel("Transit Kernel:"))
            self.transit_window_kernel_combo = QComboBox()
            self.transit_window_kernel_combo.addItems(['RBF', 'Matern32', 'Matern52'])
            self.transit_window_kernel_combo.setCurrentText('Matern32')
            transit_kernel_layout.addWidget(self.transit_window_kernel_combo)
            transit_window_layout.addLayout(transit_kernel_layout)
            
            # Drift kernel
            transit_drift_kernel_layout = QHBoxLayout()
            transit_drift_kernel_layout.addWidget(QLabel("Drift Kernel:"))
            self.transit_window_drift_kernel_combo = QComboBox()
            self.transit_window_drift_kernel_combo.addItems(['RBF', 'Matern32', 'Matern52'])
            transit_drift_kernel_layout.addWidget(self.transit_window_drift_kernel_combo)
            transit_window_layout.addLayout(transit_drift_kernel_layout)
            
            self.model_widget_map["Transit Window GP"] = self.detrend_params_stack.addWidget(transit_window_widget)
            
            # AIRS Drift parameters
            airs_widget = QWidget()
            airs_layout = QVBoxLayout(airs_widget)
            
            # Average drift parameters
            avg_layout = QHBoxLayout()
            avg_layout.addWidget(QLabel("Avg Kernel:"))
            self.airs_avg_kernel_combo = QComboBox()
            self.airs_avg_kernel_combo.addItems(['Matern32', 'RBF', 'Matern52'])
            avg_layout.addWidget(self.airs_avg_kernel_combo)
            avg_layout.addWidget(QLabel("Length Scale:"))
            self.airs_avg_length_spinbox = QDoubleSpinBox()
            self.airs_avg_length_spinbox.setRange(0.1, 10.0)
            self.airs_avg_length_spinbox.setValue(1.0)
            self.airs_avg_length_spinbox.setSingleStep(0.1)
            avg_layout.addWidget(self.airs_avg_length_spinbox)
            airs_layout.addLayout(avg_layout)
            
            # Spectral drift parameters
            spec_layout = QHBoxLayout()
            spec_layout.addWidget(QLabel("Spectral Kernel:"))
            self.airs_spec_kernel_combo = QComboBox()
            self.airs_spec_kernel_combo.addItems(['Matern32', 'RBF', 'Matern52'])
            spec_layout.addWidget(self.airs_spec_kernel_combo)
            spec_layout.addWidget(QLabel("Time Scale:"))
            self.airs_time_scale_spinbox = QDoubleSpinBox()
            self.airs_time_scale_spinbox.setRange(0.1, 10.0)
            self.airs_time_scale_spinbox.setValue(0.4)
            self.airs_time_scale_spinbox.setSingleStep(0.1)
            spec_layout.addWidget(self.airs_time_scale_spinbox)
            spec_layout.addWidget(QLabel("Wavelength Scale:"))
            self.airs_wl_scale_spinbox = QDoubleSpinBox()
            self.airs_wl_scale_spinbox.setRange(0.01, 1.0)
            self.airs_wl_scale_spinbox.setValue(0.05)
            self.airs_wl_scale_spinbox.setSingleStep(0.01)
            spec_layout.addWidget(self.airs_wl_scale_spinbox)
            airs_layout.addLayout(spec_layout)
            
            # Sparse approximation
            sparse_layout = QHBoxLayout()
            self.airs_sparse_checkbox = QCheckBox("Use Sparse Approximation")
            self.airs_sparse_checkbox.setChecked(True)
            sparse_layout.addWidget(self.airs_sparse_checkbox)
            airs_layout.addLayout(sparse_layout)
            
            self.model_widget_map["AIRS Drift (CPU)"] = self.detrend_params_stack.addWidget(airs_widget)
            if GP_GPU_ENABLED:
                self.model_widget_map["AIRS Drift (GPU)"] = self.detrend_params_stack.indexOf(airs_widget)
            
            # FGS Drift parameters
            fgs_widget = QWidget()
            fgs_layout = QHBoxLayout(fgs_widget)
            fgs_layout.addWidget(QLabel("Kernel:"))
            self.fgs_kernel_combo = QComboBox()
            self.fgs_kernel_combo.addItems(['Matern32', 'RBF', 'Matern52'])
            fgs_layout.addWidget(self.fgs_kernel_combo)
            fgs_layout.addWidget(QLabel("Length Scale:"))
            self.fgs_length_spinbox = QDoubleSpinBox()
            self.fgs_length_spinbox.setRange(0.1, 10.0)
            self.fgs_length_spinbox.setValue(1.0)
            self.fgs_length_spinbox.setSingleStep(0.1)
            fgs_layout.addWidget(self.fgs_length_spinbox)
            
            self.model_widget_map["FGS Drift (CPU)"] = self.detrend_params_stack.addWidget(fgs_widget)
            if GP_GPU_ENABLED:
                self.model_widget_map["FGS Drift (GPU)"] = self.detrend_params_stack.indexOf(fgs_widget)
            
            # Bayesian Multi-Component parameters
            bayesian_widget = QWidget()
            bayesian_layout = QVBoxLayout(bayesian_widget)
            
            # PCA components
            pca_layout = QHBoxLayout()
            pca_layout.addWidget(QLabel("PCA Components:"))
            self.bayesian_pca_spinbox = QSpinBox()
            self.bayesian_pca_spinbox.setRange(0, 5)
            self.bayesian_pca_spinbox.setValue(1)
            pca_layout.addWidget(self.bayesian_pca_spinbox)
            bayesian_layout.addLayout(pca_layout)
            
            # Iterations
            iter_layout = QHBoxLayout()
            iter_layout.addWidget(QLabel("Iterations:"))
            self.bayesian_iter_spinbox = QSpinBox()
            self.bayesian_iter_spinbox.setRange(1, 20)
            self.bayesian_iter_spinbox.setValue(7)
            iter_layout.addWidget(self.bayesian_iter_spinbox)
            bayesian_layout.addLayout(iter_layout)
            
            # Samples
            samples_layout = QHBoxLayout()
            samples_layout.addWidget(QLabel("Samples:"))
            self.bayesian_samples_spinbox = QSpinBox()
            self.bayesian_samples_spinbox.setRange(10, 1000)
            self.bayesian_samples_spinbox.setValue(100)
            samples_layout.addWidget(self.bayesian_samples_spinbox)
            bayesian_layout.addLayout(samples_layout)
            
            self.model_widget_map["Bayesian Multi-Component (CPU)"] = self.detrend_params_stack.addWidget(bayesian_widget)
            if GP_GPU_ENABLED:
                self.model_widget_map["Bayesian Multi-Component (GPU)"] = self.detrend_params_stack.indexOf(bayesian_widget)

        layout.addWidget(self.detrend_params_stack)
        detrend_group.setLayout(layout)
        return detrend_group

    def _create_analysis_group(self):
        """Creates a group box for analysis settings like masking and folding."""
        analysis_group = QGroupBox("Analysis Settings")
        layout = QVBoxLayout()

        mask_group = QGroupBox("Transit Mask Method")
        mask_layout = QVBoxLayout(mask_group)
        self.mask_method_combo = QComboBox()
        self.mask_method_combo.addItems(["empirical", "physical"])
        mask_layout.addWidget(self.mask_method_combo)
        layout.addWidget(mask_group)

        fold_group = QGroupBox("Phase Folding")
        fold_layout = QVBoxLayout(fold_group)
        fold_layout.addWidget(QLabel("Number of Bins:"))
        self.bins_spinbox = QSpinBox()
        self.bins_spinbox.setRange(10, 500)
        self.bins_spinbox.setValue(100)
        fold_layout.addWidget(self.bins_spinbox)
        layout.addWidget(fold_group)

        analysis_group.setLayout(layout)
        return analysis_group

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
            
            mask_method = self.mask_method_combo.currentText()
            detrender = self._get_detrender(self.detrend_model_combo.currentText())
            if detrender:
                self.observation.run_detrending(detrender, mask_method=mask_method)

            self.observation.run_phase_folding(n_bins=self.bins_spinbox.value())
        except (ValueError, RuntimeError) as e:
            self.statusBar().showMessage(f"ERROR: {e}", 15000)
            print(f"ERROR: {e}")
            return
        self._update_all_plots()
        self.statusBar().showMessage("Pipeline finished.", 5000)

    def _get_detrender(self, model_name):
        # Check for not implemented methods
        if "❌ Not Implemented" in model_name:
            self.statusBar().showMessage("This method is not implemented yet.", 5000)
            return None
            
        # Basic methods
        if model_name == "Polynomial": 
            return detrending.PolynomialDetrender(degree=self.poly_degree_spinbox.value())
        elif model_name == "Savitzky-Golay": 
            return detrending.SavGolDetrender(window_length=self.savgol_window_spinbox.value(), polyorder=self.savgol_order_spinbox.value())
        elif model_name == "GP (george CPU) ⚠️ Very Slow": 
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
        
        # Advanced sklearn methods
        elif model_name == "Advanced GP (2-Step)":
            if SKLEARN_GP_ENABLED: 
                self.statusBar().showMessage("Running Advanced GP Detrending (2-Step)...", 30000)
                QApplication.processEvents()
                return detrending.AdvancedGPDetrender(
                    drift_kernel=self.advanced_drift_kernel_combo.currentText(),
                    depth_kernel=self.advanced_depth_kernel_combo.currentText(),
                    noise_level=self.advanced_noise_spinbox.value()
                )
        elif model_name == "Multi-Kernel GP":
            if SKLEARN_GP_ENABLED:
                self.statusBar().showMessage("Running Multi-Kernel GP Detrending...", 30000)
                QApplication.processEvents()
                return detrending.MultiKernelDetrender(
                    average_kernel=self.multikernel_avg_kernel_combo.currentText(),
                    spectral_kernel=self.multikernel_spec_kernel_combo.currentText(),
                    use_sparse=self.multikernel_sparse_checkbox.isChecked()
                )
        elif model_name == "Transit Window GP":
            if SKLEARN_GP_ENABLED:
                self.statusBar().showMessage("Running Transit Window GP Detrending...", 30000)
                QApplication.processEvents()
                return detrending.TransitWindowDetrender(
                    transit_kernel=self.transit_window_kernel_combo.currentText(),
                    drift_kernel=self.transit_window_drift_kernel_combo.currentText()
                )
        
        # ariel_gp-style methods
        elif model_name == "AIRS Drift (CPU)":
            self.statusBar().showMessage("Running AIRS Drift Detrending (CPU)...", 30000)
            QApplication.processEvents()
            return detrending.AIRSDriftDetrender(
                use_gpu=False,
                avg_kernel=self.airs_avg_kernel_combo.currentText(),
                avg_length_scale=self.airs_avg_length_spinbox.value(),
                spectral_kernel=self.airs_spec_kernel_combo.currentText(),
                time_scale=self.airs_time_scale_spinbox.value(),
                wavelength_scale=self.airs_wl_scale_spinbox.value(),
                use_sparse=self.airs_sparse_checkbox.isChecked()
            )
        
        elif model_name == "AIRS Drift (GPU)":
            if GP_GPU_ENABLED:
                self.statusBar().showMessage("Running AIRS Drift Detrending (GPU)...", 30000)
                QApplication.processEvents()
                return detrending.AIRSDriftDetrender(
                    use_gpu=True,
                    avg_kernel=self.airs_avg_kernel_combo.currentText(),
                    avg_length_scale=self.airs_avg_length_spinbox.value(),
                    spectral_kernel=self.airs_spec_kernel_combo.currentText(),
                    time_scale=self.airs_time_scale_spinbox.value(),
                    wavelength_scale=self.airs_wl_scale_spinbox.value(),
                    use_sparse=self.airs_sparse_checkbox.isChecked()
                )
        
        elif model_name == "FGS Drift (CPU)":
            self.statusBar().showMessage("Running FGS Drift Detrending (CPU)...", 30000)
            QApplication.processEvents()
            return detrending.FGSDriftDetrender(
                use_gpu=False,
                kernel=self.fgs_kernel_combo.currentText(),
                length_scale=self.fgs_length_spinbox.value()
            )
        
        elif model_name == "FGS Drift (GPU)":
            if GP_GPU_ENABLED:
                self.statusBar().showMessage("Running FGS Drift Detrending (GPU)...", 30000)
                QApplication.processEvents()
                return detrending.FGSDriftDetrender(
                    use_gpu=True,
                    kernel=self.fgs_kernel_combo.currentText(),
                    length_scale=self.fgs_length_spinbox.value()
                )
        
        elif model_name == "Bayesian Multi-Component (CPU)":
            self.statusBar().showMessage("Running Bayesian Multi-Component Detrending (CPU)...", 60000)
            QApplication.processEvents()
            return detrending.BayesianMultiComponentDetrender(
                use_gpu=False,
                n_pca=self.bayesian_pca_spinbox.value(),
                n_iter=self.bayesian_iter_spinbox.value(),
                n_samples=self.bayesian_samples_spinbox.value()
            )
        
        elif model_name == "Bayesian Multi-Component (GPU)":
            if GP_GPU_ENABLED:
                self.statusBar().showMessage("Running Bayesian Multi-Component Detrending (GPU)...", 60000)
                QApplication.processEvents()
                return detrending.BayesianMultiComponentDetrender(
                    use_gpu=True,
                    n_pca=self.bayesian_pca_spinbox.value(),
                    n_iter=self.bayesian_iter_spinbox.value(),
                    n_samples=self.bayesian_samples_spinbox.value()
                )
        
        return None

    # --- Plotting and UI Updates ---

    def _update_all_plots(self):
        self.log_display.setText("\n".join(self.observation.calibration_log))
        self.update_image_plot()
        self.update_light_curve_plots()
    def update_image_plot(self):
        if not (self.observation and self.observation.is_loaded): return
        if self.image_cbar: self.image_cbar.remove(); self.image_cbar = None
        self.image_ax.clear()
        processed_signal = self.observation.get_data(return_type='numpy'); frame_idx = self.frame_slider.value()
        if self.calib_checkboxes["cds"].isChecked(): frame_idx //= 2
        if frame_idx >= processed_signal.shape[0]: frame_idx = 0
        img_data = processed_signal[frame_idx]; im = self.image_ax.imshow(img_data, aspect='auto', cmap='viridis', origin='lower'); self.image_cbar = self.image_canvas.figure.colorbar(im, ax=self.image_ax); self.image_ax.set_title(f"Detector Frame (Index: {frame_idx})"); self.image_canvas.draw()
    def update_light_curve_plots(self):
        if not (self.observation and self.observation.is_loaded): return
        self.update_photometry_plot(); self.update_detrended_plot(); self.update_phase_folded_plot()
    def update_photometry_plot(self):
        self.single_lc_ax.clear()
        if self.observation.light_curves is None: self.single_lc_canvas.draw(); return
        light_curves = self.observation.get_light_curves(return_type='numpy'); wavelength_col = self.wavelength_slider.value()
        if wavelength_col >= light_curves.shape[1]: wavelength_col = 0
        self.single_lc_ax.plot(self.observation.get_time_array(), light_curves[:, wavelength_col], '.-', alpha=0.8, color='dodgerblue'); self.single_lc_ax.set_title(f"Raw Light Curve (Wavelength: {wavelength_col})"); self.single_lc_ax.set_xlabel("Time (days)"); self.single_lc_ax.set_ylabel("Flux"); self.single_lc_ax.grid(True, alpha=0.3); self.single_lc_canvas.draw()
    def update_detrended_plot(self):
        self.detrended_lc_ax.clear()
        if self.observation.detrended_light_curves is None: self.detrended_lc_canvas.draw(); return
        detrended_lcs = self.observation.get_detrended_light_curves(return_type='numpy'); original_lcs = self.observation.get_light_curves(return_type='numpy'); noise_models = self.observation.get_noise_models(return_type='numpy'); wavelength_col = self.wavelength_slider.value()
        if wavelength_col >= detrended_lcs.shape[1]: wavelength_col = 0
        time_arr = self.observation.get_time_array()
        if self.zoom_checkbox.isChecked(): self.detrended_lc_ax.plot(time_arr, detrended_lcs[:, wavelength_col], '.', color='black')
        else: self.detrended_lc_ax.plot(time_arr, original_lcs[:, wavelength_col], '.', color='grey', alpha=0.2, label='Original'); self.detrended_lc_ax.plot(time_arr, noise_models[:, wavelength_col], '-', color='red', label='Noise Model'); self.detrended_lc_ax.plot(time_arr, detrended_lcs[:, wavelength_col], '.', color='black', alpha=0.8, label='Detrended'); self.detrended_lc_ax.legend()
        self.detrended_lc_ax.set_title(f"Detrended Light Curve (Wavelength: {wavelength_col})"); self.detrended_lc_ax.set_xlabel("Time (days)"); self.detrended_lc_ax.set_ylabel("Normalized Flux"); self.detrended_lc_ax.grid(True, alpha=0.3); self.detrended_lc_canvas.draw()
    def update_phase_folded_plot(self):
        self.phase_folded_ax.clear()
        if self.observation.phase_folded_lc is None: self.phase_folded_canvas.draw(); return
        folded_data = self.observation.get_phase_folded_lc(); wavelength_col = self.wavelength_slider.value()
        if wavelength_col >= len(folded_data): wavelength_col = 0
        bin_centers, binned_flux, binned_error = folded_data[wavelength_col]; self.phase_folded_ax.errorbar(bin_centers, binned_flux, yerr=binned_error, fmt='o', color='black', ecolor='gray', elinewidth=1, capsize=2); self.phase_folded_ax.axhline(1.0, color='r', linestyle='--', alpha=0.7); self.phase_folded_ax.set_title(f"Phase-Folded Transit (Wavelength: {wavelength_col})"); self.phase_folded_ax.set_xlabel("Orbital Phase"); self.phase_folded_ax.set_ylabel("Normalized Flux"); self.phase_folded_ax.grid(True, alpha=0.3); self.phase_folded_canvas.draw()
    def load_data(self):
        planet_id = self.planet_combo.currentText(); obs_id = self.obs_combo.currentText()
        if not all([planet_id, obs_id]): self.statusBar().showMessage("Error: Missing Planet or Observation ID.", 5000); return
        try:
            self.statusBar().showMessage(f"Loading data for {planet_id}..."); QApplication.processEvents()
            self.observation = DataObservation(planet_id, self.instrument_combo.currentText(), int(obs_id), self.split_combo.currentText()); self.observation.load(backend=self.backend_combo.currentText())
            if self.star_info_df is not None and int(planet_id) in self.star_info_df.index: self.info_display.setText(self.star_info_df.loc[int(planet_id)].to_string())
            is_fgs = (self.instrument_combo.currentText() == 'FGS1'); self.wavelength_slider.setEnabled(not is_fgs); self.wavelength_spinbox.setEnabled(not is_fgs)
            max_wl = self.observation.raw_signal.shape[2] - 1
            self.frame_slider.setRange(0, self.observation.raw_signal.shape[0] - 1); self.frame_spinbox.setRange(0, self.observation.raw_signal.shape[0] - 1)
            self.wavelength_slider.setRange(0, max_wl); self.wavelength_spinbox.setRange(0, max_wl)
            self.run_pipeline_and_update_plots()
        except Exception as e: self.statusBar().showMessage(f"Error loading data: {e}", 15000); print(f"ERROR: {e}")
    def populate_planet_ids(self):
        current_planet = self.planet_combo.currentText(); self.planet_combo.clear(); split = self.split_combo.currentText(); self.star_info_df = loaders.load_star_info(split)
        if self.star_info_df is None: return
        path = DATASET_DIR / split
        if path.exists():
            planet_ids = sorted([p.name for p in path.iterdir() if p.is_dir()]); self.planet_combo.addItems(planet_ids)
            if current_planet in planet_ids: self.planet_combo.setCurrentText(current_planet)
        self.populate_obs_ids()
    def populate_obs_ids(self):
        self.obs_combo.clear(); planet_id = self.planet_combo.currentText()
        if not planet_id: return
        planet_dir = DATASET_DIR / self.split_combo.currentText() / planet_id
        if not planet_dir.exists(): return
        obs_ids = sorted([f.stem.split('_')[-1] for f in planet_dir.glob(f"{self.instrument_combo.currentText()}_signal_*.parquet") if f.stem.split('_')[-1].isdigit()], key=int)
        if obs_ids: self.obs_combo.addItems(obs_ids)
    def on_detrend_model_change(self, model_name):
        index = self.model_widget_map.get(model_name, 0)
        self.detrend_params_stack.setCurrentIndex(index)
    def on_mouse_move(self, event):
        if not (event.inaxes and event.inaxes == self.image_ax and self.observation and self.observation.processed_signal is not None):
            return
            
        x, y = int(event.xdata or 0), int(event.ydata or 0)
        data = self.observation.get_data('numpy')
        
        # Check if the mouse cursor is within the valid bounds of the image data array.
        # This prevents IndexError if the mouse moves off the edge of the plot.
        if 0 <= y < data.shape[1] and 0 <= x < data.shape[2]:
            # Get the value of the pixel under the cursor for the current frame.
            pixel_value = data[self.frame_slider.value(), y, x]
            
            # Display the coordinates and value in the status bar.
            self.statusBar().showMessage(f"Pixel (x={x}, y={y}) | Value: {pixel_value:.2f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    inspector = DataInspector()
    inspector.show()
    sys.exit(app.exec())