#!/usr/bin/env python3
"""
gpp_gui.py
-----------
A premium, modern desktop GUI dashboard built with PyQt6 for running SCOPE GPP model inference.
Allows users to load datasets, select models, predict GPP values, save predictions to a CSV, 
and view/save a landcover-based performance scatterplot.
"""

import os
import sys
import subprocess

# Auto-install missing dependencies
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "joblib": "joblib",
    "PyQt6": "PyQt6",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "scikit-learn": "sklearn",
    "tabpfn": "tabpfn",
    "torch": "torch"
}

for package, import_name in REQUIRED_PACKAGES.items():
    try:
        __import__(import_name)
    except ImportError:
        print(f"📦 Installing missing dependency: {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            __import__(import_name)
            print(f"✅ Successfully installed and imported: {package}")
        except Exception as e:
            print(f"❌ Failed to install {package}: {str(e)}")

import math
import joblib
import traceback
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QTextEdit, QFrame, QSplitter, QProgressBar, QMessageBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QIcon, QColor, QPalette

import matplotlib
matplotlib.use("QtAgg")  # Ensure we use Qt backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns

# Ensure TabPfn is in import search path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# Dynamically inject the unified_model module to allow deserializing the joblib
# model without requiring an external unified_model.py script.
import sys
import types
from sklearn.base import BaseEstimator, RegressorMixin

if 'unified_model' not in sys.modules:
    unified_model_module = types.ModuleType('unified_model')
    
    class UnifiedGPPRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, pls_model, tabpfn_model, rf_model, refl_cols, met_cols=None):
            self.pls_model = pls_model
            self.tabpfn_model = tabpfn_model
            self.rf_model = rf_model
            self.refl_cols = refl_cols
            self.met_cols = met_cols if met_cols is not None else ['Ta', 'Rin', 'ea', 'SMC']

        def predict(self, X):
            if isinstance(X, pd.DataFrame):
                X_met = X[self.met_cols].values
                X_spectral = X[self.refl_cols].values
            else:
                X = np.asarray(X)
                n_met = len(self.met_cols)
                X_met = X[:, :n_met]
                X_spectral = X[:, n_met:]

            X_pls = self.pls_model.transform(X_spectral)
            X_tabpfn = np.hstack([X_met, X_pls]).astype(np.float32)
            tabpfn_pred = self.tabpfn_model.predict(X_tabpfn)
            X_rf = np.hstack([X_met, X_spectral, tabpfn_pred[:, np.newaxis]]).astype(np.float32)
            final_pred = self.rf_model.predict(X_rf)
            return final_pred

    UnifiedGPPRegressor.__module__ = 'unified_model'
    unified_model_module.UnifiedGPPRegressor = UnifiedGPPRegressor
    sys.modules['unified_model'] = unified_model_module
else:
    from unified_model import UnifiedGPPRegressor


def pearson_r(y_t, y_p):
    y_t = np.asarray(y_t)
    y_p = np.asarray(y_p)
    if len(y_t) < 2 or np.std(y_t) == 0 or np.std(y_p) == 0:
        return 0.0
    return np.corrcoef(y_t, y_p)[0, 1]


class InferenceWorker(QThread):
    """
    Worker thread that runs GPP inference in the background to keep the GUI fully responsive.
    """
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict, pd.DataFrame)
    error_signal = pyqtSignal(str)

    def __init__(self, model_path, dataset_path, output_csv_path):
        super().__init__()
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_csv_path = output_csv_path

    def run(self):
        try:
            self.log_signal.emit("🔄 Starting GPP Inference Pipeline...")
            self.progress_signal.emit(10)

            # 1. Load model
            self.log_signal.emit(f"📖 Loading Unified GPP Model from:\n   {self.model_path}")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")
            
            # Ensure TabPfn directory is available for deserialization
            sys.path.append(os.path.dirname(self.model_path))
            model = joblib.load(self.model_path)
            self.log_signal.emit("✅ Model loaded successfully.")
            self.progress_signal.emit(30)

            # 2. Load dataset
            self.log_signal.emit(f"📖 Loading Dataset from:\n   {self.dataset_path}")
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Dataset CSV file not found at: {self.dataset_path}")
            df = pd.read_csv(self.dataset_path)
            self.log_signal.emit(f"✅ Loaded dataset containing {len(df)} samples.")
            self.progress_signal.emit(50)

            # 3. Validation
            met_cols = getattr(model, "met_cols", ["Ta", "Rin", "ea", "SMC"])
            refl_cols = getattr(model, "refl_cols", None)
            if refl_cols is None:
                raise AttributeError("The loaded model does not contain standard spectral features definition.")

            self.log_signal.emit("🔍 Validating features in dataset...")
            missing_met = [col for col in met_cols if col not in df.columns]
            missing_refl = [col for col in refl_cols if col not in df.columns]

            if missing_met:
                raise ValueError(f"Missing meteorological columns in dataset: {missing_met}")
            if missing_refl:
                self.log_signal.emit(f"⚠️ Warning: Some reflectance columns missing. Trying to align features...")
                # Try finding closest match or throw error if completely mismatched
                if len(missing_refl) > 0.8 * len(refl_cols):
                    raise ValueError("Most reflectance columns are missing in dataset.")

            self.log_signal.emit("✅ Feature validation passed.")
            self.progress_signal.emit(65)

            # 4. Predict
            self.log_signal.emit("⚡ Running model prediction")
            pred = model.predict(df)
            self.log_signal.emit("✅ Prediction complete.")
            self.progress_signal.emit(85)

            # 5. Build results DataFrame
            # Look for GPP (observed) column
            obs_col = None
            for candidate in ["GPP", "GPP_observed", "GPP_observed_gpp"]:
                if candidate in df.columns:
                    obs_col = candidate
                    break
            
            if obs_col:
                obs = df[obs_col].values
                self.log_signal.emit(f"📈 Found observed GPP column '{obs_col}' for validation plot.")
            else:
                obs = np.full(len(pred), np.nan)
                self.log_signal.emit("⚠️ No observed GPP column found in dataset. Scatterplot metrics will be unavailable.")

            # Look for LandCover column
            lc_col = None
            for candidate in ["LandCover", "landcover", "Landcover", "LC"]:
                if candidate in df.columns:
                    lc_col = candidate
                    break

            if lc_col:
                lc = df[lc_col].values
            else:
                lc = np.full(len(pred), "Unknown")
                self.log_signal.emit("⚠️ No LandCover column found in dataset. Categorical scatter colors will be uniform.")

            results_df = pd.DataFrame({
                'GPP_observed': obs,
                'GPP_estimate': pred,
                'LandCover': lc
            })

            # Save results to CSV
            self.log_signal.emit(f"💾 Saving results to CSV:\n   {self.output_csv_path}")
            os.makedirs(os.path.dirname(os.path.abspath(self.output_csv_path)), exist_ok=True)
            results_df.to_csv(self.output_csv_path, index=False)
            self.log_signal.emit("✅ Results saved successfully.")

            # Compute overall stats if observed GPP is available
            metrics = {
                'samples': len(pred),
                'r': 0.0,
                'r2': 0.0,
                'rmse': 0.0,
                'rrmse': 0.0
            }

            valid_mask = ~np.isnan(obs)
            if np.sum(valid_mask) > 1:
                y_valid = obs[valid_mask]
                pred_valid = pred[valid_mask]
                r_val = pearson_r(y_valid, pred_valid)
                r2_val = r_val ** 2
                rmse_val = math.sqrt(np.mean((y_valid - pred_valid) ** 2))
                mean_y = np.mean(y_valid)
                rrmse_val = (rmse_val / mean_y * 100) if mean_y != 0 else 0.0

                metrics.update({
                    'r': r_val,
                    'r2': r2_val,
                    'rmse': rmse_val,
                    'rrmse': rrmse_val
                })

            self.progress_signal.emit(100)
            self.finished_signal.emit(metrics, results_df)

        except Exception as e:
            tb = traceback.format_exc()
            self.log_signal.emit(f"❌ Error occurred: {str(e)}")
            self.error_signal.emit(f"{str(e)}\n\n{tb}")


class GPPInferenceApp(QMainWindow):
    """
    Main Window of the Dashboard.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unified GPP Model Inference Dashboard")
        self.resize(1100, 750)
        self.init_ui()

    def init_ui(self):
        # Apply premium QSS style sheet
        self.setStyleSheet("""
            QWidget {
                background-color: #111827;
                color: #e5e7eb;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QLabel {
                font-size: 13px;
                color: #9ca3af;
            }
            QLineEdit {
                background-color: #1f2937;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 6px;
                color: #f9fafb;
            }
            QLineEdit:focus {
                border: 1px solid #6366f1;
            }
            QPushButton {
                background-color: #374151;
                color: #f9fafb;
                border: 1px solid #4b5563;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4b5563;
            }
            QPushButton:pressed {
                background-color: #1f2937;
            }
            QGroupBox {
                border: 1px solid #374151;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                color: #818cf8;
            }
        """)

        # Main Layout split into two panels: Controls (left) & Canvas plot (right)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_splitter)

        # Left Widget (Controls, logs, metrics)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(12)

        # ── Dashboard Title ────────────────────────────────────────────────
        title_label = QLabel("GPP Inference")
        title_label.setObjectName("TitleLabel")
        title_label.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: #ffffff;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #7c3aed);
            padding: 12px;
            border-radius: 8px;
        """)
        left_layout.addWidget(title_label)

        # ── TabPFN Attribution (License Requirement) ──────────────────
        attribution_label = QLabel("Built with PriorLabs-TabPFN")
        attribution_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        attribution_label.setStyleSheet("""
            font-size: 11px;
            font-weight: bold;
            color: #818cf8;
            margin-top: -6px;
            margin-bottom: 2px;
            padding-right: 4px;
        """)
        left_layout.addWidget(attribution_label)

        # ── Configuration Card (Inputs/Outputs Selection) ──────────────────
        config_frame = QFrame()
        config_frame.setStyleSheet("""
            background-color: #1f2937;
            border: 1px solid #374151;
            border-radius: 8px;
        """)
        config_layout = QGridLayout(config_frame)
        config_layout.setContentsMargins(15, 15, 15, 15)
        config_layout.setSpacing(8)

        # Paths
        default_model = os.path.join(SCRIPT_DIR, "TabPFN_GPP.joblib")
        default_dataset = os.path.join(SCRIPT_DIR, "example_inference.csv")
        default_csv_out = os.path.join(SCRIPT_DIR, "gpp_inference_results.csv")
        default_plot_out = os.path.join(SCRIPT_DIR, "gpp_inference_scatterplot.png")

        # Row 1: Model Selection
        config_layout.addWidget(QLabel("Unified Model (.joblib)"), 0, 0)
        self.model_input = QLineEdit(default_model)
        config_layout.addWidget(self.model_input, 0, 1)
        btn_model = QPushButton("Browse")
        btn_model.clicked.connect(self.browse_model)
        config_layout.addWidget(btn_model, 0, 2)

        # Row 2: Dataset Selection
        config_layout.addWidget(QLabel("Dataset File (.csv)"), 1, 0)
        self.dataset_input = QLineEdit(default_dataset)
        config_layout.addWidget(self.dataset_input, 1, 1)
        btn_dataset = QPushButton("Browse")
        btn_dataset.clicked.connect(self.browse_dataset)
        config_layout.addWidget(btn_dataset, 1, 2)

        # Row 3: Output Results CSV File Selection
        config_layout.addWidget(QLabel("Save Results CSV To"), 2, 0)
        self.output_csv_input = QLineEdit(default_csv_out)
        config_layout.addWidget(self.output_csv_input, 2, 1)
        btn_csv = QPushButton("Browse")
        btn_csv.clicked.connect(self.browse_output_csv)
        config_layout.addWidget(btn_csv, 2, 2)

        # Row 4: Output Plot File Selection
        config_layout.addWidget(QLabel("Save Scatterplot To"), 3, 0)
        self.output_plot_input = QLineEdit(default_plot_out)
        config_layout.addWidget(self.output_plot_input, 3, 1)
        btn_plot = QPushButton("Browse")
        btn_plot.clicked.connect(self.browse_output_plot)
        config_layout.addWidget(btn_plot, 3, 2)

        left_layout.addWidget(config_frame)

        # ── Statistics Metrics Panel ─────────────────────────────────────
        self.metrics_frame = QFrame()
        self.metrics_frame.setStyleSheet("""
            background-color: #111827;
            border: 1px solid #374151;
            border-radius: 8px;
            padding: 8px;
        """)
        metrics_layout = QGridLayout(self.metrics_frame)
        metrics_layout.setContentsMargins(10, 10, 10, 10)
        metrics_layout.setSpacing(10)

        # Helper method to create sub-cards for metrics
        def create_metric_card(label_txt):
            card = QFrame()
            card.setStyleSheet("""
                background-color: #1f2937;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 6px;
            """)
            layout = QVBoxLayout(card)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(2)
            
            val_lbl = QLabel("—")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #10b981;")
            
            lbl = QLabel(label_txt)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-size: 10px; color: #9ca3af; font-weight: 500;")
            
            layout.addWidget(val_lbl)
            layout.addWidget(lbl)
            return card, val_lbl

        self.card_samples, self.val_samples = create_metric_card("Samples")
        self.card_r, self.val_r = create_metric_card("Pearson r")
        self.card_r2, self.val_r2 = create_metric_card("R² Score")
        self.card_rmse, self.val_rmse = create_metric_card("RMSE")

        metrics_layout.addWidget(self.card_samples, 0, 0)
        metrics_layout.addWidget(self.card_r, 0, 1)
        metrics_layout.addWidget(self.card_r2, 0, 2)
        metrics_layout.addWidget(self.card_rmse, 0, 3)

        left_layout.addWidget(self.metrics_frame)

        # ── Pipeline Control Buttons & Progress ────────────────────────────
        self.run_button = QPushButton("🚀 Run GPP Inference Pipeline")
        self.run_button.setObjectName("RunButton")
        self.run_button.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #7c3aed);
            color: #ffffff;
            font-size: 15px;
            font-weight: bold;
            border: none;
            padding: 12px;
            border-radius: 8px;
        """)
        self.run_button.clicked.connect(self.start_inference)
        left_layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1f2937;
                border: 1px solid #374151;
                border-radius: 6px;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #6366f1;
                border-radius: 5px;
            }
        """)
        left_layout.addWidget(self.progress_bar)

        # ── Logger Output console ──────────────────────────────────────────
        left_layout.addWidget(QLabel("System Log & Status"))
        self.console_log = QTextEdit()
        self.console_log.setReadOnly(True)
        self.console_log.setObjectName("LogConsole")
        self.console_log.setStyleSheet("""
            background-color: #030712;
            border: 1px solid #1f2937;
            border-radius: 8px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 12px;
            color: #38bdf8;
            padding: 8px;
        """)
        left_layout.addWidget(self.console_log)

        # Add left widget to the splitter
        main_splitter.addWidget(left_widget)

        # ── Right Widget (Embedded matplotlib plot canvas) ────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(15, 15, 15, 15)

        title_preview = QLabel("Live Scatterplot Preview")
        title_preview.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
        right_layout.addWidget(title_preview)

        # Set up matplotlib figure and canvas
        self.fig = Figure(figsize=(7, 7), facecolor='#111827')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #111827; border: 1px solid #374151; border-radius: 8px;")
        right_layout.addWidget(self.canvas)

        # Initial placeholder plot
        self.draw_placeholder()

        main_splitter.addWidget(right_widget)

        # Adjust initial sizes: left panel 45%, right panel 55%
        main_splitter.setSizes([450, 550])

        self.log_message("💡 Dashboard initialized. Ready to predict GPP.")

    # ── Browse File Callbacks ──────────────────────────────────────────
    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Unified Model (.joblib)", "", "Joblib Files (*.joblib);;All Files (*)")
        if file_path:
            self.model_input.setText(file_path)

    def browse_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Dataset CSV", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.dataset_input.setText(file_path)

    def browse_output_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV As", "", "CSV Files (*.csv);;All Files (*)")
        if file_path:
            self.output_csv_input.setText(file_path)

    def browse_output_plot(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Scatterplot As", "", "PNG Images (*.png);;All Files (*)")
        if file_path:
            self.output_plot_input.setText(file_path)

    def log_message(self, message):
        self.console_log.append(message)
        self.console_log.ensureCursorVisible()

    # ── Draw Canvas Plots ──────────────────────────────────────────────
    def draw_placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#111827')
        ax.spines['bottom'].set_color('#4b5563')
        ax.spines['top'].set_color('#4b5563')
        ax.spines['left'].set_color('#4b5563')
        ax.spines['right'].set_color('#4b5563')
        ax.tick_params(colors='#9ca3af')
        ax.grid(True, color='#1f2937', linestyle='--')
        
        ax.text(0.5, 0.5, "Inference plot will be rendered here\nafter running predictions", 
                color='#9ca3af', ha='center', va='center', fontsize=12)
        ax.set_title("Validation Plot Preview", color='#ffffff', fontsize=14, pad=15)
        self.canvas.draw()

    def make_gpp_plot(self, df):
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.set_facecolor('#111827')
            self.fig.patch.set_facecolor('#111827')

            y = df['GPP_observed'].values
            y_pred = df['GPP_estimate'].values

            # Verify if observed GPP is available
            has_obs = not np.isnan(y).all()

            if not has_obs:
                # No observed values, plot estimates sequence
                ax.plot(y_pred, label="Predicted GPP", color="#3b82f6", lw=2)
                ax.set_xlabel("Sample index", color="#9ca3af", fontsize=11)
                ax.set_ylabel("Estimated GPP", color="#9ca3af", fontsize=11)
                ax.set_title("Estimated GPP Series", color="#ffffff", fontsize=14, pad=15)
                ax.legend(facecolor='#1f2937', edgecolor='#374151', labelcolor='#ffffff')
                ax.tick_params(colors='#9ca3af')
                ax.spines['bottom'].set_color('#4b5563')
                ax.spines['left'].set_color('#4b5563')
                ax.spines['top'].set_color('#111827')
                ax.spines['right'].set_color('#111827')
                ax.grid(True, color='#1f2937', linestyle='--')
                self.canvas.draw()

                # Save plot to output path
                plot_path = self.output_plot_input.text()
                self.fig.savefig(plot_path, dpi=300, facecolor='#111827')
                return

            # Apply layout formatting
            unique_covers = df['LandCover'].dropna().unique()
            lc_r_dict = {}
            for cover in unique_covers:
                cover_df = df[df['LandCover'] == cover]
                if len(cover_df) > 1 and np.std(cover_df['GPP_observed']) > 0 and np.std(cover_df['GPP_estimate']) > 0:
                    lc_r_dict[cover] = pearson_r(cover_df['GPP_observed'].values, cover_df['GPP_estimate'].values)
                else:
                    lc_r_dict[cover] = 0.0

            df_plot = df.copy()
            df_plot['LandCover_Legend'] = df_plot['LandCover'].map(
                lambda x: f"{x} (r={lc_r_dict[x]:.2f}, R²={lc_r_dict[x]**2:.2f})" if x in lc_r_dict else str(x)
            )

            # Define custom ordering for landcover legend: Cropland first, Forest second, then others
            ordered_covers = []
            preferred_order = ['Cropland', 'Forest', 'Grassland', 'Wetland']
            for cover in preferred_order:
                if cover in lc_r_dict:
                    ordered_covers.append(cover)
            for cover in lc_r_dict:
                if cover not in ordered_covers:
                    ordered_covers.append(cover)

            hue_order = [
                f"{cover} (r={lc_r_dict[cover]:.2f}, R²={lc_r_dict[cover]**2:.2f})" if cover in lc_r_dict else str(cover)
                for cover in ordered_covers
            ]

            # Scatterplot with Seaborn
            sns.scatterplot(
                data=df_plot,
                x='GPP_observed',
                y='GPP_estimate',
                hue='LandCover_Legend',
                hue_order=hue_order,
                palette='Set1',
                s=40,
                alpha=0.8,
                ax=ax
            )

            # Draw 1:1 dashed line
            LOWER_LIMIT = min(y.min(), y_pred.min())
            UPPER_LIMIT = max(y.max(), y_pred.max())
            margin = (UPPER_LIMIT - LOWER_LIMIT) * 0.05
            LOWER_LIMIT -= margin
            UPPER_LIMIT += margin

            ax.plot([LOWER_LIMIT, UPPER_LIMIT], [LOWER_LIMIT, UPPER_LIMIT], 'r--', lw=2, label='1:1 line')

            # Regression line
            sns.regplot(
                x=df['GPP_observed'],
                y=df['GPP_estimate'],
                scatter=False,
                color='#10b981',
                line_kws={'lw': 2.5, 'label': 'Model Fit'},
                ci=None,
                ax=ax
            )

            # Overall metrics calculations
            r_val = pearson_r(y, y_pred)
            r2 = r_val ** 2
            rmse = math.sqrt(np.mean((y - y_pred) ** 2))
            mean_y = np.mean(y)
            rrmse = (rmse / mean_y * 100) if mean_y != 0 else 0.0

            textstr = f'$r = {r_val:.2f}$\n$R^2 = {r2:.2f}$\n$RMSE = {rmse:.2f}$\n$rRMSE = {rrmse:.1f}\\%$'

            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11, color='#ffffff',
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#1f2937', edgecolor='#374151', alpha=0.95))

            # Configure design system styling
            ax.set_xlabel(r'Observed GPP [$\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$]', fontsize=11, color='#e5e7eb', fontweight='bold')
            ax.set_ylabel(r'Predicted GPP [$\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$]', fontsize=11, color='#e5e7eb', fontweight='bold')
            
            legend = ax.legend(loc='lower right', fontsize=9, frameon=True, shadow=True, facecolor='#1f2937', edgecolor='#374151')
            for text in legend.get_texts():
                text.set_color('#ffffff')
            
            ax.grid(True, color='#1f2937', linestyle='--')
            ax.set_xlim(LOWER_LIMIT, UPPER_LIMIT)
            ax.set_ylim(LOWER_LIMIT, UPPER_LIMIT)
            ax.tick_params(colors='#9ca3af')
            ax.set_aspect('equal', adjustable='box')

            # Border coloring
            ax.spines['bottom'].set_color('#4b5563')
            ax.spines['left'].set_color('#4b5563')
            ax.spines['top'].set_color('#4b5563')
            ax.spines['right'].set_color('#4b5563')

            self.fig.tight_layout()
            self.canvas.draw()

            # Save plot to output path
            plot_path = self.output_plot_input.text()
            self.fig.savefig(plot_path, dpi=300, facecolor='#111827')
            self.log_message(f"✅ Saved Scatterplot visualization to:\n   {plot_path}")

        except Exception as e:
            self.log_message(f"⚠️ Error creating plot display: {str(e)}")

    # ── Inference Execution Control ──────────────────────────────────
    def start_inference(self):
        # Prevent double trigger
        self.run_button.setEnabled(False)
        self.run_button.setText("⏳ Processing GPP Inference Pipeline...")
        self.progress_bar.setValue(5)

        model_path = self.model_input.text().strip()
        dataset_path = self.dataset_input.text().strip()
        output_csv_path = self.output_csv_input.text().strip()

        # Start thread
        self.worker = InferenceWorker(model_path, dataset_path, output_csv_path)
        self.worker.log_signal.connect(self.log_message)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self.on_inference_finished)
        self.worker.error_signal.connect(self.on_inference_error)
        self.worker.start()

    def on_inference_finished(self, metrics, results_df):
        # Update metrics indicators
        self.val_samples.setText(str(metrics['samples']))
        
        if metrics['r'] != 0.0 or metrics['r2'] != 0.0:
            self.val_r.setText(f"{metrics['r']:.3f}")
            self.val_r2.setText(f"{metrics['r2']:.3f}")
            self.val_rmse.setText(f"{metrics['rmse']:.3f}")
        else:
            self.val_r.setText("—")
            self.val_r2.setText("—")
            self.val_rmse.setText("—")

        # Update plotting
        self.make_gpp_plot(results_df)

        self.run_button.setEnabled(True)
        self.run_button.setText("🚀 Run GPP Inference Pipeline")
        self.log_message("🎉 Process execution fully complete!")
        
        # Display completion notice
        QMessageBox.information(
            self,
            "Success",
            f"GPP pipeline finished successfully!\n\nCSV file saved:\n{self.output_csv_input.text()}\n\nScatterplot saved:\n{self.output_plot_input.text()}"
        )

    def on_inference_error(self, error_details):
        self.run_button.setEnabled(True)
        self.run_button.setText("🚀 Run GPP Inference Pipeline")
        self.progress_bar.setValue(0)

        # Show warning/error messagebox
        QMessageBox.critical(
            self,
            "Inference Error",
            f"An error occurred during GPP model execution:\n\n{error_details}"
        )


def main():
    app = QApplication(sys.argv)
    window = GPPInferenceApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
