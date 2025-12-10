import os
import sys
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, QPushButton, QLabel,
    QMenuBar, QMenu, QFileDialog, QMessageBox, QComboBox, QLineEdit, QRadioButton,
    QCheckBox, QSizePolicy
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPalette, QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import pandas as pd

# Ensure parent directory (project root) is on sys.path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from projection_calculations import get_plot_data, draw_world, draw_projection, draw_side, draw_coverage
from data_manager import ToiletDataManager


def _get_git_meta(default_version="5.0.11"):
    try:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip()
        try:
            version = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip()
            if version.lower().startswith('v'):
                version = version[1:]
        except Exception:
            version = default_version
        try:
            build = subprocess.check_output(["git", "rev-list", "--count", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            build = "local"
        return version, branch, build
    except Exception:
        return default_version, "unknown", "local"

APP_VERSION, APP_BRANCH, APP_BUILD = _get_git_meta()


class SensorFOVQtApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"SENSOR SIMULATION v{APP_VERSION} ({APP_BUILD}) ({APP_BRANCH})")
        self.resize(1800, 1000)

        self._apply_dark_theme()

        self.data_manager = ToiletDataManager(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "toilet_data.csv"))

        central = QWidget()
        self.setCentralWidget(central)
        self.grid = QGridLayout(central)
        # Make left side narrower than right (plots/table)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 2)

        self._setup_title_bar()
        self._setup_results_panel()
        self._setup_params_panel()
        self._setup_plot_panel()
        self._setup_table_panel()
        self._setup_menu()

        self.refresh_timer = QTimer()
        self.refresh_timer.setInterval(2000)
        self.refresh_timer.timeout.connect(self.plot_projection)

        self.load_data()
        self.plot_projection()

    def _update_results_table(self, data: dict):
        rows = []
        rows.append(("Realistic FOV (deg)", f"{data.get('FOV_H', 0):.2f} × {data.get('FOV_V', 0):.2f}", "deg"))
        rows.append(("Water Coverage", f"{data.get('water_coverage_percent', 0):.1f}", "%"))
        rows.append(("Optimal Tilt Angle", f"{data.get('optimal_tilt_angle', 0):.1f}", "deg"))
        rows.append(("Projection Offset", f"{data.get('projection_offset', 0):.1f}", "mm"))
        rows.append(("Sensor Half Size", f"{data.get('sensor_half_width', 0):.2f} × {data.get('sensor_half_height', 0):.2f}", "mm"))
        rows.append(("Aspect Ratio", f"{data.get('sensor_aspect_ratio', '')}", "-"))
        self.results_table.setRowCount(len(rows))
        for i,(p,v,u) in enumerate(rows):
            self.results_table.setItem(i,0,QTableWidgetItem(p))
            self.results_table.setItem(i,1,QTableWidgetItem(v))
            self.results_table.setItem(i,2,QTableWidgetItem(u))

    def _apply_dark_theme(self):
        palette = QPalette()
        bg = QColor(24, 33, 58)
        base = QColor(30, 40, 70)
        text = QColor(220, 225, 235)
        accent = QColor(76, 114, 217)
        disabled = QColor(130, 140, 160)
        palette.setColor(QPalette.ColorRole.Window, bg)
        palette.setColor(QPalette.ColorRole.Base, base)
        palette.setColor(QPalette.ColorRole.AlternateBase, base)
        palette.setColor(QPalette.ColorRole.WindowText, text)
        palette.setColor(QPalette.ColorRole.Text, text)
        palette.setColor(QPalette.ColorRole.Button, base)
        palette.setColor(QPalette.ColorRole.ButtonText, text)
        palette.setColor(QPalette.ColorRole.Highlight, accent)
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255,255,255))
        palette.setColor(QPalette.ColorRole.PlaceholderText, disabled)
        self.setPalette(palette)
        self.setStyleSheet(
            "QGroupBox { background-color: rgb(30,40,70); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; margin-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 6px; color: rgb(200,210,225); }"
            "QPushButton { background-color: rgb(40,55,90); border: 1px solid rgba(255,255,255,0.06); padding: 8px 12px; border-radius: 8px; }"
            "QPushButton:hover { background-color: rgb(50,65,105); }"
            "QLineEdit, QComboBox { background-color: rgb(36,46,80); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 6px; }"
            "QHeaderView::section { background-color: rgb(36,46,80); color: rgb(220,225,235); padding: 6px; border: none; }"
        )

    def _setup_title_bar(self):
        title_group = QGroupBox()
        title_layout = QHBoxLayout(title_group)
        title_label = QLabel(f"SENSOR SIMULATION v{APP_VERSION} ({APP_BUILD}) ({APP_BRANCH})")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        help_btn = QPushButton("?")
        help_btn.setFixedWidth(40)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(help_btn)
        self.grid.addWidget(title_group, 0, 0, 1, 2)

    def _setup_results_panel(self):
        self.results_group = QGroupBox("Simulation Results")
        v = QVBoxLayout(self.results_group)
        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Value", "Unit"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.results_table)
        self.grid.addWidget(self.results_group, 1, 0)

    def _setup_params_panel(self):
        self.params_group = QGroupBox("Parameters")
        v = QGridLayout(self.params_group)
        labels = [
            ("A - Rim to Water depth (camera height) [mm]:", "133"),
            ("B - Water Spot Length [mm]:", "317.5"),
            ("C - Water Spot Width [mm]:", "266.7"),
            ("Camera Tilt [degrees]:", "30"),
            ("Margin [%]:", "10"),
            ("Shift from Water Spot Width Edge [mm]:", "0"),
            ("Required Resolution [mm/px]:", "0.22"),
            ("Dead zone [mm]:", "0.3"),
            ("Pixel pitch [um]:", "2.0"),
        ]
        self.entries = {}
        for i, (label, default) in enumerate(labels):
            v.addWidget(QLabel(label), i, 0)
            e = QLineEdit(default)
            e.setFixedWidth(120)
            self.entries[label] = e
            v.addWidget(e, i, 1)
        r = len(labels)
        v.addWidget(QLabel("Shift Axis:"), r, 0)
        axis_x = QRadioButton("X")
        axis_y = QRadioButton("Y")
        axis_x.setChecked(True)
        self.shift_axis = "X"
        axis_x.toggled.connect(lambda checked: setattr(self, 'shift_axis', 'X' if checked else getattr(self, 'shift_axis', 'Y')))
        axis_y.toggled.connect(lambda checked: setattr(self, 'shift_axis', 'Y' if checked else getattr(self, 'shift_axis', 'X')))
        v.addWidget(axis_x, r, 1)
        v.addWidget(axis_y, r, 2)
        v.addWidget(QLabel("Camera setup:"), r+1, 0)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["1x1", "1x2", "1x3", "2x2", "2x3"]) 
        self.camera_combo.setFixedWidth(100)
        v.addWidget(self.camera_combo, r+1, 1)
        self.ifov_check = QCheckBox("Enforce max projected pixel size (IFOV)")
        v.addWidget(self.ifov_check, r+2, 0, 1, 3)
        # Action buttons near parameters
        btn_apply = QPushButton("Apply")
        btn_reset = QPushButton("Reset")
        btn_save = QPushButton("Save Params")
        btn_load = QPushButton("Load Params")
        btn_apply.clicked.connect(self.plot_projection)
        btn_reset.clicked.connect(self._reset_params)
        btn_save.clicked.connect(self._save_params)
        btn_load.clicked.connect(self._load_params)
        v.addWidget(btn_apply, r+3, 0)
        v.addWidget(btn_reset, r+3, 1)
        v.addWidget(btn_save, r+4, 0)
        v.addWidget(btn_load, r+4, 1)
        self.grid.addWidget(self.params_group, 2, 0)

    def _reset_params(self):
        defaults = {
            "A - Rim to Water depth (camera height) [mm]:": "133",
            "B - Water Spot Length [mm]:": "317.5",
            "C - Water Spot Width [mm]:": "266.7",
            "Camera Tilt [degrees]:": "30",
            "Margin [%]:": "10",
            "Shift from Water Spot Width Edge [mm]:": "0",
            "Required Resolution [mm/px]:": "0.22",
            "Dead zone [mm]:": "0.3",
            "Pixel pitch [um]:": "2.0",
        }
        for label, val in defaults.items():
            if label in self.entries:
                self.entries[label].setText(val)
        self.shift_axis = "X"
        self.camera_combo.setCurrentIndex(0)
        self.ifov_check.setChecked(False)
        self.plot_projection()

    def _save_params(self):
        try:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Parameters", "", "JSON Files (*.json);;All Files (*)")
            if not file_path:
                return
            import json
            params = {label: self.entries[label].text() for label in self.entries}
            params.update({
                "ShiftAxis": self.shift_axis,
                "CameraSetup": self.camera_combo.currentText(),
                "IFOV": self.ifov_check.isChecked(),
            })
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _load_params(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Load Parameters", "", "JSON Files (*.json);;All Files (*)")
            if not file_path:
                return
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                params = json.load(f)
            for label, widget in self.entries.items():
                if label in params:
                    widget.setText(str(params[label]))
            self.shift_axis = params.get("ShiftAxis", self.shift_axis)
            cam_text = params.get("CameraSetup", self.camera_combo.currentText())
            idx = self.camera_combo.findText(cam_text)
            if idx >= 0:
                self.camera_combo.setCurrentIndex(idx)
            self.ifov_check.setChecked(bool(params.get("IFOV", False)))
            self.plot_projection()
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _setup_plot_panel(self):
        self.plot_group = QGroupBox("Plots")
        v = QVBoxLayout(self.plot_group)
        # Matplotlib canvas
        self.fig = Figure(figsize=(12, 7))
        # Use app dark theme background for graphs (no white fill)
        # Make figure fully transparent so it blends with the UI
        self.fig.set_facecolor('none')
        self.fig.patch.set_alpha(0.0)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot_group.setMinimumHeight(500)
        v.addWidget(self.canvas)
        # Zoom buttons placeholder
        zoom_bar = QHBoxLayout()
        zoom_in = QPushButton("Zoom In")
        zoom_out = QPushButton("Zoom Out")
        zoom_reset = QPushButton("Reset Zoom")
        zoom_bar.addStretch()
        zoom_bar.addWidget(zoom_in)
        zoom_bar.addWidget(zoom_out)
        zoom_bar.addWidget(zoom_reset)
        v.addLayout(zoom_bar)
        self.grid.addWidget(self.plot_group, 1, 1, 2, 1)

    def _setup_table_panel(self):
        self.table_group = QGroupBox("Toilet Database")
        v = QVBoxLayout(self.table_group)
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(len(self.data_manager.columns))
        self.table_widget.setHorizontalHeaderLabels(list(self.data_manager.columns))
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.table_widget)
        btns = QHBoxLayout()
        self.btn_export = QPushButton("Export Data")
        self.btn_import = QPushButton("Import Data")
        self.btn_export.clicked.connect(self.export_data)
        self.btn_import.clicked.connect(self.import_data)
        btns.addWidget(self.btn_export)
        btns.addWidget(self.btn_import)
        btns.addStretch()
        v.addLayout(btns)
        self.grid.addWidget(self.table_group, 3, 0, 1, 2)

    def _setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        export_action = file_menu.addAction("Export Data")
        export_action.triggered.connect(self.export_data)
        import_action = file_menu.addAction("Import Data")
        import_action.triggered.connect(self.import_data)
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        view_menu = menubar.addMenu("&View")
        refresh_action = view_menu.addAction("Refresh All")
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.refresh_all)

    # Sidebar removed to match Tkinter's grid-based panels

    # Content setup replaced by dedicated panels matching Tkinter layout

    def load_data(self):
        # Populate table from data_manager
        df = self.data_manager.data
        self.table_widget.setRowCount(len(df))
        cols = list(self.data_manager.columns)
        for i, row in df.iterrows():
            for j, col in enumerate(cols):
                val = row.get(col, "")
                self.table_widget.setItem(i, j, QTableWidgetItem(str(val)))

    def plot_projection(self):
        # Plot content mapped to four axes similar to Tkinter
        print("[PyQt6] plot_projection called")
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1.5, 0.8])
        ax_world = self.fig.add_subplot(gs[0, 0])
        ax_proj = self.fig.add_subplot(gs[0, 1])
        ax_side = self.fig.add_subplot(gs[1, 0])
        ax_cov = self.fig.add_subplot(gs[1, 1])
        for a in (ax_world, ax_proj, ax_side, ax_cov):
            # Transparent axes to blend with dark UI
            a.set_facecolor('none')
            a.patch.set_alpha(0.0)
            # High-contrast grid for readability on dark background
            a.grid(True, color='#AFC6FF', alpha=0.6, linestyle='-', linewidth=0.6)
            # Ensure axes are visible even before data
            a.plot([0, 1], [0, 0], color='#7EA2FF', linewidth=0.4)
            # Style ticks and spines for contrast
            a.tick_params(colors='#D8E2FF')
            for spine in a.spines.values():
                spine.set_edgecolor('#C7D6FF')
        # Set titles and grids upfront so axes are visible
        ax_world.set_title("World")
        ax_proj.set_title("Projection")
        ax_side.set_title("Side View")
        ax_cov.set_title("IFOV Overview")
        # Grids already set above with high contrast
        # Read parameters from UI
        def fget(label, default):
            try:
                return float(self.entries[label].text())
            except Exception:
                return default
        A = fget("A - Rim to Water depth (camera height) [mm]:", 133.0)
        B = fget("B - Water Spot Length [mm]:", 317.5)
        C = fget("C - Water Spot Width [mm]:", 266.7)
        theta = fget("Camera Tilt [degrees]:", 30.0)
        margin = fget("Margin [%]:", 10.0)
        shift_mm = fget("Shift from Water Spot Width Edge [mm]:", 0.0)
        res = fget("Required Resolution [mm/px]:", 0.22)
        deadzone = fget("Dead zone [mm]:", 0.3)
        pitch = fget("Pixel pitch [um]:", 2.0)
        cam_setup = self.camera_combo.currentText()
        print(f"[PyQt6] params A={A} B={B} C={C} theta={theta} margin={margin} shift={shift_mm} res={res} deadzone={deadzone} pitch={pitch} setup={cam_setup}")
        # Compute simulation data
        try:
            params = {
                'A': A,
                'B': B,
                'C': C,
                'Tilt': theta,
                'Margin': margin,
                'Shift': shift_mm,
                'ShiftAxis': getattr(self, 'shift_axis', 'X'),
                'Resolution': res
            }
            data = get_plot_data(params, smoothness=2)
        except Exception as e:
            print(f"[PyQt6] get_plot_data error: {e}")
            ax_cov.text(0.5, 0.5, f"Plot error: {e}", ha='center', va='center', transform=ax_cov.transAxes)
            # Fall back to basic placeholders
            from matplotlib.patches import Rectangle
            ax_world.add_patch(Rectangle((0, 0), 100, 50, fill=False, edgecolor='black'))
            ax_proj.axhline(0, color='black', linewidth=0.8)
            ax_side.axhline(0, color='black', linewidth=0.8)
            ax_side.axvline(0, color='black', linewidth=0.8)
            self.fig.tight_layout()
            self.canvas.draw()
            return
        # Use shared drawing helpers for parity with Tkinter
        try:
            draw_world(ax_world, params, data)
        except Exception as e:
            print(f"[PyQt6] draw_world error: {e}")
        try:
            draw_projection(ax_proj, params, data)
        except Exception as e:
            print(f"[PyQt6] draw_projection error: {e}")
        try:
            draw_side(ax_side, params, data)
        except Exception as e:
            print(f"[PyQt6] draw_side error: {e}")
        try:
            draw_coverage(ax_cov, params, data)
        except Exception as e:
            print(f"[PyQt6] draw_coverage error: {e}")
        self.fig.tight_layout()
        self.canvas.draw()
        # Update results table
        try:
            self._update_results_table(data)
        except Exception as e:
            print(f"[PyQt6] results table update error: {e}")

    def export_data(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Data", "", "CSV Files (*.csv);;All Files (*)")
        if not file_path:
            return
        try:
            self.data_manager.data.to_csv(file_path, index=False)
            QMessageBox.information(self, "Export", f"Data exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def import_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Data", "", "CSV Files (*.csv);;All Files (*)")
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
            self.data_manager.data = df
            self.load_data()
            QMessageBox.information(self, "Import", f"Data imported from {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", str(e))

    def refresh_all(self):
        self.plot_projection()


def main():
    app = QApplication(sys.argv)
    window = SensorFOVQtApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
