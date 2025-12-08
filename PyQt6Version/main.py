import os
import sys
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton, QLabel, QTabWidget,
    QMenuBar, QMenu, QFileDialog, QMessageBox, QComboBox
)
from PyQt6.QtCore import QTimer, Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import pandas as pd

from projection_calculations import get_plot_data
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
        self.resize(1400, 900)

        self.data_manager = ToiletDataManager(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "toilet_data.csv"))

        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QHBoxLayout(central)

        self._setup_sidebar()
        self._setup_content()
        self._setup_menu()

        self.refresh_timer = QTimer()
        self.refresh_timer.setInterval(2000)
        self.refresh_timer.timeout.connect(self.update_plots)

        self.load_data()
        self.update_plots()

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

    def _setup_sidebar(self):
        sidebar = QWidget()
        sidebar.setMinimumWidth(280)
        s_layout = QVBoxLayout(sidebar)

        portfolio_group = QGroupBox("Portfolios")
        pg_layout = QVBoxLayout()
        self.portfolio_combo = QComboBox()
        pg_layout.addWidget(self.portfolio_combo)
        self.btn_refresh = QPushButton("Refresh Prices")
        pg_layout.addWidget(self.btn_refresh)
        portfolio_group.setLayout(pg_layout)

        actions_group = QGroupBox("Actions")
        ag_layout = QVBoxLayout()
        self.btn_export = QPushButton("Export Data")
        self.btn_import = QPushButton("Import Data")
        ag_layout.addWidget(self.btn_export)
        ag_layout.addWidget(self.btn_import)
        actions_group.setLayout(ag_layout)

        s_layout.addWidget(portfolio_group)
        s_layout.addWidget(actions_group)
        s_layout.addStretch()

        self.main_layout.addWidget(sidebar)

        self.btn_refresh.clicked.connect(self.refresh_all)
        self.btn_export.clicked.connect(self.export_data)
        self.btn_import.clicked.connect(self.import_data)

    def _setup_content(self):
        content = QWidget()
        c_layout = QVBoxLayout(content)
        self.tabs = QTabWidget()
        c_layout.addWidget(self.tabs)

        # Table
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(["Asset", "Type", "Ticker", "Value"])
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        c_layout.addWidget(self.table_widget)

        # Matplotlib canvas
        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)
        c_layout.addWidget(self.canvas)

        self.main_layout.addWidget(content, 1)

    def load_data(self):
        # Example: load portfolios and populate combo
        self.portfolio_combo.clear()
        for row in self.data_manager.df.itertuples():
            self.portfolio_combo.addItem(str(row.Name))
        # Populate minimal table from data_manager
        self.table_widget.setRowCount(len(self.data_manager.df))
        for i, row in enumerate(self.data_manager.df.itertuples()):
            self.table_widget.setItem(i, 0, QTableWidgetItem(str(row.Name)))
            self.table_widget.setItem(i, 1, QTableWidgetItem("managed"))
            self.table_widget.setItem(i, 2, QTableWidgetItem(""))
            self.table_widget.setItem(i, 3, QTableWidgetItem("0.00"))

    def update_plots(self):
        # Minimal example plot using get_plot_data if available
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        try:
            data = get_plot_data(
                A=133.0, B=317.5, C=266.7, theta_deg=30.0, margin_percent=10.0,
                shift_mm=0.0, required_resolution_mm_per_px=0.22, dead_zone_mm=0.3,
                pixel_pitch_um=2.0, camera_setup="1x1", smoothness=2, shift_axis="X"
            )
            series = pd.Series(data.get('ifov_mm_per_px_series', []))
            if not series.empty:
                ax.plot(series.index, series.values, label="IFOV")
        except Exception as e:
            ax.text(0.5, 0.5, f"Plot error: {e}", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("IFOV Overview")
        ax.grid(True)
        self.canvas.draw()

    def export_data(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Data", "", "CSV Files (*.csv);;All Files (*)")
        if not file_path:
            return
        try:
            self.data_manager.df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Export", f"Data exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def import_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Data", "", "CSV Files (*.csv);;All Files (*)")
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
            self.data_manager.df = df
            self.load_data()
            QMessageBox.information(self, "Import", f"Data imported from {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", str(e))

    def refresh_all(self):
        self.update_plots()


def main():
    app = QApplication(sys.argv)
    window = SensorFOVQtApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
