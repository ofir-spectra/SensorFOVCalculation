import os
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.font import Font
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle, Arc
import numpy as np
import matplotlib.gridspec as gridspec
import pandas as pd
import logging
import sys

from projection_calculations import get_plot_data
from data_manager import ToiletDataManager
import logging
import colorsys
from image_utils import find_image_case_insensitive

# Build dynamic version from Git branch and commit count
def _get_git_meta(default_version="5.0.11"):
    try:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip()
        # Prefer latest tag as version, fallback to default
        try:
            version = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip()
            # Normalize like 5.0.11 (strip leading 'v' if present)
            if version.lower().startswith('v'):
                version = version[1:]
        except Exception:
            version = default_version
        # Build number as total commit count on the current branch
        try:
            build = subprocess.check_output(["git", "rev-list", "--count", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            build = "local"
        return version, branch, build
    except Exception:
        return default_version, "unknown", "local"

APP_VERSION, APP_BRANCH, APP_BUILD = _get_git_meta()

GRAPH_TITLE_FONTSIZE = 14
GRAPH_LABEL_FONTSIZE = 14
GRAPH_TICK_FONTSIZE = 12
GRAPH_LEGEND_FONTSIZE = 10
GRAPH_OVERLAY_FONTSIZE = 12

CAMERA_SETUP_OPTIONS = ["1x1", "1x2", "1x3", "2x2", "2x3"]

def parse_camera_setup(cam_setup_str):
    """
    Parse the camera setup string (e.g., "2x2") into nx (horizontal tiles)
    and ny (vertical tiles).
    """
    try:
        nx, ny = map(int, str(cam_setup_str).split('x'))
        return nx, ny
    except Exception:
        return 1, 1  # Default to 1x1 if parsing fails

def dead_zone_pixels(deadzone_mm, pixel_pitch_um):
    try:
        return (1000.0 * float(deadzone_mm)) / float(pixel_pitch_um)
    except Exception:
        return 0.0

def safe_float(val, default):
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default

class ProjectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"SENSOR SIMULATION v{APP_VERSION} ({APP_BUILD}) ({APP_BRANCH})")
        self.root.geometry("1900x1000")

        # Create top frame for title and help button
        top_frame = tk.Frame(root)
        top_frame.grid(row=0, column=0, columnspan=2, pady=(10, 0), sticky="ew")
        
        title_label = tk.Label(top_frame, text=f"SENSOR SIMULATION v{APP_VERSION} ({APP_BUILD}) ({APP_BRANCH})", font=("Arial", 20, "bold"))
        title_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Add help button
        help_button = tk.Button(top_frame, text="?", font=("Arial", 16, "bold"), 
                              command=self.show_help, width=2, height=1,
                              bg='lightblue', fg='black')
        help_button.pack(side=tk.RIGHT, padx=10)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "toilet_data.csv")

        self.data_manager = ToiletDataManager(csv_path)
        self.param_font = Font(family="Arial", size=16)
        self.table_font = Font(family="Arial", size=16)
        self.heading_font = Font(family="Arial", size=16, weight="bold")
        # Increase Notebook tab font size
        try:
            style = ttk.Style()
            style.configure('TNotebook.Tab', font=('Arial', 16))
        except Exception:
            pass
        self.param_labels = [
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
        self.param_keys = ["A", "B", "C", "Tilt", "Margin", "Shift", "Resolution", "DeadZone", "PixelPitch"]

        self.entries = {}
        self.shift_axis_var = tk.StringVar(value="X")
        self.camera_setup_var = tk.StringVar(value=CAMERA_SETUP_OPTIONS[0])
        self.smoothness_var = tk.IntVar(value=2)

        self.coverage_zoom = 1.0
        self.coverage_xlim = [0, 60]
        self.coverage_ylim = [5, 90]

        self.fig = None
        self.canvas = None
        self.axes = {}
        self.setup_results_frame()
        self.setup_params_tabs()
        self.setup_plot_frame()
        self.setup_table_frame()
        # Ensure grid weights so results panel is visible
        try:
            self.root.grid_columnconfigure(0, weight=1)
            self.root.grid_columnconfigure(1, weight=2)
            self.root.grid_rowconfigure(1, weight=1)  # plots + results row grows
            self.root.grid_rowconfigure(2, weight=0)  # params tabs fixed height
            self.root.grid_rowconfigure(3, weight=0)  # table row fixed
        except Exception:
            pass
        # Ensure grid weights so the params notebook is visible and resizes
        try:
            self.root.grid_columnconfigure(0, weight=1)
            self.root.grid_columnconfigure(1, weight=2)
            self.root.grid_rowconfigure(1, weight=1)  # results + plots row
            self.root.grid_rowconfigure(2, weight=1)  # parameters row
        except Exception:
            pass
        self.refresh_table()
        self.plot_projection()

    def setup_results_frame(self):
        self.results_frame = ttk.LabelFrame(self.root, text="Simulation Results", padding="10")
        self.results_frame.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="nsew")
        container = ttk.Frame(self.results_frame)
        container.pack(fill=tk.BOTH, expand=True)
        self.results_table = ttk.Treeview(
            container,
            columns=("Parameter", "Value", "Unit"),
            show='headings',
            height=12
        )
        vsb = ttk.Scrollbar(container, orient="vertical", command=self.results_table.yview)
        self.results_table.configure(yscrollcommand=vsb.set)
        self.results_table.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        for col in ("Parameter", "Value", "Unit"):
            self.results_table.heading(col, text=col)
            self.results_table.column("Parameter", anchor="w", width=290)
            self.results_table.column("Value", anchor="center", width=160)
            self.results_table.column("Unit", anchor="center", width=80)

    def update_simulation_results(self, data):
        """
        Update the simulation results table with calculated values.
        Adds resolution per tile and dead band information for better clarity.
        """
        self.results_table.delete(*self.results_table.get_children())
        mode = data.get('mode', 'IFOV')
        results = []
        if mode == 'FOV':
            # In FOV mode, show user-provided sensor resolution; group all FOV outputs together
            # fov_polygon_world now represents ONE TILE's footprint (since each tile has its own lens)
            eff_tile_w_mm = eff_tile_h_mm = 0.0
            if isinstance(data.get('fov_polygon_world'), (list, tuple)) and len(data['fov_polygon_world']) >= 3:
                pts = np.array(data['fov_polygon_world'])
                eff_tile_w_mm = float(np.max(pts[:,0]) - np.min(pts[:,0]))
                eff_tile_h_mm = float(np.max(pts[:,1]) - np.min(pts[:,1]))
            
            # Full array footprint = tile footprint × number of tiles (ignoring gaps for simplicity)
            nx, ny = parse_camera_setup(data.get('CameraSetup', '1x1'))
            eff_w_mm = eff_tile_w_mm * nx
            eff_h_mm = eff_tile_h_mm * ny
            
            # FOV outputs first (grouped one under another)
            results.extend([
                ("Naive Sensor FOV [deg]", f"{data.get('FOV_H_sensor', 0):.2f} × {data.get('FOV_V_sensor', 0):.2f}", "deg"),
                ("Effective FOV Footprint [mm]", f"{eff_w_mm:.1f} × {eff_h_mm:.1f}", "mm"),
                ("Effective FOV Footprint Per Tile [mm]", f"{eff_tile_w_mm:.1f} × {eff_tile_h_mm:.1f}", "mm"),
                ("Naive Camera FOV Per Tile [deg]", f"{(data.get('FOV_H_per_tile') or 0):.2f} × {(data.get('FOV_V_per_tile') or 0):.2f}", "deg"),
            ])
            # Then resolution-related and other metrics
            results.extend([
                ("Sensor Resolution (user)", f"{data.get('pixels_x_sensor', data.get('SensorPixelsX', 0))} × {data.get('pixels_y_sensor', data.get('SensorPixelsY', 0))}", "px"),
                ("Active Pixel Fraction (by Image Circle)", f"{100.0*data.get('active_pixel_fraction',1.0):.1f}", "%"),
                ("Sensor Resolution Per Tile", f"{data.get('pixels_x_per_tile', 0):.0f} × {data.get('pixels_y_per_tile', 0):.0f}", "px"),
                ("Dead Zone (between tiles)", f"{data.get('deadzone_px', 0):.0f}", "px"),
                ("Water Coverage", f"{data.get('water_coverage_percent', 0):.1f}", "%"),
                ("FOV Coverage", f"{data.get('fov_coverage_percent', 0):.1f}", "%"),
                ("Optimal Tilt Angle", f"{data.get('optimal_angle', 0):.1f}", "deg"),
                ("Projection Offset", f"{data.get('projection_offset', 0):.1f}", "mm"),
                ("Sensor Aspect Ratio", data.get('aspect_ratio_used', ''), "-"),
                ("Optics Diameter", f"{data.get('optics_diameter', 0):.1f}", "mm"),
                ("Maximum Projected IFOV", f"{data.get('max_ifov', 0):.4f}", "mm"),
                ("Minimum Projected IFOV", f"{data.get('min_ifov', 0):.4f}", "mm"),
                ("Sensor Size [mm]", f"{data.get('sensor_width_mm', 0):.2f} × {data.get('sensor_height_mm', 0):.2f}", "mm"),
            ])
        else:
            # IFOV mode: keep existing detailed rows
            results.extend([
                ("Realistic Sensor Resolution", f"{data['pixels_x_sensor']} × {data['pixels_y_sensor']}", "px"),
                ("Resolution Per Tile", f"{data.get('pixels_x_per_tile', 0):.0f} × {data.get('pixels_y_per_tile', 0):.0f}", "px"),
                ("Dead Zone (between tiles)", f"{data.get('deadzone_px', 0):.0f}", "px"),
                ("Naive Resolution", f"{data['pixels_x_naive']} × {data['pixels_y_naive']}", "px"),
                ("Realistic Sensor FOV", f"{data.get('FOV_H_sensor', 0):.2f} × {data.get('FOV_V_sensor', 0):.2f}", "deg"),
                ("Naive Sensor FOV", f"{data.get('FOV_H_naive', 0):.2f} × {data.get('FOV_V_naive', 0):.2f}", "deg"),
                ("Water Coverage", f"{data.get('water_coverage_percent', 0):.1f}", "%"),
                ("FOV Coverage", f"{data.get('fov_coverage_percent', 0):.1f}", "%"),
                ("Optimal Tilt Angle", f"{data.get('optimal_angle', 0):.1f}", "deg"),
                ("Projection Offset", f"{data.get('projection_offset', 0):.1f}", "mm"),
                ("Sensor Aspect Ratio", data.get('aspect_ratio_used', ''), "-"),
                ("Optics Diameter", f"{data.get('optics_diameter', 0):.1f}", "mm"),
                ("Maximum Projected IFOV", f"{data.get('max_ifov', 0):.4f}", "mm"),
                ("Minimum Projected IFOV", f"{data.get('min_ifov', 0):.4f}", "mm"),
                ("Sensor Size [mm]", f"{data.get('sensor_width_mm', 0):.2f} × {data.get('sensor_height_mm', 0):.2f}", "mm"),
            ])
        for param, value, unit in results:
            self.results_table.insert("", tk.END, values=(param, value, unit))

    def setup_params_tabs(self):
        self.param_notebook = ttk.Notebook(self.root)
        self.param_notebook.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        # Give the notebook a minimum height to avoid collapsing
        try:
            self.param_notebook.update_idletasks()
            self.param_notebook.winfo_toplevel().update_idletasks()
        except Exception:
            pass
        # IFOV-based tab (existing panel)
        self.ifov_tab = ttk.Frame(self.param_notebook)
        self.param_notebook.add(self.ifov_tab, text="IFOVBased")
        # FOV-based tab (new panel)
        self.fov_tab = ttk.Frame(self.param_notebook)
        self.param_notebook.add(self.fov_tab, text="FOVBased")
        # Build panels
        self._build_ifov_params(self.ifov_tab)
        self._build_fov_params(self.fov_tab)

    def _build_ifov_params(self, parent):
        param_frame = ttk.LabelFrame(parent, text="Parameters", padding="10")
        param_frame.pack(fill=tk.BOTH, expand=True)
        for i, (label, default) in enumerate(self.param_labels):
            lbl = tk.Label(param_frame, text=label, font=self.param_font)
            lbl.grid(row=i, column=0, sticky="e", padx=5, pady=5)
            entry = tk.Entry(param_frame, font=self.param_font, width=12)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[self.param_keys[i]] = entry
            param = self.param_keys[i]
            # Add +/- to all numeric inputs except Resolution
            if param in {"A","B","C"}:
                tk.Button(param_frame, text="−", font=self.param_font, width=2,
                          command=lambda p=param: self.adjust_param(p, -1)).grid(row=i, column=2)
                tk.Button(param_frame, text="+", font=self.param_font, width=2,
                          command=lambda p=param: self.adjust_param(p, 1)).grid(row=i, column=3)
            if param == "Tilt":
                tk.Button(param_frame, text="−", font=self.param_font, width=2,
                          command=lambda p=param: self.adjust_param(p, -1)).grid(row=i, column=2)
                tk.Button(param_frame, text="+", font=self.param_font, width=2,
                          command=lambda p=param: self.adjust_param(p, 1)).grid(row=i, column=3)
            elif param == "Margin":
                tk.Button(param_frame, text="−", font=self.param_font, width=2,
                          command=lambda p=param: self.adjust_param(p, -1)).grid(row=i, column=2)
                tk.Button(param_frame, text="+", font=self.param_font, width=2,
                          command=lambda p=param: self.adjust_param(p, 1)).grid(row=i, column=3)
            elif param == "Shift":
                tk.Button(param_frame, text="−", font=self.param_font, width=2,
                          command=lambda p=param: self.adjust_param(p, -5)).grid(row=i, column=2)
                tk.Button(param_frame, text="+", font=self.param_font, width=2,
                          command=lambda p=param: self.adjust_param(p, 5)).grid(row=i, column=3)
            elif param == "DeadZone":
                tk.Button(param_frame, text="−", font=self.param_font, width=2,
                          command=lambda p=param: self.adjust_param(p, -0.1)).grid(row=i, column=2)
                tk.Button(param_frame, text="+", font=self.param_font, width=2,
                          command=lambda p=param: self.adjust_param(p, 0.1)).grid(row=i, column=3)
            elif param == "PixelPitch":
                tk.Button(param_frame, text="−", font=self.param_font, width=2,
                          command=lambda p=param: self.adjust_param(p, -0.1)).grid(row=i, column=2)
                tk.Button(param_frame, text="+", font=self.param_font, width=2,
                          command=lambda p=param: self.adjust_param(p, 0.1)).grid(row=i, column=3)
            # Exclude Resolution from +/- controls per request
        next_row = len(self.param_labels)
        # Add lens Image Circle [mm]
        tk.Label(param_frame, text="Image Circle [mm]", font=self.param_font).grid(row=next_row, column=0, sticky="e", padx=5, pady=5)
        self.image_circle_entry_ifov = tk.Entry(param_frame, font=self.param_font, width=12)
        self.image_circle_entry_ifov.insert(0, "0")
        self.image_circle_entry_ifov.grid(row=next_row, column=1, padx=5, pady=5)
        def adjust_image_circle_ifov(delta):
            try:
                val = safe_float(self.image_circle_entry_ifov.get(), 0.0) + delta
                self.image_circle_entry_ifov.delete(0, tk.END)
                self.image_circle_entry_ifov.insert(0, f"{max(0.0, val):.1f}")
                self.plot_projection()
            except Exception:
                pass
        tk.Button(param_frame, text="−", font=self.param_font, width=2,
                  command=lambda: adjust_image_circle_ifov(-0.5)).grid(row=next_row, column=2)
        tk.Button(param_frame, text="+", font=self.param_font, width=2,
                  command=lambda: adjust_image_circle_ifov(0.5)).grid(row=next_row, column=3)

        lbl = tk.Label(param_frame, text="Shift Axis:", font=self.param_font)
        lbl.grid(row=next_row, column=0, sticky="e", padx=5, pady=5)
        radio_frame = tk.Frame(param_frame)
        radio_frame.grid(row=next_row, column=1, padx=5, pady=5, sticky="w")
        tk.Radiobutton(radio_frame, text="X", variable=self.shift_axis_var,
                       value="X", font=self.param_font, command=self.plot_projection).pack(side=tk.LEFT)
        tk.Radiobutton(radio_frame, text="Y", variable=self.shift_axis_var,
                       value="Y", font=self.param_font, command=self.plot_projection).pack(side=tk.LEFT, padx=(10,0))

        combo_label = tk.Label(param_frame, text="Camera setup:", font=self.param_font)
        combo_label.grid(row=next_row+1, column=0, sticky="e", padx=5, pady=5)
        self.camera_setup_dropdown = ttk.Combobox(param_frame, values=CAMERA_SETUP_OPTIONS, font=self.param_font,
                                                  textvariable=self.camera_setup_var, width=8, state="readonly")
        self.camera_setup_dropdown.grid(row=next_row+1, column=1, padx=5, pady=5, sticky="w")

        # Smoothness parameter for coverage curve
        smooth_lbl = tk.Label(param_frame, text="Coverage Curve Smoothness (angle step °):", font=self.param_font)
        smooth_lbl.grid(row=next_row+2, column=0, sticky="e", padx=5, pady=5)
        smooth_entry = tk.Entry(param_frame, textvariable=self.smoothness_var, font=self.param_font, width=6)
        smooth_entry.grid(row=next_row+2, column=1, padx=5, pady=5, sticky="w")
        smooth_entry.bind('<Return>', lambda e: self.plot_projection())

        self.max_res_var = tk.IntVar(value=5000) # default max 200 pixels per axis

        res_lbl = tk.Label(param_frame, text="Max Sensor Pixels (per axis):", font=self.param_font)
        res_lbl.grid(row=next_row+7, column=0, sticky="e", padx=5, pady=5)
        res_entry = tk.Entry(param_frame, textvariable=self.max_res_var, font=self.param_font, width=6)
        res_entry.grid(row=next_row+7, column=1, padx=5, pady=5, sticky="w")
        self.overlay_step_var = tk.IntVar(value=1)  # 1 = all dots, 2 = every 2nd, etc

        dot_lbl = tk.Label(param_frame, text="Dot Overlay Step:", font=self.param_font)
        dot_lbl.grid(row=next_row+8, column=0, sticky="e", padx=5, pady=5)
        dot_entry = tk.Entry(param_frame, textvariable=self.overlay_step_var, font=self.param_font, width=6)
        dot_entry.grid(row=next_row+8, column=1, padx=5, pady=5, sticky="w")

        plot_button = tk.Button(param_frame, text="Plot", font=self.param_font, command=self.plot_projection)
        plot_button.grid(row=next_row+3, column=0, columnspan=2, pady=15)

        image_path = find_image_case_insensitive("image.png") or find_image_case_insensitive("image.jpg")
        if image_path:
            try:
                from PIL import Image, ImageTk
                self.original_img = Image.open(image_path)
                self.tk_img = None
                self.img_label = tk.Label(param_frame)
                self.img_label.grid(row=next_row+4, column=0, columnspan=2, pady=10, sticky="nsew")
                def resize_image(event=None):
                    width = param_frame.winfo_width() - 40
                    height = max(100, int(param_frame.winfo_height() * 0.25))
                    if width < 50: width = 100
                    if height < 50: height = 100
                    img = self.original_img.copy()
                    img.thumbnail((width, height), Image.LANCZOS)
                    self.tk_img = ImageTk.PhotoImage(img)
                    self.img_label.config(image=self.tk_img)
                param_frame.bind('<Configure>', resize_image)
                resize_image()
            except Exception:
                img_label = tk.Label(param_frame, text="Failed to load image file", font=self.param_font, fg="red")
                img_label.grid(row=next_row+4, column=0, columnspan=2, pady=10)
        else:
            img_label = tk.Label(param_frame, text="image.png/.jpg not found", font=self.param_font, fg="red")
            img_label.grid(row=next_row+4, column=0, columnspan=2, pady=10)

        self.ifov_enforce_var = tk.BooleanVar(value=False)
        ifov_check = tk.Checkbutton(param_frame,
            text="Enforce max projected pixel size (IFOV)",
            variable=self.ifov_enforce_var,
            font=self.param_font,
            command=self.plot_projection)
        ifov_check.grid(row=next_row+5, column=0, columnspan=2, sticky="w", pady=(8,2))
        
        self.flip_image_plane_var = tk.BooleanVar(value=True)  # Default ON
        self.flip_button_ifov = tk.Button(param_frame,
            text="Flip: ON",
            font=self.param_font,
            command=self.toggle_flip_image_plane,
            bg="lightgreen")
        self.flip_button_ifov.grid(row=next_row+6, column=0, columnspan=2, sticky="ew", pady=(8,2), padx=5)

    def _build_fov_params(self, parent):
        frame = ttk.LabelFrame(parent, text="FOV-Based Parameters", padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        # Reuse core geometric params A,B,C,Tilt,Margin,Shift,DeadZone,PixelPitch
        labels = [
            ("A - Rim to Water depth (camera height) [mm]:", "133"),
            ("B - Water Spot Length [mm]:", "317.5"),
            ("C - Water Spot Width [mm]:", "266.7"),
            ("Camera Tilt [degrees]:", "30"),
            ("Margin [%]:", "10"),
            ("Shift from Water Spot Width Edge [mm]:", "0"),
            ("Dead zone [mm]:", "0.3"),
            ("Pixel pitch [um]:", "2.0"),
        ]
        self.fov_entries = {}
        for i,(label,default) in enumerate(labels):
            tk.Label(frame, text=label, font=self.param_font).grid(row=i, column=0, sticky="e", padx=5, pady=5)
            e = tk.Entry(frame, font=self.param_font, width=12)
            e.insert(0, default)
            e.grid(row=i, column=1, padx=5, pady=5)
            self.fov_entries[label] = e
            # Add +/- controls similar to IFOV tab
            lbl_key_map = {
                "A - Rim to Water depth (camera height) [mm]:": (None, 1.0),
                "B - Water Spot Length [mm]:": (None, 1.0),
                "C - Water Spot Width [mm]:": (None, 1.0),
                "Camera Tilt [degrees]:": ("Tilt", 1.0),
                "Margin [%]:": ("Margin", 1.0),
                "Shift from Water Spot Width Edge [mm]:": ("Shift", 5.0),
                "Dead zone [mm]:": ("DeadZone", 0.1),
                "Pixel pitch [um]:": ("PixelPitch", 0.1),
            }
            key_pair = lbl_key_map.get(label)
            if key_pair:
                key_name, step = key_pair
                # Use a local adjust that targets fov_entries
                def adjust_fov(label_key=label, delta=step):
                    try:
                        val = safe_float(self.fov_entries[label_key].get(), 0.0)
                        val += delta
                        # clamp deadzone >= 0
                        if label_key == "Dead zone [mm]:":
                            val = max(0.0, val)
                        self.fov_entries[label_key].delete(0, tk.END)
                        # two decimals for micrometer/mm fields, one for angles
                        fmt = "{:.2f}" if label_key in ("Dead zone [mm]:", "Pixel pitch [um]:") else "{:.1f}"
                        self.fov_entries[label_key].insert(0, fmt.format(val))
                        self.plot_projection()
                    except Exception:
                        pass
                tk.Button(frame, text="−", font=self.param_font, width=2,
                          command=lambda l=label, s=step: adjust_fov(l, -s)).grid(row=i, column=2)
                tk.Button(frame, text="+", font=self.param_font, width=2,
                          command=lambda l=label, s=step: adjust_fov(l, s)).grid(row=i, column=3)
        r = len(labels)
        # New inputs for FOV-based mode
        tk.Label(frame, text="Focal Length [mm]:", font=self.param_font).grid(row=r, column=0, sticky="e", padx=5, pady=5)
        self.focal_length_entry = tk.Entry(frame, font=self.param_font, width=12)
        self.focal_length_entry.insert(0, "6.0")
        self.focal_length_entry.grid(row=r, column=1, padx=5, pady=5)
        # +/- for focal length
        def adjust_focal(delta):
            try:
                val = safe_float(self.focal_length_entry.get(), 6.0) + delta
                self.focal_length_entry.delete(0, tk.END)
                self.focal_length_entry.insert(0, f"{val:.1f}")
                self.plot_projection()
            except Exception:
                pass
        tk.Button(frame, text="−", font=self.param_font, width=2,
                  command=lambda: adjust_focal(-0.5)).grid(row=r, column=2)
        tk.Button(frame, text="+", font=self.param_font, width=2,
                  command=lambda: adjust_focal(0.5)).grid(row=r, column=3)
        tk.Label(frame, text="Sensor Resolution [px × px]", font=self.param_font).grid(row=r+1, column=0, sticky="e", padx=5, pady=5)
        self.sensor_res_entry = tk.Entry(frame, font=self.param_font, width=12)
        self.sensor_res_entry.insert(0, "1920x1080")
        self.sensor_res_entry.grid(row=r+1, column=1, padx=5, pady=5)
        
        # Image Circle [mm] with +/- buttons
        tk.Label(frame, text="Image Circle [mm]", font=self.param_font).grid(row=r+2, column=0, sticky="e", padx=5, pady=5)
        self.image_circle_entry_fov = tk.Entry(frame, font=self.param_font, width=12)
        self.image_circle_entry_fov.insert(0, "0")
        self.image_circle_entry_fov.grid(row=r+2, column=1, padx=5, pady=5)
        def adjust_image_circle_fov(delta):
            try:
                val = safe_float(self.image_circle_entry_fov.get(), 0.0) + delta
                self.image_circle_entry_fov.delete(0, tk.END)
                self.image_circle_entry_fov.insert(0, f"{max(0.0, val):.1f}")
                self.plot_projection()
            except Exception:
                pass
        tk.Button(frame, text="−", font=self.param_font, width=2,
                  command=lambda: adjust_image_circle_fov(-0.5)).grid(row=r+2, column=2)
        tk.Button(frame, text="+", font=self.param_font, width=2,
                  command=lambda: adjust_image_circle_fov(0.5)).grid(row=r+2, column=3)
        
        # Controls
        tk.Label(frame, text="Shift Axis:", font=self.param_font).grid(row=r+3, column=0, sticky="e", padx=5, pady=5)
        radio_frame = tk.Frame(frame)
        radio_frame.grid(row=r+3, column=1, padx=5, pady=5, sticky="w")
        tk.Radiobutton(radio_frame, text="X", variable=self.shift_axis_var,
                       value="X", font=self.param_font, command=self.plot_projection).pack(side=tk.LEFT)
        tk.Radiobutton(radio_frame, text="Y", variable=self.shift_axis_var,
                       value="Y", font=self.param_font, command=self.plot_projection).pack(side=tk.LEFT, padx=(10,0))
        tk.Label(frame, text="Plot camera setup:", font=self.param_font).grid(row=r+4, column=0, sticky="e", padx=5, pady=5)
        self.camera_setup_dropdown_fov = ttk.Combobox(frame, values=CAMERA_SETUP_OPTIONS, font=self.param_font,
                                                      textvariable=self.camera_setup_var, width=8, state="readonly")
        self.camera_setup_dropdown_fov.grid(row=r+4, column=1, padx=5, pady=5, sticky="w")
        
        # Add flip button and plot button side by side
        button_frame = tk.Frame(frame)
        button_frame.grid(row=r+5, column=0, columnspan=4, pady=10)
        tk.Button(button_frame, text="Plot", font=self.param_font, command=self.plot_projection, width=15).pack(side=tk.LEFT, padx=5)
        # Flip button - default state is ON
        flip_state = self.flip_image_plane_var.get() if hasattr(self, 'flip_image_plane_var') else True
        self.flip_button_fov = tk.Button(button_frame, 
                                         text="Flip: ON" if flip_state else "Flip: OFF", 
                                         font=self.param_font, 
                                         command=self.toggle_flip_image_plane, 
                                         width=20, 
                                         bg="lightgreen" if flip_state else "lightgray")
        self.flip_button_fov.pack(side=tk.LEFT, padx=5)
        # Show toilet image in this tab as well
        image_path = find_image_case_insensitive("image.png") or find_image_case_insensitive("image.jpg")
        if image_path:
            try:
                from PIL import Image, ImageTk
                self.original_img_fov = Image.open(image_path)
                self.tk_img_fov = None
                self.img_label_fov = tk.Label(frame)
                self.img_label_fov.grid(row=r+6, column=0, columnspan=2, pady=10, sticky="nsew")
                def resize_image_fov(event=None):
                    width = frame.winfo_width() - 40
                    height = max(100, int(frame.winfo_height() * 0.25))
                    if width < 50: width = 100
                    if height < 50: height = 100
                    img = self.original_img_fov.copy()
                    img.thumbnail((width, height), Image.LANCZOS)
                    self.tk_img_fov = ImageTk.PhotoImage(img)
                    self.img_label_fov.config(image=self.tk_img_fov)
                frame.bind('<Configure>', resize_image_fov)
                resize_image_fov()
            except Exception:
                tk.Label(frame, text="Failed to load image file", font=self.param_font, fg="red").grid(row=r+6, column=0, columnspan=2, pady=10)
        else:
            tk.Label(frame, text="image.png/.jpg not found", font=self.param_font, fg="red").grid(row=r+6, column=0, columnspan=2, pady=10)

    def adjust_param(self, key, delta):
        entry = self.entries[key]
        value = safe_float(entry.get(), 0.0)
        if key in {"DeadZone", "PixelPitch","Resolution"}:
            value = round(value + delta, 2)
            if key == "DeadZone":
                value = max(0, value)  # Ensure deadzone never goes below 0
        else:
            value += delta
        entry.delete(0, tk.END)
        entry.insert(0, f"{value:.2f}" if key in {"DeadZone", "PixelPitch","Resolution"} else f"{value:.1f}")
        self.plot_projection()

    def setup_plot_frame(self):
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.grid(row=1, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        plt.ioff()
        self.fig = plt.figure(figsize=(16, 10), dpi=100)
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1.5, 0.8])
        self.axes = {
            'world': self.fig.add_subplot(gs[0, 0]),
            'proj': self.fig.add_subplot(gs[0, 1]),
            'side': self.fig.add_subplot(gs[1, 0]),
            'coverage': self.fig.add_subplot(gs[1, 1])
        }
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas.draw()
        # Add zoom-in/out buttons for coverage graph
        zoom_frame = tk.Frame(self.plot_frame)
        zoom_frame.pack(anchor="e", pady=(8, 2))
        tk.Button(zoom_frame, text="Zoom In", command=self.zoom_in_coverage).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text="Zoom Out", command=self.zoom_out_coverage).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text="Reset Zoom", command=self.reset_zoom_coverage).pack(side=tk.LEFT, padx=2)

    # --- Zoom handlers for coverage plot ---
    def zoom_in_coverage(self):
        self.coverage_zoom *= 1.25
        self.plot_projection()
    def zoom_out_coverage(self):
        self.coverage_zoom /= 1.25
        self.plot_projection()
    def reset_zoom_coverage(self):
        self.coverage_zoom = 1.0
        self.coverage_xlim = [0, 60]
        self.coverage_ylim = [5, 90]
        self.plot_projection()

    def setup_table_frame(self):
        table_frame = ttk.LabelFrame(self.root, text="Toilet Database", padding="10")
        table_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.table_columns = self.data_manager.columns
        self.tree = ttk.Treeview(table_frame, columns=self.table_columns, show='headings', height=6)
        style = ttk.Style()
        style.configure("Treeview.Heading", font=self.heading_font)
        style.configure("Treeview", font=self.table_font, rowheight=36)
        for col in self.table_columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="center")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=v_scrollbar.set)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind('<Double-1>', self.edit_cell)
        btn_frame = ttk.Frame(table_frame)
        btn_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y, anchor='n')
        tk.Button(btn_frame, text="Add Current Parameters", font=self.param_font, command=self.add_toilet).pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="Load Selected", font=self.param_font, command=self.load_selected_toilet).pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="Delete Selected", font=self.param_font, command=self.delete_selected_toilet).pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="Save to CSV", font=self.param_font, command=self.data_manager.save_data).pack(fill=tk.X, pady=5)

    def get_current_parameters(self):
        try:
            active_tab = self.param_notebook.tab(self.param_notebook.select(), "text")
            if active_tab == "IFOVBased":
                deadzone = max(0, safe_float(self.entries['DeadZone'].get(), 0.3))
                self.entries['DeadZone'].delete(0, tk.END)
                self.entries['DeadZone'].insert(0, str(deadzone))
                return {
                    'Mode': 'IFOV',
                    'A': safe_float(self.entries['A'].get(), 133),
                    'B': safe_float(self.entries['B'].get(), 317.5),
                    'C': safe_float(self.entries['C'].get(), 266.7),
                    'Tilt': safe_float(self.entries['Tilt'].get(), 30),
                    'Margin': safe_float(self.entries['Margin'].get(), 10),
                    'Shift': safe_float(self.entries['Shift'].get(), 0),
                    'ShiftAxis': self.shift_axis_var.get(),
                    'Resolution': safe_float(self.entries['Resolution'].get(), 0.22),
                    'DeadZone': deadzone,
                    'PixelPitch': safe_float(self.entries['PixelPitch'].get(), 2.0),
                    'CameraSetup': self.camera_setup_var.get(),
                    'Smoothness': self.smoothness_var.get(),
                    'EnforceIFOV': self.ifov_enforce_var.get(),
                    'MaxSensorRes': self.max_res_var.get(),
                    'OverlayStep': self.overlay_step_var.get(),
                    'ImageCircle': safe_float(self.image_circle_entry_ifov.get(), 0.0),
                }
            else:
                # FOV-based mode parameters
                def fget(label, default):
                    val = self.fov_entries[label].get()
                    return safe_float(val, default)
                A = fget("A - Rim to Water depth (camera height) [mm]:", 133)
                B = fget("B - Water Spot Length [mm]:", 317.5)
                C = fget("C - Water Spot Width [mm]:", 266.7)
                Tilt = fget("Camera Tilt [degrees]:", 30)
                Margin = fget("Margin [%]:", 10)
                Shift = fget("Shift from Water Spot Width Edge [mm]:", 0)
                DeadZone = fget("Dead zone [mm]:", 0.3)
                PixelPitch = fget("Pixel pitch [um]:", 2.0)
                # Focal length and sensor resolution
                focal_mm = safe_float(self.focal_length_entry.get(), 6.0)
                res_text = self.sensor_res_entry.get().lower().replace(' ', '')
                if 'x' in res_text:
                    try:
                        px_x, px_y = map(int, res_text.split('x'))
                    except Exception:
                        px_x, px_y = 1920, 1080
                else:
                    px_x, px_y = 1920, 1080
                return {
                    'Mode': 'FOV',
                    'A': A, 'B': B, 'C': C,
                    'Tilt': Tilt, 'Margin': Margin,
                    'Shift': Shift, 'ShiftAxis': self.shift_axis_var.get(),
                    'DeadZone': DeadZone, 'PixelPitch': PixelPitch,
                    'CameraSetup': self.camera_setup_var.get(),
                    'FocalLength': focal_mm,
                    'SensorPixelsX': px_x,
                    'SensorPixelsY': px_y,
                    # Provide a Resolution key for get_plot_data compatibility (mm/px)
                    'Resolution': PixelPitch / 1000.0,
                    'Smoothness': self.smoothness_var.get(),
                    'ImageCircle': safe_float(self.image_circle_entry_fov.get(), 0.0),
                }
        except Exception:
            messagebox.showerror("Error", "Please enter valid numeric values for all parameters")
            return None

    def toggle_flip_image_plane(self):
        """Toggle the flip state and replot"""
        current = self.flip_image_plane_var.get()
        new_state = not current
        self.flip_image_plane_var.set(new_state)
        
        # Update button appearance for both IFOV and FOV tabs
        if new_state:
            text = "Flip: ON"
            bg = "lightgreen"
        else:
            text = "Flip: OFF"
            bg="lightgray"
        
        if hasattr(self, 'flip_button_ifov'):
            self.flip_button_ifov.config(text=text, bg=bg)
        if hasattr(self, 'flip_button_fov'):
            self.flip_button_fov.config(text=text, bg=bg)
        
        self.plot_projection()
    
    def plot_projection(self):
        """
                        # initialize per-tile FOV placeholders (computed later)
                        plot_data['FOV_H_per_tile'] = None
                        plot_data['FOV_V_per_tile'] = None
        Updated plot_projection method to correctly calculate sensor resolution
        for all camera setups (e.g., 1x2, 2x2), including dead zones.
        """
        params = self.get_current_parameters()
        if params is None:
            return
        # Common derived values
        pixel_pitch_um = params.get("PixelPitch", 2.0)
        cam_setup_str = params.get("CameraSetup", "1x1")
        try:
            parts = cam_setup_str.lower().replace(" ", "").split("x")
            nx = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 1
            ny = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        except Exception:
            nx, ny = 1, 1
        deadzone_mm = params.get("Dead zone [mm]:", 0.0)
        deadzone_px = int(round(deadzone_mm / (pixel_pitch_um / 1000.0))) if pixel_pitch_um > 0 else 0
        # Use legacy get_plot_data for base geometry
        plot_data = get_plot_data(params)
        plot_data['mode'] = params.get('Mode', 'IFOV')

        # Step 1 & 2: Get initial IFOV values from projection calculations
        # Derive IFOV/FOV based on mode
        mode = params.get('Mode', 'IFOV')
        if mode == 'FOV':
            # Use sensor inputs directly: resolution and pixels are provided by user
            pixel_pitch_um = params.get("PixelPitch", 2.0)
            pixel_pitch_mm = pixel_pitch_um / 1000.0
            sensor_width_mm = pixel_pitch_mm * params.get('SensorPixelsX', 1920)
            sensor_height_mm = pixel_pitch_mm * params.get('SensorPixelsY', 1080)
            
            # Image circle is the optical limiting aperture of the lens
            img_circle_mm = max(0.0, params.get('ImageCircle', 0.0))
            
            # Effective sensor dimensions are limited by the smaller of sensor or image circle
            if img_circle_mm > 0.0:
                # Image circle diameter limits the usable sensor dimensions
                # Assume square inscribed in circle for simplicity, or use circle diameter as limit
                effective_width_mm = min(sensor_width_mm, img_circle_mm)
                effective_height_mm = min(sensor_height_mm, img_circle_mm)
                # Compute active fraction for pixel count
                circle_area = np.pi * (img_circle_mm/2.0)**2
                rect_area = sensor_width_mm * sensor_height_mm
                active_frac = min(1.0, circle_area/rect_area) if rect_area>0 else 1.0
            else:
                effective_width_mm = sensor_width_mm
                effective_height_mm = sensor_height_mm
                active_frac = 1.0
            
            focal_mm = max(params.get('FocalLength', 6.0), 1e-6)
            # FOV from focal length and EFFECTIVE sensor size (limited by image circle)
            FOV_H_sensor = np.rad2deg(2 * np.arctan((effective_width_mm/2) / focal_mm))
            FOV_V_sensor = np.rad2deg(2 * np.arctan((effective_height_mm/2) / focal_mm))
            # Do not derive target IFOV in FOV mode; respect inputs
            target_ifov = params.get('Resolution', pixel_pitch_mm)
            # Store into plot_data for consistency
            plot_data['sensor_width_mm'] = sensor_width_mm
            plot_data['sensor_height_mm'] = sensor_height_mm
            plot_data['effective_width_mm'] = effective_width_mm
            plot_data['effective_height_mm'] = effective_height_mm
            plot_data['FOV_H_sensor'] = FOV_H_sensor
            plot_data['FOV_V_sensor'] = FOV_V_sensor
            # Sensor resolution comes directly from user input
            plot_data['pixels_x_sensor'] = params.get('SensorPixelsX', 1920)
            plot_data['pixels_y_sensor'] = params.get('SensorPixelsY', 1080)
            plot_data['image_circle_mm'] = img_circle_mm
            plot_data['active_pixel_fraction'] = active_frac
            plot_data['nx'] = nx
            plot_data['ny'] = ny
            plot_data['shift_axis'] = params.get('ShiftAxis', 'Y')
            plot_data['shift'] = params.get('Shift', 0)
            plot_data['margin_percent'] = params.get('Margin', 10)
            # Compute FOV polygon on water plane and coverage (use per-TILE dimensions for projection view)
            # Since each tile has its own lens, we show one tile's FOV in the projection plot
            fov_poly = self._compute_fov_polygon(params, nx=nx, ny=ny)
            plot_data['fov_polygon_world'] = fov_poly
            coverage_pct = self._estimate_fov_coverage(params, fov_poly)
            plot_data['fov_coverage_percent'] = coverage_pct
        else:
            target_ifov = params.get("Resolution", 0.22)
        
        # Calculate distances and IFOV considering perspective distortion
        tilt_rad = np.radians(params.get("Tilt", 30))
        height = params.get("A", 133)  # Camera height in mm
        
        if tilt_rad == 0:
            initial_min_ifov = target_ifov
            initial_max_ifov = target_ifov
        else:
            # For tilted camera, calculate perspective-affected distances
            min_distance = height * np.tan(tilt_rad)  # Closest point to camera
            max_distance = height / np.cos(tilt_rad)  # Farthest point
            distance_ratio = max_distance / min_distance
            
            # Calculate IFOV considering perspective effects
            initial_max_ifov = target_ifov * distance_ratio
            initial_min_ifov = target_ifov / distance_ratio

        # Step 3: Calculate resolution requirements
        # Apply margin to the dimensions
        margin_percent = params.get('Margin', 10)
        margin_factor = 1.0 + (margin_percent / 100.0)
        scaled_width = params['C'] * margin_factor  # Water spot width with margin
        scaled_length = params['B'] * margin_factor  # Water spot length with margin

        # Fix axes alignment - X should use width (C), Y should use length (B)
        # Naive resolution is only meaningful in IFOV mode
        if mode == 'FOV':
            naive_pixels_x = params.get('SensorPixelsX', 1920)
            naive_pixels_y = params.get('SensorPixelsY', 1080)
        else:
            naive_pixels_x = int(scaled_width / target_ifov)  # X corresponds to water spot width (C)
            naive_pixels_y = int(scaled_length / target_ifov)  # Y corresponds to water spot length (B)
        
        if mode == 'IFOV':
            # Calculate scaling ratio based on IFOV
            ifov_ratio = initial_max_ifov / target_ifov
            # Calculate required pixels including perspective effects
            required_pixels_x = int(naive_pixels_x * ifov_ratio)  # Width dimension
            required_pixels_y = int(naive_pixels_y * ifov_ratio)  # Length dimension
            # Step 5: Calculate max allowed pixels per camera/tile
            max_pixels_per_tile_x = (params.get("MaxSensorRes", 5000) - (nx - 1) * deadzone_px) // nx
            max_pixels_per_tile_y = (params.get("MaxSensorRes", 5000) - (ny - 1) * deadzone_px) // ny
            # Step 6: Adjust resolution to maintain aspect ratio if needed
            scaling_ratio = 1.0
            if required_pixels_x > max_pixels_per_tile_x or required_pixels_y > max_pixels_per_tile_y:
                ratio_x = max_pixels_per_tile_x / required_pixels_x if required_pixels_x > max_pixels_per_tile_x else 1.0
                ratio_y = max_pixels_per_tile_y / required_pixels_y if required_pixels_y > max_pixels_per_tile_y else 1.0
                scaling_ratio = min(ratio_x, ratio_y)
            # Calculate final pixels per tile
            pixels_x_per_tile = int(required_pixels_x * scaling_ratio)
            pixels_y_per_tile = int(required_pixels_y * scaling_ratio)
            # Calculate total pixels including dead zones
            total_pixels_x = nx * pixels_x_per_tile + (nx - 1) * deadzone_px
            total_pixels_y = ny * pixels_y_per_tile + (ny - 1) * deadzone_px
            # Step 7: Update IFOV values based on final scaling
            final_ratio = ifov_ratio * scaling_ratio
            final_min_ifov = initial_min_ifov / final_ratio
            final_max_ifov = initial_max_ifov / final_ratio
        else:
            # FOV mode: use user sensor resolution; compute per-tile by splitting
            total_pixels_x = params.get('SensorPixelsX', 1920)
            total_pixels_y = params.get('SensorPixelsY', 1080)
            pixel_pitch_mm = pixel_pitch_um / 1000.0
            
            # Get effective sensor dimensions (already limited by image circle in FOV mode block above)
            effective_width_mm = plot_data.get('effective_width_mm', pixel_pitch_mm * total_pixels_x)
            effective_height_mm = plot_data.get('effective_height_mm', pixel_pitch_mm * total_pixels_y)
            
            # For 1x1 setup, per-tile FOV equals full sensor FOV (no dead zones, no splitting)
            if nx == 1 and ny == 1:
                # No tiles to split, use effective dimensions directly
                plot_data['FOV_H_per_tile'] = plot_data.get('FOV_H_sensor', 0.0)
                plot_data['FOV_V_per_tile'] = plot_data.get('FOV_V_sensor', 0.0)
                # Effective pixels are the same as what fits in the image circle
                eff_pixels_x_total = int(effective_width_mm / pixel_pitch_mm)
                eff_pixels_y_total = int(effective_height_mm / pixel_pitch_mm)
                pixels_x_per_tile = eff_pixels_x_total
                pixels_y_per_tile = eff_pixels_y_total
                # For projection view: tile sensor dimensions = effective sensor dimensions
                plot_data['tile_sensor_half_width_mm'] = effective_width_mm / 2.0
                plot_data['tile_sensor_half_height_mm'] = effective_height_mm / 2.0
            else:
                # Multiple tiles: account for dead zones and split
                # Convert effective dimensions back to effective pixel count
                eff_pixels_x_total = int(effective_width_mm / pixel_pitch_mm)
                eff_pixels_y_total = int(effective_height_mm / pixel_pitch_mm)
                
                # Remove dead zones from effective sensor for splitting
                usable_pixels_x = max(eff_pixels_x_total - (nx - 1) * deadzone_px, 0)
                usable_pixels_y = max(eff_pixels_y_total - (ny - 1) * deadzone_px, 0)
                pixels_x_per_tile = usable_pixels_x // nx if nx > 0 else usable_pixels_x
                pixels_y_per_tile = usable_pixels_y // ny if ny > 0 else usable_pixels_y
                
                # Per-tile FOV from effective tile dimensions (limited by image circle)
                tile_width_mm = pixel_pitch_mm * pixels_x_per_tile
                tile_height_mm = pixel_pitch_mm * pixels_y_per_tile
                focal_mm = max(params.get('FocalLength', 6.0), 1e-6)
                fov_h_tile = np.rad2deg(2 * np.arctan((tile_width_mm/2) / focal_mm))
                fov_v_tile = np.rad2deg(2 * np.arctan((tile_height_mm/2) / focal_mm))
                plot_data['FOV_H_per_tile'] = min(fov_h_tile, plot_data.get('FOV_H_sensor', fov_h_tile))
                plot_data['FOV_V_per_tile'] = min(fov_v_tile, plot_data.get('FOV_V_sensor', fov_v_tile))
                # For projection view: tile sensor dimensions
                plot_data['tile_sensor_half_width_mm'] = tile_width_mm / 2.0
                plot_data['tile_sensor_half_height_mm'] = tile_height_mm / 2.0
            
            # Store active fraction for display
            active_frac = plot_data.get('active_pixel_fraction', 1.0)
            
            final_min_ifov = initial_min_ifov
            final_max_ifov = initial_max_ifov

        # Update the plot data with all calculated values
        # Calculate sensor physical size in mm (converting from micrometers)
        pixel_pitch_mm = pixel_pitch_um / 1000.0  # Convert pixel pitch from μm to mm
        # Calculate dimensions maintaining consistent axis mapping
        sensor_width_mm = pixel_pitch_mm * total_pixels_x   # Width (X) corresponds to water spot width (C)
        sensor_height_mm = pixel_pitch_mm * total_pixels_y  # Height (Y) corresponds to water spot length (B)

        plot_data['pixels_x_sensor'] = total_pixels_x
        plot_data['pixels_y_sensor'] = total_pixels_y
        plot_data['pixels_x_per_tile'] = pixels_x_per_tile
        plot_data['pixels_y_per_tile'] = pixels_y_per_tile
        plot_data['deadzone_px'] = deadzone_px
        plot_data['min_ifov'] = final_min_ifov
        plot_data['max_ifov'] = final_max_ifov
        plot_data['naive_pixels_x'] = naive_pixels_x
        plot_data['naive_pixels_y'] = naive_pixels_y
        plot_data['sensor_width_mm'] = sensor_width_mm
        plot_data['sensor_height_mm'] = sensor_height_mm
        # Ensure scaling_ratio is defined for both modes
        if mode == 'IFOV':
            plot_data['scaling_ratio'] = scaling_ratio  # Store the final scaling ratio
        else:
            plot_data['scaling_ratio'] = 1.0

        # Calculate max_radius from the projected rectangle points
        # This gives us the actual maximum distance in the projected view
        proj_rect_pts = plot_data.get('proj_rect_pts', None)
        if proj_rect_pts is not None:
            # Provide compatibility arrays expected by projected view
            plot_data['px_sensor'] = proj_rect_pts[:, 0]
            plot_data['py_sensor'] = proj_rect_pts[:, 1]
            distances = np.sqrt(np.sum(proj_rect_pts**2, axis=1))  # Distance from optical center
            max_radius = np.max(distances)
        else:
            # Fallback to sensor diagonal if projection points aren't available
            max_radius = np.sqrt(sensor_width_mm**2 + sensor_height_mm**2) / 2
        plot_data['max_radius'] = max_radius

        # Store tile dimensions for projection view overlay (needed before plotting)
        if mode == 'FOV':
            # Calculate tile sensor dimensions for green box overlay
            pixel_pitch_mm = pixel_pitch_um / 1000.0
            effective_width_mm = plot_data.get('effective_width_mm', pixel_pitch_mm * params.get('SensorPixelsX', 1920))
            effective_height_mm = plot_data.get('effective_height_mm', pixel_pitch_mm * params.get('SensorPixelsY', 1080))
            
            if nx == 1 and ny == 1:
                plot_data['tile_sensor_half_width_mm'] = effective_width_mm / 2.0
                plot_data['tile_sensor_half_height_mm'] = effective_height_mm / 2.0
            else:
                tile_width_mm = pixel_pitch_mm * pixels_x_per_tile
                tile_height_mm = pixel_pitch_mm * pixels_y_per_tile
                plot_data['tile_sensor_half_width_mm'] = tile_width_mm / 2.0
                plot_data['tile_sensor_half_height_mm'] = tile_height_mm / 2.0

        # Debugging: Print detailed calculations for verification
        print(f"Camera Setup: {cam_setup_str} (nx={nx}, ny={ny})")
        print(f"Dead Zone (mm): {deadzone_mm}, Dead Zone (px): {deadzone_px}")
        print(f"Resolution per Tile: {pixels_x_per_tile} x {pixels_y_per_tile} px")
        print(f"Total Resolution: {total_pixels_x} x {total_pixels_y} px (including dead zones)")
        print(f"IFOV: min={final_min_ifov:.4f} mm/px, max={final_max_ifov:.4f} mm/px")

        # Clear old plots and redraw new ones
        self.update_simulation_results(plot_data)
        for ax in self.axes.values():
            ax.clear()
        self.plot_world_view(plot_data)
        self.plot_projected_view(plot_data)
        self.plot_side_view(plot_data)
        self.plot_coverage_view(plot_data)
        self.fig.tight_layout(pad=1.0)
        self.canvas.draw()

    def plot_world_view(self, data):
        ax = self.axes['world']
        if np.isnan(data.get('Xc', np.nan)) or np.all(np.isnan(data.get('rect_corners', np.nan))):
            ax.clear()
            ax.text(0.5, 0.5, "Impossible IFOV or parameter settings.\nPlease loosen requirements.",
                    fontsize=14, color='red', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return
        ax.set_aspect('equal')
        # Close the rectangle by adding the first point at the end
        rect_points = np.vstack([data['rect_corners'], data['rect_corners'][0]])
        ax.plot(rect_points[:, 0], rect_points[:, 1], 'b-', lw=2, label='Water Spot Rectangle')
        ax.add_patch(Ellipse((0, 0), data['C'], data['L'], edgecolor='r', facecolor='r', alpha=0.3, lw=2, label='Water Area'))
        camera_size = 20
        color = 'green' if data['shift_axis'] == 'X' else 'blue'
        ax.add_patch(Rectangle((data['Xc'] - camera_size/2, data['Yc'] - camera_size/2), camera_size, camera_size,
                               color=color, alpha=0.7, label=f'Camera ({data["shift_axis"]}-axis)'))
        ax.plot([data['Xc']], [data['Yc']], 'go' if data['shift_axis'] == 'X' else 'bo')
        # Fill camera FOV polygons for all tiles in world view (distinct hues)
        if data.get('mode','IFOV')=='FOV':
            nx, ny = parse_camera_setup(data.get('CameraSetup','1x1'))
            added_label = False
            
            # Draw overall image circle footprint (outline)
            fov_polygon_world = data.get('fov_polygon_world')
            if fov_polygon_world and len(fov_polygon_world) >= 3:
                try:
                    pts = np.array(fov_polygon_world)
                    ax.plot(pts[:,0], pts[:,1], 'g-', linewidth=2.5, label='Image Circle Footprint', alpha=0.8)
                    ax.fill(pts[:,0], pts[:,1], color=(0, 1, 0, 0.1))
                    
                    # Draw rays from camera to FOV corners/edge points
                    # Sample every 8th point to avoid clutter
                    cam_x, cam_y = data['Xc'], data['Yc']
                    sample_step = max(len(pts) // 8, 1)
                    for i in range(0, len(pts), sample_step):
                        ax.plot([cam_x, pts[i,0]], [cam_y, pts[i,1]], 'g--', linewidth=0.8, alpha=0.4)
                except Exception as e:
                    print(f"Error drawing FOV footprint: {e}")
            
            # Draw individual tiles only if more than 1x1 (otherwise redundant with overall footprint)
            if nx > 1 or ny > 1:
                for ty in range(max(ny,1)):
                    for tx in range(max(nx,1)):
                        tile_poly = self._compute_tile_fov_polygon(
                            data,
                            nx=nx, ny=ny,
                            tile_ix=tx, tile_iy=ty,
                            pixels_x_per_tile=data.get('pixels_x_per_tile'),
                            pixels_y_per_tile=data.get('pixels_y_per_tile')
                        )
                        try:
                            pts = np.array(tile_poly)
                            lbl = 'Camera FOV (tiles)' if not added_label else '_nolegend_'
                            hue = ((ty*max(nx,1)+tx)+1) / max(nx*ny,1)
                            rgb = colorsys.hsv_to_rgb(hue, 0.5, 0.9)
                            ax.fill(pts[:,0], pts[:,1], color=(rgb[0], rgb[1], rgb[2], 0.18), label=lbl)
                            ax.plot(pts[:,0], pts[:,1], color=rgb, alpha=0.7, linewidth=1.5, label=lbl)
                            cx, cy = np.mean(pts[:,0]), np.mean(pts[:,1])
                            ax.text(cx, cy, f"{tx+1},{ty+1}", fontsize=8, color=rgb)
                            added_label = True
                        except Exception:
                            pass
        ax.set_xlim(-300, 300)
        ax.set_ylim(-300, 300)
        ax.set_xlabel('X [mm]', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.set_ylabel('Y [mm]', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.set_title('Object Plane - Top View (World Coordinates)', fontsize=GRAPH_TITLE_FONTSIZE)
        ax.tick_params(axis='both', labelsize=GRAPH_TICK_FONTSIZE)
        ax.legend(loc='upper right', fontsize=GRAPH_LEGEND_FONTSIZE)
        ax.grid(True, alpha=0.3)

    def plot_projected_view(self, data):
        # Place this in your plot_projected_view function, replacing all previous blue/red pixel logic
        ax = self.axes['proj']

        if np.all(np.isnan(data.get('proj_rect_pts', np.nan))) or np.all(np.isnan(data.get('proj_ellipse_pts', np.nan))):
            ax.clear()
            ax.text(0.5, 0.5, "Impossible IFOV or parameter settings.\nPlease loosen requirements.",
                    fontsize=14, color='red', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return

        ax.set_aspect('equal')

        proj_rect_pts = data['proj_rect_pts']
        px, py = data['px_sensor'], data['py_sensor']
        N = proj_rect_pts.shape[0]
        color_arr = np.full((N, 3), (0.2, 0.4, 1.0))  # blue

        # Get all parameters needed for IFOV calculation from user input
        height = data.get("A", 133)  # Camera height in mm
        tilt_rad = np.radians(data.get("Tilt", 30))
        # Get current resolution from GUI parameter
        resolution_mm = float(self.entries["Resolution"].get())  # Target resolution in mm/pixel

        print("\nDEBUG VALUES:")
        print(f"Target resolution (mm/px): {resolution_mm}")
        print(f"Camera height (mm): {height}")
        print(f"Camera tilt (degrees): {np.degrees(tilt_rad):.1f}")
        print(f"Base IFOV at camera center (mm/px): {data['max_ifov']}")
        print(f"Base IFOV at edge (mm/px): {data['min_ifov']}")
        print("\nPoint Analysis (20 sample points):")
        print("Index  Position(x,y)mm    Dist(mm)  IFOV(mm/px)  Status")
        print("-" * 65)
        
        print("\nDEBUG VALUES:")
        print(f"Target resolution (mm/px): {resolution_mm}")
        print(f"Camera height (mm): {height}")
        print(f"Camera tilt (degrees): {np.degrees(tilt_rad):.1f}")
        print(f"Base IFOV at camera center (mm/px): {data['max_ifov']}")
        print("\nPoint Analysis (20 sample points):")
        print("Index  Position(x,y)mm    Dist(mm)  Angle(°)  IFOV(mm/px)  Status")
        print("-" * 75)

        if self.ifov_enforce_var.get():
            
            # Calculate indices for 20 evenly distributed points
            if N > 1:
                sample_indices = [0]  # Always include first point
                if N > 2:
                    step = (N - 1) / 18  # Calculate step size for remaining 18 points
                    sample_indices.extend([int(i * step) for i in range(1, 19)])
                sample_indices.append(N - 1)  # Always include last point
            else:
                sample_indices = [0]
            
            # For each point in our projected view
            for i in range(N):
                point = proj_rect_pts[i]
                point_x = point[0]  # X coordinate on water surface
                point_y = point[1]  # Y coordinate on water surface
                
                # Calculate distances to neighboring points
                if i > 0:
                    prev_point = proj_rect_pts[i-1]
                    dx = point_x - prev_point[0]
                    dy = point_y - prev_point[1]
                    point_dist = np.sqrt(dx*dx + dy*dy)  # Distance between consecutive points
                    
                    # First find all point-to-point distances if we haven't already
                    if i == 1:  # Only calculate once
                        all_dists = []
                        perimeter = 0
                        for j in range(1, N):
                            p1, p2 = proj_rect_pts[j], proj_rect_pts[j-1]
                            d = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                            if d > 0:  # Avoid zero distances
                                all_dists.append(d)
                                perimeter += d
                        
                        # Calculate average point spacing
                        avg_point_spacing = perimeter / N
                        min_d = min(all_dists) if all_dists else point_dist
                        max_d = max(all_dists) if all_dists else point_dist
                        
                        print("\nSpacing Analysis:")
                        print(f"Perimeter length: {perimeter:.2f} mm")
                        print(f"Average spacing: {avg_point_spacing:.4f} mm")
                        print(f"Min spacing: {min_d:.4f} mm")
                        print(f"Max spacing: {max_d:.4f} mm\n")
                else:
                    # For the first point, use the distance to the next point
                    if N > 1:
                        next_point = proj_rect_pts[1]
                        dx = next_point[0] - point_x
                        dy = next_point[1] - point_y
                        point_dist = np.sqrt(dx*dx + dy*dy)
                    else:
                        point_dist = 0
                
                # Calculate distance from optical center (0,0) to current point
                d = np.sqrt(point_x**2 + point_y**2)
                
                # Calculate IFOV at this point using linear interpolation based on distance
                point_ifov = data['min_ifov'] + (data['max_ifov'] - data['min_ifov']) * (d / data['max_radius'])

                # Print detailed debug info for sample points
                if i in sample_indices:
                    status = "RED" if point_ifov > resolution_mm else "BLUE"
                    print(f"{i:5d}  ({point_x:7.1f},{point_y:7.1f})  d={d:8.1f} mm  IFOV={point_ifov:8.4f}  {status}")
                
                # Color coding:
                # - Red: resolution at this point is worse than our target
                # - Blue: resolution at this point meets our target
                if point_ifov > resolution_mm:
                    color_arr[i] = (0.8, 0.2, 0.2)  # red

        # Draw points with higher visibility
        ax.scatter(
            proj_rect_pts[:, 0],
            proj_rect_pts[:, 1],
            color=color_arr,
            s=2,  # Increased dot size
            alpha=0.6,  # Full opacity
            label='Sensor Pixels (Blue/Red by IFOV)'
        )


      

        # --- Existing code for overlays and lines, unchanged ---
        # Coverage overlay for FOV mode: paint outside area red, inside FOV green
        try:
            active_tab = self.param_notebook.tab(self.param_notebook.select(), "text")
        except Exception:
            active_tab = "IFOVBased"
        if active_tab == "FOVBased":
            from matplotlib.patches import Polygon
            # Fill projected water ellipse lightly red
            ell = data['proj_ellipse_pts']
            ax.fill(ell[:,0], ell[:,1], color=(1,0.6,0.6,0.25), label='Water Area')
            
            # In FOV mode, project the LIMITING FACTOR (sensor or image circle) to water plane
            # then forward-project back to sensor view
            if data.get('mode','IFOV')=='FOV':
                try:
                    # Get sensor and image circle dimensions
                    sensor_w = data.get('sensor_width_mm', 0)
                    sensor_h = data.get('sensor_height_mm', 0)
                    img_circle = data.get('image_circle_mm', 0)
                    
                    # Determine limiting factor
                    sensor_diagonal = np.sqrt(sensor_w**2 + sensor_h**2)
                    is_circle_limiting = img_circle > 0 and img_circle < sensor_diagonal
                    
                    # Get camera parameters for projection (same as projection_calculations.py)
                    A = data.get('A')
                    B = data.get('B')
                    C = data.get('C')
                    theta_deg = data.get('theta_deg')
                    shift_axis = data.get('shift_axis', 'Y')
                    shift = data.get('shift', 0)
                    margin_percent = data.get('margin_percent', 0)
                    theta = np.deg2rad(theta_deg)
                    
                    # Apply margin factor (same as projection_calculations.py)
                    margin_factor = 1.0 + (margin_percent / 100.0)
                    H = A
                    L = B * margin_factor
                    W = C * margin_factor
                    
                    # Camera position (same as projection_calculations.py)
                    if shift_axis == 'X':
                        cam_pos = np.array([W/2 + shift, 0, H])
                    else:
                        cam_pos = np.array([0, L/2 + shift, H])
                    
                    # Camera orientation
                    initial_optical_axis = np.array([0, 0, -1])
                    if shift_axis == 'X':
                        Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
                        z_cam = Ry @ initial_optical_axis
                    else:
                        theta_corrected = -theta
                        Rx = np.array([[1, 0, 0], [0, np.cos(theta_corrected), -np.sin(theta_corrected)], [0, np.sin(theta_corrected), np.cos(theta_corrected)]])
                        z_cam = Rx @ initial_optical_axis
                    z_cam = z_cam / np.linalg.norm(z_cam)
                    
                    # Camera basis
                    if abs(np.dot(z_cam, [0, 1, 0])) > 0.99:
                        up_guess = np.array([1, 0, 0])
                    else:
                        up_guess = np.array([0, 1, 0])
                    x_cam = np.cross(up_guess, z_cam)
                    x_cam = x_cam / np.linalg.norm(x_cam)
                    y_cam = np.cross(z_cam, x_cam)
                    y_cam = y_cam / np.linalg.norm(y_cam)
                    
                    # Use fov_polygon_world which already contains the correct back-projected footprint
                    # (it was computed with the limiting factor applied)
                    fov_polygon_world = data.get('fov_polygon_world')
                    if fov_polygon_world and len(fov_polygon_world) >= 3:
                        # Forward-project the water plane footprint to sensor view
                        proj_polygon = []
                        for pt in fov_polygon_world:
                            p_world = np.array([pt[0], pt[1], 0])
                            v = p_world - cam_pos
                            Xc = np.dot(v, x_cam)
                            Yc = np.dot(v, y_cam)
                            Zc = np.dot(v, z_cam)
                            if abs(Zc) > 1e-10:
                                x_img = H * Xc / Zc
                                y_img = H * Yc / Zc
                                proj_polygon.append([x_img, y_img])
                        
                        if len(proj_polygon) >= 3:
                            proj_arr = np.array(proj_polygon)
                            # Close the polygon properly
                            poly_x = np.append(proj_arr[:, 0], proj_arr[0, 0])
                            poly_y = np.append(proj_arr[:, 1], proj_arr[0, 1])
                            # Draw with higher zorder to ensure visibility
                            ax.fill(poly_x, poly_y, color=(0, 1, 0, 0.25), label='Camera FOV (tile)', zorder=5)
                            ax.plot(poly_x, poly_y, color='g', alpha=0.9, linewidth=2.5, label='_nolegend_', zorder=5)
                except Exception as e:
                    print(f"Error drawing FOV overlay: {e}")
                    import traceback
                    traceback.print_exc()
                    
        ax.plot(data['proj_ellipse_pts'][:, 0], data['proj_ellipse_pts'][:, 1], 'r-', linewidth=2, label='Water Ellipse')
        ax.plot(data['proj_rect_outline'][:, 0], data['proj_rect_outline'][:, 1], 'b-', linewidth=2, label='Water Rectangle')
        ax.scatter(0, 0, color='red', s=200, marker='+', linewidth=3, label='_nolegend_')
        ax.scatter(data['ellipse_cx'], data['ellipse_cy'], color='purple', s=100, marker='x', linewidth=2, label='_nolegend_')
        ax.axhline(y=0, color='red', linestyle='-', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='red', linestyle='-', alpha=0.3, linewidth=1)
        from matplotlib.patches import Circle
        ax.add_patch(Circle((0, 0), data['optics_radius'], fill=False, color='black', lw=2, label=f"Optics (D={data['optics_diameter']:.1f}mm)"))

        # Set view limits based on optical diameter with 10% margin
        max_dim = data['optics_radius'] * 1.1  # 1.1 times the optical radius
        
        # Flip ON: flip all data AND de-flip axis coordinate values (double inversion = normal coords)
        # Flip OFF: flip only axis coordinate values (projection flips image, so compensate axis labels)
        if self.flip_image_plane_var.get():
            # Flip the data and de-flip axes = normal coordinate values
            ax.set_xlim([-max_dim, max_dim])
            ax.set_ylim([-max_dim, max_dim])
            ax.invert_xaxis()
            ax.invert_yaxis()
        else:
            # Flip only axis coordinate values - projection data stays as-is but axis labels are inverted
            ax.set_xlim([max_dim, -max_dim])
            ax.set_ylim([max_dim, -max_dim])
        ax.set_xlabel('World X [mm] (projected)', fontsize=14)
        ax.set_ylabel('World Y [mm] (projected)', fontsize=14)
        ax.set_title(f"Image Plane (Camera) - Top View (Tile: {data['aspect_ratio_used']}, User: {data['theta_deg']:.1f}°, Optimal: {data['optimal_angle']:.1f}°)", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    def _compute_fov_polygon(self, params, nx=1, ny=1):
        """Back-project ONE TILE's sensor rectangle corners to water plane (Z=0) to form FOV polygon in world mm.
        Each tile has its own lens, so projection view always shows a single tile's FOV."""
        A = float(params.get('A', 133))
        B = float(params.get('B', 317.5))
        C = float(params.get('C', 266.7))
        margin_percent = float(params.get('Margin', 10.0))
        tilt_deg = float(params.get('Tilt', 30))
        theta = np.deg2rad(tilt_deg)
        shift = float(params.get('Shift', 0))
        shift_axis = params.get('ShiftAxis','X')
        
        # Apply margin factor to match projection_calculations.py
        margin_factor = 1.0 + (margin_percent / 100.0)
        B = B * margin_factor
        C = C * margin_factor
        pixel_pitch_mm = float(params.get('PixelPitch', 2.0))/1000.0
        
        # Get FULL sensor dimensions first
        total_px_x = int(params.get('SensorPixelsX', 1920))
        total_px_y = int(params.get('SensorPixelsY', 1080))
        Sw_full = pixel_pitch_mm * total_px_x
        Sh_full = pixel_pitch_mm * total_px_y
        
        # Apply Image Circle constraint to effective full sensor
        img_circle_mm = max(0.0, float(params.get('ImageCircle', 0.0)))
        if img_circle_mm > 0.0:
            Sw_effective = min(Sw_full, img_circle_mm)
            Sh_effective = min(Sh_full, img_circle_mm)
        else:
            Sw_effective = Sw_full
            Sh_effective = Sh_full
        
        # Now compute PER-TILE dimensions (each tile has its own lens!)
        # For 1x1: tile = full sensor; for 2x2: tile = 1/4 of sensor (minus dead zones)
        deadzone_mm = float(params.get('Dead zone [mm]:', 0.0))
        deadzone_px = int(round(deadzone_mm / pixel_pitch_mm)) if pixel_pitch_mm > 0 else 0
        
        eff_px_x = int(Sw_effective / pixel_pitch_mm)
        eff_px_y = int(Sh_effective / pixel_pitch_mm)
        usable_px_x = max(eff_px_x - (nx - 1) * deadzone_px, 0)
        usable_px_y = max(eff_px_y - (ny - 1) * deadzone_px, 0)
        tile_px_x = usable_px_x // nx if nx > 0 else usable_px_x
        tile_px_y = usable_px_y // ny if ny > 0 else usable_px_y
        
        # TILE dimensions in mm (this is what one lens sees!)
        Sw = pixel_pitch_mm * tile_px_x
        Sh = pixel_pitch_mm * tile_px_y
        
        f = max(float(params.get('FocalLength', 6.0)), 1e-6)
        # Camera center
        if shift_axis=='X':
            Xc = C/2 + shift
            Yc = 0.0
            # Rotate around Y
            Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0,1,0], [-np.sin(theta),0,np.cos(theta)]])
            z_cam = Ry @ np.array([0,0,-1])
        else:
            Xc = 0.0
            Yc = B/2 + shift
            # Rotate around X - negate theta for Y-axis to match projection_calculations.py
            theta_corrected = -theta
            Rx = np.array([[1,0,0],[0,np.cos(theta_corrected),-np.sin(theta_corrected)],[0,np.sin(theta_corrected),np.cos(theta_corrected)]])
            z_cam = Rx @ np.array([0,0,-1])
        z_cam = z_cam/np.linalg.norm(z_cam)
        up_guess = np.array([0,1,0]) if abs(np.dot(z_cam,[0,1,0]))<0.99 else np.array([1,0,0])
        x_cam = np.cross(up_guess, z_cam); x_cam = x_cam/np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam); y_cam = y_cam/np.linalg.norm(y_cam)
        Cw = np.array([Xc, Yc, A])
        # Determine shape to back-project: circle if Image Circle is limiting, else rectangle
        if img_circle_mm > 0.0:
            sensor_diagonal = np.sqrt(Sw_full**2 + Sh_full**2)
            if img_circle_mm < sensor_diagonal:
                # Image Circle is limiting - back-project circle points
                radius = img_circle_mm / 2.0
                n_points = 64  # Circle resolution
                angles = np.linspace(0, 2*np.pi, n_points, endpoint=True)
                circle_points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
            else:
                # Sensor is limiting - back-project rectangle corners
                circle_points = np.array([[-Sw/2,-Sh/2],[Sw/2,-Sh/2],[Sw/2,Sh/2],[-Sw/2,Sh/2]])
        else:
            # No Image Circle - back-project rectangle corners
            circle_points = np.array([[-Sw/2,-Sh/2],[Sw/2,-Sh/2],[Sw/2,Sh/2],[-Sw/2,Sh/2]])
        
        # Back-project to water plane
        poly = []
        for x_img_mm, y_img_mm in circle_points:
            r_cam = np.array([x_img_mm, y_img_mm, f])
            r_cam = r_cam/np.linalg.norm(r_cam)
            Rw = x_cam*r_cam[0] + y_cam*r_cam[1] + z_cam*r_cam[2]
            if abs(Rw[2])<1e-9:
                t=0
            else:
                t = -Cw[2]/Rw[2]
            Pw = Cw + t*Rw
            poly.append([Pw[0], Pw[1]])
        return poly

    def _compute_tile_fov_polygon(self, params, nx=1, ny=1, tile_ix=0, tile_iy=0, pixels_x_per_tile=None, pixels_y_per_tile=None):
        """Compute FOV polygon for a single sensor tile back-projected to water plane.
        If 1x1, returns full sensor polygon. Tiles are arranged across the sensor surface.
        """
        A = float(params.get('A', 133))
        B = float(params.get('B', 317.5))
        C = float(params.get('C', 266.7))
        tilt_deg = float(params.get('Tilt', 30))
        theta = np.deg2rad(tilt_deg)
        shift = float(params.get('Shift', 0))
        shift_axis = params.get('ShiftAxis','X')
        pixel_pitch_mm = float(params.get('PixelPitch', 2.0))/1000.0
        total_px_x = int(params.get('SensorPixelsX', 1920))
        total_px_y = int(params.get('SensorPixelsY', 1080))
        
        # Apply Image Circle constraint to effective dimensions
        total_w_full = pixel_pitch_mm * total_px_x
        total_h_full = pixel_pitch_mm * total_px_y
        img_circle_mm = max(0.0, float(params.get('ImageCircle', 0.0)))
        if img_circle_mm > 0.0:
            total_w = min(total_w_full, img_circle_mm)
            total_h = min(total_h_full, img_circle_mm)
        else:
            total_w = total_w_full
            total_h = total_h_full
        
        # Use provided per-tile pixels when available
        if pixels_x_per_tile is None:
            # Compute effective pixels and split by tiles
            eff_px_x = int(total_w / pixel_pitch_mm)
            pixels_x_per_tile = max(eff_px_x // max(nx,1), 1)
        if pixels_y_per_tile is None:
            eff_px_y = int(total_h / pixel_pitch_mm)
            pixels_y_per_tile = max(eff_px_y // max(ny,1), 1)
        tile_w = pixel_pitch_mm * pixels_x_per_tile
        tile_h = pixel_pitch_mm * pixels_y_per_tile
        # Tile indices: 0..nx-1, 0..ny-1
        ix = int(tile_ix)
        iy = int(tile_iy)
        # Center of the tile in sensor mm
        cx = -total_w/2 + (ix + 0.5) * (total_w / max(nx,1))
        cy = -total_h/2 + (iy + 0.5) * (total_h / max(ny,1))
        f = max(float(params.get('FocalLength', 6.0)), 1e-6)
        # Camera center and orientation
        if shift_axis=='X':
            Xc = C/2 + shift
            Yc = 0.0
            Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0,1,0], [-np.sin(theta),0,np.cos(theta)]])
            z_cam = Ry @ np.array([0,0,-1])
        else:
            Xc = 0.0
            Yc = B/2 + shift
            # Rotate around X - negate theta for Y-axis to match projection_calculations.py
            theta_corrected = -theta
            Rx = np.array([[1,0,0],[0,np.cos(theta_corrected),-np.sin(theta_corrected)],[0,np.sin(theta_corrected),np.cos(theta_corrected)]])
            z_cam = Rx @ np.array([0,0,-1])
        z_cam = z_cam/np.linalg.norm(z_cam)
        up_guess = np.array([0,1,0]) if abs(np.dot(z_cam,[0,1,0]))<0.99 else np.array([1,0,0])
        x_cam = np.cross(up_guess, z_cam); x_cam = x_cam/np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam); y_cam = y_cam/np.linalg.norm(y_cam)
        Cw = np.array([Xc, Yc, A])
        # Tile corners in sensor mm relative to sensor center, offset by tile center
        corners = np.array([[-tile_w/2,-tile_h/2],[tile_w/2,-tile_h/2],[tile_w/2,tile_h/2],[-tile_w/2,tile_h/2]])
        corners += np.array([cx, cy])
        poly = []
        for x_img_mm,y_img_mm in corners:
            r_cam = np.array([x_img_mm, y_img_mm, f])
            r_cam = r_cam/np.linalg.norm(r_cam)
            Rw = x_cam*r_cam[0] + y_cam*r_cam[1] + z_cam*r_cam[2]
            t = -Cw[2]/Rw[2] if abs(Rw[2])>=1e-9 else 0.0
            Pw = Cw + t*Rw
            poly.append([Pw[0], Pw[1]])
        return poly
    def _estimate_fov_coverage(self, params, fov_poly):
        """Estimate coverage percent as area(FOV∩water)/area(water) by sampling the water rectangle."""
        B = float(params.get('B', 317.5))
        Cw = float(params.get('C', 266.7))
        margin = float(params.get('Margin', 10))
        W = Cw*(1+margin/100.0)
        L = B*(1+margin/100.0)
        x_min, x_max = -W/2, W/2
        y_min, y_max = -L/2, L/2
        nx, ny = 120, 120
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        area_cell = ((x_max-x_min)/(nx-1))*((y_max-y_min)/(ny-1))
        inside = 0
        total = nx*ny
        def point_in_poly(x,y,poly):
            wn=0
            for i in range(len(poly)):
                x1,y1 = poly[i]
                x2,y2 = poly[(i+1)%len(poly)]
                if y1<=y:
                    if y2>y and (x2-x1)*(y-y1)-(x-x1)*(y2-y1)>0:
                        wn+=1
                else:
                    if y2<=y and (x2-x1)*(y-y1)-(x-x1)*(y2-y1)<0:
                        wn-=1
            return wn!=0
        for xi in xs:
            for yi in ys:
                if point_in_poly(xi, yi, fov_poly):
                    inside+=1
        water_area = (x_max-x_min)*(y_max-y_min)
        est_inside_area = inside*area_cell
        return max(min(est_inside_area/water_area*100.0, 100.0), 0.0)


    def plot_side_view(self, data):
        ax = self.axes['side']
        if np.isnan(data.get('camera_center_x', np.nan)) or np.isnan(data.get('camera_center_z', np.nan)):
            ax.clear()
            ax.text(0.5, 0.5, "Impossible IFOV or parameter settings.\nPlease loosen requirements.",
                    fontsize=14, color='red', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return
        ax.set_aspect('equal')
        toilet_width = 50
        toilet_bottom = -20
        color = 'green' if data['shift_axis'] == 'X' else 'blue'
        if data['shift_axis'] == 'X':
            ax.add_patch(Rectangle((-data['W']/2, toilet_bottom), data['W'], toilet_width,
                color='lightgray', alpha=0.5, label='Toilet Bowl (side)'))
            ax.plot([-data['C']/2, data['C']/2], [0, 0], 'r-', lw=3, label='Water Surface')
        else:
            ax.add_patch(Rectangle((-data['L']/2, toilet_bottom), data['L'], toilet_width,
                color='lightgray', alpha=0.5, label='Toilet Bowl (side)'))
            ax.plot([-data['L']/2, data['L']/2], [0, 0], 'r-', lw=3, label='Water Surface')
        cam_width = 15
        sensor_half_length = cam_width / 2
        sensor_start = np.array([data['camera_center_x'], data['camera_center_z']]) - sensor_half_length * data['sensor_tangent_2d']
        sensor_end = np.array([data['camera_center_x'], data['camera_center_z']]) + sensor_half_length * data['sensor_tangent_2d']
        ax.plot([sensor_start[0], sensor_end[0]], [sensor_start[1], sensor_end[1]],
                color=color, linewidth=4, label='Camera Sensor Plane')
        purple_arrow_length = data['camera_center_z'] * 1.1
        ax.arrow(data['camera_center_x'], 0, 0, purple_arrow_length,
            head_width=8, head_length=8, fc='purple', ec='purple', linewidth=2, label='Water Normal Vector')
        axis_length = 80
        ax.arrow(data['camera_center_x'], data['camera_center_z'],
                 axis_length * data['optical_axis_normalized'][0], axis_length * data['optical_axis_normalized'][1],
                 head_width=8, head_length=8, fc='orange', ec='orange', linewidth=3,
                 label='Camera Optical Axis (⊥ to sensor)')
        ax.arrow(data['camera_center_x'], data['camera_center_z'],
                 axis_length * data['optimal_axis_2d'][0] * 0.8, axis_length * data['optimal_axis_2d'][1] * 0.8,
                 head_width=6, head_length=6, fc='cyan', ec='cyan', linewidth=2, linestyle='--',
                 label=f"Optimal Tilt Angle ({data['optimal_angle']:.1f}°)")
        if data['theta_deg'] != 0:
            arc_radius = 25
            theta1 = data['theta_deg'] if data['theta_deg'] < 0 else 0
            theta2 = 0 if data['theta_deg'] < 0 else data['theta_deg']
            arc = Arc((data['camera_center_x'], data['camera_center_z']), 2 * arc_radius, 2 * arc_radius,
                      angle=0, theta1=theta1, theta2=theta2, color='red', linewidth=3)
            ax.add_patch(arc)
            mid_angle = (theta1 + theta2) / 2
            text_x = data['camera_center_x'] + (arc_radius * 0.6) * np.cos(np.radians(mid_angle))
            text_z = data['camera_center_z'] + (arc_radius * 0.6) * np.sin(np.radians(mid_angle))
            ax.text(text_x, text_z, f"{data['theta_deg']:.0f}",
                    fontsize=12, color='red', weight='bold', ha='center', va='center')
        ax.text(0.68, 0.98, f"OPTIMAL TILT ANGLE: {data['optimal_angle']:.1f}°",
                transform=ax.transAxes, fontsize=11, color='black', weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9),
                verticalalignment='top')
        for wp_x, wp_z in [(-data['C']/2, 0), (data['C']/2, 0), (0, 0)]:
            if data['shift_axis'] == 'Y':
                wp_x = wp_z
                wp_z = 0  # Initialize wp_z to 0 for Y-axis mode
            ax.plot([data['camera_center_x'], wp_x], [data['camera_center_z'], wp_z],
                    color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_xlim(-300, 300)
        ax.set_ylim(-50, 200)
        ax.set_xlabel(f'{data["shift_axis"]} [mm] (side view)', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.set_title(f'Side View ({data["shift_axis"]}-Z plane) - User Input Angle + Optimal', fontsize=GRAPH_TITLE_FONTSIZE)
        ax.set_ylabel('Z [mm] (height)', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.tick_params(axis='both', labelsize=GRAPH_TICK_FONTSIZE)
        ax.legend(loc='upper left', fontsize=GRAPH_LEGEND_FONTSIZE)
        ax.grid(True, alpha=0.3)
        

    def show_help(self):
        """Show comprehensive help information in a new window"""
        help_window = tk.Toplevel(self.root)
        help_window.title("Sensor Simulation Help")
        help_window.geometry("1000x700")  # Made window larger for better readability

        # Create a frame with scrollbar
        main_frame = ttk.Frame(help_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)  # Increased padding

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        # Create fonts for different text elements
        title_font = Font(family="Arial", size=16, weight="bold")
        section_font = Font(family="Arial", size=14, weight="bold")
        content_font = Font(family="Arial", size=12)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Add main title
        main_title = tk.Label(scrollable_frame, text="Sensor Simulation Help Guide", font=title_font)
        main_title.pack(pady=(0, 20))

        # Help content sections
        help_sections = [
            ("Overview", """
The Sensor Simulation tool is designed for analyzing and optimizing camera sensor configurations 
for toilet bowl monitoring applications. It provides real-time visualization and calculations
for various sensor parameters and their effects on image quality and coverage.
            """),
            
            ("Main Features", """
• Interactive parameter adjustment with real-time updates
• Multi-view visualization (Top-down, Projected, Side, Coverage)
• Support for multiple camera configurations (1x1, 1x2, 2x1, 2x2)
• IFOV (Instantaneous Field of View) analysis and visualization
• Comprehensive sensor resolution calculations
• Database management for different toilet configurations
            """),
            
            ("Parameters Explanation", """
A - Rim to Water depth: Distance from the water surface to the camera mounting position
B - Water Spot Length: Length of the water surface area
C - Water Spot Width: Width of the water surface area
Camera Tilt: Angle of the camera relative to the vertical axis
Margin: Additional area percentage around the water spot
Resolution: Target resolution in mm/pixel (IFOV mode only)
Dead zone: Gap between sensor tiles in multi-sensor configurations
Pixel pitch: Physical size of sensor pixels in micrometers
Focal Length: Lens focal length in millimeters (FOV mode only)
Sensor Resolution: Physical pixel dimensions of the sensor (FOV mode only)
Image Circle: Diameter of the lens's usable image circle in millimeters
  • Limits the effective sensor area that receives light from the lens
  • If smaller than sensor diagonal, outer pixels will be dark/vignetted
  • For optimal sensor utilization, Image Circle ≥ sensor diagonal
            """),
            
            ("Calculations in Detail", """
1. IFOV Calculations:
   • Basic IFOV = pixel_pitch × (working_distance / focal_length)
   • Perspective-corrected IFOV considers camera tilt:
     - Minimum IFOV at closest point = base_IFOV / distance_ratio
     - Maximum IFOV at farthest point = base_IFOV × distance_ratio
     where distance_ratio = max_distance / min_distance

2. Resolution Calculations:
   • Naive resolution = area_dimension / target_IFOV
   • Required resolution includes perspective effects:
     required_pixels = naive_pixels × (max_IFOV / target_IFOV)

3. Multi-sensor Considerations:
   • Total pixels = n × pixels_per_tile + (n-1) × deadzone_pixels
   • Physical sensor size = pixel_pitch × total_pixels
            """),
            
            ("Visualization Guide", """
Top-Down View:
• Blue rectangle: Water spot boundary
• Red ellipse: Water coverage area
• Green/Blue square: Camera position

Projected View:
• Color-coded points showing IFOV distribution
• Blue points: Meeting resolution requirement
• Red points: Exceeding resolution requirement
• Black circle: Optical limitation boundary

Side View:
• Camera sensor plane and optical axis
• Optimal tilt angle indication
• Water surface representation

Coverage View:
• Water coverage percentage vs. tilt angle
• Optimal angle indication
• Zoom controls for detailed analysis
            """),
            
            ("Understanding Output Metrics", """
1. Active Pixel Fraction (by Image Circle):
   • Measures: Percentage of sensor area that receives light from the lens
   • Limited by: Lens optics (Image Circle diameter)
   • Formula: (Image Circle Area) / (Sensor Area) × 100%
   • Impact: Hardware/optical limitation - pixels outside the circle are unused
   • Good value: ~100% (lens fully illuminates sensor)
   • Example: 1.4mm Image Circle on 5.4mm sensor = 7% (93% wasted pixels!)

2. Water Coverage (from graph):
   • Measures: Percentage of water spot visible within camera FOV
   • Limited by: Camera geometry (position, tilt, height)
   • Calculation: Camera FOV footprint vs. water rectangle dimensions
   • Impact: Geometric/positioning issue - affects target visibility
   • Good value: >80% (most of target area visible)
   • Can be improved by: Adjusting camera placement or tilt angle

3. Key Difference:
   Active Pixel Fraction = Lens-to-Sensor fit (optical hardware match)
   Water Coverage = Camera-to-Scene geometry (positioning/alignment)
   Both can limit system performance independently!
            """),
            
            ("Advanced Features", """
1. Dual Mode Operation:
   • IFOV Mode: Design sensor based on required resolution (mm/pixel)
   • FOV Mode: Analyze existing sensor/lens combination performance

2. IFOV Enforcement:
   • Toggle visualization of resolution compliance
   • Real-time feedback on sensor coverage

3. Camera Setup Options:
   • Single sensor (1x1)
   • Dual sensor (1x2, 2x1)
   • Quad sensor (2x2)
   • Automatic dead zone handling

4. Database Management:
   • Save and load toilet configurations
   • Compare different setups
   • Export data to CSV
            """)]

        # Add sections to scrollable frame
        for title, content in help_sections:
            section_frame = ttk.Frame(scrollable_frame)
            section_frame.pack(fill=tk.X, pady=(0, 25))  # Increased spacing between sections
            
            # Section title with larger font
            tk.Label(section_frame, text=title, font=section_font).pack(anchor="w", pady=(0, 10))
            
            # Content with larger font and better spacing
            tk.Label(section_frame, text=content, 
                    wraplength=900,  # Increased wrap length for better text flow
                    justify=tk.LEFT,
                    font=content_font,
                    padx=20  # Add some indentation to content
                    ).pack(anchor="w")

        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True, padx=(0, 10))  # Added padding between canvas and scrollbar
        scrollbar.pack(side="right", fill="y")

    def plot_coverage_view(self, data):
        ax = self.axes['coverage']
        try:
            if len(data.get('coverage_angles', [])) == 0 or len(data.get('coverage_values', [])) == 0:
                ax.clear()
                ax.text(0.5, 0.5, "Impossible IFOV or parameter settings.\nPlease loosen requirements.",
                        fontsize=14, color='red', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
                return

            # Calculate the real max coverage for this curve
            if len(data['coverage_values']):
                self._max_coverage_y = max(data['coverage_values'])
            else:
                self._max_coverage_y = 100  # fallback

            ax.plot(data['coverage_angles'], data['coverage_values'], 'b-', linewidth=2, label='Water Coverage')
            ax.scatter(data['optimal_angle'], data['optimal_coverage'], color='red', s=100, zorder=5,
                       label=f'Optimal ({data["optimal_angle"]:.1f}°, {data["optimal_coverage"]:.1f}%)')
            ax.axvline(x=data['optimal_angle'], color='red', linestyle='--', alpha=0.7)
            ax.axhline(y=data['optimal_coverage'], color='red', linestyle='--', alpha=0.7)
            if 0 <= abs(data['theta_deg']) <= 60:
                current_coverage = data['water_coverage_percent']
                ax.scatter(abs(data['theta_deg']), current_coverage, color='orange', s=80, zorder=5,
                           marker='s', label=f'Current ({abs(data["theta_deg"]):.1f}°, {current_coverage:.1f}%)')

            max_angle = max(data['coverage_angles']) if len(data['coverage_angles']) else 60
            ymax = max(data['coverage_values']) if len(data['coverage_values']) else 90
            zoom = self.coverage_zoom
            xmid = (self.coverage_xlim[0] + self.coverage_xlim[1]) / 2
            ymid = (self.coverage_ylim[0] + self.coverage_ylim[1]) / 2
            xhalf = (self.coverage_xlim[1] - self.coverage_xlim[0]) / 2 / zoom
            yhalf = (self.coverage_ylim[1] - self.coverage_ylim[0]) / 2 / zoom

            # Always keep the highest coverage visible when zooming
            OFFSET = 1.0  # or set to any % offset you want
            proposed_ymax = ymid + yhalf
            min_ymax = getattr(self, "_max_coverage_y", 100) + OFFSET
            if proposed_ymax < min_ymax:
                proposed_ymax = min_ymax
                # prevent ymid-yhalf going over max
                proposed_ymin = max(proposed_ymax - 2*yhalf, 0)
            else:
                proposed_ymin = max(ymid - yhalf, 0)

            ax.set_xlim(xmid - xhalf, xmid + xhalf)
            ax.set_ylim(proposed_ymin, proposed_ymax)

            ax.set_xlabel('Tilt Angle [degrees]', fontsize=GRAPH_LABEL_FONTSIZE)
            ax.set_ylabel('Water Coverage [%]', fontsize=GRAPH_LABEL_FONTSIZE)
            ax.set_title('Water Coverage vs. Tilt Angle', fontsize=GRAPH_TITLE_FONTSIZE)
            ax.tick_params(axis='both', labelsize=GRAPH_TICK_FONTSIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=GRAPH_LEGEND_FONTSIZE)
        except Exception:
            ax.clear()
            ax.set_xlim(0, 60)
            ax.set_ylim(5, 90)
            ax.set_xlabel('Tilt Angle [degrees]', fontsize=GRAPH_LABEL_FONTSIZE)
            ax.set_ylabel('Water Coverage [%]', fontsize=GRAPH_LABEL_FONTSIZE)
            ax.set_title('Water Coverage vs. Tilt Angle', fontsize=GRAPH_TITLE_FONTSIZE)
            ax.tick_params(axis='both', labelsize=GRAPH_TICK_FONTSIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=GRAPH_LEGEND_FONTSIZE)
        

    def refresh_table(self):
        try:
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Insert new data
            if hasattr(self.data_manager, 'data') and not self.data_manager.data.empty:
                for index, row in self.data_manager.data.iterrows():
                    values = [str(val) for val in row]  # Convert all values to strings
                    self.tree.insert('', 'end', values=tuple(values))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh table: {str(e)}")
            logging.error(f"Table refresh error: {str(e)}")

    def add_toilet(self):
        params = self.get_current_parameters()
        if params is None:
            return
            
        toilet_params = {
            'Manufacturer': "DEFAULT",
            'Model': "DEFAULT",
            'Sub-Model': "DEFAULT",
            'A - Rim to Water depth (camera height) [mm]': params['A'],
            'B - Water Spot Length [mm]': params['B'],
            'C - Water Spot Width [mm]': params['C'],
            'Camera Tilt [degrees]': params['Tilt'],
            'Margin [%]': params['Margin'],
            'Shift from Water Spot Width Edge [mm]': params['Shift'],
            'Shift Axis': params['ShiftAxis'],
            'Dead Zone [mm]': params.get('DeadZone',''),
            'Required Resolution [mm/px]': params['Resolution'],
            'Pixel Pitch [um]': params['PixelPitch'],
            'Focal Length [mm]': params.get('FocalLength', ''),
            'Sensor Resolution [px×px]': f"{params.get('SensorPixelsX','')}x{params.get('SensorPixelsY','')}" if params.get('Mode')=='FOV' else '',
            'Image Circle [mm]': params.get('ImageCircle',''),
        }
        if self.is_duplicate_record(toilet_params):
            messagebox.showwarning("Duplicate Entry", "These parameters already exist in the database. "
                                                      "Please modify at least one parameter before adding.")
            return
        self.data_manager.add_toilet(toilet_params)
        self.refresh_table()
        messagebox.showinfo("Success", "Parameters added successfully!")

    def is_duplicate_record(self, new_params):
        if self.data_manager.data.empty:
            return False  # If no data exists, it can't be a duplicate
            
        technical_columns = [
            'A - Rim to Water depth (camera height) [mm]',
            'B - Water Spot Length [mm]',
            'C - Water Spot Width [mm]',
            'Camera Tilt [degrees]',
            'Margin [%]',
            'Shift from Water Spot Width Edge [mm]',
            'Shift Axis',
            'Dead Zone [mm]',
            'Required Resolution [mm/px]',
            'Pixel Pitch [um]',
            'Focal Length [mm]',
            'Sensor Resolution [px×px]',
            'Image Circle [mm]'
        ]
        
        for index, row in self.data_manager.data.iterrows():
            is_duplicate = True
            for col in technical_columns:
                if col in self.data_manager.data.columns:
                    existing_value = str(row[col]).strip()
                    new_value = str(new_params[col]).strip()
                    try:
                        existing_float = float(existing_value)
                        new_float = float(new_value)
                        if abs(existing_float - new_float) > 0.001:
                            is_duplicate = False
                            break
                    except ValueError:
                        if existing_value != new_value:
                            is_duplicate = False
                            break
            if is_duplicate:
                return True
        return False

    def load_selected_toilet(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a toilet from the table")
            return
        item = self.tree.item(selection[0])
        values = item['values']
        if len(values) >= 12:
            cols = list(self.tree['columns'])
            def set_ifov(key, col_name):
                if col_name in cols:
                    idx = cols.index(col_name)
                    self.entries[key].delete(0, tk.END)
                    self.entries[key].insert(0, str(values[idx]))
            def set_fov(label_text, col_name):
                if hasattr(self, 'fov_entries') and label_text in self.fov_entries and col_name in cols:
                    idx = cols.index(col_name)
                    e = self.fov_entries[label_text]
                    e.delete(0, tk.END)
                    e.insert(0, str(values[idx]))
            # IFOV tab fields
            set_ifov('A', 'A - Rim to Water depth (camera height) [mm]')
            set_ifov('B', 'B - Water Spot Length [mm]')
            set_ifov('C', 'C - Water Spot Width [mm]')
            set_ifov('Tilt', 'Camera Tilt [degrees]')
            set_ifov('Margin', 'Margin [%]')
            set_ifov('Shift', 'Shift from Water Spot Width Edge [mm]')
            if 'Shift Axis' in cols:
                self.shift_axis_var.set(str(values[cols.index('Shift Axis')]))
            set_ifov('Resolution', 'Required Resolution [mm/px]')
            set_ifov('PixelPitch', 'Pixel Pitch [um]')
            if 'Dead Zone [mm]' in cols:
                dz_val = str(values[cols.index('Dead Zone [mm]')])
                self.entries['DeadZone'].delete(0, tk.END)
                self.entries['DeadZone'].insert(0, dz_val)
            # FOV tab fields mirror
            set_fov('A - Rim to Water depth (camera height) [mm]:', 'A - Rim to Water depth (camera height) [mm]')
            set_fov('B - Water Spot Length [mm]:', 'B - Water Spot Length [mm]')
            set_fov('C - Water Spot Width [mm]:', 'C - Water Spot Width [mm]')
            set_fov('Camera Tilt [degrees]:', 'Camera Tilt [degrees]')
            set_fov('Margin [%]:', 'Margin [%]')
            set_fov('Shift from Water Spot Width Edge [mm]:', 'Shift from Water Spot Width Edge [mm]')
            set_fov('Dead zone [mm]:', 'Dead Zone [mm]')
            set_fov('Pixel pitch [um]:', 'Pixel Pitch [um]')
            # Image Circle
            if 'Image Circle [mm]' in cols:
                idx = cols.index('Image Circle [mm]')
                self.image_circle_entry_ifov.delete(0, tk.END)
                self.image_circle_entry_ifov.insert(0, str(values[idx]))
                self.image_circle_entry_fov.delete(0, tk.END)
                self.image_circle_entry_fov.insert(0, str(values[idx]))
            # FOV-specific fields
            if 'Focal Length [mm]' in cols:
                idx = cols.index('Focal Length [mm]')
                self.focal_length_entry.delete(0, tk.END)
                self.focal_length_entry.insert(0, str(values[idx]))
            if 'Sensor Resolution [px×px]' in cols:
                idx = cols.index('Sensor Resolution [px×px]')
                self.sensor_res_entry.delete(0, tk.END)
                self.sensor_res_entry.insert(0, str(values[idx]))
            self.plot_projection()

    def delete_selected_toilet(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a toilet to delete")
            return
        if messagebox.askyesno("Confirm", "Are you sure you want to delete the selected toilet?"):
            item_index = self.tree.index(selection[0])
            self.data_manager.delete_toilet(item_index)
            self.refresh_table()

    def edit_cell(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        rowid = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        col_index = int(column.replace('#', '')) - 1
        x, y, width, height = self.tree.bbox(rowid, column)
        value = self.tree.set(rowid, column)
        entry = tk.Entry(self.tree, font=self.table_font)
        entry.insert(0, value)
        entry.place(x=x, y=y, width=width, height=height)
        entry.focus()
        def on_enter(event):
            new_value = entry.get()
            self.tree.set(rowid, column, new_value)
            item_index = self.tree.index(rowid)
            col_name = self.tree["columns"][col_index]
            self.data_manager.update_cell(item_index, col_name, new_value)
            self.data_manager.save_data()
            entry.destroy()
        entry.bind('<Return>', on_enter)
        entry.bind('<Escape>', lambda e: entry.destroy())
        
        def handle_focus_out(event):
            # Give a small delay before destroying to allow for Return event
            entry.after(100, entry.destroy)
        
        entry.bind('<FocusOut>', handle_focus_out)

import psutil

LOCKFILE = "app.lock"

def ensure_single_instance():
    if os.path.exists(LOCKFILE):
        try:
            with open(LOCKFILE, "r") as lock:
                pid = int(lock.read().strip())
            if pid == os.getpid():
                # Current process, continue
                pass
            elif psutil.pid_exists(pid):
                # Be forgiving in dev: warn but continue by replacing lockfile
                logging.warning("Another instance appears running (pid=%s). Continuing by replacing lockfile.", pid)
                try:
                    os.remove(LOCKFILE)
                except Exception:
                    pass
            else:
                # Stale lockfile (process not running), remove it
                os.remove(LOCKFILE)
        except Exception:
            # If lockfile is corrupt, remove it anyway
            os.remove(LOCKFILE)
    # Create fresh lockfile
    with open(LOCKFILE, "w") as lock:
        lock.write(str(os.getpid()))


def cleanup_lockfile():
    if os.path.exists(LOCKFILE):
        try:
            os.remove(LOCKFILE)
            logging.debug("Lockfile removed.")
        except Exception as e:
            logging.error(f"Failed to remove lockfile: {e}")

if __name__ == "__main__":
    ensure_single_instance()
    root = tk.Tk()
    app = ProjectionApp(root)
    # Maximize window after everything is initialized
    root.update_idletasks()
    root.state('zoomed')
    try:
        root.mainloop()
    finally:
        cleanup_lockfile()
