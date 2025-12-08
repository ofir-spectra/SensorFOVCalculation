import os
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
from image_utils import find_image_case_insensitive

APP_VERSION = "V5.0.10"

GRAPH_TITLE_FONTSIZE = 14
GRAPH_LABEL_FONTSIZE = 14
GRAPH_TICK_FONTSIZE = 12
GRAPH_LEGEND_FONTSIZE = 10
GRAPH_OVERLAY_FONTSIZE = 12

CAMERA_SETUP_OPTIONS = ["1x1", "1x2", "2x2", "2x3"]

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
        self.root.title(f"SENSOR SIMULATION {APP_VERSION}")
        self.root.geometry("1900x1000")

        title_label = tk.Label(root, text=f"SENSOR SIMULATION {APP_VERSION}", font=("Arial", 20, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(10, 0))

        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "toilet_data.csv")

        self.data_manager = ToiletDataManager(csv_path)
        self.param_font = Font(family="Arial", size=14)
        self.table_font = Font(family="Arial", size=14)
        self.heading_font = Font(family="Arial", size=14, weight="bold")
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
        self.setup_params_frame()
        self.setup_plot_frame()
        self.setup_table_frame()
        self.refresh_table()
        self.plot_projection()

    def setup_results_frame(self):
        self.results_frame = ttk.LabelFrame(self.root, text="Simulation Results", padding="10")
        self.results_frame.grid(row=1, column=0, padx=10, pady=(10, 0), sticky="nsew")
        self.results_table = ttk.Treeview(
            self.results_frame,
            columns=("Parameter", "Value", "Unit"),
            show='headings',
            height=10
        )
        for col in ("Parameter", "Value", "Unit"):
            self.results_table.heading(col, text=col)
            self.results_table.column("Parameter", anchor="w", width=270)  # wider for text
            self.results_table.column("Value", anchor="center", width=140)
            self.results_table.column("Unit", anchor="center", width=70)   # narrower for 'mm', 'px'
            self.results_table.pack(fill=tk.BOTH, expand=True)

    def update_simulation_results(self, data):
        """
        Update the simulation results table with calculated values.
        Adds resolution per tile and dead band information for better clarity.
        """
        self.results_table.delete(*self.results_table.get_children())
        results = [
            ("Realistic Sensor Resolution", f"{data['pixels_x_sensor']} × {data['pixels_y_sensor']}", "px"),
            ("Resolution Per Tile", f"{data.get('pixels_x_per_tile', 0):.0f} × {data.get('pixels_y_per_tile', 0):.0f}", "px"),
            ("Dead Band (between tiles)", f"{data.get('deadzone_px', 0):.0f}", "px"),
            ("Naive Resolution", f"{data['pixels_x_naive']} × {data['pixels_y_naive']}", "px"),
            ("Realistic Sensor FOV", f"{data.get('FOV_H_sensor', 0):.2f} × {data.get('FOV_V_sensor', 0):.2f}", "deg"),
            ("Naive Sensor FOV", f"{data.get('FOV_H_naive', 0):.2f} × {data.get('FOV_V_naive', 0):.2f}", "deg"),
            ("Water Coverage", f"{data.get('water_coverage_percent', 0):.1f}", "%"),
            ("Optimal Tilt Angle", f"{data.get('optimal_angle', 0):.1f}", "deg"),
            ("Projection Offset", f"{data.get('projection_offset', 0):.1f}", "mm"),
            ("Sensor Aspect Ratio", data.get('aspect_ratio_used', ''), "-"),
            ("Optics Diameter", f"{data.get('optics_diameter', 0):.1f}", "mm"),
            ("Maximum Projected IFOV", f"{data.get('max_ifov', 0):.4f}", "mm"),
            ("Minimum Projected IFOV", f"{data.get('min_ifov', 0):.4f}", "mm"),
            ("Sensor Size [mm]", f"{data.get('sensor_width_mm', 0):.2f} × {data.get('sensor_height_mm', 0):.2f}", "mm"),
        ]
        for param, value, unit in results:
            self.results_table.insert("", tk.END, values=(param, value, unit))

    def setup_params_frame(self):
        param_frame = ttk.LabelFrame(self.root, text="Parameters", padding="10")
        param_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        for i, (label, default) in enumerate(self.param_labels):
            lbl = tk.Label(param_frame, text=label, font=self.param_font)
            lbl.grid(row=i, column=0, sticky="e", padx=5, pady=5)
            entry = tk.Entry(param_frame, font=self.param_font, width=12)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[self.param_keys[i]] = entry
            param = self.param_keys[i]
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
            elif param == "Resolution":
                tk.Button(param_frame, text="−", font=self.param_font, width=2,
                        command=lambda p=param: self.adjust_param(p, -0.01)).grid(row=i, column=2)
                tk.Button(param_frame, text="+", font=self.param_font, width=2,
                        command=lambda p=param: self.adjust_param(p, 0.01)).grid(row=i, column=3)
        next_row = len(self.param_labels)

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
            deadzone = max(0, safe_float(self.entries['DeadZone'].get(), 0.3))  # Ensure deadzone is non-negative
            self.entries['DeadZone'].delete(0, tk.END)
            self.entries['DeadZone'].insert(0, str(deadzone))  # Update the entry with validated value
            
            return {
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
            }
        except Exception:
            messagebox.showerror("Error", "Please enter valid numeric values for all parameters")
            return None

    def plot_projection(self):
        """
        Updated plot_projection method to correctly calculate sensor resolution
        for all camera setups (e.g., 1x2, 2x2), including dead zones.
        """
        params = self.get_current_parameters()
        if params is None:
            return

        # Parse Camera Setup (e.g., "2x2")
        cam_setup_str = params.get("CameraSetup", "1x1")
        nx, ny = parse_camera_setup(cam_setup_str)  # nx: horizontal tiles, ny: vertical tiles

        # Retrieve Dead Zone and Pixel Pitch
        deadzone_mm = params.get("DeadZone", 0.3)  # Dead zone between tiles in mm
        pixel_pitch_um = params.get("PixelPitch", 2.0)  # Pixel pitch in micrometers (um)

        # Calculate Dead Zone in Pixels
        deadzone_px = int((1000.0 * float(deadzone_mm)) / float(pixel_pitch_um))

        # Get plot data for a single tile (1x1 setup)
        plot_data = get_plot_data(params, smoothness=params.get('Smoothness', 2))

        # Step 1 & 2: Get initial IFOV values from projection calculations
        target_ifov = params.get("Resolution", 0.22)        # Target resolution/IFOV from user
        
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
        # Fix axes alignment - X should use width (C), Y should use length (B)
        naive_pixels_x = int(params['C'] / target_ifov)  # X corresponds to water spot width (C)
        naive_pixels_y = int(params['B'] / target_ifov)  # Y corresponds to water spot length (B)
        
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
        plot_data['scaling_ratio'] = scaling_ratio  # Store the final scaling ratio

        # Calculate max_radius from the projected rectangle points
        # This gives us the actual maximum distance in the projected view
        proj_rect_pts = plot_data.get('proj_rect_pts', None)
        if proj_rect_pts is not None:
            distances = np.sqrt(np.sum(proj_rect_pts**2, axis=1))  # Distance from optical center
            max_radius = np.max(distances)
        else:
            # Fallback to sensor diagonal if projection points aren't available
            max_radius = np.sqrt(sensor_width_mm**2 + sensor_height_mm**2) / 2
        plot_data['max_radius'] = max_radius

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
        ax.add_patch(Ellipse((0, 0), data['C'], data['L'], edgecolor='r', facecolor='r', alpha=0.3, lw=2, label='Water Spot Ellipse'))
        camera_size = 20
        color = 'green' if data['shift_axis'] == 'X' else 'blue'
        ax.add_patch(Rectangle((data['Xc'] - camera_size/2, data['Yc'] - camera_size/2), camera_size, camera_size,
                               color=color, alpha=0.7, label=f'Camera ({data["shift_axis"]}-axis)'))
        ax.plot([data['Xc']], [data['Yc']], 'go' if data['shift_axis'] == 'X' else 'bo')
        for point in data['fov_points_world']:
            ax.plot([data['Xc'], point[0]], [data['Yc'], point[1]], 'gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_xlim(-300, 300)
        ax.set_ylim(-300, 300)
        ax.set_xlabel('X [mm]', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.set_ylabel('Y [mm]', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.set_title('Top-Down View (World Coordinates)', fontsize=GRAPH_TITLE_FONTSIZE)
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

        # Get parameters needed for IFOV calculation
        target_ifov = data.get("Resolution", 0.22)
        height = data.get("A", 133)  # Camera height in mm
        tilt_rad = np.radians(data.get("Tilt", 30))
        # Calculate the reference distance (closest point to camera)
        min_distance = height * np.tan(tilt_rad) if tilt_rad > 0 else height

        if self.ifov_enforce_var.get():
            # Get our target resolution (what we want to achieve)
            resolution_mm = data.get("Resolution", 0.22)  # Target resolution in mm/pixel
            
            print("\nDEBUG VALUES:")
            print(f"Target resolution (mm/px): {resolution_mm}")
            print(f"Camera height (mm): {height}")
            print(f"Camera tilt (degrees): {np.degrees(tilt_rad):.1f}")
            print(f"Base IFOV at camera center (mm/px): {data['max_ifov']}")
            print("\nPoint Analysis (20 sample points):")
            print("Index  Position(x,y)mm    Dist(mm)  Angle(°)  IFOV(mm/px)  Status")
            print("-" * 75)
            
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
                
                # Calculate how far this point is from directly under the camera
                horizontal_distance = np.sqrt(point_x**2 + point_y**2)
                
                # Calculate viewing angle to this point
                point_angle = np.arctan2(horizontal_distance, height)
                
                # Calculate IFOV (resolution) at this point
                angle_factor = 1.0 / np.cos(point_angle)
                point_ifov = data['max_ifov'] * angle_factor**2
                
                # Print detailed debug info for sample points
                if i in sample_indices:
                    status = "RED" if point_ifov > resolution_mm else "BLUE"
                    print(f"{i:5d}  ({point_x:7.1f},{point_y:7.1f})  {horizontal_distance:8.1f}  {np.degrees(point_angle):7.1f}  {point_ifov:10.4f}  {status}")
                
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
            s=10,  # Increased dot size
            alpha=1.0,  # Full opacity
            label='Sensor Pixels (Blue/Red by IFOV)'
        )


      

        # --- Existing code for overlays and lines, unchanged ---
        ax.plot(data['proj_ellipse_pts'][:, 0], data['proj_ellipse_pts'][:, 1], 'r-', linewidth=2, label='Projected Ellipse')
        ax.plot(data['proj_rect_outline'][:, 0], data['proj_rect_outline'][:, 1], 'b-', linewidth=2, label='Projected Rectangle Outline')
        ax.plot(data['box_x_mm'], data['box_y_mm'], 'k--', linewidth=2, label=f"Realistic Sensor FOV ({data['aspect_ratio_used']})")
        ax.scatter(0, 0, color='red', s=200, marker='+', linewidth=3, label='Optical Axis (Principal Point)')
        ax.scatter(data['ellipse_cx'], data['ellipse_cy'], color='purple', s=100, marker='x', linewidth=2, label='Water Spot Center')
        ax.axhline(y=0, color='red', linestyle='-', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='red', linestyle='-', alpha=0.3, linewidth=1)
        from matplotlib.patches import Circle
        ax.add_patch(Circle((0, 0), data['optics_radius'], fill=False, color='black', lw=2, label=f"Optics Image Size (D={data['optics_diameter']:.1f} mm)"))

        ax.set_xlim([-500, 500])
        ax.set_ylim([-500, 500])
        ax.set_xlabel('World X [mm] (projected)', fontsize=14)
        ax.set_ylabel('World Y [mm] (projected)', fontsize=14)
        ax.set_title(f"Projected View - REALISTIC SENSOR {data['aspect_ratio_used']} (User: {data['theta_deg']:.1f}°, Optimal: {data['optimal_angle']:.1f}°)", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)


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
            'Required Resolution [mm/px]': params['Resolution'],
            'Pixel Pitch [um]': params['PixelPitch'],
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
            'Required Resolution [mm/px]',
            'Pixel Pitch [um]'
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
            self.entries['A'].delete(0, tk.END)
            self.entries['A'].insert(0, str(values[3]))
            self.entries['B'].delete(0, tk.END)
            self.entries['B'].insert(0, str(values[4]))
            self.entries['C'].delete(0, tk.END)
            self.entries['C'].insert(0, str(values[5]))
            self.entries['Tilt'].delete(0, tk.END)
            self.entries['Tilt'].insert(0, str(values[6]))
            self.entries['Margin'].delete(0, tk.END)
            self.entries['Margin'].insert(0, str(values[7]))
            self.entries['Shift'].delete(0, tk.END)
            self.entries['Shift'].insert(0, str(values[8]))
            self.shift_axis_var.set(str(values[9]))
            self.entries['Resolution'].delete(0, tk.END)
            self.entries['Resolution'].insert(0, str(values[10]))
            if len(values) > 11:
                self.entries['PixelPitch'].delete(0, tk.END)
                self.entries['PixelPitch'].insert(0, str(values[11]))
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
                logging.error("Another instance of the application is already running.")
                sys.exit(1)
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
    try:
        root.mainloop()
    finally:
        cleanup_lockfile()
