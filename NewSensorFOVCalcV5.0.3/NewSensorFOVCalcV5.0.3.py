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

from projection_calculations import get_plot_data
from data_manager import ToiletDataManager
from image_utils import find_image_case_insensitive

APP_VERSION = "V5.0.3"

GRAPH_TITLE_FONTSIZE = 14
GRAPH_LABEL_FONTSIZE = 14
GRAPH_TICK_FONTSIZE = 12
GRAPH_LEGEND_FONTSIZE = 10
GRAPH_OVERLAY_FONTSIZE = 12

CAMERA_SETUP_OPTIONS = ["1x1", "1x2", "2x2", "2x3"]

def parse_camera_setup(cam_setup_str):
    try:
        nx, ny = map(int, str(cam_setup_str).split('x'))
        return nx, ny
    except Exception:
        return 1, 1

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
            self.results_table.column(col, anchor="center", width=170)
        self.results_table.pack(fill=tk.BOTH, expand=True)

    def update_simulation_results(self, data):
        self.results_table.delete(*self.results_table.get_children())
        results = [
            ("Realistic Sensor Resolution", f"{data['pixels_x_sensor']} × {data['pixels_y_sensor']}", "px"),
            ("Naive Resolution", f"{data['pixels_x_naive']} × {data['pixels_y_naive']}", "px"),
            ("Realistic Sensor FOV", f"{data.get('FOV_H_sensor', 0):.2f} × {data.get('FOV_V_sensor', 0):.2f}", "deg"),
            ("Naive Sensor FOV", f"{data.get('FOV_H_naive', 0):.2f} × {data.get('FOV_V_naive', 0):.2f}", "deg"),
            ("Water Coverage", f"{data.get('water_coverage_percent', 0):.1f}", "%"),
            ("Optimal Tilt Angle", f"{data.get('optimal_angle', 0):.1f}", "deg"),
            ("Projection Offset", f"{data.get('projection_offset', 0):.1f}", "mm"),
            ("Sensor Aspect Ratio", data.get('aspect_ratio_used', ''), "-"),
            ("Optics Diameter", f"{data.get('optics_diameter', 0):.1f}", "mm"),
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

    def adjust_param(self, key, delta):
        entry = self.entries[key]
        value = safe_float(entry.get(), 0.0)
        if key in {"DeadZone", "PixelPitch"}:
            value = round(value + delta, 2)
        else:
            value += delta
        entry.delete(0, tk.END)
        entry.insert(0, f"{value:.2f}" if key in {"DeadZone", "PixelPitch"} else f"{value:.1f}")
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
            return {
                'A': safe_float(self.entries['A'].get(), 133),
                'B': safe_float(self.entries['B'].get(), 317.5),
                'C': safe_float(self.entries['C'].get(), 266.7),
                'Tilt': safe_float(self.entries['Tilt'].get(), 30),
                'Margin': safe_float(self.entries['Margin'].get(), 10),
                'Shift': safe_float(self.entries['Shift'].get(), 0),
                'ShiftAxis': self.shift_axis_var.get(),
                'Resolution': safe_float(self.entries['Resolution'].get(), 0.22),
                'DeadZone': safe_float(self.entries['DeadZone'].get(), 0.3),
                'PixelPitch': safe_float(self.entries['PixelPitch'].get(), 2.0),
                'CameraSetup': self.camera_setup_var.get(),
                'Smoothness': self.smoothness_var.get() # <-- addition
            }
        except Exception:
            messagebox.showerror("Error", "Please enter valid numeric values for all parameters")
            return None

    def plot_projection(self):
        params = self.get_current_parameters()
        if params is None:
            return
        cam_setup_str = params.get("CameraSetup", "1x1")
        nx, ny = parse_camera_setup(cam_setup_str)
        deadzone_mm = params.get("DeadZone", 0.3)
        pixel_pitch_um = params.get("PixelPitch", 2.0)
        plot_data = get_plot_data(params, smoothness=params.get('Smoothness', 2),
        smoothing_window=params.get('SmoothingWindow', 1)  # Default to 1 (no smoothing)
        )
        dz_pix = safe_float(dead_zone_pixels(deadzone_mm, pixel_pitch_um), 0)
        px_naive_ori = safe_float(plot_data.get('pixels_x_naive'), 0)
        py_naive_ori = safe_float(plot_data.get('pixels_y_naive'), 0)
        px_sensor_ori = safe_float(plot_data.get('pixels_x_sensor'), 0)
        py_sensor_ori = safe_float(plot_data.get('pixels_y_sensor'), 0)
        plot_data['pixels_x_naive'] = int(round(px_naive_ori * nx + dz_pix * (nx-1)))
        plot_data['pixels_y_naive'] = int(round(py_naive_ori * ny + dz_pix * (ny-1)))
        plot_data['pixels_x_sensor'] = int(round(px_sensor_ori * nx + dz_pix * (nx-1)))
        plot_data['pixels_y_sensor'] = int(round(py_sensor_ori * ny + dz_pix * (ny-1)))
        A = params['A']
        try:
            W_naive = px_naive_ori * params['Resolution']
            H_naive = py_naive_ori * params['Resolution']
            plot_data['FOV_H_naive'] = np.rad2deg(2 * np.arctan(0.5 * W_naive / A))
            plot_data['FOV_V_naive'] = np.rad2deg(2 * np.arctan(0.5 * H_naive / A))
            W_sensor = px_sensor_ori * params['Resolution']
            H_sensor = py_sensor_ori * params['Resolution']
            plot_data['FOV_H_sensor'] = np.rad2deg(2 * np.arctan(0.5 * W_sensor / A))
            plot_data['FOV_V_sensor'] = np.rad2deg(2 * np.arctan(0.5 * H_sensor / A))
        except Exception:
            plot_data['FOV_H_naive'] = plot_data['FOV_V_naive'] = plot_data['FOV_H_sensor'] = plot_data['FOV_V_sensor'] = 0
        for ax in self.axes.values():
            ax.clear()
        self.plot_world_view(plot_data)
        self.plot_projected_view(plot_data)
        self.plot_side_view(plot_data)
        self.plot_coverage_view(plot_data)
        self.fig.tight_layout(pad=1.0)
        self.canvas.draw()
        self.update_simulation_results(plot_data)

    def plot_world_view(self, data):
        ax = self.axes['world']
        ax.set_aspect('equal')
        ax.plot(data['rect_corners'][:, 0], data['rect_corners'][:, 1], 'b-', lw=2, label='Water Spot Rectangle')
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
        ax = self.axes['proj']
        ax.set_aspect('equal')
        ax.scatter(data['proj_rect_pts'][:, 0], data['proj_rect_pts'][:, 1], color='b', s=1, alpha=0.2, label='Projected Rectangle Pixels')
        ax.plot(data['proj_ellipse_pts'][:, 0], data['proj_ellipse_pts'][:, 1], 'r-', linewidth=2, label='Projected Ellipse')
        ax.plot(data['proj_rect_outline'][:, 0], data['proj_rect_outline'][:, 1], 'b-', linewidth=2, label='Projected Rectangle Outline')
        ax.plot(data['box_x_mm'], data['box_y_mm'], 'k--', linewidth=2, label=f"Realistic Sensor FOV ({data['aspect_ratio_used']})")
        ax.scatter(0, 0, color='red', s=200, marker='+', linewidth=3, label='Optical Axis (Principal Point)')
        ax.scatter(data['ellipse_cx'], data['ellipse_cy'], color='purple', s=100, marker='x',
                   linewidth=2, label='Water Spot Center')
        ax.axhline(y=0, color='red', linestyle='-', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='red', linestyle='-', alpha=0.3, linewidth=1)
        ax.add_patch(Circle((0, 0), data['optics_radius'], fill=False, color='black', lw=2, label=f"Optics Image Size (D={data['optics_diameter']:.1f} mm)"))
        ax.set_xlim([-500, 500])
        ax.set_ylim([-500, 500])
        ax.set_xlabel('World X [mm] (projected)', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.set_ylabel('World Y [mm] (projected)', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.set_title(
            f"Projected View - REALISTIC SENSOR {data['aspect_ratio_used']} (User: {data['theta_deg']:.1f}°, Optimal: {data['optimal_angle']:.1f}°)",
            fontsize=GRAPH_TITLE_FONTSIZE
        )
        ax.tick_params(axis='both', labelsize=GRAPH_TICK_FONTSIZE)
        ax.legend(loc='best', fontsize=GRAPH_LEGEND_FONTSIZE)
        ax.grid(True, alpha=0.3)

    def plot_side_view(self, data):
        ax = self.axes['side']
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
                wp_z = 0
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
        # Calculate the real max coverage for this curve (always respect it during zoom)
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
        try:
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
                proposed_ymin = ymid - yhalf

            ax.set_xlim(xmid - xhalf, xmid + xhalf)
            ax.set_ylim(proposed_ymin, proposed_ymax)
        except Exception:
            ax.set_xlim(0, 60)
            ax.set_ylim(5, 90)
        ax.set_xlabel('Tilt Angle [degrees]', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.set_ylabel('Water Coverage [%]', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.set_title('Water Coverage vs. Tilt Angle', fontsize=GRAPH_TITLE_FONTSIZE)
        ax.tick_params(axis='both', labelsize=GRAPH_TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=GRAPH_LEGEND_FONTSIZE)
        

    def refresh_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for index, row in self.data_manager.data.iterrows():
            self.tree.insert('', 'end', values=tuple(row))

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
            return False
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
        entry.bind('<FocusOut>', lambda e: entry.destroy())

if __name__ == "__main__":
    root = tk.Tk()
    app = ProjectionApp(root)
    root.mainloop()
