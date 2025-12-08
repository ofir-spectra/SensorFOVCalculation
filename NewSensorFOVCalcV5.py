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

from projection_calculations import calculate_projection
from data_manager import ToiletDataManager
from image_utils import find_image_case_insensitive

APP_VERSION = "V5.0.1"

# === FONT SIZE SETTINGS (change these as you like) ===
GRAPH_TITLE_FONTSIZE = 14
GRAPH_LABEL_FONTSIZE = 14
GRAPH_TICK_FONTSIZE = 12
GRAPH_LEGEND_FONTSIZE = 10
GRAPH_OVERLAY_FONTSIZE = 12

class ProjectionApp:

    def __init__(self, root):
        self.root = root
        self.root.title(f"SENSOR SIMULATION {APP_VERSION}")
        
        # Set initial window size and minimum size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = min(1900, int(screen_width * 0.9))  # 90% of screen width
        window_height = min(1000, int(screen_height * 0.9))  # 90% of screen height
        
        # Calculate center position
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        # Set geometry with position and size
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(1600, 900)
        
        # Configure window state handling
        def handle_window_state(event=None):
            if event and event.widget == self.root:
                # Update all figures
                for fig in self.figures.values():
                    fig.tight_layout(pad=1.2)
                # Force a geometry update
                self.root.update_idletasks()
        
        self.root.bind('<Configure>', handle_window_state)

        self.param_font = Font(family="Arial", size=14)
        self.table_font = Font(family="Arial", size=14)
        self.heading_font = Font(family="Arial", size=14, weight="bold")

        # Initialize parameters before any setup calls
        self.param_labels = [
            ("A - Rim to Water depth (camera height) [mm]:", "133"),
            ("B - Water Spot Length [mm]:", "317.5"),
            ("C - Water Spot Width [mm]:", "266.7"),
            ("Camera Tilt [degrees]:", "30"),
            ("Margin [%]:", "10"),
            ("Shift from Water Spot Width Edge [mm]:", "0"),
            ("Required Resolution [mm/px]:", "0.22")
        ]
        self.param_keys = ["A", "B", "C", "Tilt", "Margin", "Shift", "Resolution", "PixelPitch"]
        self.entries = {}
        self.shift_axis_var = tk.StringVar(value="X")
        self.fig = None
        self.canvas = None
        self.axes = {}

        # Configure grid: 4 rows, 3 columns
        # Row weights: title=0, plots_top=1, plots_bottom=1, table=0
        for r in range(4):
            self.root.grid_rowconfigure(r, weight=1 if r in [1, 2] else 0)

        # Column weights: parameters=0 (fixed width), plots=1 (expand)
        self.root.grid_columnconfigure(0, weight=0, minsize=420)  # Fixed width for parameters
        self.root.grid_columnconfigure(1, weight=1)  # Expand for plots
        self.root.grid_columnconfigure(2, weight=1)  # Expand for plots

        # 1. Title row (row=0, col=0-2)
        title_label = tk.Label(root, text=f"SENSOR SIMULATION {APP_VERSION}", font=("Arial", 20, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(10, 0), sticky="ew")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "toilet_data.csv")
        self.data_manager = ToiletDataManager(csv_path)

        # --- Row 1: Simulation Results (col=0) and Top Plots ---
        self.sim_results_frame = ttk.LabelFrame(self.root, text="Simulation Results", padding="10")
        self.sim_results_frame.grid(row=1, column=0, padx=10, pady=(10,0), sticky="nsew")
        
        # Configure the sim_results_frame to expand properly
        self.sim_results_frame.grid_propagate(False)  # Prevent frame from shrinking
        self.sim_results_frame.grid_columnconfigure(0, weight=1)
        self.sim_results_frame.grid_rowconfigure(0, weight=1)
        
        # Create an inner frame to hold the table
        inner_frame = ttk.Frame(self.sim_results_frame)
        inner_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        inner_frame.grid_columnconfigure(0, weight=1)
        inner_frame.grid_rowconfigure(0, weight=1)
        
        # Configure the table with fixed size
        self.sim_results_table = ttk.Treeview(inner_frame, columns=("Parameter", "Value", "Unit"), 
                                             show="headings", height=7)
        for col, width in zip(("Parameter", "Value", "Unit"), (180, 180, 180)):
            self.sim_results_table.heading(col, text=col)
            self.sim_results_table.column(col, anchor="center", width=width, minwidth=width)
        
        self.sim_results_table.grid(row=0, column=0, sticky="nsew")

        # --- Row 2: Parameters (col=0) ---
        self.setup_params_frame(row=2)
        self.param_frame.grid(row=2, column=0, padx=10, pady=(10,0), sticky="nsew")

        # Create frames for each plot with proper configuration
        self.plot_frames = {}
        plot_grid = {'world': (1,1), 'proj': (1,2), 'side': (2,1), 'coverage': (2,2)}
        
        for name, (row, col) in plot_grid.items():
            # Create main frame
            frame = ttk.Frame(self.root)
            frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Set minimum size and configure grid weights
            frame.grid_propagate(True)  # Allow frame to resize with content
            frame.grid_columnconfigure(0, weight=1)
            frame.grid_rowconfigure(0, weight=1)
            
            # Create a canvas container frame
            canvas_frame = ttk.Frame(frame)
            canvas_frame.grid(row=0, column=0, sticky="nsew")
            canvas_frame.grid_columnconfigure(0, weight=1)
            canvas_frame.grid_rowconfigure(0, weight=1)
            
            self.plot_frames[name] = canvas_frame

        self.setup_plot_frame()
        
        # Configure each plot canvas
        for name in ['world', 'proj', 'side', 'coverage']:
            canvas_widget = self.axes[name].figure.canvas.get_tk_widget()
            canvas_widget.pack(in_=self.plot_frames[name], fill='both', expand=True, padx=2, pady=2)
            
            # Configure pack propagation
            self.plot_frames[name].pack_propagate(False)

        # Get initial plot data
        params = self.get_current_parameters()
        if params:
            self.plot_data = calculate_projection(params)

        # Place zoom controls directly below coverage plot (row=2, col=2)
        self.zoom_frame = ttk.Frame(self.root)
        self.zoom_frame.grid(row=2, column=2, sticky="ew", padx=10, pady=(0, 10))
        ttk.Label(self.zoom_frame, text="Water Coverage Zoom:").pack(side='left', padx=(0, 5))
        zoom_in_btn = tk.Button(self.zoom_frame, text="+", font=self.param_font, width=3, command=lambda: self.zoom_coverage('in'))
        zoom_in_btn.pack(side='left', padx=2)
        zoom_out_btn = tk.Button(self.zoom_frame, text="-", font=self.param_font, width=3, command=lambda: self.zoom_coverage('out'))
        zoom_out_btn.pack(side='left', padx=2)

        # 3. Third row: Toilet table (row=3, col=0-2)
        self.setup_table_frame()
        self.table_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        self.refresh_table()
        self.plot_projection()




    def ensure_csv_exists():
        csv_path = "sensor_data.csv"  # This will be in the same directory as the .exe
        if not os.path.exists(csv_path):
            # Create with appropriate columns for your sensor data
            df = pd.DataFrame(columns=['Parameter', 'Value', 'Units'])
            df.to_csv(csv_path, index=False)
        return csv_path

    def setup_params_frame(self, row=1):
        self.param_frame = ttk.LabelFrame(self.root, text="Parameters", padding="10")
        self.param_frame.grid_columnconfigure(1, weight=1)  # Make entry column expandable

        for i, (label, default) in enumerate(self.param_labels):
            lbl = tk.Label(self.param_frame, text=label, font=self.param_font)
            lbl.grid(row=i, column=0, sticky="e", padx=5, pady=5)
            entry = tk.Entry(self.param_frame, font=self.param_font, width=12)
            entry.insert(0, default)
            entry.grid(row=i, column=1, sticky="ew", padx=5, pady=5)  # Make entry expand horizontally
            self.entries[self.param_keys[i]] = entry

            if self.param_keys[i] == "Tilt":
                tk.Button(self.param_frame, text="−", font=self.param_font, width=2,
                        command=lambda: self.adjust_param("Tilt", -1)).grid(row=i, column=2, padx=(0,2))
                tk.Button(self.param_frame, text="+", font=self.param_font, width=2,
                        command=lambda: self.adjust_param("Tilt", 1)).grid(row=i, column=3)
            if self.param_keys[i] == "Shift":
                tk.Button(self.param_frame, text="−", font=self.param_font, width=2,
                        command=lambda: self.adjust_param("Shift", -5)).grid(row=i, column=2, padx=(0,2))
                tk.Button(self.param_frame, text="+", font=self.param_font, width=2,
                        command=lambda: self.adjust_param("Shift", 5)).grid(row=i, column=3)
            if self.param_keys[i] == "Margin":
                tk.Button(self.param_frame, text="−", font=self.param_font, width=2,
                        command=lambda: self.adjust_param("Margin", -1)).grid(row=i, column=2, padx=(0,2))
                tk.Button(self.param_frame, text="+", font=self.param_font, width=2,
                        command=lambda: self.adjust_param("Margin", 1)).grid(row=i, column=3)
        
        shift_axis_row = len(self.param_labels)
        lbl = tk.Label(self.param_frame, text="Shift Axis:", font=self.param_font)
        lbl.grid(row=shift_axis_row, column=0, sticky="e", padx=5, pady=5)
        
        radio_frame = tk.Frame(self.param_frame)
        radio_frame.grid(row=shift_axis_row, column=1, padx=5, pady=5, sticky="w")
        tk.Radiobutton(radio_frame, text="X", variable=self.shift_axis_var, 
                    value="X", font=self.param_font, command=self.plot_projection).pack(side=tk.LEFT)
        tk.Radiobutton(radio_frame, text="Y", variable=self.shift_axis_var, 
                    value="Y", font=self.param_font, command=self.plot_projection).pack(side=tk.LEFT, padx=(10,0))
        
        plot_button = tk.Button(self.param_frame, text="Plot", font=self.param_font, command=self.plot_projection)
        plot_button.grid(row=shift_axis_row+1, column=0, columnspan=2, pady=15)

    def adjust_param(self, key, delta):
        entry = self.entries[key]
        try:
            value = float(entry.get())
        except Exception:
            value = 0
        value += delta
        entry.delete(0, tk.END)
        entry.insert(0, f"{value:.1f}")
        self.plot_projection()
            
    def setup_plot_frame(self):
        self.figures = {}
        self.axes = {}
        
        for name in ['world', 'proj', 'side', 'coverage']:
            # Create figure with a reasonable aspect ratio
            self.figures[name] = plt.Figure(figsize=(6, 6))
            self.axes[name] = self.figures[name].add_subplot(111)
            
            # Set initial properties
            self.axes[name].grid(True)
            self.axes[name].set_aspect('equal', adjustable='box')
            
            # Create canvas and enable resizing
            canvas = FigureCanvasTkAgg(self.figures[name], master=self.plot_frames[name])
            canvas.draw()
            
            # Configure responsive behavior
            def on_resize(event, fig=self.figures[name]):
                # Only adjust if the size has significantly changed
                if event.width > 1 and event.height > 1:
                    fig.set_size_inches(event.width/100, event.height/100)
                    fig.tight_layout(pad=1.2)
            
            canvas.get_tk_widget().bind('<Configure>', on_resize)

    def setup_table_frame(self):
        self.table_frame = ttk.LabelFrame(self.root, text="Toilet Database", padding="10")
        self.table_frame.grid_columnconfigure(0, weight=1)  # Make table expand horizontally
        self.table_frame.grid_rowconfigure(0, weight=1)     # Make table expand vertically

        # Create main content frame to hold tree and buttons
        content_frame = ttk.Frame(self.table_frame)
        content_frame.grid(row=0, column=0, sticky="nsew")
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)

        # Setup tree with scrollbar
        tree_frame = ttk.Frame(content_frame)
        tree_frame.grid(row=0, column=0, sticky="nsew")
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)

        self.table_columns = self.data_manager.columns
        self.tree = ttk.Treeview(tree_frame, columns=self.table_columns, show='headings', height=6)
        style = ttk.Style()
        style.configure("Treeview.Heading", font=self.heading_font)
        style.configure("Treeview", font=self.table_font, rowheight=36)

        # Configure columns to expand proportionally
        total_width = sum(len(col) for col in self.table_columns)  # Use column name lengths as weights
        for col in self.table_columns:
            weight = max(len(col), 10)  # Minimum width of 10 characters
            self.tree.heading(col, text=col)
            self.tree.column(col, width=weight*10, minwidth=100, anchor="center")

        self.tree.grid(row=0, column=0, sticky="nsew")
        
        # Add vertical scrollbar
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=v_scrollbar.set)

        # Button frame on the right
        btn_frame = ttk.Frame(content_frame)
        btn_frame.grid(row=0, column=1, padx=10, pady=5, sticky="ns")
        
        # Configure buttons
        button_texts = ["Add Current Parameters", "Load Selected", "Delete Selected", "Save to CSV"]
        button_commands = [self.add_toilet, self.load_selected_toilet, 
                         self.delete_selected_toilet, self.data_manager.save_data]
        
        for text, command in zip(button_texts, button_commands):
            btn = tk.Button(btn_frame, text=text, font=self.param_font, command=command)
            btn.pack(fill=tk.X, pady=5)

        self.tree.bind('<Double-1>', self.edit_cell)

    def get_current_parameters(self):
        try:
            return {
                'A': float(self.entries['A'].get()),
                'B': float(self.entries['B'].get()),
                'C': float(self.entries['C'].get()),
                'Tilt': float(self.entries['Tilt'].get()),
                'Margin': float(self.entries['Margin'].get()),
                'Shift': float(self.entries['Shift'].get()),
                'ShiftAxis': self.shift_axis_var.get(),
                'Resolution': float(self.entries['Resolution'].get())
            }
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for all parameters")
            return None

    def plot_projection(self):
        # Get current parameters and calculate projections
        params = self.get_current_parameters()
        if not params:
            return  # Exit if parameters are invalid

        # Get plot data and calculate projections
        plot_data = calculate_projection(params)
            
        # Clear all plots
        for ax in self.axes.values():
            ax.clear()
            ax.grid(True)
            ax.set_aspect('equal', adjustable='box')

        # Update each view with the plot data
        self.plot_world_view(plot_data)
        self.plot_projected_view(plot_data)
        self.plot_side_view(plot_data)
        self.plot_coverage_view(plot_data)
        
        # Update all canvases
        for name, fig in self.figures.items():
            fig.tight_layout(pad=1.2)
            self.axes[name].figure.canvas.draw()
            self.axes[name].figure.canvas.flush_events()

    def update_simulation_results(self, data):
        # Clear previous results
        for item in self.sim_results_table.get_children():
            self.sim_results_table.delete(item)
        # Compose results
        results = [
            ("Realistic Sensor Resolution", f"{data['pixels_x_sensor']} x {data['pixels_y_sensor']}", "px X px"),
            ("Naive Resolution", f"{data['pixels_x_naive']} x {data['pixels_y_naive']}", "px X px"),
            ("Water Coverage", f"{data['water_coverage_percent']:.1f}", "%"),
            ("Optimal Tilt Angle", f"{data['optimal_angle']:.1f}", "deg"),
            ("Projection Offset", f"{data['projection_offset']:.1f}", "mm"),
            ("Required FOV (H x V)", f"{data['fov_h_deg']:.1f} x {data['fov_v_deg']:.1f}", "deg x deg"),
        ]
        for param, value, unit in results:
            self.sim_results_table.insert("", "end", values=(param, value, unit))

    def plot_world_view(self, data):
        ax = self.axes['world']
        ax.plot(data['rect_corners'][:, 0], data['rect_corners'][:, 1], 'b-', lw=2, label='Water Spot Rectangle')
        ax.add_patch(Ellipse((0, 0), data['C'], data['L'], edgecolor='r', facecolor='r', alpha=0.3, lw=2, label='Water Spot Ellipse'))
        camera_size = 20
        color = 'green' if data['shift_axis'] == 'X' else 'blue'
        ax.add_patch(Rectangle((data['Xc'] - camera_size/2, data['Yc'] - camera_size/2), camera_size, camera_size, 
                               color=color, alpha=0.7, label=f'Camera ({data["shift_axis"]}-axis)'))
        ax.plot([data['Xc']], [data['Yc']], 'go' if data['shift_axis'] == 'X' else 'bo')
        for point in data['fov_points_world']:
            ax.plot([data['Xc'], point[0]], [data['Yc'], point[1]], 'gray', linestyle='--', linewidth=1, alpha=0.6)
        
        # Set view limits with some padding
        ax.set_xlim(-350, 350)
        ax.set_ylim(-350, 350)
        
        ax.set_xlabel('X [mm]', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.set_ylabel('Y [mm]', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.set_title('Top-Down View (World Coordinates)', fontsize=GRAPH_TITLE_FONTSIZE)
        ax.tick_params(axis='both', labelsize=GRAPH_TICK_FONTSIZE)
        ax.legend(loc='upper right', fontsize=GRAPH_LEGEND_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    def plot_projected_view(self, data):
        ax = self.axes['proj']
        ax.set_aspect('equal')
        ax.scatter(data['proj_rect_pts'][:, 0], data['proj_rect_pts'][:, 1], color='b', s=1, alpha=0.2, label='Projected Rectangle Pixels')
        ax.plot(data['proj_ellipse_pts'][:, 0], data['proj_ellipse_pts'][:, 1], 'r-', linewidth=2, label='Projected Ellipse')
        ax.plot(data['proj_rect_outline'][:, 0], data['proj_rect_outline'][:, 1], 'b-', linewidth=2, label='Projected Rectangle Outline')
        ax.plot(data['box_x_mm'], data['box_y_mm'], 'k--', linewidth=2, label=f'Realistic Sensor FOV ({data["aspect_ratio_used"]})')
        ax.scatter(0, 0, color='red', s=200, marker='+', linewidth=3, label='Optical Axis (Principal Point)')
        ax.scatter(data['ellipse_cx'], data['ellipse_cy'], color='purple', s=100, marker='x', 
                   linewidth=2, label='Water Spot Center')
        ax.axhline(y=0, color='red', linestyle='-', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='red', linestyle='-', alpha=0.3, linewidth=1)
        ax.add_patch(Circle((0, 0), data['optics_radius'], fill=False, color='black', lw=2, 
                           label=f'Optics Image Size (D={data["optics_diameter"]:.1f} mm)'))

        ax.set_xlim([-500, 500])
        ax.set_ylim([-500, 500])
        ax.set_xlabel('World X [mm] (projected)', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.set_ylabel('World Y [mm] (projected)', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.set_title(f'Projected View - REALISTIC SENSOR {data["aspect_ratio_used"]} (User: {data["theta_deg"]:.1f}°, Optimal: {data["optimal_angle"]:.1f}°)', fontsize=GRAPH_TITLE_FONTSIZE)
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
                head_width=8, head_length=8, fc='purple', ec='purple', linewidth=2, 
                label='Water Normal Vector')

        axis_length = 80
        ax.arrow(data['camera_center_x'], data['camera_center_z'], 
                 axis_length * data['optical_axis_normalized'][0], axis_length * data['optical_axis_normalized'][1],
                 head_width=8, head_length=8, fc='orange', ec='orange', linewidth=3,
                 label='Camera Optical Axis (⊥ to sensor)')

        ax.arrow(data['camera_center_x'], data['camera_center_z'],
                 axis_length * data['optimal_axis_2d'][0] * 0.8, axis_length * data['optimal_axis_2d'][1] * 0.8,
                 head_width=6, head_length=6, fc='cyan', ec='cyan', linewidth=2, linestyle='--',
                 label=f'Optimal Tilt Angle ({data["optimal_angle"]:.1f}°)')

        if data['theta_deg'] != 0:
            arc_radius = 25
            theta1 = data['theta_deg'] if data['theta_deg'] < 0 else 0
            theta2 = 0 if data['theta_deg'] < 0 else data['theta_deg']
            
            arc = Arc((data['camera_center_x'], data['camera_center_z']), 2*arc_radius, 2*arc_radius,
                      angle=0, theta1=theta1, theta2=theta2, color='red', linewidth=3)
            ax.add_patch(arc)

            mid_angle = (theta1 + theta2) / 2
            text_x = data['camera_center_x'] + (arc_radius * 0.6) * np.cos(np.radians(mid_angle))
            text_z = data['camera_center_z'] + (arc_radius * 0.6) * np.sin(np.radians(mid_angle))
            ax.text(text_x, text_z, f'{data["theta_deg"]:.0f}',
                    fontsize=12, color='red', weight='bold', ha='center', va='center')

        ax.text(0.68, 0.98, f'OPTIMAL TILT ANGLE: {data["optimal_angle"]:.1f}°',
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
        ax.set_ylim(-50, 200)  # Limited to 200mm as requested
        ax.set_xlabel(f'{data["shift_axis"]} [mm] (side view)', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.set_title(f'Side View ({data["shift_axis"]}-Z plane) - User Input Angle + Optimal', fontsize=GRAPH_TITLE_FONTSIZE)
        ax.set_ylabel('Z [mm] (height)', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.tick_params(axis='both', labelsize=GRAPH_TICK_FONTSIZE)
        ax.legend(loc='upper left', fontsize=GRAPH_LEGEND_FONTSIZE)
        ax.grid(True, alpha=0.3)

    def plot_coverage_view(self, data):
        ax = self.axes['coverage']
        ax.plot(data['coverage_angles'], data['coverage_values'], 'b-', linewidth=2, label='Water Coverage')
        ax.scatter(data['optimal_angle'], data['optimal_coverage'], color='red', s=100, zorder=5, 
                   label=f'Optimal ({data["optimal_angle"]:.1f}°, {data["optimal_coverage"]:.1f}%)')
        ax.axvline(x=data['optimal_angle'], color='red', linestyle='--', alpha=0.7)
        ax.axhline(y=data['optimal_coverage'], color='red', linestyle='--', alpha=0.7)
        if 0 <= abs(data['theta_deg']) <= 60:
            current_coverage = data['water_coverage_percent']
            ax.scatter(abs(data['theta_deg']), current_coverage, color='orange', s=80, zorder=5,
                       marker='s', label=f'Current ({abs(data["theta_deg"]):.1f}°, {current_coverage:.1f}%)')
        ax.set_xlabel('Tilt Angle [degrees]', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.set_ylabel('Water Coverage [%]', fontsize=GRAPH_LABEL_FONTSIZE)
        ax.set_title('Water Coverage vs. Tilt Angle', fontsize=GRAPH_TITLE_FONTSIZE)
        ax.tick_params(axis='both', labelsize=GRAPH_TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=GRAPH_LEGEND_FONTSIZE)
        ax.set_xlim(0, 60)
        # Set default y-limits to 20-80%
        if not hasattr(self, 'coverage_ylim'):
            self.coverage_ylim = [20, 80]
        ax.set_ylim(self.coverage_ylim)
    def zoom_coverage(self, direction):
        # direction: 'in' or 'out'
        if not hasattr(self, 'coverage_ylim'):
            self.coverage_ylim = [20, 80]
        y0, y1 = self.coverage_ylim
        center = (y0 + y1) / 2
        span = (y1 - y0)
        if direction == 'in':
            new_span = max(5, span * 0.7)
        else:
            new_span = min(100, span * 1.3)
        new_y0 = max(0, center - new_span / 2)
        new_y1 = min(100, center + new_span / 2)
        # Clamp to [0, 100]
        if new_y0 < 0: new_y0 = 0
        if new_y1 > 100: new_y1 = 100
        self.coverage_ylim = [new_y0, new_y1]
        self.plot_projection()
        
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
            'Required Resolution [mm/px]': params['Resolution']
        }
        
        if self.is_duplicate_record(toilet_params):
            messagebox.showwarning("Duplicate Entry", 
                                 "These parameters already exist in the database. "
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
        """Allow editing of table cells by double-clicking - FIXED INDENTATION"""
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
            
        rowid = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        col_index = int(column.replace('#','')) - 1
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
