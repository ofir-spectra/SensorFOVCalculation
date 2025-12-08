import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.font import Font
import numpy as np
import pandas as pd
import os

class ProjectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Projection Parameter Input")
        self.root.geometry("1800x1000")
        self.csv_file = "toilet_data.csv"

        # Set fonts
        self.param_font = Font(family="Arial", size=16)
        self.table_font = Font(family="Arial", size=16)
        self.heading_font = Font(family="Arial", size=18, weight="bold")

        # Parameters
        self.param_labels = [
            ("A - Rim to Water depth (camera height) [mm]:", "133"),
            ("B - Water Spot Length [mm]:", "317.5"),
            ("C - Water Spot Width [mm]:", "266.7"),
            ("Camera Tilt [degrees]:", "-30"),
            ("Margin [%]:", "10"),
            ("Shift from Water Spot Width Edge [mm]:", "0")
        ]
        self.param_keys = ["A", "B", "C", "Tilt", "Margin", "Shift"]
        self.entries = {}

        # Layout
        self.setup_params_frame()
        self.setup_plot_frame()
        self.setup_table_frame()

        # Data
        self.toilet_data = self.load_toilet_data()
        self.refresh_table()

        # Initial plot
        self.plot_projection()

    def setup_params_frame(self):
        param_frame = ttk.LabelFrame(self.root, text="Parameters", padding="10")
        param_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        for i, (label, default) in enumerate(self.param_labels):
            lbl = tk.Label(param_frame, text=label, font=self.param_font)
            lbl.grid(row=i, column=0, sticky="e", padx=5, pady=5)
            entry = tk.Entry(param_frame, font=self.param_font, width=12)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[self.param_keys[i]] = entry
        plot_button = tk.Button(param_frame, text="Plot", font=self.param_font, command=self.plot_projection)
        plot_button.grid(row=len(self.param_labels), column=0, columnspan=2, pady=15)

    def setup_plot_frame(self):
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    def setup_table_frame(self):
        table_frame = ttk.LabelFrame(self.root, text="Toilet Database", padding="10")
        table_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        columns = ('Manufacturer', 'Model', 'Sub-Model', 'H', 'L', 'W', 'Tilt', 'Margin', 'Shift')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        style = ttk.Style()
        style.configure("Treeview.Heading", font=self.heading_font)
        style.configure("Treeview", font=self.table_font, rowheight=36)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=130, anchor="center")

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=v_scrollbar.set)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Table editing
        self.tree.bind('<Double-1>', self.edit_cell)

        # Table buttons
        btn_frame = ttk.Frame(table_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="Add Current Parameters", font=self.param_font, command=self.add_toilet).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Load Selected", font=self.param_font, command=self.load_selected_toilet).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Delete Selected", font=self.param_font, command=self.delete_selected_toilet).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save to CSV", font=self.param_font, command=self.save_toilet_data).pack(side=tk.LEFT, padx=5)

    def get_current_parameters(self):
        try:
            A = float(self.entries['A'].get())
            B = float(self.entries['B'].get())
            C = float(self.entries['C'].get())
            tilt = float(self.entries['Tilt'].get())
            margin = float(self.entries['Margin'].get())
            shift = float(self.entries['Shift'].get())
            return A, B, C, tilt, margin, shift
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for all parameters")
            return None

    def plot_projection(self):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Ellipse
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        params = self.get_current_parameters()
        if params is None:
            return
        W, L, H, theta_deg, margin_percent, shift = params
        theta = np.deg2rad(theta_deg)
        # Camera at rightmost edge of water spot, plus shift, plus tilt offset
        Xc = W/2 + shift 
        Yc = 0
        cam_pos = np.array([Xc, Yc, H])

        # --- Projected view calculation ---
        optical_axis = -cam_pos
        z_cam = optical_axis / np.linalg.norm(optical_axis)
        if abs(np.dot(z_cam, [0, 1, 0])) > 0.99:
            up_guess = np.array([1, 0, 0])
        else:
            up_guess = np.array([0, 1, 0])
        x_cam = np.cross(up_guess, z_cam)
        x_cam = x_cam / np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        y_cam = y_cam / np.linalg.norm(y_cam)
        rect_corners = np.array([
            [-W/2, -L/2, 0],
            [ W/2, -L/2, 0],
            [ W/2,  L/2, 0],
            [-W/2,  L/2, 0]
        ])
        proj_rect = np.zeros((4, 2))
        for i in range(4):
            v = rect_corners[i] - cam_pos
            v_norm = v / np.linalg.norm(v)
            x_p = np.dot(v_norm, x_cam)
            y_p = np.dot(v_norm, y_cam)
            z_p = np.dot(v_norm, z_cam)
            proj_rect[i] = [x_p / z_p, y_p / z_p]
        a = W / 2
        b = L / 2
        t = np.linspace(0, 2 * np.pi, 200)
        ellipse_points = np.column_stack((a * np.cos(t), b * np.sin(t), np.zeros_like(t)))
        proj_ellipse = np.zeros((len(t), 2))
        for i in range(len(t)):
            v = ellipse_points[i] - cam_pos
            v_norm = v / np.linalg.norm(v)
            x_p = np.dot(v_norm, x_cam)
            y_p = np.dot(v_norm, y_cam)
            z_p = np.dot(v_norm, z_cam)
            proj_ellipse[i] = [x_p / z_p, y_p / z_p]
        min_x = np.min(proj_ellipse[:, 0])
        max_x = np.max(proj_ellipse[:, 0])
        min_y = np.min(proj_ellipse[:, 1])
        max_y = np.max(proj_ellipse[:, 1])
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        half_width = (max_x - min_x) / 2 * (1 + margin_percent / 100)
        half_height = (max_y - min_y) / 2 * (1 + margin_percent / 100)
        box_x = np.array([cx - half_width, cx + half_width, cx + half_width, cx - half_width])
        box_y = np.array([cy - half_height, cy - half_height, cy + half_height, cy + half_height])
        FOV_H_margin = np.rad2deg(np.arctan2(np.max(box_x), 1) - np.arctan2(np.min(box_x), 1))
        FOV_V_margin = np.rad2deg(np.arctan2(np.max(box_y), 1) - np.arctan2(np.min(box_y), 1))
        d = H / np.cos(theta)
        proj_rect_mm = proj_rect * d
        proj_ellipse_mm = proj_ellipse * d
        box_x_mm = box_x * d
        box_y_mm = box_y * d

        # Project camera position to image plane (for overlay)
        v = -cam_pos  # from camera to origin
        v_norm = v / np.linalg.norm(v)
        x_p = np.dot(v_norm, x_cam)
        y_p = np.dot(v_norm, y_cam)
        z_p = np.dot(v_norm, z_cam)
        cam_proj_2d = np.array([x_p / z_p, y_p / z_p]) * d  # scale to mm

        # --- World view calculation ---
        # Water spot rectangle for world view
        rect_corners_world = np.array([
            [-W/2, -L/2],
            [ W/2, -L/2],
            [ W/2,  L/2],
            [-W/2,  L/2],
            [-W/2, -L/2]
        ])
        # Ellipse for world view
        ellipse_world = Ellipse((0, 0), W, L, edgecolor='r', facecolor='r', alpha=0.3, lw=2)

        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        import matplotlib.pyplot as plt
        fig, (ax_world, ax_proj) = plt.subplots(1, 2, figsize=(18, 8))
        # --- World view (top-down) ---
        ax_world.set_aspect('equal')
        ax_world.plot(rect_corners_world[:,0], rect_corners_world[:,1], color='b', lw=2, label="Water Spot Rectangle")
        ax_world.add_patch(Ellipse((0, 0), W, L, edgecolor='r', facecolor='r', alpha=0.3, lw=2, label="Water Spot Ellipse"))
        # Camera as a green rectangle
        camera_size = 20
        ax_world.add_patch(Rectangle((Xc-camera_size/2, Yc-camera_size/2), camera_size, camera_size, color='green', alpha=0.7, label="Camera"))
        ax_world.plot([Xc], [Yc], 'go')
        ax_world.set_xlim(-W, W+W/2)
        ax_world.set_ylim(-L, L)
        ax_world.set_xlabel('X [mm]')
        ax_world.set_ylabel('Y [mm]')
        ax_world.set_title('Top-Down View (World Coordinates)')
        ax_world.legend(loc='upper right')
        ax_world.grid(True, alpha=0.3)

        # --- Projected view (as seen by camera) ---
        ax_proj.axis('equal')
        ax_proj.fill(proj_rect_mm[:, 0], proj_rect_mm[:, 1], color=(0.7, 0.9, 1), edgecolor='b', linewidth=2, alpha=0.3, label='Projected Rectangle (Trapezoid)')
        ax_proj.fill(proj_ellipse_mm[:, 0], proj_ellipse_mm[:, 1], color='r', edgecolor='r', linewidth=2, alpha=0.5, label='Projected Ellipse (Oval)')
        box_x_mm_closed = np.append(box_x_mm, box_x_mm[0])
        box_y_mm_closed = np.append(box_y_mm, box_y_mm[0])
        ax_proj.plot(box_x_mm_closed, box_y_mm_closed, 'k--', linewidth=2, label='Ellipse+Margin FOV')
        # Camera position is always at the center
        camera_size = 20  # mm
        ax_proj.add_patch(Rectangle(
            (cam_proj_2d[0] - camera_size/2, cam_proj_2d[1] - camera_size/2),
            camera_size, camera_size,
            fill=True, color='green', alpha=0.7, label='Camera Optical Axis'
        ))
        max_abs_x = np.max(np.abs(np.concatenate([proj_rect_mm[:, 0], proj_ellipse_mm[:, 0], box_x_mm, [cam_proj_2d[0]]])))
        max_abs_y = np.max(np.abs(np.concatenate([proj_rect_mm[:, 1], proj_ellipse_mm[:, 1], box_y_mm, [cam_proj_2d[1]]])))
        max_range = max(max_abs_x, max_abs_y) + 50
        ax_proj.set_xlim(-max_range, max_range)
        ax_proj.set_ylim(-max_range, max_range)
        ax_proj.axhline(0, color=(0.5, 0.5, 0.5), linestyle='--', linewidth=1)
        ax_proj.axvline(0, color=(0.5, 0.5, 0.5), linestyle='--', linewidth=1)
        ax_proj.set_xlabel('Horizontal projection [mm]')
        ax_proj.set_ylabel('Vertical projection [mm]')
        ax_proj.set_title(f'Projected View (Camera Optical Axis)\nEllipse+Margin FOV_H = {FOV_H_margin:.1f}°, FOV_V = {FOV_V_margin:.1f}°')
        ax_proj.legend(loc='best')
        ax_proj.grid(True, alpha=0.3)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def load_toilet_data(self):
        if os.path.exists(self.csv_file):
            try:
                return pd.read_csv(self.csv_file)
            except:
                pass
        return pd.DataFrame(columns=['Manufacturer', 'Model', 'Sub-Model', 'H', 'L', 'W', 'Tilt', 'Margin', 'Shift'])

    def save_toilet_data(self):
        try:
            self.toilet_data.to_csv(self.csv_file, index=False)
            messagebox.showinfo("Success", f"Data saved to {self.csv_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")

    def refresh_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for index, row in self.toilet_data.iterrows():
            self.tree.insert('', 'end', values=tuple(row))

    def add_toilet(self):
        params = self.get_current_parameters()
        if params is None:
            return
        W, L, H, tilt, margin, shift = params
        new_row = {
            'Manufacturer': "DEFAULT",
            'Model': "DEFAULT",
            'Sub-Model': "DEFAULT",
            'H': H, 'L': L, 'W': W,
            'Tilt': tilt, 'Margin': margin, 'Shift': shift
        }
        self.toilet_data = pd.concat([self.toilet_data, pd.DataFrame([new_row])], ignore_index=True)
        self.refresh_table()

    def load_selected_toilet(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a toilet from the table")
            return
        item = self.tree.item(selection[0])
        values = item['values']
        self.entries['H'].delete(0, tk.END)
        self.entries['H'].insert(0, str(values[3]))
        self.entries['L'].delete(0, tk.END)
        self.entries['L'].insert(0, str(values[4]))
        self.entries['W'].delete(0, tk.END)
        self.entries['W'].insert(0, str(values[5]))
        self.entries['Tilt'].delete(0, tk.END)
        self.entries['Tilt'].insert(0, str(values[6]))
        self.entries['Margin'].delete(0, tk.END)
        self.entries['Margin'].insert(0, str(values[7]))
        self.entries['Shift'].delete(0, tk.END)
        self.entries['Shift'].insert(0, str(values[8]))
        self.plot_projection()

    def delete_selected_toilet(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a toilet to delete")
            return
        if messagebox.askyesno("Confirm", "Are you sure you want to delete the selected toilet?"):
            item_index = self.tree.index(selection[0])
            self.toilet_data = self.toilet_data.drop(self.toilet_data.index[item_index]).reset_index(drop=True)
            self.refresh_table()

    def edit_cell(self, event):
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
            # Update DataFrame as well
            item_index = self.tree.index(rowid)
            col_name = self.tree["columns"][col_index]
            self.toilet_data.at[item_index, col_name] = new_value
            entry.destroy()
        entry.bind('<Return>', on_enter)
        entry.bind('<FocusOut>', lambda e: entry.destroy())

if __name__ == "__main__":
    root = tk.Tk()
    app = ProjectionApp(root)
    root.mainloop()
