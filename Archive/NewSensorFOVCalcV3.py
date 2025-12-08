import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.font import Font
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageTk

def find_image_case_insensitive(basename):
    folder = os.path.dirname(os.path.abspath(__file__))
    for fname in os.listdir(folder):
        if fname.lower() == basename.lower():
            return os.path.join(folder, fname)
    return None


class ProjectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Projection Parameter Input")
        self.root.geometry("1900x1000")
        self.csv_file = "toilet_data.csv"

        # Set fonts
        self.param_font = Font(family="Arial", size=14)
        self.table_font = Font(family="Arial", size=14)
        self.heading_font = Font(family="Arial", size=14, weight="bold")

        # Parameters: label, default, key
        self.param_labels = [
            ("A - Rim to Water depth (camera height) [mm]:", "133"),
            ("B - Water Spot Length [mm]:", "317.5"),
            ("C - Water Spot Width [mm]:", "266.7"),
            ("Camera Tilt [degrees]:", "-30"),
            ("Margin [%]:", "10"),
            ("Shift from Water Spot Width Edge [mm]:", "0"),
            ("Required Resolution [mm/px]:", "0.22"),
            ("Pixel Pitch [um]:", "1.2")
        ]
        self.param_keys = ["A", "B", "C", "Tilt", "Margin", "Shift", "Resolution", "PixelPitch"]
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

    # --- Add image below the parameters ---
        image_path = find_image_case_insensitive("image.png")
        if not image_path:
            image_path = find_image_case_insensitive("image.jpg")
        if image_path:
            try:
                self.original_img = Image.open(image_path)
                self.tk_img = None  # Will be set in resize callback
                self.img_label = tk.Label(param_frame)
                self.img_label.grid(row=len(self.param_labels)+1, column=0, columnspan=2, pady=10, sticky="nsew")

                def resize_image(event=None):
                    # Get available width (2 columns) and a reasonable height
                    width = param_frame.winfo_width() - 40
                    height = max(100, int(param_frame.winfo_height() * 0.25))
                    if width < 50: width = 100
                    if height < 50: height = 100
                    img = self.original_img.copy()
                    img.thumbnail((width, height), Image.LANCZOS)
                    self.tk_img = ImageTk.PhotoImage(img)
                    self.img_label.config(image=self.tk_img)

                param_frame.bind('<Configure>', resize_image)
                resize_image()  # Initial draw
            except Exception as e:
                img_label = tk.Label(param_frame, text="Failed to load image file", font=self.param_font, fg="red")
                img_label.grid(row=len(self.param_labels)+1, column=0, columnspan=2, pady=10)
        else:
            img_label = tk.Label(param_frame, text="image.png/.jpg not found", font=self.param_font, fg="red")
            img_label.grid(row=len(self.param_labels)+1, column=0, columnspan=2, pady=10)
            
    def setup_plot_frame(self):
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    def setup_table_frame(self):
        table_frame = ttk.LabelFrame(self.root, text="Toilet Database", padding="10")
        table_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        self.table_columns = (
            'Manufacturer',
            'Model',
            'Sub-Model',
            'A - Rim to Water depth (camera height) [mm]',
            'B - Water Spot Length [mm]',
            'C - Water Spot Width [mm]',
            'Camera Tilt [degrees]',
            'Margin [%]',
            'Shift from Water Spot Width Edge [mm]'
        )
        self.tree = ttk.Treeview(table_frame, columns=self.table_columns, show='headings', height=8)
        style = ttk.Style()
        style.configure("Treeview.Heading", font=self.heading_font)
        style.configure("Treeview", font=self.table_font, rowheight=36)

        for col in self.table_columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=180, anchor="center")

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=v_scrollbar.set)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Table editing
        self.tree.bind('<Double-1>', self.edit_cell)

        # Table buttons in a single column (vertical)
        btn_frame = ttk.Frame(table_frame)
        btn_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y, anchor='n')
        tk.Button(btn_frame, text="Add Current Parameters", font=self.param_font, command=self.add_toilet).pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="Load Selected", font=self.param_font, command=self.load_selected_toilet).pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="Delete Selected", font=self.param_font, command=self.delete_selected_toilet).pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="Save to CSV", font=self.param_font, command=self.save_toilet_data).pack(fill=tk.X, pady=5)

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

    def plot_projection(self, x_range_mm=(-350, 350), y_range_mm=(-350, 350), n_ticks=9):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Ellipse, Circle
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import numpy as np

        params = self.get_current_parameters()
        if params is None:
            return
        A, B, C, theta_deg, margin_percent, shift = params
        H = A
        L = B
        W = C
        theta = np.deg2rad(theta_deg)

        # Camera position in world coordinates
        Xc = W / 2 + shift
        Yc = 0
        cam_pos = np.array([Xc, Yc, H])

        # Camera orientation: look at (0,0,0), then tilt about Y axis
        look_vec = np.array([0, 0, 0]) - cam_pos
        look_vec = look_vec / np.linalg.norm(look_vec)
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        z_cam = Ry @ look_vec
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

        # Project a 3D point to the camera image plane (sensor coordinates, optical axis at (0,0))
        def project_point(p_world):
            v = p_world - cam_pos
            Xc = np.dot(v, x_cam)
            Yc = np.dot(v, y_cam)
            Zc = np.dot(v, z_cam)
            x_img = H * Xc / Zc
            y_img = H * Yc / Zc
            return np.array([x_img, y_img])

        # Rectangle grid (for "pixels" of the rectangle)
        rect_res = 100
        x_rect = np.linspace(-W/2, W/2, rect_res)
        y_rect = np.linspace(-L/2, L/2, rect_res)
        xx_rect, yy_rect = np.meshgrid(x_rect, y_rect)
        rect_points = np.column_stack([xx_rect.ravel(), yy_rect.ravel(), np.zeros(rect_res**2)])
        proj_rect_pts = np.array([project_point(p) for p in rect_points])

        # Ellipse outline
        ell_res = 200
        angles = np.linspace(0, 2*np.pi, ell_res)
        ell_x = (C/2) * np.cos(angles)
        ell_y = (L/2) * np.sin(angles)
        ellipse_points = np.column_stack([ell_x, ell_y, np.zeros(ell_res)])
        proj_ellipse_pts = np.array([project_point(p) for p in ellipse_points])

        # For outline of rectangle (project corners)
        rect_corners = np.array([
            [-W/2, -L/2, 0],
            [ W/2, -L/2, 0],
            [ W/2,  L/2, 0],
            [-W/2,  L/2, 0],
            [-W/2, -L/2, 0]
        ])
        proj_rect_outline = np.array([project_point(p) for p in rect_corners])

        # Compute bounding box of projected ellipse (for FOV + margin)
        min_x, max_x = np.min(proj_ellipse_pts[:, 0]), np.max(proj_ellipse_pts[:, 0])
        min_y, max_y = np.min(proj_ellipse_pts[:, 1]), np.max(proj_ellipse_pts[:, 1])
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        half_width = (max_x - min_x) / 2 * (1 + margin_percent / 100)
        half_height = (max_y - min_y) / 2 * (1 + margin_percent / 100)

        # FOV calculation (using projected mm values)
        FOV_H = np.rad2deg(2 * np.arctan(half_width / H))
        FOV_V = np.rad2deg(2 * np.arctan(half_height / H))

        # FOV margin box in mm (centered on ellipse center in projected mm space)
        box_x_mm = np.array([cx - half_width, cx + half_width, cx + half_width, cx - half_width, cx - half_width])
        box_y_mm = np.array([cy - half_height, cy - half_height, cy + half_height, cy + half_height, cy - half_height])
        box_pts = np.column_stack([box_x_mm, box_y_mm])

        # Shift all projected data so that the center of the ellipse+margin FOV is at (0,0)
        shift_x = (cx)
        shift_y = (cy)
        proj_rect_pts -= np.array([shift_x, shift_y])
        proj_ellipse_pts -= np.array([shift_x, shift_y])
        proj_rect_outline -= np.array([shift_x, shift_y])
        box_x_mm -= shift_x
        box_y_mm -= shift_y
        box_pts -= np.array([shift_x, shift_y])

        # The optical axis is at (0,0) in the projected view by construction
        optical_axis_proj = np.array([0, 0])

        # --- Projected axis ticks: project world x/y grid lines onto the image plane ---
        xticks_world = np.linspace(x_range_mm[0], x_range_mm[1], n_ticks)
        yticks_world = np.linspace(y_range_mm[0], y_range_mm[1], n_ticks)
        # Project x ticks (y=0), then shift
        xtick_proj = np.array([project_point([x, 0, 0])[0] for x in xticks_world]) - shift_x
        # Project y ticks (x=0), then shift
        ytick_proj = np.array([project_point([0, y, 0])[1] for y in yticks_world]) - shift_y

        # --- Axis limits: ensure optics FOV is centered ---
        all_x = np.concatenate([proj_rect_pts[:, 0], proj_ellipse_pts[:, 0], box_x_mm])
        all_y = np.concatenate([proj_rect_pts[:, 1], proj_ellipse_pts[:, 1], box_y_mm])
        pad_x = 0.05 * (all_x.max() - all_x.min())
        pad_y = 0.05 * (all_y.max() - all_y.min())
        xlim_proj = [all_x.min() - pad_x, all_x.max() + pad_x]
        ylim_proj = [all_y.min() - pad_y, all_y.max() + pad_y]

        # --- Optics image size circle ---
        # Find the maximum distance from the optical axis to any point on the FOV margin box
        dists = np.sqrt(box_x_mm**2 + box_y_mm**2)
        optics_radius = np.max(dists)
        optics_diameter = 2 * optics_radius

        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, (ax_world, ax_proj) = plt.subplots(1, 2, figsize=(18, 8))

        # --- World view (top-down) ---
        ax_world.set_aspect('equal')
        ax_world.plot(rect_corners[:, 0], rect_corners[:, 1], 'b-', lw=2, label='Water Spot Rectangle')
        ax_world.add_patch(Ellipse((0, 0), C, L, edgecolor='r', facecolor='r', alpha=0.3, lw=2, label='Water Spot Ellipse'))
        camera_size = 20
        ax_world.add_patch(Rectangle((Xc - camera_size/2, Yc - camera_size/2), camera_size, camera_size, color='green', alpha=0.7, label='Camera'))
        ax_world.plot([Xc], [Yc], 'go')
        ax_world.set_xlim(-300, 300)
        ax_world.set_ylim(-300, 300)
        ax_world.set_xlabel('X [mm]')
        ax_world.set_ylabel('Y [mm]')
        ax_world.set_title('Top-Down View (World Coordinates)')
        ax_world.legend(loc='upper right')
        ax_world.grid(True, alpha=0.3)

        # --- Projected view (camera image plane, optics FOV centered) ---
        ax_proj.set_aspect('equal')
        ax_proj.scatter(proj_rect_pts[:, 0], proj_rect_pts[:, 1], color='b', s=1, alpha=0.2, label='Projected Rectangle Pixels')
        ax_proj.plot(proj_ellipse_pts[:, 0], proj_ellipse_pts[:, 1], 'r-', linewidth=2, label='Projected Ellipse')
        ax_proj.plot(proj_rect_outline[:, 0], proj_rect_outline[:, 1], 'b-', linewidth=2, label='Projected Rectangle Outline')
        ax_proj.plot(box_x_mm, box_y_mm, 'k--', linewidth=2, label='Ellipse + Margin FOV')
        ax_proj.scatter(0, 0, color='green', s=100, label='Optical Axis (Principal Point)')
        # Draw optics image size circle
        circle = Circle((0, 0), optics_radius, fill=False, color='magenta', lw=2, label=f'Optics Image Size (D={optics_diameter:.1f} mm)')
        ax_proj.add_patch(circle)
        # Annotate diameter
        ax_proj.annotate(f"D = {optics_diameter:.1f} mm", xy=(optics_radius * 0.7, 0), color='magenta', fontsize=12, fontweight='bold')

        # Set axis limits so optics FOV is centered
        ax_proj.set_xlim(xlim_proj)
        ax_proj.set_ylim(ylim_proj)

        # Set custom ticks: at projected positions, but labeled with world coordinates
        ax_proj.set_xticks(xtick_proj)
        ax_proj.set_xticklabels([f"{x:.0f}" for x in xticks_world])
        ax_proj.set_yticks(ytick_proj)
        ax_proj.set_yticklabels([f"{y:.0f}" for y in yticks_world])

        ax_proj.set_xlabel('World X [mm] (projected)')
        ax_proj.set_ylabel('World Y [mm] (projected)')
        ax_proj.set_title(f'Projected View (Optical Axis at Center of FOV)\nEllipse+Margin FOV_H = {FOV_H:.1f}°, FOV_V = {FOV_V:.1f}°')
        ax_proj.legend(loc='best')
        ax_proj.grid(True, alpha=0.3)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        plt.close(fig)






    def load_toilet_data(self):
        if os.path.exists(self.csv_file):
            try:
                return pd.read_csv(self.csv_file)
            except:
                pass
        return pd.DataFrame(columns=self.table_columns)

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
        A, B, C, tilt, margin, shift = params
        new_row = {
            'Manufacturer': "DEFAULT",
            'Model': "DEFAULT",
            'Sub-Model': "DEFAULT",
            'A - Rim to Water depth (camera height) [mm]': A,
            'B - Water Spot Length [mm]': B,
            'C - Water Spot Width [mm]': C,
            'Camera Tilt [degrees]': tilt,
            'Margin [%]': margin,
            'Shift from Water Spot Width Edge [mm]': shift
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
