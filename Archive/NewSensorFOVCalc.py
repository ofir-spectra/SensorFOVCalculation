import tkinter as tk
from tkinter import ttk
import numpy as np

def plot_projection():
    import matplotlib.pyplot as plt
    # Get and validate user input
    try:
        W = float(entry_W.get())
        L = float(entry_L.get())
        H = float(entry_H.get())
        theta_deg = float(entry_theta.get())
        margin_percent = float(entry_margin.get())
    except ValueError:
        return  # Optionally, show an error dialog

    theta = np.deg2rad(theta_deg)
    Xc = H * np.tan(theta)
    Yc = 0
    cam_pos = np.array([Xc, Yc, H])
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
    plt.figure(figsize=(8, 8))
    plt.axis('equal')
    plt.fill(proj_rect_mm[:, 0], proj_rect_mm[:, 1], color=(0.7, 0.9, 1), edgecolor='b', linewidth=2, alpha=0.3, label='Projected Rectangle (Trapezoid)')
    plt.fill(proj_ellipse_mm[:, 0], proj_ellipse_mm[:, 1], color='r', edgecolor='r', linewidth=2, alpha=0.5, label='Projected Ellipse (Oval)')
    box_x_mm_closed = np.append(box_x_mm, box_x_mm[0])
    box_y_mm_closed = np.append(box_y_mm, box_y_mm[0])
    plt.plot(box_x_mm_closed, box_y_mm_closed, 'k--', linewidth=2, label='Ellipse+Margin FOV')
    max_abs_x = np.max(np.abs(np.concatenate([proj_rect_mm[:, 0], proj_ellipse_mm[:, 0], box_x_mm])))
    max_abs_y = np.max(np.abs(np.concatenate([proj_rect_mm[:, 1], proj_ellipse_mm[:, 1], box_y_mm])))
    max_range = max(max_abs_x, max_abs_y) + 10
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    plt.axhline(0, color=(0.5, 0.5, 0.5), linestyle='--', linewidth=1)
    plt.axvline(0, color=(0.5, 0.5, 0.5), linestyle='--', linewidth=1)
    plt.xlabel('Horizontal projection [mm]')
    plt.ylabel('Vertical projection [mm]')
    plt.title(f'Projected Water Surface and Oval on Camera Image Plane\nEllipse+Margin FOV_H = {FOV_H_margin:.1f}°, FOV_V = {FOV_V_margin:.1f}°')
    plt.legend(loc='best')
    plt.grid(True)

    # Embed plot in Tkinter window
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    for widget in plot_frame.winfo_children():
        widget.destroy()
    fig = plt.gcf()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    plt.close(fig)

# --- TKINTER GUI ---
root = tk.Tk()
root.title("Projection Parameter Input")

# Frame for plot
plot_frame = ttk.Frame(root)
plot_frame.grid(row=0, column=2, rowspan=10, padx=10, pady=10, sticky="nsew")

# Labels and entries for each parameter
labels = [
    ("Rectangle Width (W, mm):", "206"),
    ("Rectangle Length (L, mm):", "264"),
    ("Camera Height (H, mm):", "168"),
    ("Camera Tilt (deg):", "30"),
    ("Margin Percent:", "10")
]
entries = []

for i, (label_text, default) in enumerate(labels):
    label = ttk.Label(root, text=label_text)
    label.grid(row=i, column=0, sticky="e", padx=5, pady=5)
    entry = ttk.Entry(root)
    entry.insert(0, default)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries.append(entry)

entry_W, entry_L, entry_H, entry_theta, entry_margin = entries

plot_button = ttk.Button(root, text="Plot", command=plot_projection)
plot_button.grid(row=len(labels), column=0, columnspan=2, pady=10)

root.mainloop()
