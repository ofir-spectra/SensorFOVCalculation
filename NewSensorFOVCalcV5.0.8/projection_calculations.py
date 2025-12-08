import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle, Arc

# Allowed aspect ratios (width:height)
ALLOWED_ASPECT_RATIOS = {
    "3:2": 3/2,
    "2:3": 2/3,
    "4:3": 4/3,
    "3:4": 3/4
}

def compute_max_projected_pixel_size(sensor_half_width, sensor_half_height, pixels_x, pixels_y, cam_pos, x_cam, y_cam, z_cam, H):
    """
    Compute the largest projected pixel size (corner-to-corner) on the water surface.
    """
    hw, hh = sensor_half_width, sensor_half_height
    x_edges = np.linspace(-hw, hw, pixels_x + 1)
    y_edges = np.linspace(-hh, hh, pixels_y + 1)
    def project_to_world(xi, yi):
        # Ray from camera through sensor at (xi, yi)
        ray_dir = (xi * x_cam + yi * y_cam + H * z_cam)
        norm_dir = np.linalg.norm(ray_dir)
        if norm_dir < 1e-8:
            return np.array([np.nan, np.nan])
        ray_dir = ray_dir / norm_dir
        # Water plane: z = 0
        if ray_dir[2] == 0:
            return np.array([np.nan, np.nan])
        t = -cam_pos[2] / ray_dir[2]
        point = cam_pos + t * ray_dir
        return point[:2]  # X, Y on water plane
    max_size = 0

    corner_ij = [
        (0, 0),
        (pixels_x-1, 0),
        (pixels_x-1, pixels_y-1),
        (0, pixels_y-1)
    ]
    max_size = 0
    for i, j in corner_ij:
        corners = [
            (x_edges[i], y_edges[j]),
            (x_edges[i+1], y_edges[j]),
            (x_edges[i+1], y_edges[j+1]),
            (x_edges[i], y_edges[j+1])
        ]
        scene_pts = np.array([project_to_world(x, y) for (x, y) in corners])
        lengths = [np.linalg.norm(scene_pts[k] - scene_pts[(k+1)%4]) for k in range(4)]
        max_pix_len = max(lengths)
        if max_pix_len > max_size:
            max_size = max_pix_len
    print(f"Largest projected IFOV (corners only): {max_size:.6f} mm")
    return max_size
    

def smooth_curve(y, window_size=3):
    """Moving average smoothing, window_size must be odd"""
    if window_size < 2 or window_size > len(y):
        return y
    pad = window_size // 2
    padded = np.pad(y, (pad, pad), mode='reflect')
    smooth = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
    # 'valid' convolution shortens by window_size-1, so output is len(y)
    # If your version differs, just use slicing: 
    if len(smooth) > len(y):
        extra = len(smooth)-len(y)
        start = extra//2
        smooth = smooth[start:start+len(y)]
    return smooth

def find_optimal_angle_for_coverage(params, angle_range=(0, 45), angle_step=1, smoothing_window=1):
    """
    Find the tilt angle that maximizes water coverage percentage,
    optionally smoothing the curve before finding the max.
    """
    # Get the angle array and coverage array over the angle range
    angles, coverages = calculate_water_coverage_curve(
        params, angle_range=angle_range, angle_step=angle_step
    )
    # Apply smoothing if needed
    if smoothing_window > 1 and len(angles) >= smoothing_window:
        coverages = smooth_curve(coverages, window_size=smoothing_window)
    # Find index of max smoothed coverage
    max_idx = int(np.argmax(coverages))
    return float(angles[max_idx]), float(coverages[max_idx])
    
    return best_angle, best_coverage

def calculate_water_coverage_curve(params, angle_range=(0, 45), angle_step=1, smoothing_window=1):
    """Calculate water coverage for a range of angles to create a plot"""
    angles = []
    coverages = []
    
    for angle in np.arange(angle_range[0], angle_range[1] + angle_step, angle_step):
        test_params = params.copy()
        test_params['Tilt'] = angle
        
        try:
            coverage = calculate_water_coverage_for_angle(test_params)
            angles.append(angle)
            coverages.append(coverage)
        except Exception:
            continue
    
    angles_np = np.array(angles)
    coverages_np = np.array(coverages)
    # Apply smoothing if requested
    if smoothing_window > 1 and len(coverages_np) >= smoothing_window:
        smoothed = smooth_curve(coverages_np, window_size=smoothing_window)
        # Center the angles array to match the smoothed coverage size
        offset = (len(angles_np) - len(smoothed)) // 2
        angles_np = angles_np[offset : offset + len(smoothed)]
        coverages_np = smoothed
    return angles_np, coverages_np

def calculate_water_coverage_for_angle(params):
    """Calculate water coverage percentage for a given tilt angle with realistic sensor"""
    A = params['A']
    B = params['B']
    C = params['C']
    theta_deg = params['Tilt']
    margin_percent = params['Margin']
    try:
        margin_percent = float(margin_percent)
    except Exception:
        margin_percent = 0.0
    shift = params['Shift']
    shift_axis = params['ShiftAxis']
    resolution = params['Resolution']
    
    margin_factor = 1.0 + (margin_percent / 100.0)
    H = A
    L = B * margin_factor
    W = C * margin_factor
    theta = np.deg2rad(theta_deg)

    # Camera position based on shift axis
    if shift_axis == 'X':
        Xc = W / 2 + shift
        Yc = 0
    else:  # Y-axis
        Xc = 0
        Yc = L / 2 + shift
    cam_pos = np.array([Xc, Yc, H])

    # Camera optical axis determined by user's tilt angle
    initial_optical_axis = np.array([0, 0, -1])  # Straight down
    
    # Apply tilt rotation based on shift axis
    if shift_axis == 'X':
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        z_cam = Ry @ initial_optical_axis
    else:  # Y-axis
        theta_corrected = -theta
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_corrected), -np.sin(theta_corrected)],
            [0, np.sin(theta_corrected), np.cos(theta_corrected)]
        ])
        z_cam = Rx @ initial_optical_axis
    
    z_cam = z_cam / np.linalg.norm(z_cam)

    # Camera basis vectors
    if abs(np.dot(z_cam, [0, 1, 0])) > 0.99:
        up_guess = np.array([1, 0, 0])
    else:
        up_guess = np.array([0, 1, 0])
    x_cam = np.cross(up_guess, z_cam)
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    y_cam = y_cam / np.linalg.norm(y_cam)

    # Project a 3D point to the camera image plane
    def project_point(p_world):
        v = p_world - cam_pos
        Xc = np.dot(v, x_cam)
        Yc = np.dot(v, y_cam)
        Zc = np.dot(v, z_cam)
        if abs(Zc) < 1e-10:
            return np.array([0, 0])
        x_img = H * Xc / Zc
        y_img = H * Yc / Zc
        return np.array([x_img, y_img])

    # Ellipse outline
    ell_res = 200
    angles = np.linspace(0, 2*np.pi, ell_res)
    ell_x = (C/2) * np.cos(angles)
    ell_y = (L/2) * np.sin(angles)
    ellipse_points = np.column_stack([ell_x, ell_y, np.zeros(ell_res)])
    proj_ellipse_pts = np.array([project_point(p) for p in ellipse_points])

    # Water spot bounds
    ellipse_min_x, ellipse_max_x = np.min(proj_ellipse_pts[:, 0]), np.max(proj_ellipse_pts[:, 0])
    ellipse_min_y, ellipse_max_y = np.min(proj_ellipse_pts[:, 1]), np.max(proj_ellipse_pts[:, 1])
    ellipse_cx = (ellipse_min_x + ellipse_max_x) / 2
    ellipse_cy = (ellipse_min_y + ellipse_max_y) / 2
    ellipse_half_width = (ellipse_max_x - ellipse_min_x) / 2
    ellipse_half_height = (ellipse_max_y - ellipse_min_y) / 2

    # Calculate required FOV with margin
    water_extent_from_optical_axis_x = max(abs(ellipse_min_x), abs(ellipse_max_x))
    water_extent_from_optical_axis_y = max(abs(ellipse_min_y), abs(ellipse_max_y))
    
    required_half_width = water_extent_from_optical_axis_x * (1 + margin_percent / 100)
    required_half_height = water_extent_from_optical_axis_y * (1 + margin_percent / 100)

    # REALISTIC SENSOR: Find smallest standard aspect ratio sensor that fits required FOV
    sensor_half_width, sensor_half_height, aspect_ratio_used = find_realistic_sensor_size(
        required_half_width, required_half_height)

    # Calculate water coverage with realistic sensor
    coverage_grid_size = 100
    sensor_x = np.linspace(-sensor_half_width, sensor_half_width, coverage_grid_size)
    sensor_y = np.linspace(-sensor_half_height, sensor_half_height, coverage_grid_size)
    sensor_xx, sensor_yy = np.meshgrid(sensor_x, sensor_y)
    
    water_pixels = 0
    total_pixels = coverage_grid_size * coverage_grid_size
    
    for i in range(coverage_grid_size):
        for j in range(coverage_grid_size):
            pixel_x = sensor_xx[i, j]
            pixel_y = sensor_yy[i, j]
            
            rel_x = pixel_x - ellipse_cx
            rel_y = pixel_y - ellipse_cy
            
            if (rel_x / ellipse_half_width)**2 + (rel_y / ellipse_half_height)**2 <= 1:
                water_pixels += 1
    
    coverage_percentage = (water_pixels / total_pixels) * 100
    return coverage_percentage

def find_realistic_sensor_size(required_half_width, required_half_height):
    best_sensor = None
    best_area = float('inf')
    best_aspect_name = None

    for aspect_name, aspect_ratio in ALLOWED_ASPECT_RATIOS.items():
        # Landscape: width >= height
        if aspect_ratio >= 1:
            # Try to fit by width first
            min_half_width = max(required_half_width, required_half_height * aspect_ratio)
            min_half_height = min_half_width / aspect_ratio
        else:
            # Portrait: height > width
            min_half_height = max(required_half_height, required_half_width / aspect_ratio)
            min_half_width = min_half_height * aspect_ratio

        area = min_half_width * min_half_height
        if area < best_area:
            best_area = area
            best_sensor = (min_half_width, min_half_height)
            best_aspect_name = aspect_name

    return best_sensor[0], best_sensor[1], best_aspect_name


def calculate_projection(params, x_range_mm=(-350, 350), y_range_mm=(-350, 350), n_ticks=9):
    A = params['A']
    B = params['B']
    C = params['C']
    theta_deg = params['Tilt']
    margin_percent = params['Margin']
    try:
        margin_percent = float(margin_percent)
    except Exception:
        margin_percent = 0.0
    shift = params['Shift']
    shift_axis = params['ShiftAxis']
    resolution = params['Resolution']
    
    margin_factor = 1.0 + (margin_percent / 100.0)
    H = A
    L = B * margin_factor
    W = C * margin_factor
    theta = np.deg2rad(theta_deg)

    # Camera position based on shift axis
    if shift_axis == 'X':
        Xc = W / 2 + shift
        Yc = 0
    else:  # Y-axis
        Xc = 0
        Yc = L / 2 + shift
    cam_pos = np.array([Xc, Yc, H])

    # Camera optical axis determined by user's tilt angle
    initial_optical_axis = np.array([0, 0, -1])  # Straight down
    
    # Apply tilt rotation based on shift axis
    if shift_axis == 'X':
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        z_cam = Ry @ initial_optical_axis
    else:  # Y-axis
        theta_corrected = -theta
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_corrected), -np.sin(theta_corrected)],
            [0, np.sin(theta_corrected), np.cos(theta_corrected)]
        ])
        z_cam = Rx @ initial_optical_axis
    
    z_cam = z_cam / np.linalg.norm(z_cam)

    # Calculate OPTIMAL TILT ANGLE based on maximum water coverage
    optimal_angle, optimal_coverage = find_optimal_angle_for_coverage(params, angle_step=smoothness, smoothing_window=smoothing_window)

    # Calculate water coverage curve for plotting
    coverage_angles, coverage_values = calculate_water_coverage_curve(params)

    # Apply correct sign convention to optimal angle
    if shift_axis == 'X':
        optimal_angle_corrected = -optimal_angle if Xc > 0 else optimal_angle
    else:
        optimal_angle_corrected = -optimal_angle if Yc > 0 else optimal_angle

    # Camera basis vectors
    if abs(np.dot(z_cam, [0, 1, 0])) > 0.99:
        up_guess = np.array([1, 0, 0])
    else:
        up_guess = np.array([0, 1, 0])
    x_cam = np.cross(up_guess, z_cam)
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    y_cam = y_cam / np.linalg.norm(y_cam)

    # Project a 3D point to the camera image plane
    def project_point(p_world):
        v = p_world - cam_pos
        Xc = np.dot(v, x_cam)
        Yc = np.dot(v, y_cam)
        Zc = np.dot(v, z_cam)
        if abs(Zc) < 1e-10:
            return np.array([0, 0])
        x_img = H * Xc / Zc
        y_img = H * Yc / Zc
        return np.array([x_img, y_img])

    # Rectangle grid
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

    # Rectangle corners
    rect_corners = np.array([
        [-W/2, -L/2, 0], [ W/2, -L/2, 0], [ W/2,  L/2, 0], [-W/2,  L/2, 0], [-W/2, -L/2, 0]
    ])
    proj_rect_outline = np.array([project_point(p) for p in rect_corners])

    # Water spot bounds for reference
    ellipse_min_x, ellipse_max_x = np.min(proj_ellipse_pts[:, 0]), np.max(proj_ellipse_pts[:, 0])
    ellipse_min_y, ellipse_max_y = np.min(proj_ellipse_pts[:, 1]), np.max(proj_ellipse_pts[:, 1])
    ellipse_cx = (ellipse_min_x + ellipse_max_x) / 2
    ellipse_cy = (ellipse_min_y + ellipse_max_y) / 2
    ellipse_half_width = (ellipse_max_x - ellipse_min_x) / 2
    ellipse_half_height = (ellipse_max_y - ellipse_min_y) / 2

    # Calculate required FOV with margin
    water_extent_from_optical_axis_x = max(abs(ellipse_min_x), abs(ellipse_max_x))
    water_extent_from_optical_axis_y = max(abs(ellipse_min_y), abs(ellipse_max_y))
    
    required_half_width = water_extent_from_optical_axis_x * (1 + margin_percent / 100)
    required_half_height = water_extent_from_optical_axis_y * (1 + margin_percent / 100)

    # REALISTIC SENSOR: Find smallest standard aspect ratio sensor that fits required FOV
    sensor_half_width, sensor_half_height, aspect_ratio_used = find_realistic_sensor_size(
        required_half_width, required_half_height)

    # FOV box coordinates - symmetrical around optical axis (0,0)
    box_x_mm = np.array([-sensor_half_width, sensor_half_width, sensor_half_width, 
                         -sensor_half_width, -sensor_half_width])
    box_y_mm = np.array([-sensor_half_height, -sensor_half_height, sensor_half_height, 
                         sensor_half_height, -sensor_half_height])

    # FOV calculation based on realistic sensor dimensions
    FOV_H = np.rad2deg(2 * np.arctan(sensor_half_width / H))
    FOV_V = np.rad2deg(2 * np.arctan(sensor_half_height / H))

    # Sensor resolution calculation based on realistic sensor dimensions
    sensor_width_mm = 2 * sensor_half_width
    sensor_height_mm = 2 * sensor_half_height
    
    pixels_x_sensor = round(sensor_width_mm / resolution)
    pixels_y_sensor = round(sensor_height_mm / resolution)

    px = int(np.ceil(sensor_width_mm / resolution))
    py = int(np.ceil(sensor_height_mm / resolution))

    # Calculate water coverage efficiency with realistic sensor
    def calculate_water_coverage_efficiency():
        coverage_grid_size = 50
        sensor_x = np.linspace(-sensor_half_width, sensor_half_width, coverage_grid_size)
        sensor_y = np.linspace(-sensor_half_height, sensor_half_height, coverage_grid_size)
        sensor_xx, sensor_yy = np.meshgrid(sensor_x, sensor_y)
        
        water_pixels = 0
        total_pixels = coverage_grid_size * coverage_grid_size
        
        for i in range(coverage_grid_size):
            for j in range(coverage_grid_size):
                pixel_x = sensor_xx[i, j]
                pixel_y = sensor_yy[i, j]
                
                rel_x = pixel_x - ellipse_cx
                rel_y = pixel_y - ellipse_cy
                
                if (rel_x / ellipse_half_width)**2 + (rel_y / ellipse_half_height)**2 <= 1:
                    water_pixels += 1
        
        coverage_percentage = (water_pixels / total_pixels) * 100
        return coverage_percentage, water_pixels, total_pixels

    water_coverage_percent, water_pixels, total_sensor_pixels = calculate_water_coverage_efficiency()

    # Keep previous naive calculation for comparison
    ellipse_width_with_margin = ellipse_half_width * 2 * (1 + margin_percent / 100)
    ellipse_height_with_margin = ellipse_half_height * 2 * (1 + margin_percent / 100)
    pixels_x_naive = round(ellipse_width_with_margin / resolution)
    pixels_y_naive = round(ellipse_height_with_margin / resolution)

    # Optics radius calculation based on sensor bounds
    all_x_coords = np.concatenate([box_x_mm, proj_ellipse_pts[:, 0]])
    all_y_coords = np.concatenate([box_y_mm, proj_ellipse_pts[:, 1]])
    dists = np.sqrt(all_x_coords**2 + all_y_coords**2)
    optics_radius = np.max(dists)
    optics_diameter = 2 * optics_radius

    # Projected axis ticks
    xticks_world = np.linspace(x_range_mm[0], x_range_mm[1], n_ticks)
    yticks_world = np.linspace(y_range_mm[0], y_range_mm[1], n_ticks)
    xtick_proj = np.array([project_point([x, 0, 0])[0] for x in xticks_world])
    ytick_proj = np.array([project_point([0, y, 0])[1] for y in yticks_world])

    # Fixed window size for projected view
    PROJ_LIMIT = 500
    xlim_proj = [-PROJ_LIMIT, PROJ_LIMIT]
    ylim_proj = [-PROJ_LIMIT, PROJ_LIMIT]

    # BALANCED LAYOUT: Create figure with proper sizing
    fig = plt.figure(figsize=(10, 6), dpi=100)  # Use this exact size
    
    ax_world = plt.subplot2grid((2, 2), (0, 0))      # Top left
    ax_proj = plt.subplot2grid((2, 2), (0, 1))       # Top right  
    ax_side = plt.subplot2grid((2, 2), (1, 0))       # Bottom left
    ax_coverage = plt.subplot2grid((2, 2), (1, 1))   # Bottom right

    # World view (top-down)
    ax_world.set_aspect('equal')
    ax_world.plot(rect_corners[:, 0], rect_corners[:, 1], 'b-', lw=2, label='Water Spot Rectangle')
    ax_world.add_patch(Ellipse((0, 0), C, L, edgecolor='r', facecolor='r', alpha=0.3, lw=2, label='Water Spot Ellipse'))
    camera_size = 20
    color = 'green' if shift_axis == 'X' else 'blue'
    ax_world.add_patch(Rectangle((Xc - camera_size/2, Yc - camera_size/2), camera_size, camera_size, 
                               color=color, alpha=0.7, label=f'Camera ({shift_axis}-axis)'))
    ax_world.plot([Xc], [Yc], 'go' if shift_axis == 'X' else 'bo')
    
    # FOV lines to water spot edges
    fov_points_world = [
        [-C/2, -L/2, 0], [C/2, -L/2, 0], [C/2, L/2, 0], [-C/2, L/2, 0],
        [0, -L/2, 0], [0, L/2, 0], [-C/2, 0, 0], [C/2, 0, 0]
    ]
    for point in fov_points_world:
        ax_world.plot([Xc, point[0]], [Yc, point[1]], 'gray', linestyle='--', linewidth=1, alpha=0.6)
    
    ax_world.set_xlim(-300, 300)
    ax_world.set_ylim(-300, 300)
    ax_world.set_xlabel('X [mm]')
    ax_world.set_ylabel('Y [mm]')
    ax_world.set_title('Top-Down View (World Coordinates)')
    ax_world.legend(loc='upper right', fontsize=8)
    ax_world.grid(True, alpha=0.3)

    # Side view
    ax_side.set_aspect('equal')
    
    toilet_width = 50
    toilet_bottom = -20
    if shift_axis == 'X':
        ax_side.add_patch(Rectangle((-W/2, toilet_bottom), W, toilet_width, 
                                color='lightgray', alpha=0.5, label='Toilet Bowl (side)'))
        ax_side.plot([-C/2, C/2], [0, 0], 'r-', lw=3, label='Water Surface')
    else:
        ax_side.add_patch(Rectangle((-L/2, toilet_bottom), L, toilet_width, 
                                color='lightgray', alpha=0.5, label='Toilet Bowl (side)'))
        ax_side.plot([-L/2, L/2], [0, 0], 'r-', lw=3, label='Water Surface')
    
    cam_width = 15
    if shift_axis == 'X':
        side_cam_x = Xc
        side_cam_z = H
        optical_axis_2d = np.array([z_cam[0], z_cam[2]])
    else:
        side_cam_x = Yc
        side_cam_z = H
        optical_axis_2d = np.array([z_cam[1], z_cam[2]])

    camera_center_x = side_cam_x
    camera_center_z = side_cam_z
    
    sensor_normal_2d = optical_axis_2d / np.linalg.norm(optical_axis_2d)
    sensor_tangent_2d = np.array([-sensor_normal_2d[1], sensor_normal_2d[0]])
    sensor_half_length = cam_width / 2
    sensor_start = np.array([camera_center_x, camera_center_z]) - sensor_half_length * sensor_tangent_2d
    sensor_end = np.array([camera_center_x, camera_center_z]) + sensor_half_length * sensor_tangent_2d

    ax_side.plot([sensor_start[0], sensor_end[0]], [sensor_start[1], sensor_end[1]], 
                color=color, linewidth=4, label='Camera Sensor Plane')

    purple_arrow_length = camera_center_z * 1.1
    ax_side.arrow(camera_center_x, 0, 0, purple_arrow_length, 
                head_width=8, head_length=8, fc='purple', ec='purple', linewidth=2, 
                label='Water Normal Vector')

    axis_length = 80
    optical_axis_normalized = optical_axis_2d / np.linalg.norm(optical_axis_2d)
    ax_side.arrow(camera_center_x, camera_center_z, 
                 axis_length * optical_axis_normalized[0], axis_length * optical_axis_normalized[1],
                 head_width=8, head_length=8, fc='orange', ec='orange', linewidth=3,
                 label='Camera Optical Axis (⊥ to sensor)')

    # Optimal tilt angle vector with correct direction
    optimal_axis_2d = np.array([np.sin(np.radians(optimal_angle_corrected)), -np.cos(np.radians(optimal_angle_corrected))])
    ax_side.arrow(camera_center_x, camera_center_z,
                 axis_length * optimal_axis_2d[0] * 0.8, axis_length * optimal_axis_2d[1] * 0.8,
                 head_width=6, head_length=6, fc='cyan', ec='cyan', linewidth=2, linestyle='--',
                 label=f'Optimal Tilt Angle ({optimal_angle:.1f}°)')

    arc_radius = 25
    if theta_deg != 0:
        if theta_deg < 0:
            theta1 = theta_deg
            theta2 = 0
        else:
            theta1 = 0
            theta2 = theta_deg
        
        arc = Arc((camera_center_x, camera_center_z), 2*arc_radius, 2*arc_radius,
                  angle=0, theta1=theta1, theta2=theta2, color='red', linewidth=3)
        ax_side.add_patch(arc)

        mid_angle = (theta1 + theta2) / 2
        text_x = camera_center_x + (arc_radius * 0.6) * np.cos(np.radians(mid_angle))
        text_z = camera_center_z + (arc_radius * 0.6) * np.sin(np.radians(mid_angle))
        ax_side.text(text_x, text_z, f'{theta_deg:.0f}',
                    fontsize=12, color='red', weight='bold', ha='center', va='center')

    # Side view text
    ax_side.text(0.02, 0.98, f'OPTIMAL TILT ANGLE: {optimal_angle:.1f}°',
                transform=ax_side.transAxes, fontsize=11, color='black', weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9),
                verticalalignment='top')

    water_edge_points = [(-C/2, 0), (C/2, 0), (0, 0)]
    for wp_x, wp_z in water_edge_points:
        if shift_axis == 'Y':
            wp_x = wp_z
            wp_z = 0
        ax_side.plot([camera_center_x, wp_x], [camera_center_z, wp_z], 
                    color='gray', linestyle='--', linewidth=1, alpha=0.6)
    
    ax_side.set_xlim(-300, 300)
    ax_side.set_ylim(-50, H + 120)
    if shift_axis == 'X':
        ax_side.set_xlabel('X [mm] (side view)')
        ax_side.set_title('Side View (X-Z plane) - User Input Angle + Optimal')
    else:
        ax_side.set_xlabel('Y [mm] (side view)')
        ax_side.set_title('Side View (Y-Z plane) - User Input Angle + Optimal')
    ax_side.set_ylabel('Z [mm] (height)')
    ax_side.legend(loc='upper right', fontsize=8)
    ax_side.grid(True, alpha=0.3)

    # Projected view with realistic sensor
    ax_proj.set_aspect('equal')
    ax_proj.scatter(proj_rect_pts[:, 0], proj_rect_pts[:, 1], color='b', s=1, alpha=0.2, label='Projected Rectangle Pixels')
    ax_proj.plot(proj_ellipse_pts[:, 0], proj_ellipse_pts[:, 1], 'r-', linewidth=2, label='Projected Ellipse')
    ax_proj.plot(proj_rect_outline[:, 0], proj_rect_outline[:, 1], 'b-', linewidth=2, label='Projected Rectangle Outline')
    
    # Realistic sensor FOV box centered around optical axis
    ax_proj.plot(box_x_mm, box_y_mm, 'k--', linewidth=2, label=f'Realistic Sensor FOV ({aspect_ratio_used})')
    
    # Show both optical axis and projection center for comparison
    ax_proj.scatter(0, 0, color='red', s=200, marker='+', linewidth=3, label='Optical Axis (Principal Point)')
    ax_proj.scatter(ellipse_cx, ellipse_cy, color='purple', s=100, marker='x', 
                   linewidth=2, label='Water Spot Center')
    
    # Cross-hairs to clearly show optical axis
    ax_proj.axhline(y=0, color='red', linestyle='-', alpha=0.3, linewidth=1)
    ax_proj.axvline(x=0, color='red', linestyle='-', alpha=0.3, linewidth=1)
    
    circle = Circle((0, 0), optics_radius, fill=False, color='black', lw=2, 
                   label=f'Optics Image Size (D={optics_diameter:.1f} mm)')
    ax_proj.add_patch(circle)

    # Enhanced overlay with realistic sensor information
    resolution_threshold = 2500
    is_bad_resolution = max(pixels_x_sensor, pixels_y_sensor) > resolution_threshold
    
    # Create all overlay lines with realistic sensor info
    if is_bad_resolution:
        overlay_lines = [
            "BAD TILT/SHIFT VALUES",
            f"REALISTIC SENSOR ({aspect_ratio_used}): {pixels_x_sensor} x {pixels_y_sensor} [px]",
            f"NAIVE RESOLUTION: {pixels_x_naive} x {pixels_y_naive} [px]", 
            f"WATER COVERAGE: {water_coverage_percent:.1f}%",
            f"OPTIMAL TILT ANGLE: {optimal_angle:.1f}°",
            f"PROJECTION OFFSET: {np.sqrt(ellipse_cx**2 + ellipse_cy**2):.1f} mm from optical axis"
        ]
    else:
        overlay_lines = [
            f"REALISTIC SENSOR ({aspect_ratio_used}): {pixels_x_sensor} x {pixels_y_sensor} [px]",
            f"NAIVE RESOLUTION: {pixels_x_naive} x {pixels_y_naive} [px]",
            f"WATER COVERAGE: {water_coverage_percent:.1f}%",
            f"OPTIMAL TILT ANGLE: {optimal_angle:.1f}°", 
            f"PROJECTION OFFSET: {np.sqrt(ellipse_cx**2 + ellipse_cy**2):.1f} mm from optical axis"
        ]

    # Join all lines into single text with newlines
    overlay_text = "\n".join(overlay_lines)
    
    # Create the text box with tight padding
    ax_proj.text(
        0.02, 0.98, overlay_text,
        fontsize=9,  # Smaller font to fit better
        color='black',
        fontweight='bold',
        ha='left',
        va='top',
        transform=ax_proj.transAxes,
        bbox=dict(
            boxstyle='round,pad=0.2',
            facecolor='white',
            edgecolor='black',
            alpha=0.9
        ),
        zorder=10
    )

    ax_proj.set_xlim(xlim_proj)
    ax_proj.set_ylim(ylim_proj)
    
    # Smart tick filtering
    visible_xticks = []
    visible_xtick_labels = []
    visible_yticks = []
    visible_ytick_labels = []
    
    for i, (tick_pos, world_pos) in enumerate(zip(xtick_proj, xticks_world)):
        if xlim_proj[0] <= tick_pos <= xlim_proj[1]:
            visible_xticks.append(tick_pos)
            visible_xtick_labels.append(f"{world_pos:.0f}")
    
    for i, (tick_pos, world_pos) in enumerate(zip(ytick_proj, yticks_world)):
        if ylim_proj[0] <= tick_pos <= ylim_proj[1]:
            visible_yticks.append(tick_pos)
            visible_ytick_labels.append(f"{world_pos:.0f}")
    
    if visible_xticks:
        ax_proj.set_xticks(visible_xticks)
        ax_proj.set_xticklabels(visible_xtick_labels)
    else:
        ax_proj.set_xticks([])
        
    if visible_yticks:
        ax_proj.set_yticks(visible_yticks)
        ax_proj.set_yticklabels(visible_ytick_labels)
    else:
        ax_proj.set_yticks([])

    ax_proj.set_xlabel('World X [mm] (projected)')
    ax_proj.set_ylabel('World Y [mm] (projected)')
    ax_proj.set_title(f'Projected View - REALISTIC SENSOR {aspect_ratio_used} (User: {theta_deg:.1f}°, Optimal: {optimal_angle:.1f}°)\nSensor FOV_H = {FOV_H:.1f}°, FOV_V = {FOV_V:.1f}°')
    ax_proj.legend(loc='best', fontsize=8)
    ax_proj.grid(True, alpha=0.3)

    # Water Coverage vs. Angle Plot
    ax_coverage.plot(coverage_angles, coverage_values, 'b-', linewidth=2, label='Water Coverage')
    ax_coverage.scatter(optimal_angle, optimal_coverage, color='red', s=100, zorder=5, 
                       label=f'Optimal ({optimal_angle:.1f}°, {optimal_coverage:.1f}%)')
    ax_coverage.axvline(x=optimal_angle, color='red', linestyle='--', alpha=0.7)
    ax_coverage.axhline(y=optimal_coverage, color='red', linestyle='--', alpha=0.7)
    
    # Mark current user angle if it's within range
    if 0 <= abs(theta_deg) <= 60:
        current_coverage = water_coverage_percent
        ax_coverage.scatter(abs(theta_deg), current_coverage, color='orange', s=80, zorder=5,
                           marker='s', label=f'Current ({abs(theta_deg):.1f}°, {current_coverage:.1f}%)')
    
    ax_coverage.set_xlabel('Tilt Angle [degrees]')
    ax_coverage.set_ylabel('Water Coverage [%]')
    ax_coverage.set_title('Water Coverage vs. Tilt Angle')
    ax_coverage.grid(True, alpha=0.3)
    ax_coverage.legend(fontsize=8)
    ax_coverage.set_xlim(0, 60)
    ax_coverage.set_ylim(0, max(100, np.max(coverage_values) * 1.1))

    # Adjust subplot spacing to prevent truncation
    plt.tight_layout(pad=1.5)
    
    # Return enhanced data including coverage curve
    return fig, {
        'pixels_x_naive': pixels_x_naive,
        'pixels_y_naive': pixels_y_naive, 
        'pixels_x': pixels_x_sensor,
        'pixels_y': pixels_y_sensor,
        'FOV_H': FOV_H,
        'FOV_V': FOV_V,
        'optics_diameter': optics_diameter,
        'optimal_tilt_angle': optimal_angle,
        'is_bad_resolution': is_bad_resolution,
        'projection_offset': np.sqrt(ellipse_cx**2 + ellipse_cy**2),
        'water_coverage_percent': water_coverage_percent,
        'water_pixels': water_pixels,
        'total_sensor_pixels': total_sensor_pixels,
        'sensor_aspect_ratio': aspect_ratio_used,
        'sensor_half_width': sensor_half_width,
        'sensor_half_height': sensor_half_height,
        'coverage_angles': coverage_angles,
        'coverage_values': coverage_values
    }

def get_plot_data(params, x_range_mm=(-350, 350), y_range_mm=(-350, 350), n_ticks=9, smoothness=2, smoothing_window=1):
    import numpy as np

    A = params['A']
    B = params['B']
    C = params['C']
    theta_deg = params['Tilt']
    margin_percent = params['Margin']
    try:
        margin_percent = float(margin_percent)
    except Exception:
        margin_percent = 0.0
    shift = params['Shift']
    shift_axis = params['ShiftAxis']
    resolution = params['Resolution']
    margin_factor = 1.0 + (margin_percent / 100.0)
    H = A
    L = B * margin_factor
    W = C * margin_factor
    theta = np.deg2rad(theta_deg)

    # Camera position based on shift axis
    if shift_axis == 'X':
        Xc = W / 2 + shift
        Yc = 0
    else:
        Xc = 0
        Yc = L / 2 + shift
    cam_pos = np.array([Xc, Yc, H])

    # Camera optical axis
    initial_optical_axis = np.array([0, 0, -1])
    if shift_axis == 'X':
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        z_cam = Ry @ initial_optical_axis
    else:
        theta_corrected = -theta
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_corrected), -np.sin(theta_corrected)],
            [0, np.sin(theta_corrected), np.cos(theta_corrected)]
        ])
        z_cam = Rx @ initial_optical_axis
    z_cam = z_cam / np.linalg.norm(z_cam)

    # Optimal tilt/coverage
    from projection_calculations import find_optimal_angle_for_coverage, calculate_water_coverage_curve
    optimal_angle, optimal_coverage = find_optimal_angle_for_coverage(params, smoothing_window=smoothing_window)
    coverage_angles, coverage_values = calculate_water_coverage_curve(params, smoothing_window=smoothing_window)
    if shift_axis == 'X':
        optimal_angle_corrected = -optimal_angle if Xc > 0 else optimal_angle
    else:
        optimal_angle_corrected = -optimal_angle if Yc > 0 else optimal_angle

    # Camera basis vectors
    if abs(np.dot(z_cam, [0, 1, 0])) > 0.99:
        up_guess = np.array([1, 0, 0])
    else:
        up_guess = np.array([0, 1, 0])
    x_cam = np.cross(up_guess, z_cam)
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    y_cam = y_cam / np.linalg.norm(y_cam)

    def project_point(p_world):
        v = p_world - cam_pos
        Xc_ = np.dot(v, x_cam)
        Yc_ = np.dot(v, y_cam)
        Zc_ = np.dot(v, z_cam)
        if abs(Zc_) < 1e-10:
            return np.array([0, 0])
        x_img = H * Xc_ / Zc_
        y_img = H * Yc_ / Zc_
        return np.array([x_img, y_img])

    # Plotting arrays
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

    # Rectangle corners
    rect_corners = np.array([
        [-W/2, -L/2, 0], [ W/2, -L/2, 0], [ W/2, L/2, 0], [-W/2, L/2, 0], [-W/2, -L/2, 0]
    ])
    proj_rect_outline = np.array([project_point(p) for p in rect_corners])

    # Water spot bounds for reference
    ellipse_min_x, ellipse_max_x = np.min(proj_ellipse_pts[:, 0]), np.max(proj_ellipse_pts[:, 0])
    ellipse_min_y, ellipse_max_y = np.min(proj_ellipse_pts[:, 1]), np.max(proj_ellipse_pts[:, 1])
    ellipse_cx = (ellipse_min_x + ellipse_max_x) / 2
    ellipse_cy = (ellipse_min_y + ellipse_max_y) / 2
    ellipse_half_width = (ellipse_max_x - ellipse_min_x) / 2
    ellipse_half_height = (ellipse_max_y - ellipse_min_y) / 2

    # Sensor sizing
    from projection_calculations import find_realistic_sensor_size
    water_extent_from_optical_axis_x = max(abs(ellipse_min_x), abs(ellipse_max_x))
    water_extent_from_optical_axis_y = max(abs(ellipse_min_y), abs(ellipse_max_y))
    required_half_width = water_extent_from_optical_axis_x * (1 + margin_percent / 100)
    required_half_height = water_extent_from_optical_axis_y * (1 + margin_percent / 100)
    sensor_half_width, sensor_half_height, aspect_ratio_used = find_realistic_sensor_size(
        required_half_width, required_half_height
    )
    sensor_width_mm = 2 * sensor_half_width
    sensor_height_mm = 2 * sensor_half_height
    px = int(np.ceil(sensor_width_mm / resolution))
    py = int(np.ceil(sensor_height_mm / resolution))

    # FOV box coordinates
    box_x_mm = np.array([-sensor_half_width, sensor_half_width, sensor_half_width, -sensor_half_width, -sensor_half_width])
    box_y_mm = np.array([-sensor_half_height, -sensor_half_height, sensor_half_height, sensor_half_height, -sensor_half_height])

    enforce_ifov = params.get('EnforceIFOV', False)
    pixel_pitch_um = params.get("PixelPitch", 2.0)
    pixels_x = max(2, int(np.ceil(sensor_width_mm / resolution)))
    pixels_y = max(2, int(np.ceil(sensor_height_mm / resolution)))
    max_pixels = params.get('MaxSensorRes', 5000)
    px = min(px, max_pixels)
    py = min(py, max_pixels)

    if not enforce_ifov:
        pixels_x_sensor = int(pixels_x)
        pixels_y_sensor = int(pixels_y)
        max_ifov = resolution
    else:
        from projection_calculations import compute_max_projected_pixel_size
        max_ifov = compute_max_projected_pixel_size(
            sensor_half_width, sensor_half_height, pixels_x, pixels_y,
            cam_pos, x_cam, y_cam, z_cam, H
        )
        ifov_factor = max(1.0, max_ifov / resolution)
        new_pixels_x = int(np.ceil(pixels_x * ifov_factor))
        new_pixels_y = int(np.ceil(pixels_y * ifov_factor))
        limit_factor = max(new_pixels_x / max_pixels, new_pixels_y / max_pixels, 1.0)
        if limit_factor > 1.0:
            new_pixels_x = min(new_pixels_x, max_pixels)
            new_pixels_y = min(new_pixels_y, max_pixels)
            max_ifov = resolution * limit_factor
        pixels_x_sensor = int(new_pixels_x)
        pixels_y_sensor = int(new_pixels_y)

    ellipse_width_with_margin = ellipse_half_width * 2 * (1 + margin_percent / 100)
    ellipse_height_with_margin = ellipse_half_height * 2 * (1 + margin_percent / 100)
    pixels_x_naive = round(ellipse_width_with_margin / resolution)
    pixels_y_naive = round(ellipse_height_with_margin / resolution)

    def calculate_water_coverage_efficiency():
        coverage_grid_size = 100
        sensor_x = np.linspace(-sensor_half_width, sensor_half_width, coverage_grid_size)
        sensor_y = np.linspace(-sensor_half_height, sensor_half_height, coverage_grid_size)
        sensor_xx, sensor_yy = np.meshgrid(sensor_x, sensor_y)
        water_pixels = 0
        total_pixels = coverage_grid_size * coverage_grid_size
        for i in range(coverage_grid_size):
            for j in range(coverage_grid_size):
                pixel_x = sensor_xx[i, j]
                pixel_y = sensor_yy[i, j]
                rel_x = pixel_x - ellipse_cx
                rel_y = pixel_y - ellipse_cy
                if (rel_x / ellipse_half_width)**2 + (rel_y / ellipse_half_height)**2 <= 1:
                    water_pixels += 1
        coverage_percentage = (water_pixels / total_pixels) * 100
        return coverage_percentage, water_pixels, total_pixels

    water_coverage_percent, water_pixels, total_sensor_pixels = calculate_water_coverage_efficiency()

    # Optics calculations
    all_x_coords = np.concatenate([box_x_mm, proj_ellipse_pts[:, 0]])
    all_y_coords = np.concatenate([box_y_mm, proj_ellipse_pts[:, 1]])
    dists = np.sqrt(all_x_coords**2 + all_y_coords**2)
    optics_radius = np.max(dists)
    optics_diameter = 2 * optics_radius

    # FOV points for world view
    fov_points_world = [
        [-C/2, -L/2, 0], [C/2, -L/2, 0], [C/2, L/2, 0], [-C/2, L/2, 0],
        [0, -L/2, 0], [0, L/2, 0], [-C/2, 0, 0], [C/2, 0, 0]
    ]

    # Side view calculations
    if shift_axis == 'X':
        side_cam_x = Xc
        side_cam_z = H
        optical_axis_2d = np.array([z_cam[0], z_cam[2]])
    else:
        side_cam_x = Yc
        side_cam_z = H
        optical_axis_2d = np.array([z_cam[1], z_cam[2]])
    camera_center_x = side_cam_x
    camera_center_z = side_cam_z
    sensor_normal_2d = optical_axis_2d / np.linalg.norm(optical_axis_2d)
    sensor_tangent_2d = np.array([-sensor_normal_2d[1], sensor_normal_2d[0]])
    optical_axis_normalized = optical_axis_2d / np.linalg.norm(optical_axis_2d)
    optimal_axis_2d = np.array([np.sin(np.radians(optimal_angle_corrected)), -np.cos(np.radians(optimal_angle_corrected))])

    # IFOV Map Calculation
    ifov_map = None
    min_ifov = 0.0
    max_ifov_for_map = 0.0
    if params.get("EnforceIFOV", False):
        px = int(np.ceil(2 * sensor_half_width / resolution))
        py = int(np.ceil(2 * sensor_half_height / resolution))
        x_edges = np.linspace(-sensor_half_width, sensor_half_width, px + 1)
        y_edges = np.linspace(-sensor_half_height, sensor_half_height, py + 1)
        ifov_map = np.zeros((px, py))
        for j in range(min(10, py)):
            for i in range(px):
                corners = [
                    (x_edges[i], y_edges[j]),
                    (x_edges[i+1], y_edges[j]),
                    (x_edges[i+1], y_edges[j+1]),
                    (x_edges[i], y_edges[j+1])
                ]
                world_xy = []
                for x_s, y_s in corners:
                    ray = x_s * x_cam + y_s * y_cam + H * z_cam
                    ray = ray / np.linalg.norm(ray)
                    t = -cam_pos[2] / ray[2]
                    pt = cam_pos + t * ray
                    world_xy.append(pt[:2])
                world_xy = np.array(world_xy)
                edges = [np.linalg.norm(world_xy[k] - world_xy[(k+1)%4]) for k in range(4)]
                ifov_map[i, j] = max(edges)
        for j in range(max(py-10, 0), py):
            for i in range(px):
                corners = [
                    (x_edges[i], y_edges[j]),
                    (x_edges[i+1], y_edges[j]),
                    (x_edges[i+1], y_edges[j+1]),
                    (x_edges[i], y_edges[j+1])
                ]
                world_xy = []
                for x_s, y_s in corners:
                    ray = x_s * x_cam + y_s * y_cam + H * z_cam
                    ray = ray / np.linalg.norm(ray)
                    t = -cam_pos[2] / ray[2]
                    pt = cam_pos + t * ray
                    world_xy.append(pt[:2])
                world_xy = np.array(world_xy)
                edges = [np.linalg.norm(world_xy[k] - world_xy[(k+1)%4]) for k in range(4)]
                ifov_map[i, j] = max(edges)
        for i in range(min(10, px)):
            for j in range(py):
                corners = [
                    (x_edges[i], y_edges[j]),
                    (x_edges[i+1], y_edges[j]),
                    (x_edges[i+1], y_edges[j+1]),
                    (x_edges[i], y_edges[j+1])
                ]
                world_xy = []
                for x_s, y_s in corners:
                    ray = x_s * x_cam + y_s * y_cam + H * z_cam
                    ray = ray / np.linalg.norm(ray)
                    t = -cam_pos[2] / ray[2]
                    pt = cam_pos + t * ray
                    world_xy.append(pt[:2])
                world_xy = np.array(world_xy)
                edges = [np.linalg.norm(world_xy[k] - world_xy[(k+1)%4]) for k in range(4)]
                ifov_map[i, j] = max(edges)
        for i in range(max(px-10, 0), px):
            for j in range(py):
                corners = [
                    (x_edges[i], y_edges[j]),
                    (x_edges[i+1], y_edges[j]),
                    (x_edges[i+1], y_edges[j+1]),
                    (x_edges[i], y_edges[j+1])
                ]
                world_xy = []
                for x_s, y_s in corners:
                    ray = x_s * x_cam + y_s * y_cam + H * z_cam
                    ray = ray / np.linalg.norm(ray)
                    t = -cam_pos[2] / ray[2]
                    pt = cam_pos + t * ray
                    world_xy.append(pt[:2])
                world_xy = np.array(world_xy)
                edges = [np.linalg.norm(world_xy[k] - world_xy[(k+1)%4]) for k in range(4)]
                ifov_map[i, j] = max(edges)
        # Proper, robust minimal IFOV calculation (excluding zeros and NaNs)
        if ifov_map is not None:
            valid_ifov_vals = ifov_map[np.isfinite(ifov_map) & (ifov_map > 0)]
            min_ifov = float(np.min(valid_ifov_vals)) if valid_ifov_vals.size > 0 else 0.0
            max_ifov_for_map = float(np.max(valid_ifov_vals)) if valid_ifov_vals.size > 0 else 0.0
    pixel_pitch_um = float(params.get("PixelPitch", 2.0))
    pixels_x_sensor = int(pixels_x_sensor)
    pixels_y_sensor = int(pixels_y_sensor)
    sensor_width_mm = pixels_x_sensor * pixel_pitch_um / 1000.0
    sensor_height_mm = pixels_y_sensor * pixel_pitch_um / 1000.0
    # Return everything needed for downstream plotting
    return {
        'A': A, 'B': B, 'C': C, 'H': H, 'L': L, 'W': W,
        'theta_deg': theta_deg, 'shift_axis': shift_axis,
        'Xc': Xc, 'Yc': Yc,
        'rect_corners': rect_corners,
        'proj_rect_pts': proj_rect_pts,
        'proj_ellipse_pts': proj_ellipse_pts,
        'proj_rect_outline': proj_rect_outline,
        'box_x_mm': box_x_mm,
        'box_y_mm': box_y_mm,
        'fov_points_world': fov_points_world,
        'ellipse_cx': ellipse_cx,
        'ellipse_cy': ellipse_cy,
        'ellipse_half_width': ellipse_half_width,
        'ellipse_half_height': ellipse_half_height,
        'aspect_ratio_used': aspect_ratio_used,
        'pixels_x_sensor': pixels_x_sensor,
        'pixels_y_sensor': pixels_y_sensor,
        'pixels_x_naive': pixels_x_naive,
        'pixels_y_naive': pixels_y_naive,
        'max_ifov': max_ifov_for_map if enforce_ifov else max_ifov,
        'min_ifov': min_ifov,
        'water_coverage_percent': water_coverage_percent,
        'optimal_angle': optimal_angle,
        'optimal_coverage': optimal_coverage,
        'coverage_angles': coverage_angles,
        'coverage_values': coverage_values,
        'optics_radius': optics_radius,
        'optics_diameter': optics_diameter,
        'projection_offset': np.sqrt(ellipse_cx**2 + ellipse_cy**2),
        'camera_center_x': camera_center_x,
        'camera_center_z': camera_center_z,
        'sensor_tangent_2d': sensor_tangent_2d,
        'optical_axis_normalized': optical_axis_normalized,
        'optimal_axis_2d': optimal_axis_2d,
        'sensor_width_mm': sensor_width_mm,
        'sensor_height_mm': sensor_height_mm,
        'ifov_map': ifov_map,
        'px_sensor': px,
        'py_sensor': py
    }
