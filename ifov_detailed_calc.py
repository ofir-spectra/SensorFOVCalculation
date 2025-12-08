import numpy as np

def show_detailed_calculations(height, tilt_angle_deg, water_width=266.7, water_length=317.5):
    """Show all calculation steps with numerical values"""
    print("\n" + "=" * 80)
    print(f"DETAILED CALCULATIONS FOR TILT ANGLE: {tilt_angle_deg}°")
    print("=" * 80)
    
    # Step 1: Convert angle to radians
    tilt_rad = np.radians(tilt_angle_deg)
    print(f"\nStep 1: Convert tilt angle to radians")
    print(f"tilt_rad = {tilt_angle_deg}° × π/180")
    print(f"tilt_rad = {tilt_rad:.4f} radians")
    
    # Step 2: Calculate distances
    if tilt_angle_deg == 0:
        min_distance = height
        max_distance = height
        print(f"\nStep 2: Calculate distances (vertical camera)")
        print(f"min_distance = max_distance = height = {height} mm")
    else:
        # Minimum distance (closest point)
        min_distance = height * np.tan(tilt_rad)
        print(f"\nStep 2a: Calculate minimum distance")
        print(f"min_distance = height × tan(tilt_rad)")
        print(f"min_distance = {height} × tan({tilt_rad:.4f})")
        print(f"min_distance = {min_distance:.1f} mm")
        
        # Maximum distance (farthest point)
        max_distance = height / np.cos(tilt_rad)
        print(f"\nStep 2b: Calculate maximum distance")
        print(f"max_distance = height / cos(tilt_rad)")
        print(f"max_distance = {height} / cos({tilt_rad:.4f})")
        print(f"max_distance = {max_distance:.1f} mm")
    
    # Step 3: Calculate IFOV
    base_resolution = 0.22  # mm/pixel
    print(f"\nStep 3: Calculate IFOV")
    print(f"Base resolution = {base_resolution} mm/pixel")
    
    if tilt_angle_deg == 0:
        min_ifov = base_resolution
        max_ifov = base_resolution
        print("For vertical camera:")
        print(f"min_ifov = max_ifov = base_resolution = {base_resolution} mm/pixel")
    else:
        # Calculate IFOV considering perspective effects
        min_ifov = base_resolution * (min_distance / max_distance)
        print("\nStep 3a: Calculate minimum IFOV")
        print(f"min_ifov = base_resolution × (min_distance / max_distance)")
        print(f"min_ifov = {base_resolution} × ({min_distance:.1f} / {max_distance:.1f})")
        print(f"min_ifov = {min_ifov:.4f} mm/pixel")
        
        max_ifov = base_resolution * (max_distance / min_distance)
        print("\nStep 3b: Calculate maximum IFOV")
        print(f"max_ifov = base_resolution × (max_distance / min_distance)")
        print(f"max_ifov = {base_resolution} × ({max_distance:.1f} / {min_distance:.1f})")
        print(f"max_ifov = {max_ifov:.4f} mm/pixel")
    
    # Step 4: Calculate naive pixel requirements
    naive_pixels_x = int(water_width / base_resolution)
    naive_pixels_y = int(water_length / base_resolution)
    print("\nStep 4: Calculate naive pixel requirements")
    print(f"naive_pixels_x = water_width / base_resolution")
    print(f"naive_pixels_x = {water_width} / {base_resolution}")
    print(f"naive_pixels_x = {naive_pixels_x} pixels")
    print(f"\nnaive_pixels_y = water_length / base_resolution")
    print(f"naive_pixels_y = {water_length} / {base_resolution}")
    print(f"naive_pixels_y = {naive_pixels_y} pixels")
    
    if tilt_angle_deg != 0:
        # Step 5: Calculate required pixels considering projection
        ifov_ratio = max_ifov / base_resolution
        print("\nStep 5: Calculate required pixels with projection")
        print(f"ifov_ratio = max_ifov / base_resolution")
        print(f"ifov_ratio = {max_ifov:.4f} / {base_resolution}")
        print(f"ifov_ratio = {ifov_ratio:.2f}")
        
        required_pixels_x = int(naive_pixels_x * ifov_ratio)
        required_pixels_y = int(naive_pixels_y * ifov_ratio)
        print(f"\nrequired_pixels_x = naive_pixels_x × ifov_ratio")
        print(f"required_pixels_x = {naive_pixels_x} × {ifov_ratio:.2f}")
        print(f"required_pixels_x = {required_pixels_x} pixels")
        print(f"\nrequired_pixels_y = naive_pixels_y × ifov_ratio")
        print(f"required_pixels_y = {naive_pixels_y} × {ifov_ratio:.2f}")
        print(f"required_pixels_y = {required_pixels_y} pixels")

# Input parameters
height = 133  # mm (camera height)
tilt_angles = [0, 10, 20, 30]  # degrees

for angle in tilt_angles:
    show_detailed_calculations(height, angle)
