import numpy as np

def calculate_resolution_steps(height, tilt_angle_deg, target_resolution=0.22):
    """Show detailed resolution calculation steps"""
    print(f"\n{'='*80}")
    print(f"RESOLUTION CALCULATIONS FOR {tilt_angle_deg}° TILT")
    print(f"{'='*80}")
    
    # Step 1: Initial values
    print("\nStep 1: Initial Parameters")
    print(f"Height: {height} mm")
    print(f"Target Resolution: {target_resolution} mm/pixel")
    
    # Step 2: Calculate distances and raw IFOV ratio
    tilt_rad = np.radians(tilt_angle_deg)
    if tilt_angle_deg == 0:
        raw_ratio = 1.0
        print(f"\nStep 2: At 0° tilt")
        print(f"Raw ratio = 1.0 (no perspective effect)")
    else:
        min_distance = height * np.tan(tilt_rad)
        max_distance = height / np.cos(tilt_rad)
        raw_ratio = max_distance / min_distance
        print(f"\nStep 2: Calculate perspective ratio")
        print(f"Minimum distance = {min_distance:.2f} mm")
        print(f"Maximum distance = {max_distance:.2f} mm")
        print(f"Raw ratio = max_distance / min_distance = {max_distance:.2f} / {min_distance:.2f} = {raw_ratio:.4f}")
    
    # Step 3: Calculate IFOV values
    min_ifov = target_resolution / raw_ratio
    max_ifov = target_resolution * raw_ratio
    print(f"\nStep 3: Calculate IFOV values")
    print(f"Minimum IFOV = target_resolution / ratio = {target_resolution} / {raw_ratio:.4f} = {min_ifov:.4f} mm/pixel")
    print(f"Maximum IFOV = target_resolution × ratio = {target_resolution} × {raw_ratio:.4f} = {max_ifov:.4f} mm/pixel")
    
    # Step 4: Calculate required pixels for water spot dimensions
    water_width = 266.7  # mm
    water_length = 317.5  # mm
    
    # Naive resolution (without perspective)
    naive_pixels_x = water_width / target_resolution
    naive_pixels_y = water_length / target_resolution
    print(f"\nStep 4: Calculate naive pixel requirements (without perspective)")
    print(f"Naive X pixels = water_width / target_resolution = {water_width} / {target_resolution} = {naive_pixels_x:.1f}")
    print(f"Naive Y pixels = water_length / target_resolution = {water_length} / {target_resolution} = {naive_pixels_y:.1f}")
    
    # Required resolution with perspective
    required_pixels_x = naive_pixels_x * raw_ratio
    required_pixels_y = naive_pixels_y * raw_ratio
    print(f"\nStep 5: Calculate actual pixel requirements (with perspective)")
    print(f"Required X pixels = naive_pixels_x × ratio = {naive_pixels_x:.1f} × {raw_ratio:.4f} = {required_pixels_x:.1f}")
    print(f"Required Y pixels = naive_pixels_y × ratio = {naive_pixels_y:.1f} × {raw_ratio:.4f} = {required_pixels_y:.1f}")
    
    # Final truncated values
    truncated_pixels_x = int(required_pixels_x)
    truncated_pixels_y = int(required_pixels_y)
    print(f"\nStep 6: Final truncated resolution")
    print(f"Final X resolution = {truncated_pixels_x} pixels")
    print(f"Final Y resolution = {truncated_pixels_y} pixels")
    
    # Calculate actual achieved resolution
    achieved_res_x = water_width / truncated_pixels_x
    achieved_res_y = water_length / truncated_pixels_y
    print(f"\nStep 7: Achieved resolution")
    print(f"Achieved X resolution = {water_width} mm / {truncated_pixels_x} pixels = {achieved_res_x:.4f} mm/pixel")
    print(f"Achieved Y resolution = {water_length} mm / {truncated_pixels_y} pixels = {achieved_res_y:.4f} mm/pixel")

# Calculate for different angles
angles = [0, 10, 20, 30]
for angle in angles:
    calculate_resolution_steps(133, angle)
