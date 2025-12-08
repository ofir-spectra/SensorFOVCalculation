import numpy as np

def calculate_ifov(height, tilt_angle_deg, water_width=266.7, water_length=317.5):
    """Calculate detailed IFOV and resolution calculations for given parameters"""
    tilt_rad = np.radians(tilt_angle_deg)
    base_resolution = 0.22  # mm/pixel (target resolution)
    
    results = {
        'tilt_deg': tilt_angle_deg,
        'tilt_rad': tilt_rad,
        'height': height,
        'base_resolution': base_resolution
    }
    
    if tilt_angle_deg == 0:
        # For vertical camera (0 degrees tilt)
        results.update({
            'min_distance': height,
            'max_distance': height,
            'min_ifov': base_resolution,
            'max_ifov': base_resolution,
            'naive_pixels_x': int(water_width / base_resolution),
            'naive_pixels_y': int(water_length / base_resolution)
        })
    else:
        # Step 1: Calculate distances for tilted camera
        min_distance = height * np.tan(tilt_rad)  # Closest point to camera
        max_distance = height / np.cos(tilt_rad)  # Farthest point
        
        # Step 2: Calculate IFOV considering perspective effects
        min_ifov = base_resolution * (min_distance / max_distance)
        max_ifov = base_resolution * (max_distance / min_distance)
        
        # Step 3: Calculate naive pixel requirements
        naive_pixels_x = int(water_width / base_resolution)
        naive_pixels_y = int(water_length / base_resolution)
        
        # Step 4: Calculate required pixels considering projection
        ifov_ratio = max_ifov / base_resolution
        required_pixels_x = int(naive_pixels_x * ifov_ratio)
        required_pixels_y = int(naive_pixels_y * ifov_ratio)
        
        results.update({
            'min_distance': min_distance,
            'max_distance': max_distance,
            'min_ifov': min_ifov,
            'max_ifov': max_ifov,
            'naive_pixels_x': naive_pixels_x,
            'naive_pixels_y': naive_pixels_y,
            'ifov_ratio': ifov_ratio,
            'required_pixels_x': required_pixels_x,
            'required_pixels_y': required_pixels_y
        })
    
    return results

# Input parameters
height = 133  # mm (camera height)
water_width = 266.7  # mm (water spot width - C)
water_length = 317.5  # mm (water spot length - B)
tilt_angles = [0, 10, 20, 30]  # degrees

print("IFOV and Resolution Calculations for Different Tilt Angles")
print("-------------------------------------------------------")
print(f"Camera Height (A): {height} mm")
print(f"Water Spot Width (C): {water_width} mm")
print(f"Water Spot Length (B): {water_length} mm")
print("\nDetailed Calculations for Each Tilt Angle:")

for angle in tilt_angles:
    results = calculate_ifov(height, angle, water_width, water_length)
    print("\n" + "=" * 80)
    print(f"TILT ANGLE: {angle}°")
    print("=" * 80)
    
    # Basic geometry
    print("\n1. Basic Geometry:")
    print(f"   - Tilt angle (degrees): {results['tilt_deg']:.1f}°")
    print(f"   - Tilt angle (radians): {results['tilt_rad']:.4f} rad")
    print(f"   - Camera height: {results['height']:.1f} mm")
    
    # Distances
    print("\n2. Distance Calculations:")
    print(f"   - Minimum distance to surface: {results['min_distance']:.1f} mm")
    print(f"   - Maximum distance to surface: {results['max_distance']:.1f} mm")
    print(f"   - Distance ratio (max/min): {results['max_distance']/results['min_distance']:.2f}")
    
    # IFOV calculations
    print("\n3. IFOV Calculations:")
    print(f"   - Base resolution (target): {results['base_resolution']:.3f} mm/px")
    print(f"   - Minimum IFOV: {results['min_ifov']:.4f} mm/px")
    print(f"   - Maximum IFOV: {results['max_ifov']:.4f} mm/px")
    print(f"   - IFOV ratio (max/min): {results['max_ifov']/results['min_ifov']:.2f}")
    
    # Resolution requirements
    print("\n4. Resolution Requirements:")
    print(f"   - Naive resolution (X): {results['naive_pixels_x']} pixels")
    print(f"   - Naive resolution (Y): {results['naive_pixels_y']} pixels")
    if angle != 0:
        print(f"   - IFOV ratio for scaling: {results['ifov_ratio']:.2f}")
        print(f"   - Required resolution (X): {results['required_pixels_x']} pixels")
        print(f"   - Required resolution (Y): {results['required_pixels_y']} pixels")
