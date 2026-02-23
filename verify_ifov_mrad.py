#!/usr/bin/env python3
"""
Verification script for IFOV display in both mm and mrad
"""

def convert_ifov_to_mrad(ifov_mm, camera_height_mm):
    """Convert IFOV from mm to milliradians"""
    if camera_height_mm > 0:
        ifov_rad = ifov_mm / camera_height_mm
        ifov_mrad = ifov_rad * 1000
        return ifov_mrad
    return 0

print("=" * 80)
print("IFOV CONVERSION TO MILLIRADIANS")
print("=" * 80)
print()

# Example values from IFOV mode with default parameters
camera_height_mm = 133.0
max_ifov_mm = 0.22
min_ifov_mm = 0.04256

max_ifov_mrad = convert_ifov_to_mrad(max_ifov_mm, camera_height_mm)
min_ifov_mrad = convert_ifov_to_mrad(min_ifov_mm, camera_height_mm)

print(f"Camera Height: {camera_height_mm} mm")
print()
print(f"IFOV RESULTS:")
print(f"  Max IFOV: {max_ifov_mm:.4f} mm / {max_ifov_mrad:.4f} mrad")
print(f"  Min IFOV: {min_ifov_mm:.4f} mm / {min_ifov_mrad:.4f} mrad")
print()

# More examples with different camera heights
print("=" * 80)
print("SENSITIVITY TO CAMERA HEIGHT")
print("=" * 80)
print()

heights_mm = [100, 133, 200, 300]
ifov_mm = 0.22

for h in heights_mm:
    mrad = convert_ifov_to_mrad(ifov_mm, h)
    print(f"Height: {h:3d} mm  →  IFOV: {ifov_mm:.4f} mm / {mrad:.4f} mrad")
print()

# Show why mrad is useful
print("=" * 80)
print("WHY MILLIRADIANS ARE USEFUL")
print("=" * 80)
print()
print("Angular resolution is independent of distance!")
print()
print("With IFOV = 1.65 mrad:")
print("  At distance 133 mm:  Size on object ≈ 0.22 mm")
print("  At distance 530 mm:  Size on object ≈ 0.88 mm")
print("  At distance 1000 mm: Size on object ≈ 1.65 mm")
print()
print("The angle (mrad) stays the same, only the projection size changes.")
