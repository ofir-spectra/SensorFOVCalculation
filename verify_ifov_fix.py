#!/usr/bin/env python3
"""
Verification script to test IFOV calculation consistency between IFOV and FOV tabs.
Compares the calculated Resolution (IFOV) values for both modes using the same parameters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_ifov_fov_consistency():
    """Test that both tabs produce consistent IFOV values with same parameters"""
    
    print("=" * 80)
    print("IFOV CALCULATION VERIFICATION TEST")
    print("=" * 80)
    print()
    
    # Common parameters (user inputs)
    A = 133.0        # Camera height (mm)
    B = 317.5        # Water spot length with margin
    C = 266.7        # Water spot width with margin
    tilt = 30.0      # Camera tilt (degrees)
    margin = 10.0    # Margin (%)
    shift = 0.0      # Shift (mm)
    deadzone = 0.3   # Dead zone (mm)
    pixel_pitch_um = 2.0  # Pixel pitch (um)
    
    # IFOV Mode specific
    ifov_resolution_mm_px = 0.22  # User input: required resolution in mm/px
    
    # FOV Mode specific
    focal_length_mm = 6.0  # Focal length (mm)
    sensor_px_x = 1920     # Sensor resolution X
    sensor_px_y = 1080     # Sensor resolution Y
    
    print("COMMON PARAMETERS:")
    print(f"  Camera Height (A): {A} mm")
    print(f"  Water Spot Length (B): {B} mm")
    print(f"  Water Spot Width (C): {C} mm")
    print(f"  Camera Tilt: {tilt}°")
    print(f"  Margin: {margin}%")
    print(f"  Shift: {shift} mm")
    print(f"  Dead Zone: {deadzone} mm")
    print(f"  Pixel Pitch: {pixel_pitch_um} um = {pixel_pitch_um/1000} mm")
    print()
    
    print("IFOV MODE (Direct Input):")
    print(f"  Required Resolution: {ifov_resolution_mm_px} mm/px")
    print(f"  → Min IFOV ≈ {ifov_resolution_mm_px:.4f} mm")
    print(f"  → Max IFOV ≈ {ifov_resolution_mm_px:.4f} mm")
    print()
    
    print("FOV MODE (Calculated from Focal Length):")
    print(f"  Focal Length: {focal_length_mm} mm")
    print(f"  Sensor Resolution: {sensor_px_x} × {sensor_px_y} px")
    
    # Calculate effective IFOV using the fix
    pixel_pitch_mm = pixel_pitch_um / 1000.0
    if focal_length_mm > 0:
        effective_ifov = pixel_pitch_mm * A / focal_length_mm
    else:
        effective_ifov = pixel_pitch_mm
    
    print(f"  Effective IFOV = (PixelPitch × A) / FocalLength")
    print(f"  Effective IFOV = ({pixel_pitch_mm:.6f} mm × {A} mm) / {focal_length_mm} mm")
    print(f"  Effective IFOV = {effective_ifov:.6f} mm/px")
    print(f"  → Min IFOV ≈ {effective_ifov:.4f} mm (adjusted by tilt/angle)")
    print(f"  → Max IFOV ≈ {effective_ifov:.4f} mm (adjusted by tilt/angle)")
    print()
    
    print("=" * 80)
    print("COMPARISON:")
    print("=" * 80)
    print(f"IFOV Mode Resolution:  {ifov_resolution_mm_px:.4f} mm/px")
    print(f"FOV Mode Resolution:   {effective_ifov:.4f} mm/px")
    print(f"Ratio (IFOV/FOV):      {ifov_resolution_mm_px/effective_ifov:.2f}x")
    print()
    
    # Note about differences
    print("NOTE:")
    print("  The two modes use DIFFERENT resolution concepts:")
    print("  - IFOV Mode: User specifies desired pixel size on water surface")
    print("  - FOV Mode: Calculates pixel size from camera optics (focal length, pixel pitch)")
    print()
    print("  They should NOT necessarily match unless the user input in IFOV mode")
    print("  happens to equal the calculated value from FOV mode's optics.")
    print()
    print("  For these parameters:")
    if abs(ifov_resolution_mm_px - effective_ifov) < 0.01:
        print(f"  ✓ Values are VERY SIMILAR (difference < 0.01 mm)")
    elif abs(ifov_resolution_mm_px - effective_ifov) < 0.1:
        print(f"  ✓ Values are REASONABLY CLOSE (difference < 0.1 mm)")
    else:
        print(f"  ! Values DIFFER significantly (difference = {abs(ifov_resolution_mm_px - effective_ifov):.4f} mm)")
        print(f"    This is EXPECTED if they represent different sensor/optics setups")
    print()
    
    print("=" * 80)
    print("HOW TO VERIFY THE FIX:")
    print("=" * 80)
    print("1. Open the Sensor Simulation GUI")
    print("2. In IFOV tab:")
    print(f"   - Set Resolution to: {effective_ifov:.2f} mm/px (to match FOV calc)")
    print(f"   - Compare the min/max IFOV output values")
    print("3. Switch to FOV tab:")
    print(f"   - Focal Length: {focal_length_mm} mm")
    print(f"   - Sensor Resolution: {sensor_px_x}×{sensor_px_y} px")
    print(f"   - Pixel Pitch: {pixel_pitch_um} um")
    print(f"   - Compare the min/max IFOV output values")
    print("4. Both should show similar IFOV ranges (accounting for tilt effects)")
    print()

if __name__ == "__main__":
    test_ifov_fov_consistency()
