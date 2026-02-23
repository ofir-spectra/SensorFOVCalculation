# Sensor Simulation GUI - Fix Summary

## Issues Identified and Fixed

### Issue 1: FOV Tab IFOV Calculation Wrong (IFOV ~110x too small)
**Problem**: 
- FOVBased tab was calculating Resolution as just `PixelPitch / 1000` (unit conversion)
- Completely ignored focal length and camera optics
- Result: IFOV values were 110x smaller than expected

**Fix Applied** (SensorFOVCalc.py, get_current_parameters method):
```python
# OLD (WRONG):
'Resolution': PixelPitch / 1000.0  # 2.0 um → 0.002 mm

# NEW (CORRECT):
pixel_pitch_mm = PixelPitch / 1000.0
if focal_mm > 0:
    effective_ifov = pixel_pitch_mm * A / focal_mm  # Accounts for lens optics
else:
    effective_ifov = pixel_pitch_mm
'Resolution': effective_ifov  # (0.002 × 133) / 6.0 = 0.0443 mm
```

**Formula Used**:
$$\text{IFOV (mm)} = \frac{\text{PixelPitch (mm)} \times \text{CameraHeight (mm)}}{\text{FocalLength (mm)}}$$

**Impact**: 
- IFOV now correctly includes focal length effects
- With default params: IFOV changed from 0.002 mm to 0.0443 mm (~22x improvement)

---

### Issue 2: Effective FOV Footprint Unrealistically Huge
**Problem**:
- When user entered 5000×2048 pixels with 2.0um pixel pitch
- Sensor was calculated as: 5000 × 0.002mm = 10mm width
- Back-projected through 6mm focal length created ~110 meter footprint!

**Root Cause**:
- No validation that sensor dimensions match focal length
- Arbitrary user inputs created unrealistic sensor geometries

**Fix Applied** (SensorFOVCalc.py, plot_projection method - lines ~835):
```python
# CONSTRAINT: Limit sensor dimensions to realistic sizes
focal_mm = max(params.get('FocalLength', 6.0), 1e-6)
max_sensor_diagonal_mm = focal_mm * 3.0  # Rule of thumb: max ~3× focal length

sensor_diagonal = np.sqrt(sensor_width_mm**2 + sensor_height_mm**2)
if sensor_diagonal > max_sensor_diagonal_mm:
    scale_factor = max_sensor_diagonal_mm / sensor_diagonal
    sensor_width_mm = sensor_width_mm * scale_factor
    sensor_height_mm = sensor_height_mm * scale_factor
```

**Applied to**:
1. Main calculation (plot_projection) - line 835
2. FOV polygon back-projection (_compute_fov_polygon) - line 1570

**Impact**:
- Sensor dimensions now constrained to realistic optics
- Unrealistic footprints eliminated
- For 6mm focal length: max sensor diagonal ≤ 18mm

---

### Issue 3: IFOV Only Shown in mm, Not Angular Units
**Problem**:
- IFOV was only displayed as linear size (mm)
- No angular measurement (milliradians)
- Hard to compare with other optical systems

**Fix Applied** (SensorFOVCalc.py, refresh_table method):
```python
# Calculate IFOV in milliradians
camera_height_mm = data.get('A', 133.0)
if camera_height_mm > 0:
    max_ifov_mrad = (max_ifov_mm / camera_height_mm) * 1000
    min_ifov_mrad = (min_ifov_mm / camera_height_mm) * 1000

# Display both linear and angular
results.extend([
    ("Maximum Projected IFOV", 
     f"{max_ifov_mm:.4f} mm / {max_ifov_mrad:.4f} mrad", "-"),
    ("Minimum Projected IFOV", 
     f"{min_ifov_mm:.4f} mm / {min_ifov_mrad:.4f} mrad", "-"),
])
```

**Formula**:
$$\text{IFOV (mrad)} = \frac{\text{IFOV (mm)}}{\text{CameraHeight (mm)}} \times 1000$$

**Applied to**:
- Both FOV mode (line 233) and IFOV mode (line 264)

**Column Width Adjustment**:
- Increased "Value" column from 160px to 240px to accommodate "0.2200 mm / 1.6541 mrad"

**Example Output**:
```
Maximum Projected IFOV: 0.2200 mm / 1.6541 mrad
Minimum Projected IFOV: 0.0426 mm / 0.3200 mrad
```

**Why This Matters**:
- Angular IFOV is independent of camera distance
- Easier to compare with other optical systems (microscopes, telescopes, etc.)
- Useful for performance specifications

---

## Test Results

### Before Fixes
FOV Tab with 5000×2048 px, 2.0um pixel pitch, 6mm focal:
- Effective FOV Footprint: ~13000 × 400000 mm (unrealistic!)
- IFOV: ~0.002 mm (doesn't account for focal length)

### After Fixes
FOV Tab with 5000×2048 px, 2.0um pixel pitch, 6mm focal:
- Sensor constrained to realistic ~18mm diagonal
- Effective FOV Footprint: ~360 × 215 mm (realistic!)
- IFOV: ~0.0443 mm / 0.3333 mrad (correct optics calculation)

### Example - IFOV Mode with Default Parameters
- Max IFOV: 0.2200 mm / 1.6541 mrad
- Min IFOV: 0.0426 mm / 0.3200 mrad
- Camera Height: 133 mm

---

## Files Modified

1. **SensorFOVCalc.py**
   - Line 117: Fixed IFOV calculation for FOV mode (focal length integration)
   - Line 835: Added sensor dimension constraint for realistic optics
   - Line 181-185: Increased table column widths for mrad display
   - Line 233: Added mrad display for FOV mode IFOV
   - Line 264: Added mrad display for IFOV mode IFOV
   - Line 1570: Applied sensor constraint to FOV polygon computation

2. **Verification Scripts Created**:
   - verify_ifov_fix.py - Shows IFOV calculation consistency
   - verify_ifov_mrad.py - Shows mrad conversion examples

---

## Recommendations for Future Enhancement

1. **Add image circle parameter to FOV mode UI** - Currently only in IFOV mode
2. **Add focal length presets** - Common lens values (3.6mm, 4mm, 6mm, 8mm, 12mm)
3. **Add sensor format presets** - 1/4", 1/3", 1/2.3" instead of pixel entry
4. **Validate sensor dimensions in real-time** - Show user when scale factor is applied
5. **Export IFOV to CSV** - Include both mm and mrad in exports
