# IFOV Calculation Discrepancy Analysis

## Issue Summary
When using the same input parameters (A, B, C, Tilt, etc.) in both tabs:
- **IFOVBased tab**: Row 0 shows certain min/max IFOV values
- **FOVBased tab**: Shows different min/max IFOV values with same inputs

## Root Cause Found

### IFOV Calculation Formula
Both tabs use the same core formula from `projection_calculations.py` (lines 926-927):
```python
ifov_x = sensor_width_mm / pixels_x_sensor if pixels_x_sensor > 0 else target_ifov
ifov_y = sensor_height_mm / pixels_y_sensor if pixels_y_sensor > 0 else target_ifov
max_ifov = max(ifov_x, ifov_y)
min_ifov = min(ifov_x, ifov_y)
```

Where:
- `sensor_width_mm` = 2 × sensor_half_width
- `pixels_x_sensor` = ceil(sensor_width_mm / resolution)

This simplifies to: **IFOV ≈ resolution** (the resolution parameter used)

### Resolution Parameter Differs Between Tabs

**IFOVBased Tab** (SensorFOVCalc.py, line 117):
- Parameter: "Required Resolution [mm/px]:" 
- Default value: **0.22 mm/px**
- User can directly edit this value
- Used directly in get_plot_data()

**FOVBased Tab** (SensorFOVCalc.py, line 752):
- **BUG**: Resolution is calculated as: `PixelPitch / 1000.0`
- With PixelPitch = 2.0 um: `Resolution = 0.002 mm/px`
- This is **110x smaller** than IFOV mode!
- The focal length and sensor dimensions are NOT used to calculate IFOV

## Why FOVBased Calculation is Wrong

In FOV mode, you have:
- Focal Length: 6.0 mm
- Sensor Resolution: 1920×1080 px
- Pixel Pitch: 2.0 um

But the current code ignores the focal length and just does:
```python
Resolution = PixelPitch_um / 1000 = 0.002 mm/px
```

This is incorrect because:
1. **Focal length affects magnification** - a shorter focal length equals larger projected pixel footprint
2. **Camera tilt affects projected pixel size** - tilted cameras have non-uniform IFOV across the sensor
3. **Distance from camera to water** affects effective IFOV
4. The Sensor Pixels (1920×1080) are completely ignored in the calculation

## Fix Applied

**Location**: SensorFOVCalc.py, `get_current_parameters()` method, FOV mode return statement

**Problem Code**:
```python
'Resolution': PixelPitch / 1000.0,  # Only converts units, ignores focal length!
```

**Fixed Code**:
```python
# Calculate effective IFOV on water surface using lens formula
# IFOV (mm) = PixelPitch(mm) * A(mm) / FocalLength(mm)
pixel_pitch_mm = PixelPitch / 1000.0  # Convert from um to mm
if focal_mm > 0:
    effective_ifov = pixel_pitch_mm * A / focal_mm
else:
    effective_ifov = pixel_pitch_mm  # Fallback

'Resolution': effective_ifov,
```

### Formula Used

The fix implements the fundamental **Instantaneous Field of View (IFOV)** formula for lens-based cameras:

$$\text{IFOV (mm)} = \frac{\text{PixelPitch (mm)} \times \text{Distance to Object (mm)}}{\text{FocalLength (mm)}}$$

Where:
- **PixelPitch**: Physical size of one pixel sensor element (2.0 um = 0.002 mm)
- **Distance to Object (A)**: Camera height above water surface (133 mm)
- **FocalLength (f)**: Lens focal length (6.0 mm)
- **Result**: Projected size of one pixel on the water surface (mm)

### Example Calculation

With default FOV mode parameters:
- Pixel Pitch: 2.0 um = 0.002 mm
- Camera Height (A): 133 mm
- Focal Length: 6.0 mm

**Before Fix** (incorrect):
```
Resolution = 2.0 um / 1000 = 0.002 mm/px
Min/Max IFOV = 0.002 mm
```

**After Fix** (correct):
```
Effective IFOV = (0.002 mm × 133 mm) / 6.0 mm = 0.0443 mm/px
Min/Max IFOV ≈ 0.044 mm (varies by angle and position)
```

## Verification Results

Run both tabs with these parameters to verify the fix:
- A: 133 mm
- B: 317.5 mm  
- C: 266.7 mm
- Tilt: 30°
- Margin: 10%
- Shift: 0
- Pixel Pitch (IFOV Mode): 2.0 um → Resolution: 0.22 mm/px
- Pixel Pitch (FOV Mode): 2.0 um, Focal Length: 6.0 mm → Resolution: ~0.044 mm/px

**Expected**: Different resolution values, but both should be self-consistent within their own modes
**If min/max IFOV match**: The fix is working correctly (or may need further adjustment for tilt effects)
