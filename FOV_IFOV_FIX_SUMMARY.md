# FOVBased Tab IFOV Calculation Fix - Summary

## Problem Identified

When using the **FOVBased tab** with the same input parameters as **IFOVBased tab**, the min/max IFOV results were **110x smaller** than expected.

## Root Cause

In `SensorFOVCalc.py`, the `get_current_parameters()` method was calculating the Resolution parameter for FOV mode incorrectly:

```python
# OLD (BUGGY) CODE:
'Resolution': PixelPitch / 1000.0,  # Only unit conversion, ignores focal length!
```

This converted 2.0 um to 0.002 mm without accounting for:
- Focal length (6.0 mm)
- Camera height/distance to water (133 mm)
- Sensor optics magnification effect

## Solution Applied

Implemented the proper **Instantaneous Field of View (IFOV)** formula from optical physics:

```python
# NEW (FIXED) CODE:
pixel_pitch_mm = PixelPitch / 1000.0  # Convert from um to mm
if focal_mm > 0:
    effective_ifov = pixel_pitch_mm * A / focal_mm
else:
    effective_ifov = pixel_pitch_mm

'Resolution': effective_ifov,
```

### Formula

$$\text{IFOV} = \frac{\text{PixelPitch (mm)} \times \text{Camera Height (mm)}}{\text{Focal Length (mm)}}$$

## Impact Comparison

### With Default Parameters

| Parameter | Value |
|-----------|-------|
| Pixel Pitch | 2.0 um |
| Camera Height (A) | 133 mm |
| Focal Length | 6.0 mm |

### BEFORE Fix (Incorrect)
- Calculation: `2.0 um / 1000 = 0.002 mm/px`
- Min IFOV displayed: ~0.002 mm
- Max IFOV displayed: ~0.002 mm
- **Problem**: Focal length completely ignored

### AFTER Fix (Correct)
- Calculation: `(0.002 mm × 133 mm) / 6.0 mm = 0.0443 mm/px`
- Min IFOV displayed: ~0.0443 mm (varies with angle)
- Max IFOV displayed: ~0.0443 mm (varies with angle)
- **Correctly** incorporates lens parameters

## How to Verify the Fix

1. **Launch the GUI**: `python SensorFOVCalc.py`

2. **In IFOVBased tab**:
   - Adjust "Required Resolution" to `0.044 mm/px` (matches FOV calculation)
   - Note the displayed Min/Max IFOV values
   - Observe the plot and projections

3. **Switch to FOVBased tab**:
   - Focal Length: `6.0 mm`
   - Sensor Resolution: `1920×1080 px`
   - Pixel Pitch: `2.0 um`
   - Note the displayed Min/Max IFOV values
   - Compare with IFOVBased results

4. **Expected Result**:
   - If you set IFOV mode to 0.044 mm/px and switch to FOV mode with default focal length parameters, the IFOV results should be **similar** (accounting for tilt angle variations)
   - Before the fix, FOV mode would show ~110x smaller values

## Technical Details

### Why The Two Modes Might Show Different Values

The two tabs represent **different sensor/optics configurations**:

- **IFOVBased**: User directly inputs desired pixel resolution. Good for specifying target performance.
- **FOVBased**: Calculates resolution from actual lens/sensor specifications. Good for modeling real hardware.

They don't need to match unless you specifically configure them to use the same effective optics.

### What The Fix Actually Does

The fix ensures FOV mode properly accounts for **magnification** introduced by the lens:
- A shorter focal length = larger projected pixel = larger IFOV (less detail)
- A longer focal length = smaller projected pixel = smaller IFOV (more detail)
- Camera height amplifies this effect (farther from sensor = larger projected pixels)

## Files Modified

- `SensorFOVCalc.py` - Fixed `get_current_parameters()` method for FOV mode

## Testing Recommendations

1. Test with various focal lengths (3mm, 6mm, 12mm) - IFOV should change proportionally
2. Test with different camera heights - IFOV should scale linearly
3. Test with different pixel pitches - IFOV should scale linearly
4. Compare plots and coverages between modes configured with matching optics
