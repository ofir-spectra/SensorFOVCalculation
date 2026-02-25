# IFOV PDF Quick Reference

## PDF Document Generated ✓

**File**: `IFOV_Calculations_Detailed.pdf` (18.9 KB)

---

## What's Included in the PDF

### ✓ Complete Mathematical Formulations
- Effective IFOV formula: IFOV = (p × A) / f
- Angular IFOV formula: IFOV_mrad = (IFOV_mm / A) × 1000
- Field of View calculations from focal length
- Sensor dimension calculations

### ✓ Step-by-Step Calculations
**Your Example Parameters:**
- **A = 146 mm** (camera height)
- **B = 541 mm** (water spot length)
- **C = 268 mm** (water spot width)
- Focal Length: 6.0 mm
- Pixel Pitch: 2.0 μm
- Sensor: 1920×1080 pixels
- Tilt: 30°

### ✓ Calculated Results

#### FOV Mode (Optics-Based):
- **IFOV: 0.0487 mm/pixel or 0.3335 mrad**
- FOV: 35.47° × 20.40°
- Sensor Size: 3.84 × 2.16 mm

#### IFOV Mode (Performance-Based):
- **IFOV: 0.22 mm/pixel or 1.5068 mrad**
- Required Resolution: 3,030 × 3,608 pixels
- Designed for target performance

### ✓ Practical Examples
- Distance sensitivity table (100mm to 1000mm)
- Pixel projection on water surface at different distances
- Angular resolution vs distance

---

## How to Customize the PDF

### Option 1: Quick Edit Script (Recommended)

Edit `generate_ifov_pdf.py` around **line 620** to change parameters:

```python
# Define parameters
A = 146  # ← Change camera height
B = 541  # ← Change water spot length
C = 268  # ← Change water spot width
margin_pct = 10
tilt_deg = 30
pixel_pitch_um = 2.0
focal_length = 6.0
sensor_px_x = 1920
sensor_px_y = 1080
```

### Option 2: Run with Custom Example

After editing, run:
```bash
python generate_ifov_pdf.py
```

A new PDF will be generated with your custom values.

---

## Example Scenarios

### Scenario 1: Different Camera Height
```python
A = 200  # Farther from water
# Result: IFOV increases proportionally
# IFOV_mrad decreases: (0.0487/200)*1000 = 0.2435 mrad
```

### Scenario 2: Different Focal Length
```python
focal_length = 4.0  # Shorter focal length = wider field of view
# Result: IFOV decreases (smaller pixels at same distance)
# IFOV = (0.002 * 146) / 4.0 = 0.073 mm
```

### Scenario 3: Different Sensor
```python
sensor_px_x = 2560  # Higher resolution
sensor_px_y = 1440
# Result: FOV calculated from new sensor size
# Sensor diagonal may need adjustment if > 3×f
```

---

## Key Values from PDF

### Your Configuration Results:

**FOV Mode (Realistic Optics):**
```
Focal Length:        6.0 mm
Sensor Size:         3.84 × 2.16 mm  
Field of View:       35.47° × 20.40°
IFOV (linear):       0.0487 mm/pixel
IFOV (angular):      0.3335 mrad
```

**IFOV Mode (Performance Target):**
```
Target Resolution:   0.22 mm/pixel
IFOV (angular):      1.5068 mrad
Required Pixels:     ~3,030 × 3,608 (with tilt)
```

**Comparison:**
```
Ratio (IFOV Mode / FOV Mode): 4.5×
Different use cases - both valid
```

---

## Angular IFOV Interpretation

The "mrad" value is most useful for comparing across different camera heights:

**Same angular resolution (0.3335 mrad) projects to:**
- At A=146mm: 0.0487 mm pixel size
- At A=292mm: 0.0974 mm pixel size  
- At A=584mm: 0.1948 mm pixel size

This makes it easy to scale designs up or down!

---

## Document Sections Map

| Section | Title | Use For |
|---------|-------|---------|
| 1 | Introduction | Understanding IFOV concepts |
| 2 | Core Formulas | Reference formulas for calculations |
| 3 | IFOV Mode | Direct resolution specification |
| 4 | FOV Mode | Optics-based calculations |
| **5** | **Detailed Example** | **Your specific calculation** |
| 6 | Results & Interpretation | Mode comparison |
| 7 | Angular Resolution | mrad conversions & sensitivity |

---

## Technical Notes

### Formula Explanation - IFOV = (p × A) / f

- **p** = pixel pitch in mm (smaller = more detail)
- **A** = camera height in mm (larger = smaller projection)
- **f** = focal length in mm (smaller = wider field)
- **Result** = size of one pixel projected on subject

#### Example:
- Pixel: 2.0 μm = 0.002 mm
- Height: 146 mm above water
- Lens: 6.0 mm focal length

IFOV = (0.002 × 146) / 6.0 = **0.0487 mm/pixel**

Each pixel covers about 0.0487 mm on the water surface.

### Why Milliradians?

**1.0 mrad = 1/1000 of a radian**

Advantages:
- ✓ Independent of distance
- ✓ Standard in optics community
- ✓ Easy to compare camera systems
- ✓ Direct relationship to angle

---

## Files Included

1. **IFOV_Calculations_Detailed.pdf** - Main documentation
2. **generate_ifov_pdf.py** - Script to create custom PDFs
3. **PDF_CONTENTS.md** - Detailed PDF section guide
4. **This file** - Quick reference

---

## Next Steps

1. **Review the PDF** at `IFOV_Calculations_Detailed.pdf`
2. **Understand the formulas** in Section 2
3. **Study your example** in Section 5
4. **Customize parameters** in `generate_ifov_pdf.py` if needed
5. **Reference mrad values** from Section 7 for other distances

---

**Generated**: February 25, 2026  
**Status**: ✓ Complete and committed to repository
