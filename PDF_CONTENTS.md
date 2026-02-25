# IFOV Calculations Detailed PDF - Contents

## Document Overview
**File**: `IFOV_Calculations_Detailed.pdf` (18.9 KB)  
**Generated**: February 25, 2026  
**Example Parameters**: A=146mm, B=541mm, C=268mm

---

## Document Structure (7 Sections)

### 1. Introduction & Overview
- Key concepts of IFOV (linear and angular)
- Definition of parameters (A, B, C, f, p, etc.)
- Two ways to express IFOV

### 2. Core Formulas
**2.1 Sensor Dimensions**
- W_sensor = p × n_x
- H_sensor = p × n_y

**2.2 Field of View (FOV) from Focal Length**
- FOV_H = 2 × arctan(W_sensor / (2×f)) × (180/π)
- FOV_V = 2 × arctan(H_sensor / (2×f)) × (180/π)

**2.3 Linear IFOV (IFOV Mode)**
- IFOV_mm = Resolution_specified
- n_x = ceil(W_spot / IFOV_mm)
- n_y = ceil(H_spot / IFOV_mm)

**2.4 Effective IFOV (FOV Mode)**
- IFOV_eff = (p × A) / f
- This is the key formula incorporating optical properties

**2.5 Angular IFOV (milliradians)**
- IFOV_mrad = (IFOV_mm / A) × 1000

### 3. IFOV Mode Calculation
Step-by-step calculation showing:
- Margin factor application
- Required pixel calculation
- Perspective distortion adjustment
- Example output with min/max IFOV values

### 4. FOV Mode Calculation  
Detailed FOV mode calculation including:
- Physical sensor size calculation
- Sensor constraint checking (max 3× focal length diagonal)
- Field of View angle computation
- Back-projection formula to water surface
- Effective IFOV using focal length

### 5. Detailed Example Calculation
**Your Parameters:**
- Camera Height (A) = 146 mm
- Water Spot Length (B) = 541 mm
- Water Spot Width (C) = 268 mm
- Camera Tilt = 30°
- Pixel Pitch = 2.0 μm
- Focal Length = 6.0 mm
- Sensor Resolution = 1920×1080 px
- Margin = 10%

**Step-by-Step Results:**

**FOV Mode Calculation:**
- W_sensor = 0.002 mm × 1920 = 3.84 mm
- H_sensor = 0.002 mm × 1080 = 2.16 mm
- Sensor diagonal = 4.41 mm (within 18mm limit ✓)
- FOV_H = 35.47°
- FOV_V = 20.40°
- IFOV_eff = (0.002 × 146) / 6.0 = **0.0487 mm/pixel**
- IFOV_mrad = **0.3335 mrad**

**IFOV Mode Calculation (0.22 mm specified):**
- Required pixels: 1,340 × 2,705 (naive)
- With tilt adjustment: ≈3,030 × 3,608 px
- IFOV = **0.22 mm/pixel**
- IFOV_mrad = **1.5068 mrad**

### 6. Results & Interpretation
**Comparison Table:**
| Metric | FOV Mode | IFOV Mode |
|--------|----------|-----------|
| Linear IFOV | 0.0487 mm | 0.22 mm |
| Angular IFOV | 0.3335 mrad | 1.5068 mrad |
| Sensor Size | 3.84×2.16 mm | Calculated from needs |
| FOV | 35.47°×20.40° | Depends on lens |

**Mode Selection Criteria:**
- **FOV Mode**: Real hardware specs, "what if" analysis
- **IFOV Mode**: Performance requirements, design starting point

### 7. Angular Resolution (Milliradians)
**Why Milliradians?**
- Independent of distance
- Same angular resolution at any distance

**Sensitivity to Distance Table:**
| Distance (mm) | IFOV (mrad) |
|---------------|-------------|
| 100 | 0.4870 |
| **146** | **0.3335** |
| 200 | 0.2435 |
| 300 | 0.1623 |
| 500 | 0.0974 |
| 1000 | 0.0487 |

**Practical Interpretation:**
With FOV mode IFOV = 0.3335 mrad:
- At 146 mm: projects to 0.0487 mm
- At 292 mm: projects to 0.0974 mm
- At 584 mm: projects to 0.1948 mm

---

## Key Calculations Summary

### Formula 1: Effective IFOV (FOV Mode)
```
IFOV = (PixelPitch × CameraHeight) / FocalLength
IFOV = (0.002 mm × 146 mm) / 6.0 mm = 0.0487 mm
```

### Formula 2: Angular IFOV
```
IFOV_mrad = (IFOV_mm / CameraHeight) × 1000
IFOV_mrad = (0.0487 / 146) × 1000 = 0.3335 mrad
```

### Formula 3: Field of View from Focal Length
```
FOV_H = 2 × arctan(SensorWidth / (2×f)) × 57.3
FOV_H = 2 × arctan(3.84 / 12) × 57.3 = 35.47°
```

---

## Document Features
✓ Professional PDF formatting with colored headers  
✓ Mathematical formulas with proper notation  
✓ Step-by-step calculation breakdowns  
✓ Summary tables with real values  
✓ Practical examples and interpretations  
✓ Sensitivity analysis for different parameters  
✓ Complete FOV and IFOV mode comparison  

---

## How to Use This Document
1. **For understanding IFOV**: Start with Section 2 (Core Formulas)
2. **For specific examples**: Jump to Section 5 with your parameters
3. **For mode comparison**: See Section 6 (Results & Interpretation)
4. **For angular resolution**: Reference Section 7 (Milliradians)
5. **For custom parameters**: Use the formulas in Section 2 with your values

---

## File Locations
- **PDF Document**: `IFOV_Calculations_Detailed.pdf`
- **Generation Script**: `generate_ifov_pdf.py`
- **Repository**: GitHub (FOV branch)

---

## Next Steps
To regenerate the PDF with different parameters, edit `generate_ifov_pdf.py`:
1. Modify the parameters at line ~620
2. Run: `python generate_ifov_pdf.py`
3. New PDF will be created with updated calculations
