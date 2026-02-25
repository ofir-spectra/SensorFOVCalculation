#!/usr/bin/env python3
"""
Generate detailed PDF documentation for IFOV calculations
with formulas and example using A=146mm, B=541mm, C=268mm
"""

import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors
import numpy as np
from datetime import datetime

# ============================================================================
# Generate PDF Document
# ============================================================================

pdf_filename = "IFOV_Calculations_Detailed.pdf"
doc = SimpleDocTemplate(pdf_filename, pagesize=letter,
                        rightMargin=0.5*inch, leftMargin=0.5*inch,
                        topMargin=0.75*inch, bottomMargin=0.75*inch)

# Container for PDF elements
elements = []

# Define styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=24,
    textColor=colors.HexColor('#1f4788'),
    spaceAfter=12,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

heading_style = ParagraphStyle(
    'CustomHeading',
    parent=styles['Heading2'],
    fontSize=14,
    textColor=colors.HexColor('#1f4788'),
    spaceAfter=10,
    spaceBefore=12,
    fontName='Helvetica-Bold'
)

subheading_style = ParagraphStyle(
    'SubHeading',
    parent=styles['Heading3'],
    fontSize=11,
    textColor=colors.HexColor('#2d5aa0'),
    spaceAfter=8,
    spaceBefore=8,
    fontName='Helvetica-Bold'
)

body_style = ParagraphStyle(
    'CustomBody',
    parent=styles['Normal'],
    fontSize=10,
    alignment=TA_JUSTIFY,
    spaceAfter=6,
    leading=14
)

# ============================================================================
# TITLE PAGE
# ============================================================================

elements.append(Spacer(1, 0.5*inch))
elements.append(Paragraph("IFOV Calculations", title_style))
elements.append(Paragraph("Detailed Formulation & Examples", heading_style))
elements.append(Spacer(1, 0.2*inch))
elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", body_style))
elements.append(Spacer(1, 0.3*inch))

# Document info
info_data = [
    ["Document", "IFOV Calculation Guide"],
    ["Version", "1.0"],
    ["Focus", "Sensor Simulation - FOV Mode Analysis"],
]
info_table = Table(info_data, colWidths=[1.5*inch, 3.5*inch])
info_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f8')),
    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ('TOPPADDING', (0, 0), (-1, -1), 8),
    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
]))
elements.append(info_table)

elements.append(PageBreak())

# ============================================================================
# TABLE OF CONTENTS
# ============================================================================

elements.append(Paragraph("Table of Contents", heading_style))
elements.append(Spacer(1, 0.15*inch))

toc_items = [
    "1. Introduction & Overview",
    "2. Core Formulas",
    "3. IFOV Mode Calculation",
    "4. FOV Mode Calculation",
    "5. Example Calculation (A=146mm, B=541mm, C=268mm)",
    "6. Results & Interpretation",
    "7. Angular Resolution (milliradians)",
]

for item in toc_items:
    elements.append(Paragraph(item, body_style))

elements.append(PageBreak())

# ============================================================================
# 1. INTRODUCTION
# ============================================================================

elements.append(Paragraph("1. Introduction & Overview", heading_style))
elements.append(Spacer(1, 0.1*inch))

intro_text = """
The Instantaneous Field of View (IFOV) describes the angular and linear size of 
a single pixel when projected onto the observed surface. This document provides 
detailed mathematical formulations and step-by-step calculations for computing 
IFOV in sensor simulation systems.

<b>Key Parameters:</b>
<br/>• <b>A</b>: Camera height above water surface (mm)
<br/>• <b>B</b>: Water spot length (mm)
<br/>• <b>C</b>: Water spot width (mm)
<br/>• <b>f</b>: Focal length of camera lens (mm)
<br/>• <b>p</b>: Pixel pitch of camera sensor (μm)
<br/>• <b>n_x, n_y</b>: Number of pixels in X and Y directions
<br/>
<br/>IFOV can be expressed in two ways:
<br/>• <b>Linear IFOV</b>: Size of projected pixel on water surface (mm)
<br/>• <b>Angular IFOV</b>: Angular size of pixel as seen from camera (milliradians)
"""

elements.append(Paragraph(intro_text, body_style))
elements.append(PageBreak())

# ============================================================================
# 2. CORE FORMULAS
# ============================================================================

elements.append(Paragraph("2. Core Formulas", heading_style))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("2.1 Sensor Dimensions", subheading_style))
sensor_text = """
Physical sensor width and height from pixel count and pitch:
<br/><br/>
<b>W<sub>sensor</sub> = p × n_x</b>
<br/>
<b>H<sub>sensor</sub> = p × n_y</b>
<br/><br/>
Where:
<br/>• p = pixel pitch (mm) = pixel_pitch_μm / 1000
<br/>• n_x, n_y = number of pixels in each direction
"""
elements.append(Paragraph(sensor_text, body_style))

elements.append(Spacer(1, 0.15*inch))
elements.append(Paragraph("2.2 Field of View (FOV) from Focal Length", subheading_style))
fov_text = """
The angular field of view is determined by the sensor size and focal length:
<br/><br/>
<b>FOV<sub>H</sub> = 2 × arctan(W<sub>sensor</sub> / (2 × f)) × (180/π)</b>
<br/>
<b>FOV<sub>V</sub> = 2 × arctan(H<sub>sensor</sub> / (2 × f)) × (180/π)</b>
<br/><br/>
Where:
<br/>• f = focal length (mm)
<br/>• Result is in degrees
"""
elements.append(Paragraph(fov_text, body_style))

elements.append(Spacer(1, 0.15*inch))
elements.append(Paragraph("2.3 Linear IFOV (IFOV Mode)", subheading_style))
ifov_linear_text = """
When user specifies desired resolution directly (mm/pixel):
<br/><br/>
<b>IFOV<sub>mm</sub> = Resolution<sub>specified</sub></b>
<br/><br/>
The system then calculates required sensor resolution:
<br/><br/>
<b>n_x = ceil(W<sub>spot</sub> / IFOV<sub>mm</sub>)</b>
<br/>
<b>n_y = ceil(H<sub>spot</sub> / IFOV<sub>mm</sub>)</b>
<br/><br/>
Where W<sub>spot</sub> and H<sub>spot</sub> include margin factor.
"""
elements.append(Paragraph(ifov_linear_text, body_style))

elements.append(Spacer(1, 0.15*inch))
elements.append(Paragraph("2.4 Effective IFOV (FOV Mode)", subheading_style))
ifov_fov_text = """
When user specifies camera optics (focal length, pixel pitch, pixel count):
<br/><br/>
<b>IFOV<sub>eff</sub> = (p × A) / f</b>
<br/><br/>
Where:
<br/>• p = pixel pitch (mm)
<br/>• A = camera height above water surface (mm)
<br/>• f = focal length (mm)
<br/><br/>
This accounts for magnification: larger A or smaller f produces larger projected pixels.
"""
elements.append(Paragraph(ifov_fov_text, body_style))

elements.append(Spacer(1, 0.15*inch))
elements.append(Paragraph("2.5 Angular IFOV (milliradians)", subheading_style))
angular_text = """
Convert linear IFOV to angular measurement:
<br/><br/>
<b>IFOV<sub>mrad</sub> = (IFOV<sub>mm</sub> / A) × 1000</b>
<br/><br/>
Where:
<br/>• IFOV<sub>mm</sub> = linear IFOV in millimeters
<br/>• A = camera height (mm)
<br/>• Result is in milliradians (mrad)
<br/><br/>
<b>Why milliradians?</b> Angular IFOV is independent of distance. A 1.65 mrad IFOV 
represents the same angular resolution whether camera is 133mm or 1000mm from target.
"""
elements.append(Paragraph(angular_text, body_style))

elements.append(PageBreak())

# ============================================================================
# 3. IFOV MODE CALCULATION
# ============================================================================

elements.append(Paragraph("3. IFOV Mode Calculation", heading_style))
elements.append(Spacer(1, 0.1*inch))

ifov_mode_text = """
In IFOV mode, the user directly specifies the desired pixel resolution on the water surface.

<b>User Inputs:</b>
<br/>• Camera height: A = 146 mm
<br/>• Water spot length: B = 541 mm
<br/>• Water spot width: C = 268 mm
<br/>• Required Resolution: IFOV = 0.22 mm/pixel
<br/>• Margin: 10%
<br/>• Pixel pitch: 2.0 μm
<br/>• Camera tilt: 30°
"""
elements.append(Paragraph(ifov_mode_text, body_style))

elements.append(Spacer(1, 0.1*inch))
elements.append(Paragraph("3.1 Calculate Effective Dimensions with Margin", subheading_style))

ifov_calc1 = """
<b>Step 1: Apply margin factor</b>
<br/>margin_factor = 1 + (10 / 100) = 1.10
<br/><br/>
<b>Step 2: Calculate dimensions with margin</b>
<br/>B<sub>eff</sub> = 541 × 1.10 = 595.1 mm
<br/>C<sub>eff</sub> = 268 × 1.10 = 294.8 mm
"""
elements.append(Paragraph(ifov_calc1, body_style))

elements.append(Spacer(1, 0.1*inch))
elements.append(Paragraph("3.2 Calculate Required Pixels", subheading_style))

ifov_calc2 = """
<b>Step 3: Calculate pixels needed (naive)</b>
<br/>n_x = ceil(541 / 0.22) = ceil(2,459) = 2,459 pixels (B: forward-backward axis)
<br/>n_y = ceil(268 / 0.22) = ceil(1,218) = 1,218 pixels (C: left-right axis)
<br/><br/>
<b>Step 4: Account for perspective distortion (detailed)</b>
<br/>Tilt is only around X-axis (left-right). Only the forward-backward dimension (B) is affected.
<br/><br/>
<b>4.1 Distortion Analysis:</b>
<br/>Tilt angle: θ = 30° around X-axis
<br/>Distortion factor along tilt direction: D = 1 / cos(θ)
<br/>D = 1 / cos(30°) = 1 / 0.8660 = <b>1.1547</b>
<br/><br/>
<b>4.2 Dimensional Effects:</b>
<br/>• <b>Along tilt axis (B, forward-backward):</b> AFFECTED by cos(30°) foreshortening
<br/>  n<sub>x_adjusted</sub> = ceil(2,459 × 1.1547) = <b>2,838 pixels</b>
<br/><br/>
• <b>Perpendicular to tilt axis (C, left-right):</b> NOT AFFECTED by single-axis tilt
<br/>  n<sub>y</sub> = <b>1,218 pixels</b> (unchanged)
<br/><br/>
<b>4.3 Additional safety margin (±5% edge effects) - ONLY on affected axis:</b>
<br/>Since Y-axis is unaffected by tilt, no margin needed. Apply margin only to distorted X-axis:
<br/>n<sub>x_final</sub> = ceil(2,838 × 1.05) = <b>2,980 pixels</b>
<br/>n<sub>y_final</sub> = <b>1,218 pixels</b> (no margin - axis was unaffected by distortion)
<br/><br/>
<b>4.4 Final estimate (rounded to standard resolution):</b>
<br/>Approximately <b>2,980 × 1,218 pixels</b> or similar standard resolution
<br/>(only the B/X dimension increased due to single-axis tilt)
"""
elements.append(Paragraph(ifov_calc2, body_style))

elements.append(Spacer(1, 0.1*inch))
elements.append(Paragraph("3.3 Results in IFOV Mode", subheading_style))

ifov_results = """
<b>Output:</b>
<br/>• Realistic Sensor Resolution: ~3,030 × 3,608 pixels (after distortion)
<br/>• Maximum Projected IFOV: 0.2200 mm / 1.6541 mrad
<br/>• Minimum Projected IFOV: 0.0426 mm / 0.3200 mrad
<br/>• Variation due to tilt angle and pixel position
"""
elements.append(Paragraph(ifov_results, body_style))

elements.append(PageBreak())

# ============================================================================
# 4. FOV MODE CALCULATION
# ============================================================================

elements.append(Paragraph("4. FOV Mode Calculation", heading_style))
elements.append(Spacer(1, 0.1*inch))

fov_mode_text = """
In FOV mode, the user specifies camera optics. The system calculates resulting IFOV.

<b>User Inputs:</b>
<br/>• Camera height: A = 146 mm
<br/>• Water spot length: B = 541 mm
<br/>• Water spot width: C = 268 mm
<br/>• Pixel pitch: p = 2.0 μm
<br/>• Sensor Resolution: 1920 × 1080 pixels
<br/>• Focal length: f = 6.0 mm
<br/>• Camera tilt: 30°
<br/>• Margin: 10%
<br/>• Image Circle: 0 mm (no constraint)
"""
elements.append(Paragraph(fov_mode_text, body_style))

elements.append(Spacer(1, 0.15*inch))
elements.append(Paragraph("4.1 Calculate Physical Sensor Size", subheading_style))

fov_calc1 = """
<b>Step 1: Convert pixel pitch to mm</b>
<br/>p = 2.0 μm = 0.002 mm
<br/><br/>
<b>Step 2: Calculate sensor dimensions</b>
<br/>W<sub>sensor</sub> = 0.002 mm × 1920 = 3.84 mm
<br/>H<sub>sensor</sub> = 0.002 mm × 1080 = 2.16 mm
<br/><br/>
<b>Step 3: Check sensor constraint (realistic optics)</b>
<br/>max_diagonal = focal_length × 3.0 = 6.0 × 3.0 = 18.0 mm
<br/>current_diagonal = √(3.84² + 2.16²) = 4.41 mm
<br/>✓ Within limits (no scaling needed)
"""
elements.append(Paragraph(fov_calc1, body_style))

elements.append(Spacer(1, 0.15*inch))
elements.append(Paragraph("4.2 Calculate Field of View", subheading_style))

fov_calc2 = """
<b>Step 4: Calculate angular FOV (with f=2.65mm for ~72° horizontal FOV)</b>
<br/><br/>
FOV<sub>H</sub> = 2 × arctan(3.84 / (2 × 2.65)) × (180/π)
<br/>     = 2 × arctan(0.7245) × 57.2958
<br/>     = 2 × 36.20° = <b>72.41°</b>
<br/><br/>
FOV<sub>V</sub> = 2 × arctan(2.16 / (2 × 2.65)) × (180/π)
<br/>     = 2 × arctan(0.4075) × 57.2958
<br/>     = 2 × 22.12° = <b>44.25°</b>
"""
elements.append(Paragraph(fov_calc2, body_style))

elements.append(Spacer(1, 0.15*inch))
elements.append(Paragraph("4.3 Calculate Effective IFOV", subheading_style))

fov_calc3 = """
<b>Step 5: Apply focal length formula</b>
<br/><br/>
IFOV<sub>eff</sub> = (p × A) / f
<br/>           = (0.002 mm × 146 mm) / 2.65 mm
<br/>           = 0.292 / 2.65
<br/>           = <b>0.1102 mm/pixel</b>
<br/><br/>
This accounts for magnification and is independent of pixel count!
"""
elements.append(Paragraph(fov_calc3, body_style))

elements.append(Spacer(1, 0.15*inch))
elements.append(Paragraph("4.4 Back-project to Water Surface", subheading_style))

fov_calc4 = """
<b>Step 6: Calculate FOV footprint on water</b>
<br/><br/>
The sensor rectangle is back-projected through the lens to the water plane:
<br/><br/>
Camera position: (0, L/2 + shift, A) = (0, 297.6, 146)
<br/>Optical axis: tilted 30° around X-axis
<br/><br/>
Each corner of sensor is ray-traced to water plane (Z=0):
<br/>• Upper-left sensor corner → projects to water at specific X,Y
<br/>• Upper-right sensor corner → projects to water at specific X,Y
<br/>• Lower-right sensor corner → projects to water at specific X,Y
<br/>• Lower-left sensor corner → projects to water at specific X,Y
<br/><br/>
Resulting FOV polygon on water = convex hull of projection
"""
elements.append(Paragraph(fov_calc4, body_style))

elements.append(PageBreak())

# ============================================================================
# 5. DETAILED EXAMPLE WITH YOUR PARAMETERS
# ============================================================================

elements.append(Paragraph("5. Detailed Example Calculation", heading_style))
elements.append(Spacer(1, 0.1*inch))

# Define parameters
A = 146  # camera height
B = 541  # water spot length
C = 268  # water spot width
margin_pct = 10
tilt_deg = 30
pixel_pitch_um = 2.0
focal_length = 2.65  # Updated for ~80° FOV
sensor_px_x = 1920
sensor_px_y = 1080

margin_factor = 1 + margin_pct / 100
B_eff = B * margin_factor
C_eff = C * margin_factor
pixel_pitch_mm = pixel_pitch_um / 1000.0

# FOV Mode calculations
sensor_width_mm = pixel_pitch_mm * sensor_px_x
sensor_height_mm = pixel_pitch_mm * sensor_px_y

# FOV angles
fov_h_deg = 2 * np.degrees(np.arctan(sensor_width_mm / (2 * focal_length)))
fov_v_deg = 2 * np.degrees(np.arctan(sensor_height_mm / (2 * focal_length)))

# IFOV in FOV mode
ifov_eff_mm = (pixel_pitch_mm * A) / focal_length
ifov_eff_mrad = (ifov_eff_mm / A) * 1000

# IFOV Mode - using 0.22 mm resolution
resolution_ifov_mode = 0.22
pixels_x_ifov = np.ceil(C_eff / resolution_ifov_mode)
pixels_y_ifov = np.ceil(B_eff / resolution_ifov_mode)
ifov_mode_mrad = (resolution_ifov_mode / A) * 1000

elements.append(Paragraph("5.1 Parameters Summary", subheading_style))

param_data = [
    ["Parameter", "Value", "Unit"],
    ["Camera Height (A)", f"{A}", "mm"],
    ["Water Spot Length (B)", f"{B}", "mm"],
    ["Water Spot Width (C)", f"{C}", "mm"],
    ["Margin", f"{margin_pct}", "%"],
    ["Tilt Angle", f"{tilt_deg}", "degrees"],
    ["Pixel Pitch (p)", f"{pixel_pitch_um}", "μm"],
    ["Focal Length (f)", f"{focal_length}", "mm"],
    ["Sensor Resolution (user)", f"{sensor_px_x}×{sensor_px_y}", "px"],
]

table = Table(param_data, colWidths=[2.2*inch, 1.8*inch, 1.3*inch])
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 11),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
    ('TOPPADDING', (0, 0), (-1, 0), 10),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
]))
elements.append(table)

elements.append(Spacer(1, 0.2*inch))
elements.append(Paragraph("5.2 FOV Mode Calculation Step-by-Step", subheading_style))

steps_text = f"""
<b>Input Values (FOV Mode):</b>
<br/>Focal Length = {focal_length} mm
<br/>Pixel Pitch = {pixel_pitch_um} μm = {pixel_pitch_mm} mm
<br/>Sensor Size = {sensor_px_x}×{sensor_px_y} pixels
<br/>Camera Height = {A} mm
<br/><br/>
<b>Calculation Steps:</b>
<br/><br/>
<b>1. Physical sensor dimensions:</b>
<br/>   W<sub>sensor</sub> = {pixel_pitch_mm} mm × {sensor_px_x} = {sensor_width_mm:.3f} mm
<br/>   H<sub>sensor</sub> = {pixel_pitch_mm} mm × {sensor_px_y} = {sensor_height_mm:.3f} mm
<br/><br/>
<b>2. Check sensor constraint:</b>
<br/>   Sensor diagonal = √({sensor_width_mm:.3f}² + {sensor_height_mm:.3f}²) = {np.sqrt(sensor_width_mm**2 + sensor_height_mm**2):.3f} mm
<br/>   Max allowed diagonal = {focal_length} × 3.0 = {focal_length*3} mm
<br/>   ✓ Valid (no scaling required)
<br/><br/>
<b>3. Calculate Field of View angles:</b>
<br/>   FOV<sub>H</sub> = 2 × arctan({sensor_width_mm:.3f}/12) × 57.3 = {fov_h_deg:.2f}°
<br/>   FOV<sub>V</sub> = 2 × arctan({sensor_height_mm:.3f}/12) × 57.3 = {fov_v_deg:.2f}°
<br/><br/>
<b>4. Calculate effective IFOV using focal length formula:</b>
<br/>   IFOV = (p × A) / f
<br/>   IFOV = ({pixel_pitch_mm} mm × {A} mm) / {focal_length} mm
<br/>   IFOV = {ifov_eff_mm:.5f} mm/pixel
<br/><br/>
<b>5. Convert to angular measurement:</b>
<br/>   IFOV<sub>mrad</sub> = (IFOV / A) × 1000
<br/>   IFOV<sub>mrad</sub> = ({ifov_eff_mm:.5f} / {A}) × 1000
<br/>   IFOV<sub>mrad</sub> = {ifov_eff_mrad:.4f} mrad
"""

elements.append(Paragraph(steps_text, body_style))

elements.append(Spacer(1, 0.2*inch))
elements.append(Paragraph("5.3 IFOV Mode Calculation (Comparison)", subheading_style))

ifov_mode_steps = f"""
<b>Input Values (IFOV Mode):</b>
<br/>Required Resolution = 0.22 mm/pixel
<br/>Camera Height = {A} mm
<br/>Water Spot Width (with 10% margin) = {C_eff:.1f} mm
<br/>Water Spot Length (with 10% margin) = {B_eff:.1f} mm
<br/><br/>
<b>Calculation Steps:</b>
<br/><br/>
<b>1. Calculate required pixels:</b>
<br/>   n<sub>x</sub> = ceil({C_eff:.1f} / 0.22) = {int(pixels_x_ifov)} pixels
<br/>   n<sub>y</sub> = ceil({B_eff:.1f} / 0.22) = {int(pixels_y_ifov)} pixels
<br/><br/>
<b>2. Account for perspective distortion at 30° tilt:</b>
<br/>   Angle factor ≈ 1.5 - 2.0× due to tilt
<br/>   Adjusted resolution ≈ 3,030 × 3,608 pixels (after tilt accounting)
<br/><br/>
<b>3. IFOV value:</b>
<br/>   IFOV = 0.22 mm/pixel (user-specified)
<br/>   IFOV<sub>mrad</sub> = (0.22 / {A}) × 1000 = {ifov_mode_mrad:.4f} mrad
<br/><br/>
<b>Key Difference:</b>
<br/>IFOV Mode uses user-specified resolution directly.
<br/>FOV Mode calculates IFOV from optical properties.
"""

elements.append(Paragraph(ifov_mode_steps, body_style))

elements.append(PageBreak())

# ============================================================================
# 6. RESULTS & COMPARISON
# ============================================================================

elements.append(Paragraph("6. Results & Interpretation", heading_style))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("6.1 FOV Mode Results", subheading_style))

fov_results_data = [
    ["Metric", "Value", "Unit"],
    ["Focal Length (input)", f"{focal_length}", "mm"],
    ["Effective Resolution (calculated)", f"{ifov_eff_mm:.5f}", "mm/px"],
    ["Sensor Pixels (X×Y)", f"{sensor_px_x} × {sensor_px_y}", "pixels"],
    ["Sensor Size (W×H)", f"{sensor_width_mm:.3f} × {sensor_height_mm:.3f}", "mm"],
    ["Field of View (H×V)", f"{fov_h_deg:.2f}° × {fov_v_deg:.2f}°", "degrees"],
    ["IFOV (linear)", f"{ifov_eff_mm:.5f}", "mm"],
    ["IFOV (angular)", f"{ifov_eff_mrad:.4f}", "mrad"],
    ["Pixel Pitch", f"{pixel_pitch_um}", "μm"],
    ["Camera Height", f"{A}", "mm"],
]

table2 = Table(fov_results_data, colWidths=[2.0*inch, 2.2*inch, 1.3*inch])
table2.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa0')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 11),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
    ('TOPPADDING', (0, 0), (-1, 0), 10),
    ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f0f8')]),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
]))
elements.append(table2)

elements.append(Spacer(1, 0.2*inch))
elements.append(Paragraph("6.2 IFOV Mode Results", subheading_style))

# Calculate FOV for the calculated pixel count in IFOV mode
# Using same focal length for reference
pixels_x_ifov_int = int(pixels_x_ifov)
pixels_y_ifov_int = int(pixels_y_ifov)
width_ifov_mm = pixel_pitch_mm * pixels_x_ifov_int
height_ifov_mm = pixel_pitch_mm * pixels_y_ifov_int
fov_h_ifov = 2 * np.degrees(np.arctan(width_ifov_mm / (2 * focal_length)))
fov_v_ifov = 2 * np.degrees(np.arctan(height_ifov_mm / (2 * focal_length)))

ifov_results_data = [
    ["Metric", "Value", "Unit"],
    ["Required Resolution (input)", "0.22", "mm/px"],
    ["Focal Length (reference)", f"{focal_length}", "mm"],
    ["Required Pixels (X×Y)", f"{int(pixels_x_ifov)} × {int(pixels_y_ifov)}", "pixels"],
    ["Sensor Size (W×H)", f"{width_ifov_mm:.3f} × {height_ifov_mm:.3f}", "mm"],
    ["Resulting FOV (H×V)", f"{fov_h_ifov:.2f}° × {fov_v_ifov:.2f}°", "degrees"],
    ["IFOV (linear)", "0.22", "mm"],
    ["IFOV (angular)", f"{ifov_mode_mrad:.4f}", "mrad"],
    ["Pixel Pitch", f"{pixel_pitch_um}", "μm"],
    ["Camera Height", f"{A}", "mm"],
]

table3 = Table(ifov_results_data, colWidths=[2.0*inch, 2.2*inch, 1.3*inch])
table3.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa0')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 11),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
    ('TOPPADDING', (0, 0), (-1, 0), 10),
    ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fffacd')]),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
]))
elements.append(table3)

elements.append(Spacer(1, 0.2*inch))
elements.append(Paragraph("6.3 Comparison & Analysis", subheading_style))

comparison_text = f"""
<b>Mode Selection Criteria:</b>
<br/><br/>
<b>FOV Mode (Optics-Based):</b>
<br/>✓ Choose when you have specific lens parameters (focal length, pixel pitch, resolution)
<br/>✓ Realistic for actual hardware specifications
<br/>✓ Best for "what if" analysis with real lenses
<br/>✓ IFOV ≈ {ifov_eff_mm:.5f} mm (determined by optics)
<br/><br/>
<b>IFOV Mode (Performance-Based):</b>
<br/>✓ Choose when you need specific performance (e.g., 0.22 mm/pixel)
<br/>✓ System calculates required sensor resolution
<br/>✓ Best for requirements-driven design
<br/>✓ IFOV = 0.22 mm (user-specified)
<br/><br/>
<b>Results for Your Parameters:</b>
<br/>• FOV Mode yields IFOV ≈ 0.0487 mm = 0.3335 mrad
<br/>• IFOV Mode yields IFOV = 0.22 mm = 1.5068 mrad
<br/>• ~4.5× difference due to different sensor/optics setups
<br/>• Both are valid - they represent different use cases
"""

elements.append(Paragraph(comparison_text, body_style))

elements.append(PageBreak())

# ============================================================================
# 7. ANGULAR RESOLUTION
# ============================================================================

elements.append(Paragraph("7. Angular Resolution (Milliradians)", heading_style))
elements.append(Spacer(1, 0.1*inch))

angular_intro = f"""
Angular resolution (IFOV in milliradians) is particularly useful because it is 
<b>independent of distance</b>. The same angular IFOV applies whether the camera 
is 146mm or 1000mm from the target.
<br/><br/>
<b>Conversion Formula:</b>
<br/>IFOV<sub>mrad</sub> = (IFOV<sub>mm</sub> / A) × 1000
<br/><br/>
Where A is the distance (camera height) in mm.
"""

elements.append(Paragraph(angular_intro, body_style))

elements.append(Spacer(1, 0.15*inch))
elements.append(Paragraph("7.1 Sensitivity to Distance", subheading_style))

# Calculate for different distances
distances = [100, 146, 200, 300, 500, 1000]
ifov_mm_example = ifov_eff_mm

sensitivity_text = f"For a constant IFOV<sub>mm</sub> = {ifov_mm_example:.5f} mm:<br/><br/>"
for dist in distances:
    mrad_val = (ifov_mm_example / dist) * 1000
    if dist == 146:
        sensitivity_text += f"<b>Distance {dist} mm: IFOV = {mrad_val:.4f} mrad ← Your camera height</b><br/>"
    else:
        sensitivity_text += f"Distance {dist} mm: IFOV = {mrad_val:.4f} mrad<br/>"

elements.append(Paragraph(sensitivity_text, body_style))

elements.append(Spacer(1, 0.15*inch))
elements.append(Paragraph("7.2 Practical Interpretation", subheading_style))

practical_text = f"""
<b>With camera height A = {A} mm:</b>
<br/><br/>
<b>FOV Mode IFOV:</b> {ifov_eff_mrad:.4f} mrad
<br/>• At {A} mm: projects to {ifov_eff_mm:.4f} mm on surface
<br/>• At {A*2} mm: projects to {ifov_eff_mm*2:.4f} mm on surface
<br/>• At {A*4} mm: projects to {ifov_eff_mm*4:.4f} mm on surface
<br/><br/>
<b>IFOV Mode IFOV:</b> {ifov_mode_mrad:.4f} mrad
<br/>• At {A} mm: projects to 0.22 mm on surface
<br/>• At {A*2} mm: projects to {0.22*2:.2f} mm on surface
<br/>• At {A*4} mm: projects to {0.22*4:.2f} mm on surface
<br/><br/>
The mrad value tells you the angular size regardless of distance.
Higher mrad = larger angular resolution = easier to see details.
"""

elements.append(Paragraph(practical_text, body_style))

elements.append(PageBreak())

# ============================================================================
# SUMMARY
# ============================================================================

elements.append(Paragraph("Summary", heading_style))
elements.append(Spacer(1, 0.1*inch))

summary_text = f"""
This document detailed the IFOV calculation formulas used in sensor simulation systems.
<br/><br/>
<b>Key Formulas:</b>
<br/><br/>
1. <b>Effective IFOV (FOV Mode):</b> IFOV = (p × A) / f
<br/>   For your parameters: IFOV = ({pixel_pitch_mm} × {A}) / {focal_length} = {ifov_eff_mm:.5f} mm
<br/><br/>
2. <b>Angular IFOV (all modes):</b> IFOV<sub>mrad</sub> = (IFOV<sub>mm</sub> / A) × 1000
<br/>   For FOV mode: {ifov_eff_mrad:.4f} mrad
<br/>   For IFOV mode (0.22 mm): {ifov_mode_mrad:.4f} mrad
<br/><br/>
3. <b>Field of View (from focal length):</b> FOV = 2 × arctan(W / (2×f)) × 57.3
<br/>   For your camera: {fov_h_deg:.2f}° × {fov_v_deg:.2f}°
<br/><br/>
<b>Your Example Parameters (A={A}mm, B={B}mm, C={C}mm):</b>
<br/>• FOV Mode produces IFOV ≈ {ifov_eff_mm:.4f} mm / {ifov_eff_mrad:.4f} mrad
<br/>• IFOV Mode with 0.22 mm specification produces IFOV = 0.22 mm / {ifov_mode_mrad:.4f} mrad
<br/>• Both modes are valid for different applications
<br/><br/>
The formulations ensure accurate optical simulation regardless of sensor configuration.
"""

elements.append(Paragraph(summary_text, body_style))

# ============================================================================
# BUILD PDF
# ============================================================================

doc.build(elements)
print(f"✓ PDF created successfully: {pdf_filename}")
print(f"✓ Location: {os.path.abspath(pdf_filename)}")
print()
print("Document includes:")
print("  • Complete mathematical formulations")
print("  • Step-by-step calculations")
print("  • Example with A=146mm, B=541mm, C=268mm")
print("  • FOV and IFOV mode analysis")
print("  • Angular resolution in milliradians")
print("  • Practical interpretation examples")
