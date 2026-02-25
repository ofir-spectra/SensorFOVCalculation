#!/usr/bin/env python3
"""
Dynamic PDF Report Generator for IFOV calculations
Can be called with custom parameters from the GUI
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
import numpy as np
from datetime import datetime


def generate_pdf_report(params_dict, output_filename=None):
    """
    Generate a comprehensive PDF report based on current simulation parameters.
    
    Args:
        params_dict: Dictionary with keys:
            - A: Camera height (mm)
            - B: Water spot length (mm)
            - C: Water spot width (mm)
            - Tilt: Tilt angle (degrees)
            - Margin: Margin percentage (%)
            - PixelPitch: Pixel pitch (μm)
            - Mode: 'FOV' or 'IFOV'
            - For FOV mode:
                - FocalLength: Focal length (mm)
                - SensorPixelsX: Sensor X resolution (pixels)
                - SensorPixelsY: Sensor Y resolution (pixels)
            - For IFOV mode:
                - Resolution: Required resolution (mm/pixel)
        output_filename: Output PDF filename (default: timestamp-based)
    
    Returns:
        Tuple (success: bool, filename: str, message: str)
    """
    try:
        # Extract parameters with defaults
        A = float(params_dict.get('A', 146))
        B = float(params_dict.get('B', 541))
        C = float(params_dict.get('C', 268))
        tilt_deg = float(params_dict.get('Tilt', 30))
        margin_pct = float(params_dict.get('Margin', 10))
        pixel_pitch_um = float(params_dict.get('PixelPitch', 2.0))
        mode = params_dict.get('Mode', 'FOV')
        
        # Generate default filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"IFOV_Report_{timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(output_filename, pagesize=letter,
                                rightMargin=0.5*inch, leftMargin=0.5*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        
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
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            leading=14
        )
        
        # ====================================================================
        # TITLE PAGE
        # ====================================================================
        
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph("IFOV Calculations Report", title_style))
        elements.append(Paragraph("Sensor Simulation System", heading_style))
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", body_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Mode info
        mode_text = "IFOV-Based (Performance-Driven)" if mode == "IFOV" else "FOV-Based (Optics-Driven)"
        info_data = [
            ["Simulation Mode", mode_text],
            ["Report Type", "Dynamic Parameters Report"],
            ["Generated From", "Live GUI Parameters"],
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
        
        # ====================================================================
        # SECTION 1: INPUT PARAMETERS
        # ====================================================================
        
        elements.append(Paragraph("1. Input Parameters", heading_style))
        elements.append(Spacer(1, 0.1*inch))
        
        margin_factor = 1 + margin_pct / 100
        B_eff = B * margin_factor
        C_eff = C * margin_factor
        pixel_pitch_mm = pixel_pitch_um / 1000.0
        
        # Common parameters table
        common_data = [
            ["Parameter", "Value", "Unit", "Description"],
            ["Camera Height (A)", f"{A:.1f}", "mm", "Distance from camera to water surface"],
            ["Water Spot Length (B)", f"{B:.1f}", "mm", "Longitudinal coverage needed"],
            ["Water Spot Width (C)", f"{C:.1f}", "mm", "Lateral coverage needed"],
            ["Margin", f"{margin_pct:.1f}", "%", "Safety margin on coverage area"],
            ["Effective Length (B')", f"{B_eff:.1f}", "mm", "B with margin applied"],
            ["Effective Width (C')", f"{C_eff:.1f}", "mm", "C with margin applied"],
            ["Tilt Angle", f"{tilt_deg:.1f}", "°", "Camera tilt around X-axis"],
            ["Pixel Pitch", f"{pixel_pitch_um:.2f}", "μm", "Individual pixel size"],
        ]
        
        common_table = Table(common_data, colWidths=[1.8*inch, 1.2*inch, 0.9*inch, 1.6*inch])
        common_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        elements.append(common_table)
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Mode-specific parameters
        if mode == "FOV":
            focal_length = float(params_dict.get('FocalLength', 2.65))
            sensor_px_x = int(params_dict.get('SensorPixelsX', 1920))
            sensor_px_y = int(params_dict.get('SensorPixelsY', 1080))
            
            elements.append(Paragraph("1.1 FOV Mode - Optical Parameters", heading_style))
            
            fov_data = [
                ["Parameter", "Value", "Unit"],
                ["Focal Length", f"{focal_length:.2f}", "mm"],
                ["Sensor Resolution (X)", f"{sensor_px_x}", "pixels"],
                ["Sensor Resolution (Y)", f"{sensor_px_y}", "pixels"],
            ]
            
            fov_table = Table(fov_data, colWidths=[2.2*inch, 1.8*inch, 1.3*inch])
            fov_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f0f8')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ]))
            elements.append(fov_table)
            
        else:  # IFOV mode
            resolution = float(params_dict.get('Resolution', 0.22))
            
            elements.append(Paragraph("1.1 IFOV Mode - Performance Parameters", heading_style))
            
            ifov_data = [
                ["Parameter", "Value", "Unit"],
                ["Required Resolution", f"{resolution:.4f}", "mm/pixel"],
                ["Required Pixels (X)", f"{int(np.ceil(C_eff / resolution))}", "pixels"],
                ["Required Pixels (Y)", f"{int(np.ceil(B_eff / resolution))}", "pixels"],
            ]
            
            ifov_table = Table(ifov_data, colWidths=[2.2*inch, 1.8*inch, 1.3*inch])
            ifov_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fffacd')]),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ]))
            elements.append(ifov_table)
        
        elements.append(PageBreak())
        
        # ====================================================================
        # SECTION 2: CALCULATIONS
        # ====================================================================
        
        elements.append(Paragraph("2. Key Calculations", heading_style))
        elements.append(Spacer(1, 0.1*inch))
        
        if mode == "FOV":
            # FOV mode calculations
            sensor_width_mm = pixel_pitch_mm * sensor_px_x
            sensor_height_mm = pixel_pitch_mm * sensor_px_y
            
            fov_h_deg = 2 * np.degrees(np.arctan(sensor_width_mm / (2 * focal_length)))
            fov_v_deg = 2 * np.degrees(np.arctan(sensor_height_mm / (2 * focal_length)))
            
            ifov_eff_mm = (pixel_pitch_mm * A) / focal_length
            ifov_eff_mrad = (ifov_eff_mm / A) * 1000
            
            calc_text = f"""
<b>2.1 Physical Sensor Dimensions:</b>
<br/>Width = {pixel_pitch_mm} mm × {sensor_px_x} = {sensor_width_mm:.3f} mm
<br/>Height = {pixel_pitch_mm} mm × {sensor_px_y} = {sensor_height_mm:.3f} mm
<br/><br/>
<b>2.2 Field of View Calculations:</b>
<br/>FOV<sub>H</sub> = 2 × arctan({sensor_width_mm:.3f} / (2 × {focal_length})) × 57.3 = <b>{fov_h_deg:.2f}°</b>
<br/>FOV<sub>V</sub> = 2 × arctan({sensor_height_mm:.3f} / (2 × {focal_length})) × 57.3 = <b>{fov_v_deg:.2f}°</b>
<br/><br/>
<b>2.3 Effective IFOV (Focal Length Formula):</b>
<br/>IFOV = (p × A) / f
<br/>IFOV = ({pixel_pitch_mm} mm × {A} mm) / {focal_length} mm
<br/>IFOV = <b>{ifov_eff_mm:.5f} mm/pixel</b>
<br/><br/>
<b>2.4 Angular Resolution:</b>
<br/>IFOV<sub>mrad</sub> = (IFOV / A) × 1000
<br/>IFOV<sub>mrad</sub> = ({ifov_eff_mm:.5f} / {A}) × 1000
<br/>IFOV<sub>mrad</sub> = <b>{ifov_eff_mrad:.4f} mrad</b>
"""
            
        else:
            # IFOV mode calculations
            resolution = float(params_dict.get('Resolution', 0.22))
            pixels_x = int(np.ceil(C_eff / resolution))
            pixels_y = int(np.ceil(B_eff / resolution))
            ifov_mode_mrad = (resolution / A) * 1000
            
            # Single-axis distortion (30° tilt around X-axis)
            # Only Y-axis (forward-backward) is affected; X-axis (left-right) is NOT affected
            distortion_factor = 1 / np.cos(np.radians(tilt_deg))
            pixels_y_adjusted = int(np.ceil(pixels_y * distortion_factor))
            # Apply 5% safety margin only to affected (distorted) dimension
            pixels_y_final = int(np.ceil(pixels_y_adjusted * 1.05))
            
            calc_text = f"""
<b>2.1 Required Pixels (Coverage Based):</b>
<br/>Pixels<sub>X</sub> (Width) = ceil({C_eff:.1f} / {resolution}) = {pixels_x} pixels
<br/>Pixels<sub>Y</sub> (Length) = ceil({B_eff:.1f} / {resolution}) = {pixels_y} pixels
<br/><br/>
<b>2.2 Distortion Correction ({tilt_deg}° Single-Axis Tilt):</b>
<br/>Distortion Factor = 1 / cos({tilt_deg}°) = {distortion_factor:.4f}
<br/>Pixels<sub>Y_adjusted</sub> = ceil({pixels_y} × {distortion_factor:.4f}) = {pixels_y_adjusted} pixels
<br/><b>Note:</b> Only Y-axis (forward-backward) affected by X-axis tilt; X-axis (left-right) unchanged
<br/><br/>
<b>2.3 Safety Margin (5% edge effects - ONLY on affected axis):</b>
<br/>Pixels<sub>Y_final</sub> = ceil({pixels_y_adjusted} × 1.05) = {pixels_y_final} pixels
<br/>Pixels<sub>X_final</sub> = {pixels_x} pixels (unchanged - unaffected by tilt)
<br/><br/>
<b>2.4 Angular Resolution:</b>
<br/>IFOV<sub>mrad</sub> = ({resolution} / {A}) × 1000
<br/>IFOV<sub>mrad</sub> = <b>{ifov_mode_mrad:.4f} mrad</b>
<br/><br/>
<b>2.5 Required Sensor:</b>
<br/>Approximately {pixels_x} × {pixels_y_final} pixels
<br/>(or nearest standard resolution supporting this pixel count)
"""
        
        elements.append(Paragraph(calc_text, body_style))
        
        elements.append(PageBreak())
        
        # ====================================================================
        # SECTION 3: SUMMARY
        # ====================================================================
        
        elements.append(Paragraph("3. Summary", heading_style))
        elements.append(Spacer(1, 0.1*inch))
        
        summary_text = f"""
<b>Simulation Configuration:</b>
<br/>• Mode: {mode_text}
<br/>• Camera Height: {A} mm
<br/>• Coverage Area: {B} × {C} mm (with {margin_pct}% margin: {B_eff:.1f} × {C_eff:.1f} mm)
<br/>• Tilt Angle: {tilt_deg}°
<br/>• Pixel Pitch: {pixel_pitch_um} μm
<br/><br/>
This report provides detailed calculations of the sensor requirements and 
optical properties based on the entered parameters. Use these specifications 
for sensor procurement, lens selection, and system validation.
<br/><br/>
<b>Report Notes:</b>
<br/>• All calculations assume perpendicular projection at specified tilt angle
<br/>• Angular measurements (mrad) are independent of distance
<br/>• Higher resolution/lower IFOV allows detection of smaller features
<br/>• Distortion factor accounts for perspective compression due to tilt
"""
        
        elements.append(Paragraph(summary_text, body_style))
        
        # ====================================================================
        # BUILD PDF
        # ====================================================================
        
        doc.build(elements)
        
        return (True, output_filename, f"✓ PDF report generated: {os.path.basename(output_filename)}")
        
    except Exception as e:
        return (False, "", f"✗ Error generating PDF: {str(e)}")


if __name__ == "__main__":
    # Example usage
    test_params_fov = {
        'A': 146,
        'B': 541,
        'C': 268,
        'Tilt': 30,
        'Margin': 10,
        'PixelPitch': 2.0,
        'Mode': 'FOV',
        'FocalLength': 2.65,
        'SensorPixelsX': 1920,
        'SensorPixelsY': 1080,
    }
    
    success, filename, msg = generate_pdf_report(test_params_fov)
    print(msg)
