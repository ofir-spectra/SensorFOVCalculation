# Sensor FOV Calculation (Outsense)

A practical simulator to analyze camera sensor field-of-view (FOV), IFOV resolution, and water coverage on a target area. Built to support design decisions around sensor placement, tilt, aspect ratio, and pixel pitch.

## Key Features
- Visualize top-down, side, and projected views of the scene.
- Compute IFOV across the area with dead zones and margins.
- Evaluate water spot coverage vs tilt and optimal angle.
- Load predefined toilet geometry data from `toilet_data.csv`.
- Simple Tkinter GUI for parameter tweaking and quick iterations.

## Versioning in Title
The app title now shows `v<version> build <build> (<branch>)` where:
- `version`: latest git tag (fallback `5.0.11`).
- `build`: commit count on current branch (fallback `local`).
- `branch`: current git branch (fallback `unknown`).

## Getting Started

### Prerequisites
- Python 3.10+
- Packages: `matplotlib`, `numpy`, `pandas`

Install packages:
```powershell
python -m pip install matplotlib numpy pandas
```

### Run the App
```powershell
$env:PYTHONPATH = "c:\Users\ofirn\OneDrive\Documents\Work\ConsultingServices\Outsense\Python\NewSensorCalculation"
python "c:\Users\ofirn\OneDrive\Documents\Work\ConsultingServices\Outsense\Python\NewSensorCalculation\NewSensorFOVCalc.py"
```
Alternatively, launch the Tkinter GUI directly:
```powershell
python "c:\Users\ofirn\OneDrive\Documents\Work\ConsultingServices\Outsense\Python\NewSensorCalculation\SensorFOVCalc.py"
```

### Parameters Overview
- **A (Camera height)**: Rim-to-water depth [mm]
- **B (Water spot length)**: Target length [mm]
- **C (Water spot width)**: Target width [mm]
- **Camera Tilt**: Degrees relative to vertical
- **Margin**: Extra coverage percentage
- **Shift**: Offset from water spot edge [mm]
- **Required Resolution**: Target IFOV [mm/px] (IFOV mode only)
- **Dead zone**: Non-usable area between tiles [mm]
- **Pixel pitch**: Sensor pixel size [μm]
- **Focal Length**: Lens focal length [mm] (FOV mode only)
- **Sensor Resolution**: Physical pixel dimensions [px×px] (FOV mode only)
- **Image Circle**: Diameter of lens's usable image circle [mm]
  - Limits the effective sensor area receiving light
  - If smaller than sensor diagonal, outer pixels are vignetted
  - For full sensor utilization: Image Circle ≥ sensor diagonal

### VS Code Tasks
Use the preconfigured tasks for one-click runs:
- "Run GUI" launches `SensorFOVCalc.py`
- "Run CLI" launches `NewSensorFOVCalc.py`
 - "Run PyQt6 GUI" launches `PyQt6Version/main.py` (experimental, slate theme)

### PyQt6 Experimental GUI
- Location: `PyQt6Version/main.py`
- Theme: Slate grey UI inspired by the screenshots; plots use white backgrounds.
- Panels: Title, Simulation Results, Parameters, Plots (4 subplots), Toilet Database.
- Run:
```powershell
python "${workspaceFolder}/PyQt6Version/main.py"
```
If imports fail when running from the folder directly, ensure the project root is on `PYTHONPATH` or run via the VS Code task.

### Screenshots
Screenshots live under `docs/`.

- `docs/pyqt6-overview.png` — Slate UI overview (title, panels)
- `docs/pyqt6-plots.png` — Four plots (World, Projection, Side, Coverage)
- `docs/tkinter-overview.png` — Tkinter UI overview

Quick capture:
1. Run the GUI (`Run GUI` or `Run PyQt6 GUI`).
2. Press `PrtScn` and paste into an image editor, or use `Snipping Tool`.
3. Save images to `docs/` with the names above.

## Repository Layout
- `SensorFOVCalc.py`: Tkinter GUI app (shows version/build in title)
- `NewSensorFOVCalc.py`: CLI-style entry (prints debug, runs calcs)
- `projection_calculations.py`: Core math and plotting data helpers
- `data_manager.py`: CSV data loader and management
- `image_utils.py`: Utility for resolving image paths
- `toilet_data.csv`: Geometry and configuration data

## Dual Operating Modes

### IFOV Mode
Design a sensor based on required resolution:
- Specify target resolution (mm/pixel)
- Calculate required sensor dimensions and pixel count
- Account for perspective distortion and tilt effects

### FOV Mode
Analyze existing sensor/lens combinations:
- Input actual sensor resolution and focal length
- Calculate field of view and coverage
- Evaluate Image Circle constraints on usable pixels

## Understanding Output Metrics

### Active Pixel Fraction (by Image Circle)
- **What it measures**: Percentage of sensor area that receives light from the lens
- **Limited by**: Lens optics (Image Circle diameter)
- **Good value**: ~100% (lens fully illuminates sensor)
- **Impact**: Hardware limitation - wasted pixels outside the circle
- **Fix**: Choose a lens with Image Circle ≥ sensor diagonal

### Water Coverage
- **What it measures**: Percentage of water spot visible in camera FOV
- **Limited by**: Camera geometry (position, tilt, height)
- **Good value**: >80% (most of target visible)
- **Impact**: Positioning issue - affects scene visibility
- **Fix**: Adjust camera placement, tilt, or height

**Key Difference**: Active Pixel Fraction is about lens-to-sensor optical match, while Water Coverage is about camera-to-scene geometric alignment. Both can independently limit performance.

## Aim of This Tool
Support engineering decisions for camera sensor systems by simulating coverage, resolution, and geometry interactions. Provides quick parameter sweeps, visual feedback, and derived metrics (IFOV min/max, FOV angles, coverage percentages) to optimize tilt, height, aspect ratio, and lens selection under real-world constraints.

## Notes
- If the repository isn’t a git repo or tags are missing, the title falls back to `v5.0.11 build local (unknown)`.
- Adjust defaults in `SensorFOVCalc.py` under `_get_git_meta()` if needed.

## Documentation
- User Manual: `CameraSim-UM.docx` (in the root folder). It provides background, parameter descriptions, and step-by-step usage. If you update the UI or parameters, please reflect changes in this document.
