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

## Repository Layout
- `SensorFOVCalc.py`: Tkinter GUI app (shows version/build in title)
- `NewSensorFOVCalc.py`: CLI-style entry (prints debug, runs calcs)
- `projection_calculations.py`: Core math and plotting data helpers
- `data_manager.py`: CSV data loader and management
- `image_utils.py`: Utility for resolving image paths
- `toilet_data.csv`: Geometry and configuration data

## Aim of This Tool
Support engineering decisions for a future camera sensor by simulating coverage, resolution, and geometry interactions. It offers quick parameter sweeps, visual feedback, and derived metrics (e.g., IFOV min/max), helping choose tilt, height, and aspect ratio under constraints.

## Notes
- If the repository isn’t a git repo or tags are missing, the title falls back to `v5.0.11 build local (unknown)`.
- Adjust defaults in `SensorFOVCalc.py` under `_get_git_meta()` if needed.
