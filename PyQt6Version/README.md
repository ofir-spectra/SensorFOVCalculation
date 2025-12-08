# PyQt6 Version (Experimental)

This folder contains an experimental PyQt6 port of the Sensor FOV Simulation. It keeps the original Tkinter app intact while providing a modern Qt-based UI to evaluate migration impacts.

## Run
```powershell
python "c:\Users\ofirn\OneDrive\Documents\Work\ConsultingServices\Outsense\Python\NewSensorCalculation\PyQt6Version\main.py"
```

## Notes
- Pulls version/build/branch from Git to show in the window title: `v<version> (<build>) (<branch>)`.
- Embeds a Matplotlib plot via `FigureCanvasQTAgg`.
- Uses `ToiletDataManager` and `get_plot_data` from the root modules to reuse logic.
- Minimal table populates data from `toilet_data.csv` for demonstration.

## Next Steps
- Replace placeholder table with real asset/portfolio model.
- Port auto-update threads to `QThread` with signals.
- Apply dark/light theme via Qt stylesheets or `qdarktheme`.
