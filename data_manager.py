import pandas as pd
import os
from tkinter import messagebox
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_executable_dir():
    """Get the directory where the executable is located"""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return os.path.dirname(sys.executable)
    else:
        # Running as script
        return os.path.dirname(os.path.abspath(__file__))
    
class ToiletDataManager:
    def __init__(self, csv_file):
        self.csv_file =  os.path.join(get_executable_dir(), "toilet_data.csv")
        self.columns = (
            'Manufacturer',
            'Model',
            'Sub-Model',
            'A - Rim to Water depth (camera height) [mm]',
            'B - Water Spot Length [mm]',
            'C - Water Spot Width [mm]',
            'Camera Tilt [degrees]',
            'Margin [%]',
            'Shift from Water Spot Width Edge [mm]',
            'Shift Axis',
            'Dead Zone [mm]',
            'Required Resolution [mm/px]',
            'Pixel Pitch [um]',
            'Focal Length [mm]',
            'Sensor Resolution [px×px]',
            'Image Circle [mm]'
        )
        self.data = self.load_data()
    
    def load_data(self):
        if os.path.exists(self.csv_file):
            try:
                data = pd.read_csv(self.csv_file)
                # Handle old CSV files missing new columns
                for col in self.columns:
                    if col not in data.columns:
                        if col == 'Shift Axis':
                            data[col] = 'X'
                        elif col == 'Dead Zone [mm]':
                            data[col] = 0.3
                        elif col == 'Required Resolution [mm/px]':
                            data[col] = 0.22
                        elif col == 'Pixel Pitch [um]':
                            data[col] = 1.2
                        elif col == 'Focal Length [mm]':
                            data[col] = ''
                        elif col == 'Sensor Resolution [px×px]':
                            data[col] = ''
                        elif col == 'Image Circle [mm]':
                            data[col] = ''
                        else:
                            data[col] = ''
                
                # Reorder columns to match expected order
                data = data.reindex(columns=self.columns, fill_value='')
                return data
            except Exception as e:
                print(f"Error loading CSV: {e}")
        return pd.DataFrame(columns=self.columns)
    
    def save_data(self):
        try:
            # Ensure all columns exist and are in correct order
            for col in self.columns:
                if col not in self.data.columns:
                    self.data[col] = ''
            
            # Reorder columns
            self.data = self.data.reindex(columns=self.columns, fill_value='')
            
            # Replace any NaN values with empty strings before saving
            self.data = self.data.fillna('')
            
            # Save to CSV
            self.data.to_csv(self.csv_file, index=False)
            messagebox.showinfo("Success", f"Data saved to {self.csv_file}")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")
            return False
    
    def add_toilet(self, toilet_params):
        # Ensure all required columns are present with proper values
        complete_params = {}
        for col in self.columns:
            if col in toilet_params:
                complete_params[col] = toilet_params[col]
            else:
                complete_params[col] = ''
        
        # Create new DataFrame row
        new_row = pd.DataFrame([complete_params], columns=self.columns)
        
        # FIXED: Use pd.concat properly and reset index
        if self.data.empty:
            self.data = new_row.copy()
        else:
            self.data = pd.concat([self.data, new_row], ignore_index=True)
    
        # Ensure data integrity
        self.data = self.data.fillna('')
    
    def delete_toilet(self, index):
        if index < len(self.data):
            self.data = self.data.drop(self.data.index[index]).reset_index(drop=True)
    
    def update_cell(self, index, column, value):
        if index < len(self.data) and column in self.data.columns:
            self.data.at[index, column] = value
