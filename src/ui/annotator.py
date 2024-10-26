import pandas as pd
import json
from pathlib import Path
import gradio as gr
from datetime import datetime
import shutil

class TranscriptionAnnotator:
    def __init__(self):
        self.df = None
        self.excel_file = None
        self.current_index = 0
        self.selected_column = None
        self.upload_dir = Path("uploads")
        self.upload_dir.mkdir(exist_ok=True)
        self.codebook_path = self.upload_dir / "codebook.json"

    def backup_existing_codebook(self):
        if self.codebook_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.upload_dir / f"backup_{timestamp}_codebook.json"
            shutil.copy2(self.codebook_path, backup_path)

    def upload_file(self, file):
        if not file:
            return "No file uploaded", [], []
        
        try:
            self.excel_file = file.name
            # Backup existing codebook if it exists
            self.backup_existing_codebook()
            
            # Delete the existing codebook.json after backup
            if self.codebook_path.exists():
                self.codebook_path.unlink()
            
            # Get sheet names
            xl = pd.ExcelFile(self.excel_file)
            sheet_names = xl.sheet_names
            
            return "File uploaded successfully", sheet_names, []
        except Exception as e:
            return f"Error loading file: {str(e)}", [], []

    def get_columns(self, sheet_name):
        if not self.excel_file or not sheet_name:
            return []
        try:
            temp_df = pd.read_excel(self.excel_file, sheet_name=sheet_name)
            return temp_df.columns.tolist()
        except:
            return []

    def load_settings(self, sheet_name, column_name):
        if not self.excel_file:
            return "No file loaded", gr.DataFrame(visible=False), "", [], []
        
        try:
            temp_df = pd.read_excel(self.excel_file, sheet_name=sheet_name)
            
            if column_name not in temp_df.columns:
                return f"Column '{column_name}' not found in sheet", gr.DataFrame(visible=False), "", [], []
            
            # Create new DataFrame with only ID and specified column
            self.df = pd.DataFrame()
            self.df['ID'] = range(1, len(temp_df) + 1)
            # Rename the selected column to "text"
            self.df['text'] = temp_df[column_name]
            # Add is_reviewed column with default value False
            self.df['is_reviewed'] = False
            
            self.selected_column = 'text'  # Update selected column name
            self.current_index = 0
            
            # Create initial codebook
            self.create_codebook()
            
            # Initialize the first text to display
            initial_text = self.df.iloc[0]['text']
            
            # Get existing codes for dropdowns
            codes = [code["name"] for code in self.load_codebook()]
            
            return f"Settings applied and codebook created at {self.codebook_path}", gr.DataFrame(value=self.df), initial_text, codes, codes
        
        except Exception as e:
            return f"Error applying settings: {str(e)}", gr.DataFrame(visible=False), "", [], []

    def create_codebook(self):
        if not self.codebook_path.exists():
            codebook = {
                "created_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "dataset": Path(self.excel_file).name if self.excel_file else "",
                "codes": []
            }
            with open(self.codebook_path, 'w') as f:
                json.dump(codebook, f, indent=4)

    def load_codebook(self):
        if not self.codebook_path.exists():
            return []
        with open(self.codebook_path, 'r') as f:
            return json.load(f)['codes']

    def add_code(self, code_name, code_values, code_description):
        if not code_name:
            return "Please enter a code name", self.load_codebook(), gr.DataFrame(value=self.df), [], []
        
        try:
            with open(self.codebook_path, 'r') as f:
                codebook = json.load(f)

            # Convert string of values to list
            values = [v.strip() for v in code_values.split(',') if v.strip()]

            # Add new code
            new_code = {
                "name": code_name,
                "description": code_description,
                "values": values
            }
            
            codebook['codes'].append(new_code)

            with open(self.codebook_path, 'w') as f:
                json.dump(codebook, f, indent=4)

            # Add column to DataFrame
            if self.df is not None:
                self.df[code_name] = ""

            # Get updated list of code names
            codes = [code["name"] for code in codebook['codes']]

            return f"Added code: {code_name}", self.load_codebook(), gr.DataFrame(value=self.df), codes, codes
        except Exception as e:
            return f"Error adding code: {str(e)}", self.load_codebook(), gr.DataFrame(value=self.df), [], []

    def delete_code(self, code_name):
        if not self.codebook_path.exists():
            return "Codebook does not exist.", self.load_codebook(), [], []

        try:
            with open(self.codebook_path, 'r') as f:
                codebook = json.load(f)

            # Remove code from codebook
            codebook['codes'] = [code for code in codebook['codes'] if code['name'] != code_name]

            with open(self.codebook_path, 'w') as f:
                json.dump(codebook, f, indent=4)

            # Remove column from DataFrame
            if self.df is not None and code_name in self.df.columns:
                self.df.drop(columns=[code_name], inplace=True)

            # Get updated list of code names
            codes = [code["name"] for code in codebook['codes']]

            return f"Deleted code: {code_name}", self.load_codebook(), gr.DataFrame(value=self.df), codes, codes
        except Exception as e:
            return f"Error deleting code: {str(e)}", self.load_codebook(), gr.DataFrame(value=self.df), [], []

    def refresh_codes(self):
        if not self.codebook_path.exists():
            return [], []
        try:
            with open(self.codebook_path, 'r') as f:
                codebook = json.load(f)
            categories = [code['name'] for code in codebook['codes']]
            return categories, []
        except Exception as e:
            return [], []

    def get_code_values(self, code_name):
        if not self.codebook_path.exists() or not code_name:
            return []
        try:
            with open(self.codebook_path, 'r') as f:
                codebook = json.load(f)
                for code in codebook['codes']:
                    if code['name'] == code_name:
                        # Return only the value part, not the full dictionary
                        return [v['value'] for v in code['values']]
            return []
        except Exception as e:
            print(f"Error getting code values: {e}")
            return []

    def save_annotation(self, code_name, value):
        if self.df is None:
            return "Please load data first", None
        
        if not code_name or not value:
            return "Please select both category and value", None
            
        try:
            # Verify the code exists in the codebook
            with open(self.codebook_path, 'r') as f:
                codebook = json.load(f)
                code_exists = any(code['name'] == code_name for code in codebook['codes'])
                if not code_exists:
                    return f"Code {code_name} not found in codebook", None
            
            # Save the annotation
            self.df.at[self.current_index, code_name] = value
            # Set is_reviewed to True when annotation is saved
            self.df.at[self.current_index, 'is_reviewed'] = True
            
            return f"Saved {value} for {code_name} at index {self.current_index}", self.df
        except Exception as e:
            return f"Error saving annotation: {str(e)}", None

    def navigate_transcripts(self, direction):
        if self.df is None or self.selected_column is None:
            return None, None
        
        try:
            if direction == "next":
                self.current_index = min(self.current_index + 1, len(self.df) - 1)
            else:
                self.current_index = max(self.current_index - 1, 0)
                
            return self.df.iloc[self.current_index]['text'], self.current_index
        except Exception as e:
            print(f"Error navigating transcripts: {e}")
            return None, None

    def save_excel(self):
        if self.df is None:
            return None, "No data to save"
        try:
            output_path = self.upload_dir / "annotated_transcripts.xlsx"
            self.df.to_excel(output_path, index=False)
            return str(output_path), "File saved successfully"
        except Exception as e:
            return None, f"Error saving file: {str(e)}"
        
    def add_code(self, code_name, code_description):
        if not code_name:
            return "Please enter a code name", self.load_codebook(), [], []
        
        try:
            with open(self.codebook_path, 'r') as f:
                codebook = json.load(f)
            
            new_code = {
                "name": code_name,
                "description": code_description,
                "values": []
            }
            
            codebook['codes'].append(new_code)
            
            with open(self.codebook_path, 'w') as f:
                json.dump(codebook, f, indent=4)
            
            if self.df is not None:
                self.df[code_name] = ""
            
            # Get updated list of code names for dropdowns
            codes = [code["name"] for code in codebook['codes']]
            
            return (
                f"Added code: {code_name}",  # status message
                self.load_codebook(),        # codebook JSON
                codes,                       # code_select dropdown choices
                codes                        # delete_code_select dropdown choices
            )
        except Exception as e:
            return f"Error adding code: {str(e)}", self.load_codebook(), [], []

    def add_value_to_code(self, code_name, value_name, value_description):
        if not code_name or not value_name:
            return "Please select a code and enter a value", self.load_codebook()
        
        try:
            with open(self.codebook_path, 'r') as f:
                codebook = json.load(f)
            
            for code in codebook['codes']:
                if code['name'] == code_name:
                    new_value = {
                        "value": value_name,
                        "description": value_description
                    }
                    code['values'].append(new_value)
                    break
            
            with open(self.codebook_path, 'w') as f:
                json.dump(codebook, f, indent=4)
            
            return f"Added value {value_name} to {code_name}", self.load_codebook()
        except Exception as e:
            return f"Error adding value: {str(e)}", self.load_codebook()

    def delete_value_from_code(self, code_name, value_name):
        try:
            with open(self.codebook_path, 'r') as f:
                codebook = json.load(f)
            
            for code in codebook['codes']:
                if code['name'] == code_name:
                    code['values'] = [v for v in code['values'] 
                                    if v['value'] != value_name]
                    break
            
            with open(self.codebook_path, 'w') as f:
                json.dump(codebook, f, indent=4)
            
            return f"Deleted value {value_name} from {code_name}", self.load_codebook()
        except Exception as e:
            return f"Error deleting value: {str(e)}", self.load_codebook()