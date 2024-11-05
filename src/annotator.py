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
        self.backup_path = self.upload_dir / "temp_annotations.xlsx"

    def backup_existing_codebook(self):
        if self.codebook_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.upload_dir / f"backup_{timestamp}_codebook.json"
            shutil.copy2(self.codebook_path, backup_path)

    def create_new_codebook(self):
        """Creates a new empty codebook and saves it"""
        codebook = {
            "created_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "dataset": Path(self.excel_file).name if self.excel_file else "",
            "codes": []
        }
        
        #self.backup_existing_codebook() ## Disabling backup
        with open(self.codebook_path, 'w') as f:
            json.dump(codebook, f, indent=4)
        return "New empty codebook created successfully"

    def upload_file(self, file, codebook_file=None):
        if not file:
            return "No file uploaded", [], []
        try:
            self.excel_file = file.name
            if codebook_file:
                codebook_status = self.upload_codebook(codebook_file)
                if "Error" in codebook_status:
                    return codebook_status, [], []
            elif not self.codebook_path.exists():
                self.create_codebook()
            
            xl = pd.ExcelFile(self.excel_file)
            sheet_names = xl.sheet_names
            return "File uploaded successfully", sheet_names, []
        except Exception as e:
            return f"Error loading file: {str(e)}", [], []

    def upload_codebook(self, file):
        if not file:
            return "No codebook uploaded"
        try:
            with open(file.name, 'r') as f:
                codebook = json.load(f)
                if not all(required in codebook for required in ['created_at', 'codes']):
                    raise ValueError("Invalid codebook format")
            
            #self.backup_existing_codebook() ## Disabling backup
            shutil.copy2(file.name, self.codebook_path)
            return "Codebook uploaded and loaded successfully"
        except json.JSONDecodeError:
            return "Error: Invalid JSON format in codebook file"
        except Exception as e:
            return f"Error uploading codebook: {str(e)}"

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
            
            self.df = pd.DataFrame()
            self.df['ID'] = range(1, len(temp_df) + 1)
            self.df['text'] = temp_df[column_name]
            self.df['is_reviewed'] = False
            self.selected_column = 'text'
            self.current_index = 0
            
            #initial_text = self.df.iloc[0]['text']
            status_msg = "Loaded df. This will be backed up as: "+str(self.backup_path)
            self.backup_df() 
            return status_msg, self.df#, initial_text, codes, codes
            
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

    def get_code_values(self, code_name):
        if not self.codebook_path.exists() or not code_name:
            return []
        try:
            with open(self.codebook_path, 'r') as f:
                codebook = json.load(f)
                for code in codebook['codes']:
                    if code['attribute'] == code_name:
                        return [v['category'] for v in code['categories']]
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
            with open(self.codebook_path, 'r') as f:
                codebook = json.load(f)
                code_exists = any(code['attribute'] == code_name for code in codebook['codes'])
            if not code_exists:
                return f"Code {code_name} not found in codebook", None
            
            self.df.at[self.current_index, code_name] = value
            self.df.at[self.current_index, 'is_reviewed'] = True
            self.backup_df()
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
        """Saves the current data to an excel file for download"""
        if self.df is None:
            return None, "No data to save"
        try:
            output_path = self.upload_dir / "annotated_transcripts.xlsx"
            self.df.to_excel(output_path, index=False)
            return str(output_path), "File saved successfully"
        except Exception as e:
            return None, f"Error saving file: {str(e)}"
        
    def save_codebook(self):
        """Saves the current codebook to a file for download"""
        if not self.codebook_path.exists():
            return None, "No codebook to save"
        
        try:
            # Get current date and time
            import datetime
            now = datetime.datetime.now()

            # Format datetime as string
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

            # Append timestamp to filename
            output_path = self.upload_dir / f"codebook_{timestamp}.json"

            # Copy the current codebook to the output path
            shutil.copy2(self.codebook_path, output_path)
            return str(output_path), "Codebook saved successfully"
        except Exception as e:
            return None, f"Error saving codebook: {str(e)}"
        
    def get_sortable_columns(self):
        if self.df is not None:
            return self.df.columns.tolist()
        return []
    
    def backup_df(self):
        """Automatically backup the DataFrame to Excel"""
        if self.df is not None:
            try:
                self.df.to_excel(self.backup_path, index=False)
            except Exception as e:
                print(f"Error backing up DataFrame: {e}")