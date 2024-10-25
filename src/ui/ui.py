import gradio as gr
import pandas as pd
from annotator import TranscriptionAnnotator

def process_df_for_display(df):
    if df is None:
        return None
    try:
        # Convert the input to a pandas DataFrame
        if isinstance(df, pd.DataFrame):
            df_display = df.copy()
        else:
            # If it's a Gradio DataFrame or other format, convert to pandas DataFrame
            df_display = pd.DataFrame(df.value if hasattr(df, 'value') else df)
            
        # Constrain text column width by truncating long text
        for column in df_display.columns:
            if df_display[column].dtype == 'object':  # Check if column contains text
                df_display[column] = df_display[column].astype(str).apply(
                    lambda x: x[:500] + '...' if len(x) > 500 else x
                )
        
        return df_display
    except Exception as e:
        print(f"Error processing DataFrame: {e}")
        return None

def create_ui():
    annotator = TranscriptionAnnotator()
    
    with gr.Blocks() as demo:
        gr.Markdown("## 📝 Fannotate")
        with gr.Tabs():
            # Upload Tab
            with gr.Tab("Upload Data"):
                file_upload = gr.File(label="Upload Excel File")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
            
            # Settings Tab
            with gr.Tab("Settings"):
                with gr.Row():
                    sheet_select = gr.Dropdown(label="Select Sheet", choices=[], interactive=True, allow_custom_value=True)
                    column_select = gr.Dropdown(label="Select Column", choices=[], interactive=True, allow_custom_value=True)
                
                load_settings_btn = gr.Button("Apply and create codebook")
                settings_status = gr.Textbox(label="Settings Status", interactive=False)
                preview_df = gr.DataFrame(interactive=False, visible=False)
            
            # Codebook Tab
            with gr.Tab("Codebook"):
                with gr.Row():
                    code_name = gr.Textbox(label="Code Name")
                    code_values = gr.Textbox(label="Code Values (comma-separated)")
                code_description = gr.TextArea(label="Code Description")
                
                with gr.Row():
                    add_code_btn = gr.Button("Add Code")
                    reload_codebook_btn_1 = gr.Button("Reload Codebook")
                
                with gr.Row():
                    delete_code_select = gr.Dropdown(label="Select Code to Delete", choices=[], interactive=True, allow_custom_value=True)
                    delete_code_btn = gr.Button("Delete Code")
                
                code_status = gr.Textbox(label="Status", interactive=False)
                codes_display = gr.JSON(label="Current Codes")

            # LLM Auto-fill Tab
            with gr.Tab("LLM Auto-fill"):
                with gr.Row():
                    llm_code_select = gr.Dropdown(
                        label="Select Category to Auto-fill", 
                        choices=[], 
                        interactive=True, 
                        allow_custom_value=True
                    )
                    llm_reload_btn = gr.Button("Reload Categories")
                
                llm_instruction = gr.TextArea(
                    label="Instruction for LLM",
                    placeholder="Enter instructions for the LLM to follow when auto-filling annotations...",
                    interactive=True
                )
    
                auto_fill_btn = gr.Button("Auto-fill Selected Category")
            
            # Annotation Editor Tab
            with gr.Tab("Annotation Editor"):
                with gr.Row():
                    prev_btn = gr.Button("Previous")
                    next_btn = gr.Button("Next")
                    current_index = gr.Number(value=0, label="Current Index", interactive=False)
                
                # Annotation controls
                with gr.Row():
                    code_select = gr.Dropdown(label="Select Category", choices=[], interactive=True, allow_custom_value=True)
                    value_select = gr.Dropdown(label="Select Value", choices=[], interactive=True, allow_custom_value=True)
                    reload_codebook_btn_2 = gr.Button("Reload Codebook")
                
                annotate_next_btn = gr.Button("Annotate and continue to next")
                annotation_status = gr.Textbox(label="Annotation Status", interactive=False)
                
                # Text content moved to bottom
                transcript_box = gr.TextArea(label="Text Content", interactive=False)
            
            # Review Tab
            with gr.Tab("Review"):
                # Custom CSS for the table
                gr.HTML("""
                    <style>
                    .review-table-container .table-cell {
                        max-width: 500px;
                        white-space: normal;
                        word-wrap: break-word;
                        overflow-wrap: break-word;
                    }
                    </style>
                """)
                
                review_df = gr.DataFrame(
                    label="Annotation Overview",
                    interactive=False,
                    value=lambda: process_df_for_display(annotator.df),
                    elem_classes=["review-table-container"]
                )
                
                refresh_review_btn = gr.Button("Refresh Overview")
            
            # Download Tab
            with gr.Tab("Download"):
                download_btn = gr.Button("Download Annotated File")
                download_output = gr.File(label="Download")
                download_status = gr.Textbox(label="Status", interactive=False)
        
        # Event handlers
        def upload_and_get_sheets(file):
            status, sheets, _ = annotator.upload_file(file)
            return status, gr.Dropdown(choices=sheets, allow_custom_value=True)
        
        file_upload.change(
            fn=upload_and_get_sheets,
            inputs=[file_upload],
            outputs=[upload_status, sheet_select]
        )
        
        def update_columns(sheet):
            columns = annotator.get_columns(sheet)
            return gr.Dropdown(choices=columns, allow_custom_value=True)
        
        sheet_select.change(
            fn=update_columns,
            inputs=[sheet_select],
            outputs=[column_select]
        )
        
        load_settings_btn.click(
            fn=annotator.load_settings,
            inputs=[sheet_select, column_select],
            outputs=[settings_status, preview_df, transcript_box, code_select, delete_code_select]
        )
        
        llm_reload_btn.click(
            fn=lambda: [code["name"] for code in annotator.load_codebook()],
            outputs=[llm_code_select]
        )

        def update_value_choices(code_name):
            if not code_name:
                return gr.Dropdown(choices=[])
            values = annotator.get_code_values(code_name)
            return gr.Dropdown(choices=values, value=None, allow_custom_value=True)
        
        code_select.change(
            fn=update_value_choices,
            inputs=[code_select],
            outputs=[value_select]
        )
        
        add_code_btn.click(
            fn=annotator.add_code,
            inputs=[code_name, code_values, code_description],
            outputs=[code_status, codes_display, code_select, delete_code_select]
        )
        
        delete_code_btn.click(
            fn=annotator.delete_code,
            inputs=[delete_code_select],
            outputs=[code_status, codes_display, code_select, delete_code_select]
        )

        refresh_review_btn.click(
            fn=lambda: process_df_for_display(annotator.df),
            outputs=[review_df]
        )
        
        
        def refresh_codebook_display():
            try:
                codes = annotator.load_codebook()
                return codes
            except Exception as e:
                print(f"Error refreshing codebook display: {e}")
                return []

        def refresh_annotation_dropdowns():
            try:
                categories = [code["name"] for code in annotator.load_codebook()]
                return (
                    gr.Dropdown(choices=categories, value=None, allow_custom_value=True),
                    gr.Dropdown(choices=[], value=None, allow_custom_value=True)
                )
            except Exception as e:
                print(f"Error refreshing dropdowns: {e}")
                return gr.Dropdown(), gr.Dropdown()
        
        # Connect reload buttons to their specific functions
        reload_codebook_btn_1.click(
            fn=refresh_codebook_display,
            outputs=[codes_display]
        )

        reload_codebook_btn_2.click(
            fn=refresh_annotation_dropdowns,
            outputs=[code_select, value_select]
        )
        
        def annotate_and_next(code_name, value):
            try:
                # First save the annotation
                status, _ = annotator.save_annotation(code_name, value)
                # Then navigate to next
                text, idx, _ = annotator.navigate_transcripts("next")
                return status, text, idx
            except Exception as e:
                print(f"Error in annotate_and_next: {e}")
                return "Error during annotation", "", 0
        
        annotate_next_btn.click(
            fn=annotate_and_next,
            inputs=[code_select, value_select],
            outputs=[annotation_status, transcript_box, current_index]
        )
        
        def navigate_and_update(direction):
            text, idx, df = annotator.navigate_transcripts(direction)
            if df is not None:
                df_display = process_df_for_display(df)
            else:
                df_display = None
            return text, idx, df_display
        
        prev_btn.click(
            fn=lambda: navigate_and_update("prev"),
            outputs=[transcript_box, current_index]
        )
        
        next_btn.click(
            fn=lambda: navigate_and_update("next"),
            outputs=[transcript_box, current_index]
        )
        
        download_btn.click(
            fn=annotator.save_excel,
            outputs=[download_output, download_status]
        )
    
    return demo