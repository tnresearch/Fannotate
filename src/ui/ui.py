import gradio as gr
import pandas as pd
import json
from annotator import TranscriptionAnnotator
from lm import query_llm, query_constrained_llm, batch_process_transcripts

# annotation object
annotator = TranscriptionAnnotator()


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
        
        # Special handling for 'text' column - truncate to 25 characters
        if 'text' in df_display.columns:
            df_display['text'] = df_display['text'].astype(str).apply(
                lambda x: x[:25] + '...' if len(x) > 25 else x
            )
        
        # For other text columns, keep the 500 character limit
        for column in df_display.columns:
            if column != 'text' and df_display[column].dtype == 'object':
                df_display[column] = df_display[column].astype(str).apply(
                    lambda x: x[:500] + '...' if len(x) > 500 else x
                )
        
        return df_display
    
    except Exception as e:
        print(f"Error processing DataFrame: {e}")
        return None
    

def generate_prompt(code_name):
    if not code_name:
        return "Please select a category first"
    try:
        codebook = annotator.load_codebook()
        selected_code = None
        
        # Clean the code name for comparison
        clean_name = clean_column_name(code_name)
        
        # Find the selected code in the codebook
        for code in codebook:
            if clean_column_name(code['name']) == clean_name:
                selected_code = code
                break
                
        if not selected_code:
            return f"Selected category '{code_name}' not found in codebook"
            
        # Generate the prompt
        prompt = "Please classify the text within one of the following categories:\n\n"
        prompt += json.dumps(selected_code, indent=2)
        prompt += "\n\nText: "
        
        return prompt
        
    except Exception as e:
        print(f"Error generating prompt: {e}")
        return f"Error generating prompt: {str(e)}"


def clean_column_name(name):
    # If name is a list, join it into a string
    if isinstance(name, list):
        name = "".join(name)
    # Return the cleaned string
    return name.strip().replace('[','').replace(']','').replace("'", '').replace(" ", '_')

def autofill_from_codebook(code_name, instruction):
    if not code_name or not instruction:
        return "Please select a category and generate a prompt first"
    try:
        # Clean the category name and create the output column name
        clean_name = clean_column_name(code_name)
        output_column = f"autofill_{clean_name}"
        
        # Process each row
        results = []
        for idx, row in annotator.df.iterrows():
            full_prompt = instruction + str(row['text'])
            response = query_llm(full_prompt)
            results.append(response)
            
        # Add results to the DataFrame
        annotator.df[output_column] = results
        return f"Auto-fill completed. Results stored in column: {output_column}"
    except Exception as e:
        print(f"Error in auto-fill process: {e}")
        return f"Error during auto-fill: {str(e)}"

def process_with_llm(instruction, values, output_column):
    if not output_column:
        return "Please specify an output column name"
    try:
        df, status = batch_process_transcripts(
            annotator.df,
            instruction,
            annotator.selected_column,
            output_column,
            values
        )
        if df is not None:
            annotator.df = df
        return status
    except Exception as e:
        return f"Error: {str(e)}"

def update_value_choices(code_name):
    if not code_name:
        return gr.Dropdown(choices=[])
    values = annotator.get_code_values(code_name)
    return gr.Dropdown(choices=values, value=None, allow_custom_value=True)



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

def annotate_and_next(code_name, value):
    try:
        if not code_name or not value:
            return "Please select both category and value", None, None, None
        
        # First save the annotation
        status, df = annotator.save_annotation(code_name, value)
        if not status.startswith("Saved"):
            return status, None, None, None
            
        # Then navigate to next
        text, idx = annotator.navigate_transcripts("next")
        
        # Get review status for new index
        review_status_text = "✅" if annotator.df.iloc[idx]['is_reviewed'] else "❌"
        
        # Update dropdowns for next annotation
        return (
            status,  # annotation status
            text,    # transcript text
            idx,     # current index
            review_status_text  # review status
        )
    except Exception as e:
        print(f"Error in annotate_and_next: {e}")
        return "Error during annotation", None, None, "❌"


# New function for custom tab
def custom_batch_process(instruction, values, output_column):
    if not output_column:
        return "Please specify an output column name"
    try:
        df, status = batch_process_transcripts(
            annotator.df,
            instruction,
            annotator.selected_column,
            output_column,
            values
        )
        if df is not None:
            annotator.df = df
        return status
    except Exception as e:
        return f"Error: {str(e)}"


def navigate_and_update(direction):
    try:
        text, idx = annotator.navigate_transcripts(direction)
        if text is None or idx is None:
            return None, None, "❌"
            
        # Get review status for current index
        review_status_text = "✅" if annotator.df.iloc[idx]['is_reviewed'] else "❌"
        return text, idx, review_status_text
    except Exception as e:
        print(f"Error in navigate_and_update: {e}")
        return None, None, "❌"


def apply_settings(sheet, column, url, api_key, model):
    # Update LLM settings
    from lm import update_llm_config
    update_llm_config(url, api_key, model)
    
    # Apply other settings
    status, preview, transcript, codes1, codes2 = annotator.load_settings(sheet, column)
    
    # Get initial review status
    initial_review_status = "✅" if annotator.df.iloc[0]['is_reviewed'] else "❌"
    
    # Get current codebook for display
    current_codebook = annotator.load_codebook()
    
    return (
        f"Settings applied successfully. LLM endpoint: {url}, Model: {model}\n{status}",
        preview,
        transcript,
        codes1,
        codes2,
        initial_review_status,
        codes1,  # for llm_code_select
        current_codebook  # Add this output for codebook display
    )



"""
##############
"""

def create_ui():
    
    with gr.Blocks() as demo:
        gr.Markdown("## 📝 Fannotate")
        with gr.Tabs():
            # Upload Tab
            with gr.Tab("📁 Upload Data"):
                with gr.Row():
                    file_upload = gr.File(label="Upload Excel File")
                    codebook_upload = gr.File(label="Upload Codebook (Optional)")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
            
            # Settings Tab 
            with gr.Tab("⚙️ Settings"):
                with gr.Row():
                    sheet_select = gr.Dropdown(label="Select Sheet", choices=[], interactive=True)
                    column_select = gr.Dropdown(label="Select Column", choices=[], interactive=True)
                
                with gr.Row():
                    gr.Markdown("### LLM Settings")
                
                with gr.Row():
                    llm_url = gr.Textbox(
                        label="LLM Endpoint URL",
                        value="http://192.168.50.155:8000/v1/",
                        placeholder="Enter LLM endpoint URL"
                    )
                    llm_api_key = gr.Textbox(
                        label="API Key",
                        value="token-abc123",
                        placeholder="Enter API key"
                    )
                    llm_model = gr.Textbox(
                        label="Model Name",
                        value="google/gemma-2-2b-it",
                        placeholder="Enter model name"
                    )
                
                with gr.Row():
                    load_settings_btn = gr.Button("Apply Settings")
                    settings_status = gr.Textbox(label="Settings Status", interactive=False)
                
                preview_df = gr.DataFrame(interactive=False, visible=False)
            
            # Codebook Tab
            with gr.Tab("📓 Codebook"):
                with gr.Row():
                    with gr.Column(scale=1):
                        code_name = gr.Textbox(label="Code Name")
                        code_description = gr.TextArea(label="Code Description")
                        
                        with gr.Row():
                            add_code_btn = gr.Button("Add Code")
                            reload_codebook_btn_1 = gr.Button("Reload Codebook")
                        
                        gr.Markdown("---")
                        
                        with gr.Row():
                            delete_code_select = gr.Dropdown(
                                label="Select Code to Delete",
                                choices=[],
                                interactive=True,
                                allow_custom_value=True
                            )
                            delete_code_btn = gr.Button("Delete Code")
                        
                        code_status = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Column(scale=1):
                        edit_code_select = gr.Dropdown(
                            label="Select Code to Edit Values",
                            choices=[],
                            interactive=True,
                            allow_custom_value=True
                        )
                        value_name = gr.Textbox(label="Value")
                        value_description = gr.TextArea(label="Value Description")
                        
                        with gr.Row():
                            add_value_btn = gr.Button("Add Value")
                            delete_value_btn = gr.Button("Delete Value")
                        
                        value_select = gr.Dropdown(
                            label="Select Value to Edit/Delete",
                            choices=[],
                            interactive=True,
                            allow_custom_value=True
                        )
                
                codes_display = gr.JSON(label="Current Codes")

            # LLM Auto-fill Tab
            with gr.Tab("🤖 Auto-fill"):
                with gr.Row():
                    llm_code_select = gr.Dropdown(
                        label="Select Category to Auto-fill",
                        choices=[],
                        interactive=True,
                        allow_custom_value=True
                    )
                    llm_reload_btn = gr.Button("Reload Categories")
                
                with gr.Row():
                    generate_prompt_btn = gr.Button("Generate Prompt")
                    auto_fill_btn = gr.Button("Auto-fill from Codebook")
                
                with gr.Row():
                    llm_instruction = gr.TextArea(
                        label="Codebook instruction for LLM",
                        placeholder="Full prompt.",
                        interactive=True
                    )
                    progress_bar = gr.Textbox(
                        label="Progress",
                        interactive=False
                    )
                
            # LLM Auto-fill Tab
            with gr.Tab("🤖 Custom"):
                with gr.Row():
                    custom_output_column = gr.Textbox(
                        label="Output Column Name",
                        placeholder="Enter name for the new column",
                        interactive=True
                    )
                    custom_instruction = gr.TextArea(
                        label="Instruction for LLM",
                        placeholder="Enter instructions for the LLM to follow when auto-filling annotations...",
                        interactive=True
                    )
                with gr.Row():
                    custom_values = gr.TextArea(
                        label="Valid labels",
                        placeholder="Enter values",
                        interactive=True
                    )
                    custom_process_btn = gr.Button("Process with Custom Instructions")
                    custom_progress = gr.Textbox(
                        label="Progress",
                        interactive=False
                    )
                    custom_output = gr.TextArea(
                        label="LLM Response",
                        interactive=False
                    )
            
            # Annotation Editor Tab
            with gr.Tab("✏️ Annotation review"):
                with gr.Row():
                    prev_btn = gr.Button("Previous")
                    next_btn = gr.Button("Next")
                    current_index = gr.Number(value=0, label="Current Index", interactive=False)
                    review_status = gr.Textbox(label="Review Status", interactive=False)
                
                # Annotation controls
                with gr.Row():
                    code_select = gr.Dropdown(label="Select Category", choices=[], interactive=True, allow_custom_value=True)
                    value_select = gr.Dropdown(label="Select Value", choices=[], interactive=True, allow_custom_value=True)
                    reload_codebook_btn_2 = gr.Button("Reload Codebook")
                
                annotate_next_btn = gr.Button("Annotate and continue to next")
                annotation_status = gr.Textbox(label="Annotation Status", interactive=False)
                
                # Text content moved to bottom
                transcript_box = gr.TextArea(label="Text Content", interactive=False)     

            
            # Stats tab showing the performance overview
            with gr.Tab("📊 Status"): #🔍
                gr.Markdown("Status information will be displayed here")
     
            
            # Download Tab
            with gr.Tab("💾 Download"):
                download_btn = gr.Button("Download Annotated File")
                download_output = gr.File(label="Download")
                download_status = gr.Textbox(label="Status", interactive=False)
        
        # refresh
        def refresh_all_code_dropdowns():
            try:
                codes = [code["name"] for code in annotator.load_codebook()]
                return (
                    gr.Dropdown(choices=codes),  # for edit_code_select
                    gr.Dropdown(choices=codes),  # for delete_code_select
                    gr.Dropdown(choices=codes)   # for code_select
                )
            except Exception as e:
                print(f"Error refreshing code dropdowns: {e}")
                return (
                    gr.Dropdown(choices=[]),
                    gr.Dropdown(choices=[]),
                    gr.Dropdown(choices=[])
                )


        # Event handlers
        #         
        def update_columns(sheet):
            columns = annotator.get_columns(sheet)
            return gr.Dropdown(choices=columns, allow_custom_value=True)
        
        # def handle_codebook_upload(file, codebook_file):
        #     codebook_status = ""
        #     if codebook_file:
        #         codebook_status = annotator.upload_codebook(codebook_file)
        #         current_codebook = annotator.load_codebook()
        #     else:
        #         current_codebook = []
            
        #     status, sheets = annotator.upload_file(file, codebook_file)
        #     return (
        #         f"{status}\n{codebook_status}", 
        #         gr.Dropdown(choices=sheets),
        #         current_codebook
        #     )

        def handle_codebook_upload(file, codebook_file):
            codebook_status = ""
            if codebook_file:
                codebook_status = annotator.upload_codebook(codebook_file)
                current_codebook = annotator.load_codebook()
            else:
                current_codebook = []
            
            # The upload_file method returns 3 values, but we only need the first two
            status, sheets, _ = annotator.upload_file(file, codebook_file)
            return (
                f"{status}\n{codebook_status}", 
                gr.Dropdown(choices=sheets),
                current_codebook
            )

        # Update the file upload event handler
        # file_upload.change(
        #     fn=handle_codebook_upload,
        #     inputs=[file_upload, codebook_upload],
        #     outputs=[upload_status, sheet_select, codes_display]
        # )
        file_upload.change(
            fn=handle_codebook_upload,
            inputs=[file_upload, codebook_upload],
            outputs=[upload_status, sheet_select, codes_display]
        )
                
        sheet_select.change(
            fn=update_columns,
            inputs=[sheet_select],
            outputs=[column_select]
        )
        
        load_settings_btn.click(
            fn=apply_settings,
            inputs=[
                sheet_select,
                column_select,
                llm_url,
                llm_api_key,
                llm_model
            ],
            outputs=[
                settings_status,
                preview_df,
                transcript_box,
                code_select,
                delete_code_select,
                review_status,
                llm_code_select,
                codes_display  # Add this output
            ]
        )

        # Event handler for custom tab
        custom_process_btn.click(
            fn=custom_batch_process,
            inputs=[custom_instruction, custom_values, custom_output_column],
            outputs=[custom_progress]
        )

        llm_reload_btn.click(
            fn=lambda: [code["name"] for code in annotator.load_codebook()],
            outputs=[llm_code_select]
        )
        
        code_select.change(
            fn=update_value_choices,
            inputs=[code_select],
            outputs=[value_select]
        )
        
        

        # add_code button click handler
        add_code_btn.click(
            fn=annotator.add_code,
            inputs=[code_name, code_description],
            outputs=[code_status, codes_display]
        ).then(
            fn=refresh_all_code_dropdowns,
            outputs=[edit_code_select, delete_code_select, code_select]
        )

        # add_value button click handler
        add_value_btn.click(
            fn=annotator.add_value_to_code,
            inputs=[edit_code_select, value_name, value_description],
            outputs=[code_status, codes_display]
        ).then(
            fn=update_value_choices,  # Use the same function as above
            inputs=[edit_code_select],
            outputs=[value_select]
        )
        
        delete_code_btn.click(
            fn=annotator.delete_code,
            inputs=[delete_code_select],
            outputs=[code_status, codes_display, code_select, delete_code_select]
        )

        edit_code_select.change(
            fn=update_value_choices,
            inputs=[edit_code_select],
            outputs=[value_select]
        )


        delete_value_btn.click(
            fn=annotator.delete_value_from_code,
            inputs=[edit_code_select, value_select],
            outputs=[code_status, codes_display]
        )
        
        auto_fill_btn.click(
            fn=autofill_from_codebook,
            inputs=[llm_code_select, llm_instruction],
            outputs=[progress_bar]
        )

        generate_prompt_btn.click(
            fn=generate_prompt,
            inputs=[llm_code_select],
            outputs=[llm_instruction]
        )
        
        
        reload_codebook_btn_1.click(
            fn=refresh_all_code_dropdowns,
            outputs=[edit_code_select, delete_code_select, code_select]
        )

        reload_codebook_btn_2.click(
            fn=refresh_annotation_dropdowns,
            outputs=[code_select, value_select]
        )

         
        annotate_next_btn.click(
            fn=annotate_and_next,
            inputs=[code_select, value_select],
            outputs=[annotation_status, transcript_box, current_index, review_status]
        )

        
        prev_btn.click(
            fn=lambda: navigate_and_update("prev"),
            outputs=[transcript_box, current_index, review_status]
        )

        next_btn.click(
            fn=lambda: navigate_and_update("next"),
            outputs=[transcript_box, current_index, review_status]
        )
        
        download_btn.click(
            fn=annotator.save_excel,
            outputs=[download_output, download_status]
        )
    
    

    return demo