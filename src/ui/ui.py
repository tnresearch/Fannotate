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
        if isinstance(df, pd.DataFrame):
            df_display = df.copy()
        else:
            df_display = pd.DataFrame(df.value if hasattr(df, 'category') else df)
        
        if 'text' in df_display.columns:
            df_display['text'] = df_display['text'].astype(str).apply(
                lambda x: x[:25] + '...' if len(x) > 25 else x)
        
        for column in df_display.columns:
            if column != 'text' and df_display[column].dtype == 'object':
                df_display[column] = df_display[column].astype(str).apply(
                    lambda x: x[:500] + '...' if len(x) > 500 else x)
        
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
        clean_name = clean_column_name(code_name)
        for code in codebook:
            if clean_column_name(code['attribute']) == clean_name:
                selected_code = code
                break
        if not selected_code:
            return f"Selected category '{code_name}' not found in codebook"
        prompt = "Please classify the text within one of the following categories:\n\n"
        prompt += json.dumps(selected_code, indent=2)
        prompt += "\n\nText: "
        return prompt
    except Exception as e:
        print(f"Error generating prompt: {e}")
        return f"Error generating prompt: {str(e)}"

def clean_column_name(name):
    if isinstance(name, list):
        name = "".join(name)
    return name.strip().replace('[','').replace(']','').replace("'", '').replace(" ", '_')




# def autofill_from_codebook(code_name, instruction):
#     if not code_name or not instruction:
#         return "Please select a category and generate a prompt first"
#     try:
#         clean_name = clean_column_name(code_name)
#         output_column = f"autofill_{clean_name}"
#         results = []
#         for idx, row in annotator.df.iterrows():
#             full_prompt = instruction + str(row['text'])
#             response = query_llm(full_prompt)
#             results.append(response)
#         annotator.df[output_column] = results
#         return f"Auto-fill completed. Results stored in column: {output_column}"
#     except Exception as e:
#         print(f"Error in auto-fill process: {e}")
#         return f"Error during auto-fill: {str(e)}"

def get_category_values(code_name):
    """Get the values for a specific category from the codebook"""
    try:
        codebook = annotator.load_codebook()
        for code in codebook:
            if code['attribute'] == code_name:
                return [v['category'] for v in code['categories']]
        return []
    except Exception as e:
        print(f"Error getting category values: {e}")
        return []

def autofill_from_codebook(code_name, instruction):
    if not code_name or not instruction:
        return "Please select a category and generate a prompt first"
    
    try:
        # Get the valid values for the selected category
        valid_values = get_category_values(code_name)
        if not valid_values:
            return "No valid values found for the selected category"
            
        clean_name = clean_column_name(code_name)
        output_column = f"autofill_{clean_name}"
        results = []
        
        # Use batch_process_transcripts instead of individual queries
        df, status = batch_process_transcripts(
            annotator.df,
            instruction,
            'text',  # assuming this is the column with the text to process
            output_column,
            valid_values
        )
        
        if df is not None:
            annotator.df = df
            return f"Auto-fill completed. Results stored in column: {output_column}"
        else:
            return f"Error during auto-fill: {status}"
            
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
            values)
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
        categories = [code["attribute"] for code in annotator.load_codebook()]
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
        status, df = annotator.save_annotation(code_name, value)
        if not status.startswith("Saved"):
            return status, None, None, None
        text, idx = annotator.navigate_transcripts("next")
        review_status_text = "✅" if annotator.df.iloc[idx]['is_reviewed'] else "❌"
        return status, text, idx, review_status_text
    except Exception as e:
        print(f"Error in annotate_and_next: {e}")
        return "Error during annotation", None, None, "❌"

def custom_batch_process(instruction, values, output_column):
    if not output_column:
        return "Please specify an output column name"
    try:
        df, status = batch_process_transcripts(
            annotator.df,
            instruction,
            annotator.selected_column,
            output_column,
            values)
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
        review_status_text = "✅" if annotator.df.iloc[idx]['is_reviewed'] else "❌"
        return text, idx, review_status_text
    except Exception as e:
        print(f"Error in navigate_and_update: {e}")
        return None, None, "❌"

def apply_settings(sheet, column, url, api_key, model):
    from lm import update_llm_config
    
    # Update LLM configuration
    update_llm_config(url, api_key, model)
    
    # Load settings and get initial data
    status, preview, transcript, codes1, codes2 = annotator.load_settings(sheet, column)
    
    # Get the current codebook and extract codes
    current_codebook = annotator.load_codebook()
    codes = [code["attribute"] for code in current_codebook]
    
    # Get initial review status
    initial_review_status = "✅" if annotator.df.iloc[0]['is_reviewed'] else "❌"
    
    return (
        f"Settings applied successfully. LLM endpoint: {url}, Model: {model}\n{status}",
        process_df_for_display(preview),  # Make sure DataFrame is properly formatted
        transcript,
        gr.Dropdown(choices=codes),  # Update code_select dropdown
        gr.Dropdown(choices=codes),  # Update value_select dropdown
        initial_review_status,
        gr.Dropdown(choices=codes),  # Update llm_code_select dropdown
        current_codebook  # Update codes_display
    )

def handle_new_codebook():
    """Handles the creation of a new empty codebook"""
    try:
        status = annotator.create_new_codebook()
        current_codebook = annotator.load_codebook()
        codes = [code["attribute"] for code in current_codebook]
        return (
            status,
            current_codebook,
            gr.Dropdown(choices=codes),
            gr.Dropdown(choices=codes),
            gr.Dropdown(choices=codes)
        )
    except Exception as e:
        return (
            f"Error creating new codebook: {str(e)}",
            [],
            gr.Dropdown(choices=[]),
            gr.Dropdown(choices=[]),
            gr.Dropdown(choices=[])
        )

def handle_codebook_upload(file, codebook_file):
    codebook_status = ""
    current_codebook = []
    codes = []
    
    if codebook_file:
        codebook_status = annotator.upload_codebook(codebook_file)
        current_codebook = annotator.load_codebook()
        codes = [code["attribute"] for code in current_codebook]
    
    status, sheets, _ = annotator.upload_file(file, codebook_file)
    
    return (
        f"{status}\n{codebook_status}",
        gr.Dropdown(choices=sheets),
        current_codebook,
        gr.Dropdown(choices=codes),
        gr.Dropdown(choices=codes),
        gr.Dropdown(choices=codes)
    )

def reload_llm_categories():
    """Helper function to reload categories for LLM tab"""
    try:
        codes = [code["attribute"] for code in annotator.load_codebook()]
        return gr.Dropdown(choices=codes, value=None, allow_custom_value=True)
    except Exception as e:
        print(f"Error reloading LLM categories: {e}")
        return gr.Dropdown(choices=[], value=None, allow_custom_value=True)


"""
#############################
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
                    new_codebook_btn = gr.Button("New Codebook")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)

            # Settings Tab
            with gr.Tab("⚙️ Settings"):
                with gr.Row():
                    sheet_select = gr.Dropdown(label="Select Sheet", choices=[], interactive=True)
                    column_select = gr.Dropdown(label="Select Column", choices=[], interactive=True)
                with gr.Row():
                    gr.Markdown("### LLM Settings")
                with gr.Row():
                    llm_url = gr.Textbox(label="LLM Endpoint URL", value="http://192.168.50.155:8000/v1/", placeholder="Enter LLM endpoint URL")
                    llm_api_key = gr.Textbox(label="API Key", value="token-abc123", placeholder="Enter API key")
                    llm_model = gr.Textbox(label="Model Name", value="google/gemma-2-2b-it", placeholder="Enter model name")
                with gr.Row():
                    load_settings_btn = gr.Button("Apply Settings")
                    settings_status = gr.Textbox(label="Settings Status", interactive=False)
                preview_df = gr.DataFrame(interactive=False, visible=False)

            # Simplified Codebook Tab
            with gr.Tab("📓 Codebook"):
                codes_display = gr.JSON(label="Current Codebook")

            # LLM Auto-fill Tab
            with gr.Tab("🤖 Auto-fill"):
                with gr.Row():
                    llm_code_select = gr.Dropdown(label="Select Category to Auto-fill", choices=[], interactive=True, allow_custom_value=True)
                    llm_reload_btn = gr.Button("Reload Categories")
                with gr.Row():
                    generate_prompt_btn = gr.Button("Generate Prompt")
                    auto_fill_btn = gr.Button("Auto-fill from Codebook")
                with gr.Row():
                    llm_instruction = gr.TextArea(label="Codebook instruction for LLM", placeholder="Full prompt.", interactive=True)
                    progress_bar = gr.Textbox(label="Progress", interactive=False)

            # Custom Tab
            with gr.Tab("🤖 Custom"):
                with gr.Row():
                    custom_output_column = gr.Textbox(label="Output Column Name", placeholder="Enter name for the new column", interactive=True)
                    custom_instruction = gr.TextArea(label="Instruction for LLM", placeholder="Enter instructions for the LLM to follow when auto-filling annotations...", interactive=True)
                with gr.Row():
                    custom_values = gr.TextArea(label="Valid labels", placeholder="Enter values", interactive=True)
                    custom_process_btn = gr.Button("Process with Custom Instructions")
                    custom_progress = gr.Textbox(label="Progress", interactive=False)
                custom_output = gr.TextArea(label="LLM Response", interactive=False)

            # Annotation Editor Tab
            with gr.Tab("✏️ Annotation review"):
                with gr.Row():
                    prev_btn = gr.Button("Previous")
                    next_btn = gr.Button("Next")
                    current_index = gr.Number(value=0, label="Current Index", interactive=False)
                    review_status = gr.Textbox(label="Review Status", interactive=False)
                with gr.Row():
                    code_select = gr.Dropdown(label="Select Category", choices=[], interactive=True, allow_custom_value=True)
                    value_select = gr.Dropdown(label="Select Value", choices=[], interactive=True, allow_custom_value=True)
                    reload_codebook_btn_2 = gr.Button("Reload Codebook")
                    annotate_next_btn = gr.Button("Annotate and continue to next")
                    annotation_status = gr.Textbox(label="Annotation Status", interactive=False)
                transcript_box = gr.TextArea(label="Text Content", interactive=False)

            # Stats Tab
            with gr.Tab("📊 Status"):
                gr.Markdown("Status information will be displayed here")

            # Download Tab
            with gr.Tab("💾 Download"):
                download_btn = gr.Button("Download Annotated File")
                download_output = gr.File(label="Download")
                download_status = gr.Textbox(label="Status", interactive=False)

        # Event handlers
        file_upload.change(
            fn=handle_codebook_upload,
            inputs=[file_upload, codebook_upload],
            outputs=[
                upload_status, 
                sheet_select, 
                codes_display,
                code_select,
                value_select,
                llm_code_select
            ]
        )

        codebook_upload.change(
            fn=handle_codebook_upload,
            inputs=[file_upload, codebook_upload],
            outputs=[
                upload_status, 
                sheet_select, 
                codes_display,
                code_select,
                value_select,
                llm_code_select
            ]
        )

        new_codebook_btn.click(
            fn=handle_new_codebook,
            outputs=[
                upload_status,
                codes_display,
                code_select,
                value_select,
                llm_code_select
            ]
        )

        sheet_select.change(
            fn=lambda x: gr.Dropdown(choices=annotator.get_columns(x)),
            inputs=[sheet_select],
            outputs=[column_select]
        )

        load_settings_btn.click(
            fn=apply_settings,
            inputs=[sheet_select, column_select, llm_url, llm_api_key, llm_model],
            outputs=[settings_status, preview_df, transcript_box, code_select, value_select, review_status, llm_code_select, codes_display]
        )

        code_select.change(
            fn=update_value_choices,
            inputs=[code_select],
            outputs=[value_select]
        )

        llm_reload_btn.click(
            fn=reload_llm_categories,
            outputs=[llm_code_select]
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

        custom_process_btn.click(
            fn=custom_batch_process,
            inputs=[custom_instruction, custom_values, custom_output_column],
            outputs=[custom_progress]
        )

        download_btn.click(
            fn=annotator.save_excel,
            outputs=[download_output, download_status]
        )

        return demo