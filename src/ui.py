import gradio as gr
import pandas as pd
import json
from annotator import TranscriptionAnnotator
from lm import batch_process_transcripts

# annotation object
annotator = TranscriptionAnnotator()

#############################
# UI and event triggers
#############################

def create_ui():
    with gr.Blocks(theme=gr.themes.Default()) as demo:
    #with gr.Blocks(theme=gr.themes.Monochrome(spacing_size="sm", radius_size="none",primary_hue="blue",font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"])) as demo:
        gr.Markdown("## 📝 Fannotate")
        
        with gr.Tabs():
            # Upload Tab
            with gr.Tab("📜 Getting started"):
                try:
                    # with open("instructions.md", "r") as f:
                    #     instructions = f.read()
                    gr.Markdown("""## Overview
Fannotate is a tool for *faster* text annotation aided by LLMs. Central to Fannotate is the *codebook* (annotation guidelines) which help the LLM do an initial guess at the categories and labels. Fannotate lets you create any attributes to the text you'd like, which can then help you annotate the text faster, without having to read the whole transcript.""")
                    gr.Image("../bin/procedure.png", 
                            label=None,  # Removes the label
                            show_label=False,  # Ensures no label space is reserved
                            container=False,  # Removes the container box
                            show_download_button=False,  # Removes the download button
                            interactive=False,  # Prevents user interaction
                            height="auto",  # Adjusts height automatically
                            width="700px"#"100%"  # Takes full width of parent container
                            )
                    # gr.Markdown(instructions)
                    gr.Markdown("<br><br><a href='https://github.com/tnresearch/Fannotate/blob/main/userguide.md'>User guide</a>")
                except FileNotFoundError:
                    gr.Markdown("Instructions file not found. Please create instructions.md")
            with gr.Tab("📁 Upload Data"):
                with gr.Row():
                    gr.Markdown("## Upload data")
                
                with gr.Row():
                    gr.Markdown("Upload the dataset and codebook, or initialize a new codebook in the codebook tab.")
                
                with gr.Row():
                    with gr.Column():
                        file_upload = gr.File(label="Upload Excel File")
                    with gr.Column():
                        sheet_select = gr.Dropdown(label="Select Sheet", choices=[], interactive=True)
                        column_select = gr.Dropdown(label="Select Column", choices=[], interactive=True) 
                        upload_status = gr.Textbox(label="Upload Status", interactive=False)
                        load_data_btn = gr.Button("Create annotation table", variant="primary")#Load Data
                # with gr.Row():
                #     gr.Markdown("## Upload codebook")
                # with gr.Row():
                #     with gr.Column():
                #         codebook_upload = gr.File(label="Upload Codebook (Optional)")
                
                #with gr.Row():
                
                with gr.Row():
                    gr.Markdown("## Data preview")
                with gr.Row():
                    preview_df = gr.DataFrame(interactive=False, visible=True)

            # Settings Tab

            # with gr.Tab("⚙️ Settings"):
            #     with gr.Row():
            #         sheet_select = gr.Dropdown(label="Select Sheet", choices=[], interactive=True)
            #         column_select = gr.Dropdown(label="Select Column", choices=[], interactive=True)
            #     with gr.Row():
            #         gr.Markdown("### LLM Settings")
            #         gr.Markdown("Frigg is found at: ```http://172.16.16.48:8000/v1/```")
            #     with gr.Row():
            #         llm_url = gr.Textbox(label="LLM Endpoint URL", value="http://192.168.50.155:8000/v1/", placeholder="Enter LLM endpoint URL")
            #         llm_api_key = gr.Textbox(label="API Key", value="token-abc123", placeholder="Enter API key")
            #         llm_model = gr.Textbox(label="Model Name", value="google/gemma-2-2b-it", placeholder="Enter model name")
            #     with gr.Row():
            #         load_settings_btn = gr.Button("Apply Settings", variant="primary")
            #         settings_status = gr.Textbox(label="Settings Status", interactive=False)
            #     preview_df = gr.DataFrame(interactive=False, visible=False)
            
            with gr.Tab("⚙️ Settings"):
                with gr.Row():
                    gr.Markdown("### LLM Settings")
                    gr.Markdown("Frigg is found at: ```http://172.16.16.48:8000/v1/```")
                
                with gr.Row():
                    llm_url = gr.Textbox(label="LLM Endpoint URL", 
                                        value="http://192.168.50.155:8000/v1/")
                    llm_api_key = gr.Textbox(label="API Key", 
                                            value="token-abc123")
                    llm_model = gr.Textbox(label="Model Name", 
                                        value="google/gemma-2-2b-it")
                
                with gr.Row():
                    apply_llm_settings_btn = gr.Button("Apply LLM Settings", variant="primary")
                    settings_status = gr.Textbox(label="Settings Status", interactive=False)

            # Simplified Codebook Tab
            with gr.Tab("📓 Codebook"):
                with gr.Row():
                    gr.Markdown("## Annotation codebook")
                with gr.Row():
                    gr.Markdown("Upload existing codebook or initialize a new one.")
                with gr.Row():
                    codebook_upload = gr.File(label="Upload Codebook (Optional)")
                    new_codebook_btn = gr.Button("New Codebook")
                with gr.Row():
                    codes_display = gr.JSON(label="Current Codebook")
                # with gr.Row():
                #     gr.Markdown("Initialize a new, empty codebook.")
                # with gr.Row():

            # LLM Auto-fill Tab
            with gr.Tab("🤖 Auto-fill"):
                with gr.Row():
                    gr.Markdown("## Auto annotation")
                with gr.Row():
                    gr.Markdown("<span style='color: darkgrey'>Automated annotation of the text using the codebook.</span>")
                with gr.Row():
                    llm_code_select = gr.Dropdown(label="Select Category to Auto-fill", choices=[], interactive=True, allow_custom_value=True)
                    llm_reload_btn = gr.Button("Reload Categories")
                with gr.Row():
                    generate_prompt_btn = gr.Button("Generate Prompt")
                    auto_fill_btn = gr.Button("Auto-fill from Codebook", variant="primary")
                with gr.Row():
                    llm_instruction = gr.TextArea(label="Codebook instruction for LLM", placeholder="Full prompt.", interactive=True)
                    progress_bar = gr.Textbox(label="Progress", interactive=False)

            # Custom Tab
            with gr.Tab("🤖 Custom-fill"):
                with gr.Row():
                    gr.Markdown("## Custom annotation")
                with gr.Row():
                    gr.Markdown("<span style='color: darkgrey'>Write a custom prompt to generate a new attribute to the text (summary, category, keywords, ...).</span>")
                with gr.Row():
                    custom_output_column = gr.Textbox(label="Output Column Name", placeholder="Enter name for the new column", interactive=True)
                with gr.Row():
                    custom_instruction = gr.TextArea(label="Instruction for LLM", placeholder="Enter instructions for the LLM to follow when auto-filling annotations...", interactive=True)
                with gr.Row():
                    custom_values = gr.TextArea(label="Valid labels", placeholder="Enter values", interactive=True)
                with gr.Row():
                    custom_process_btn = gr.Button("Process with Custom Instructions", variant="primary")
                    custom_progress = gr.Textbox(label="Progress", interactive=False)

            # Annotation Editor Tab
            with gr.Tab("✏️ Review"):
                with gr.Row():
                    gr.Markdown("## Annotation review")
                with gr.Row():
                    gr.Markdown("<span style='color: darkgrey'>Manual review of the annotated data.</span>")
                with gr.Row():
                    prev_btn = gr.Button("Previous")
                    next_btn = gr.Button("Next")
                with gr.Row():
                    current_index = gr.Number(value=0, label="Current Index", interactive=False)
                    review_status = gr.Textbox(label="Review Status", interactive=False)
                    annotation_status = gr.Textbox(label="Annotation Status", interactive=False)
                with gr.Row():
                    code_select = gr.Dropdown(label="Select Category", choices=[], interactive=True, allow_custom_value=True)
                    value_select = gr.Dropdown(label="Select Value", choices=[], interactive=True, allow_custom_value=True)
                    reload_codebook_btn_2 = gr.Button("Reload Codebook")
                    annotate_next_btn = gr.Button("Annotate and continue to next", variant="primary")
                transcript_box = gr.TextArea(label="Text Content", interactive=False)
                with gr.Row():
                    gr.Markdown("## Autofill summary")
                with gr.Row():
                    autofill_summary = gr.TextArea(label="Auto-fill Summary", interactive=False)
                    customfill_summary = gr.TextArea(label="Custom-fill Summary", interactive=False)

            # Stats Tab
            with gr.Tab("📊 Status"):
                gr.Markdown("Status information will be displayed here")

            # Download Tab
            with gr.Tab("💾 Download"):
                with gr.Row():
                    gr.Markdown("## Download data")
                
                with gr.Row():
                    gr.Markdown("Download the annotated data and codebook.")
                    
                with gr.Row():
                    download_btn = gr.Button("Download Annotated File", variant="primary")
                    codebook_download_btn = gr.Button("Download Codebook", variant="secondary")
                    
                with gr.Row():
                    download_output = gr.File(label="Download")
                    codebook_output = gr.File(label="Codebook Download")
                    
                download_status = gr.Textbox(label="Status", interactive=False)

        
        #############################
        # Event handlers
        #############################

        ### Upload tab

        def load_data(sheet, column):
            """Load data with selected sheet and column"""
            try:
                status, preview = annotator.load_settings(sheet, column)
                current_codebook = annotator.load_codebook()
                codes = [code["attribute"] for code in current_codebook]
                
                return (
                    status,
                    process_df_for_display(preview),
                    gr.Dropdown(choices=codes),
                    gr.Dropdown(choices=codes),
                    gr.Dropdown(choices=codes),
                    current_codebook
                )
            except Exception as e:
                return (
                    f"Error loading data: {str(e)}",
                    None,
                    gr.Dropdown(choices=[]),
                    gr.Dropdown(choices=[]),
                    gr.Dropdown(choices=[]),
                    []
                )
            
        def handle_file_upload(file, codebook_file):
            """Handle initial file upload and codebook"""
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
                gr.Dropdown(choices=codes),
                process_df_for_display(None)  # Reset preview DataFrame
            )
        
        # File upload handler
        file_upload.change(
            fn=handle_file_upload,
            inputs=[file_upload, codebook_upload],
            outputs=[upload_status, sheet_select, codes_display, 
                    code_select, value_select, llm_code_select, preview_df]
        )

        # Sheet selection handler
        sheet_select.change(
            fn=lambda x: gr.Dropdown(choices=annotator.get_columns(x)),
            inputs=[sheet_select],
            outputs=[column_select]
        )

        # Load data button handler
        load_data_btn.click(
            fn=load_data,
            inputs=[sheet_select, column_select],
            outputs=[upload_status, preview_df, code_select, 
                    value_select, llm_code_select, codes_display]
        )

        ######## codebook

        codebook_upload.change(fn=handle_file_upload, 
                               inputs=[file_upload, codebook_upload], 
                               outputs=[upload_status, sheet_select, codes_display, code_select, value_select, llm_code_select])
        
        def handle_new_codebook():
            """
            Purpose: Creates and initializes a new empty codebook. Used in the Codebook tab when starting a new annotation project from scratch.
            Inputs: None
            Outputs: Status message, new codebook content, and three Gradio Dropdowns updated with empty category lists
            """
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
            
        new_codebook_btn.click(fn=handle_new_codebook, 
                               outputs=[upload_status, codes_display, code_select, value_select, llm_code_select])



        ### Settings

        def apply_llm_settings(url, api_key, model):
            from lm import update_llm_config
            update_llm_config(url, api_key, model)
            return f"LLM settings applied successfully. Endpoint: {url}, Model: {model}"

        apply_llm_settings_btn.click(
            fn=apply_llm_settings,
            inputs=[llm_url, llm_api_key, llm_model],
            outputs=[settings_status]
        )

        def process_df_for_display(df):
            """
            Purpose: Formats a DataFrame for display in the UI by truncating long text fields for better readability. Used whenever a DataFrame needs to be shown in the interface, particularly after file uploads or data processing.
            Inputs: df - The DataFrame to be processed for display
            Outputs: A formatted DataFrame with truncated text fields, or None if processing fails
            """
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
        
        def update_value_choices(code_name):
            """
            Purpose: Updates the value dropdown menu based on the selected category. Used in the Review tab when selecting annotation values.
            Inputs: code_name - The currently selected category
            Outputs: A Gradio Dropdown component with updated choices
            """
            if not code_name:
                return gr.Dropdown(choices=[])
            values = annotator.get_code_values(code_name)
            return gr.Dropdown(choices=values, value=None, allow_custom_value=True)
        
        sheet_select.change(fn=lambda x: gr.Dropdown(choices=annotator.get_columns(x)), 
                            inputs=[sheet_select], 
                            outputs=[column_select])
        
        code_select.change(fn=update_value_choices, 
                           inputs=[code_select], outputs=[value_select])

        def reload_llm_categories():
            """
            Purpose: Refreshes the category dropdown in the LLM interface. Used in the Auto-fill tab to ensure category selections are current.
            Inputs: None
            Outputs: Gradio Dropdown component updated with current categories from the codebook
            """
            try:
                codes = [code["attribute"] for code in annotator.load_codebook()]
                return gr.Dropdown(choices=codes, value=None, allow_custom_value=True)
            except Exception as e:
                print(f"Error reloading LLM categories: {e}")
                return gr.Dropdown(choices=[], value=None, allow_custom_value=True)
            
        llm_reload_btn.click(fn=reload_llm_categories, 
                             outputs=[llm_code_select])

        def get_category_values(code_name):
            """
            Purpose: Retrieves all possible values for a given category from the codebook. Used in dropdown menus and validation during annotation.
            Inputs: code_name - The name of the category whose values are needed
            Outputs: A list of valid values for the category, or an empty list if none are found
            """
            try:
                codebook = annotator.load_codebook()
                for code in codebook:
                    if code['attribute'] == code_name:
                        return [v['category'] for v in code['categories']]
                return []
            except Exception as e:
                print(f"Error getting category values: {e}")
                return []
        
        def clean_column_name(name):
            """
            Purpose: Sanitizes column names by removing special characters and spaces. Used throughout the application when handling column names for consistency in data processing and storage.
            Inputs: name - A string or list containing the column name to be cleaned
            Outputs: A cleaned string suitable for use as a column name
            """
            if isinstance(name, list):
                name = "".join(name)
            return name.strip().replace('[','').replace(']','').replace("'", '').replace(" ", '_')

        
        def autofill_from_codebook(code_name, instruction):
            """
            Purpose: Automatically annotates text using the LLM based on codebook categories. Used in the Auto-fill tab to batch process annotations for a selected category.
            Inputs: code_name - The category to use for annotation, instruction - The prompt for the LLM
            Outputs: A status message indicating the success or failure of the auto-fill process
            """
            if not code_name or not instruction:
                return "Please select a category and generate a prompt first"
            try:
                # Get the valid values for the selected category
                valid_values = get_category_values(code_name)
                if not valid_values:
                    return "No valid values found for the selected category"
                    
                clean_name = clean_column_name(code_name)
                output_column = f"autofill_{clean_name}"
                
                # Create status message with constrained values
                values_str = ", ".join(valid_values)
                status_msg = f"Processing with LLM constrained to the following values: [{values_str}]"
                
                # Use batch_process_transcripts with the constrained values
                df, process_status = batch_process_transcripts(
                    annotator.df,
                    instruction,
                    'text',
                    output_column,
                    valid_values
                )
                
                if df is not None:
                    annotator.df = df
                    # back up
                    annotator.backup_df()
                    return f"{status_msg}\n\nAuto-fill completed. Results stored in column: {output_column}"
                else:
                    return f"{status_msg}\n\nError during auto-fill: {process_status}"
                    
            except Exception as e:
                print(f"Error in auto-fill process: {e}")
                return f"Error during auto-fill: {str(e)}"
            
        auto_fill_btn.click(fn=autofill_from_codebook, 
                            inputs=[llm_code_select, llm_instruction], 
                            outputs=[progress_bar])


        def generate_prompt(code_name):
            """
            Purpose: Creates a structured prompt for the LLM based on the selected category from the codebook. Used in the Auto-fill tab when generating instructions for automated annotation.
            Inputs: code_name - The category name selected from the codebook
            Outputs: A formatted prompt string containing the category details, or an error message if generation fails
            """
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
            
        generate_prompt_btn.click(fn=generate_prompt, 
                                  inputs=[llm_code_select], 
                                  outputs=[llm_instruction])

        def refresh_annotation_dropdowns():
            """
            Purpose: Updates all dropdown menus that contain category choices from the codebook. Used whenever the codebook is modified to ensure all dropdowns reflect current categories.
            Inputs: None
            Outputs: Two Gradio Dropdown components - one for category selection and one for value selection
            """
            try:
                categories = [code["attribute"] for code in annotator.load_codebook()]
                return (
                    gr.Dropdown(choices=categories, value=None, allow_custom_value=True),
                    gr.Dropdown(choices=[], value=None, allow_custom_value=True)
                )
            except Exception as e:
                print(f"Error refreshing dropdowns: {e}")
                return gr.Dropdown(), gr.Dropdown()
            
        reload_codebook_btn_2.click(fn=refresh_annotation_dropdowns, 
                                    outputs=[code_select, value_select])
        

        # Custom tab
        def custom_batch_process(instruction, values, output_column):
            """
            Purpose: Processes all texts using custom LLM instructions. Used in the Custom tab for flexible batch processing with user-defined instructions and output categories.
            Inputs: instruction - Custom LLM prompt, values - Allowed output values, output_column - Name for the new column
            Outputs: Status message indicating success or failure of the batch process
            """
            if not output_column:
                return "Please specify an output column name"
            # specific column name for custom annotations
            output_column = "custom_"+output_column
            try:
                df, status = batch_process_transcripts(
                    annotator.df,
                    instruction,
                    annotator.selected_column,
                    output_column,
                    values)
                if df is not None:
                    annotator.df = df
                    # back up
                    annotator.backup_df()
                return status
            except Exception as e:
                return f"Error: {str(e)}"
            
        custom_process_btn.click(fn=custom_batch_process, 
                                 inputs=[custom_instruction, custom_values, custom_output_column], 
                                 outputs=[custom_progress])
        
        ### Annotation tab

        def get_autofill_summary(index):
            """Get concatenated values from all autofill columns for the current index"""
            try:
                if annotator.df is None or index >= len(annotator.df):
                    return ""
                    
                # Get all columns that start with 'autofill_'
                autofill_cols = [col for col in annotator.df.columns if col.startswith('autofill_')]
                
                if not autofill_cols:
                    return "No auto-fill annotations found"
                    
                # Build summary string
                summary = []
                for col in autofill_cols:
                    value = annotator.df.at[index, col]
                    # Clean column name by removing 'autofill_' prefix
                    clean_col = col.replace('autofill_', '')
                    summary.append(f"{clean_col}: {value}")
                    
                return "\n".join(summary)
                
            except Exception as e:
                return f"Error getting auto-fill summary: {str(e)}"
        def get_customfill_summary(index):
            """Get concatenated values from all autofill columns for the current index"""
            try:
                if annotator.df is None or index >= len(annotator.df):
                    return ""
                    
                # Get all columns that start with 'autofill_'
                autofill_cols = [col for col in annotator.df.columns if col.startswith('custom_')]
                
                if not autofill_cols:
                    return "No custom-fill annotations found"
                    
                # Build summary string
                summary = []
                for col in autofill_cols:
                    value = annotator.df.at[index, col]
                    # Clean column name by removing 'autofill_' prefix
                    clean_col = col.replace('custom_', '')
                    summary.append(f"{clean_col}: {value}")
                    
                return "\n".join(summary)
                
            except Exception as e:
                return f"Error getting custom-fill summary: {str(e)}"
        

        def annotate_and_next(code_name, value):
            """
            Purpose: Saves the current annotation and automatically moves to the next text entry. Used in the Review tab when annotating texts sequentially.
            Inputs: code_name - The category being annotated, value - The selected value for the annotation
            Outputs: Status message, next text content, current index, and review status indicator (✅/❌)
            """
            try:
                if not code_name or not value:
                    return "Please select both category and value", None, None, None, ""
                    
                status, df = annotator.save_annotation(code_name, value)
                if not status.startswith("Saved"):
                    return status, None, None, None, ""
                    
                text, idx = annotator.navigate_transcripts("next")
                review_status_text = "✅" if annotator.df.iloc[idx]['is_reviewed'] else "❌"
                autofill_summary = get_autofill_summary(idx)
                customfill_summary = get_customfill_summary(idx)
                
                return status, text, idx, review_status_text, autofill_summary, customfill_summary
                
            except Exception as e:
                print(f"Error in annotate_and_next: {e}")
                return "Error during annotation", None, None, "❌", ""
            
        annotate_next_btn.click(fn=annotate_and_next, 
                                inputs=[code_select, value_select], 
                                outputs=[annotation_status, transcript_box, current_index, review_status, autofill_summary, customfill_summary])

        def navigate_and_update(direction):
            """
            Purpose: Handles navigation between text entries in the review interface. Used for moving between texts during manual review.
            Inputs: direction - Either "prev" or "next" to indicate navigation direction
            Outputs: Text content, current index, and review status indicator (✅/❌)
            """
            try:
                text, idx = annotator.navigate_transcripts(direction)
                if text is None or idx is None:
                    return None, None, "❌", ""
                    
                review_status_text = "✅" if annotator.df.iloc[idx]['is_reviewed'] else "❌"
                autofill_summary = get_autofill_summary(idx)
                customfill_summary = get_customfill_summary(idx)
                
                return text, idx, review_status_text, autofill_summary, customfill_summary
                
            except Exception as e:
                print(f"Error in navigate_and_update: {e}")
                return None, None, "❌", ""

        prev_btn.click(fn=lambda: navigate_and_update("prev"), 
                       outputs=[transcript_box, current_index, review_status, autofill_summary, customfill_summary])

        next_btn.click(fn=lambda: navigate_and_update("next"), 
                       outputs=[transcript_box, current_index, review_status, autofill_summary, customfill_summary])
        
        ### Download tab

        download_btn.click(fn=annotator.save_excel, 
                           outputs=[download_output, download_status])
        
        codebook_download_btn.click(fn=annotator.save_codebook,
                        outputs=[codebook_output, download_status])
        return demo