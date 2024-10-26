import gradio as gr
import pandas as pd
from annotator import TranscriptionAnnotator
from lm import query_llm, query_constrained_llm, batch_process_transcripts

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

def create_ui():
    annotator = TranscriptionAnnotator()
    
    with gr.Blocks() as demo:
        gr.Markdown("## 📝 Fannotate")
        with gr.Tabs():
            # Upload Tab
            with gr.Tab("📁 Upload Data"):
                file_upload = gr.File(label="Upload Excel File")
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

                    llm_instruction = gr.TextArea(
                    label="Codebook instruction for LLM",
                    placeholder="Full prompt.",
                    interactive=True
                    )
                    auto_fill_btn = gr.Button("Auto-fill from Codebook")
                    
                progress_bar = gr.Textbox(
                    label="Progress",
                    interactive=False
                )

            # LLM Auto-fill Tab
            with gr.Tab("🤖 Custom"):
                with gr.Row():
                    # llm_code_select = gr.Dropdown(
                    #     label="Select Category to Auto-fill",
                    #     choices=[],
                    #     interactive=True,
                    #     allow_custom_value=True
                    # )
                    output_column = gr.Textbox(
                        label="Output Column Name",
                        placeholder="Enter name for the new column",
                        interactive=True
                    )                    

                llm_instruction = gr.TextArea(
                    label="Instruction for LLM",
                    placeholder="Enter instructions for the LLM to follow when auto-filling annotations...",
                    interactive=True
                )
                
                with gr.Row():
                    values = gr.TextArea(
                        label="Valid labels",
                        placeholder="Enter values",
                        interactive=True
                    )
                    auto_fill_btn = gr.Button("Auto-fill Selected Category")
                    
                progress_bar = gr.Textbox(
                    label="Progress",
                    interactive=False
                )
                llm_output = gr.TextArea(
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
            
            # Download Tab
            with gr.Tab("💾 Download"):
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
        
        def apply_settings(sheet, column, url, api_key, model):
            # Update LLM settings
            from lm import update_llm_config
            update_llm_config(url, api_key, model)
            
            # Apply other settings
            status, preview, transcript, codes1, codes2 = annotator.load_settings(sheet, column)
            
            # Get initial review status
            initial_review_status = "✅" if annotator.df.iloc[0]['is_reviewed'] else "❌"
            
            return (
                f"Settings applied successfully. LLM endpoint: {url}, Model: {model}\n{status}",
                preview,
                transcript,
                codes1,
                codes2,
                initial_review_status
            )

        # Update the load_settings_btn click handler
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
                review_status
            ]
        )
                
        # auto_fill_btn.click(
        #     fn=query_constrained_llm,
        #     inputs=[llm_instruction, values],
        #     outputs=[llm_output]
        # )

        # auto_fill_btn.click(
        #     fn=query_llm,
        #     inputs=[llm_instruction],
        #     outputs=[llm_output]
        # )

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

        auto_fill_btn.click(
            fn=process_with_llm,
            inputs=[llm_instruction, values, output_column],
            outputs=[llm_output]
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
        
        def refresh_all_code_dropdowns():
            try:
                codes = [code["name"] for code in annotator.load_codebook()]
                return {
                    edit_code_select: gr.Dropdown(choices=codes),
                    delete_code_select: gr.Dropdown(choices=codes),
                    code_select: gr.Dropdown(choices=codes)
                }
            except Exception as e:
                print(f"Error refreshing code dropdowns: {e}")
                return {
                    edit_code_select: gr.Dropdown(choices=[]),
                    delete_code_select: gr.Dropdown(choices=[]),
                    code_select: gr.Dropdown(choices=[])
                }

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

        def update_value_choices(code_name):
            if not code_name:
                return gr.Dropdown(choices=[])
            values = [v['value'] for v in annotator.get_code_values(code_name)]
            return gr.Dropdown(choices=values)

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
            fn=refresh_all_code_dropdowns,
            outputs=[edit_code_select, delete_code_select, code_select]
        )

        reload_codebook_btn_2.click(
            fn=refresh_annotation_dropdowns,
            outputs=[code_select, value_select]
        )
        
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
        
        annotate_next_btn.click(
            fn=annotate_and_next,
            inputs=[code_select, value_select],
            outputs=[annotation_status, transcript_box, current_index, review_status]
        )
        
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