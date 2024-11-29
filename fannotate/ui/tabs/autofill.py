import gradio as gr
from ..utils.display import clean_column_name

def create_autofill_tab(annotator):
    """Creates and returns the auto-fill tab interface"""
    with gr.Tab("🤖 Auto-fill"):
        with gr.Row():
            gr.Markdown("## Auto annotation")
            
        with gr.Row():
            gr.Markdown("Automated annotation of the text using the codebook.")
            
        with gr.Row():
            # Dropdown to trigger prompt generation on change
            llm_code_select = gr.Dropdown(
                label="Select Category to Auto-fill", 
                choices=[], 
                interactive=True, 
                allow_custom_value=True
            )
            llm_reload_btn = gr.Button("🔄 Refresh Categories")
            
        with gr.Row():
            auto_fill_btn = gr.Button("Auto-fill from Codebook", variant="primary")
            
        with gr.Row():
            llm_instruction = gr.TextArea(
                label="Codebook instruction for LLM", 
                placeholder="Full prompt.", 
                interactive=True
            )
            progress_bar = gr.Textbox(label="Progress", interactive=False)

        ############################################################
        # Event handlers
        ############################################################

        def reload_llm_categories():
            """Refreshes the category dropdown in the LLM interface"""
            try:
                codes = [code["attribute"] for code in annotator.load_codebook()]
                return gr.Dropdown(choices=codes, value=None, allow_custom_value=True)
            except Exception as e:
                print(f"Error reloading LLM categories: {e}")
                return gr.Dropdown(choices=[], value=None, allow_custom_value=True)

        def get_category_values(code_name):
            """Retrieves all possible values for a given category"""
            try:
                codebook = annotator.load_codebook()
                for code in codebook:
                    if code['attribute'] == code_name:
                        return [v['category'] for v in code['categories']]
                return []
            except Exception as e:
                print(f"Error getting category values: {e}")
                return []

        def create_prompt_from_json(json_data):
            """Converts a JSON codebook entry into a formatted prompt string"""
            try:
                prompt = json_data.get('instruction', '')
                
                # Replace <<categories>> tag with formatted categories
                categories_text = ""
                for category in json_data.get('categories', []):
                    categories_text += f"- {category['category']}: {category['description']}\n\n"
                
                prompt = prompt.replace('<<categories>>', categories_text.strip())
                
                # The <<text>> tag will be replaced later when processing each transcript
                return prompt
                
            except Exception as e:
                print(f"Error creating prompt: {str(e)}")
                return None

        def generate_prompt(code_name):
            """Creates a structured prompt for the LLM"""
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
                
                prompt = create_prompt_from_json(selected_code)
                
                if prompt is None:
                    return "Error generating prompt"
                    
                return prompt
                
            except Exception as e:
                print(f"Error generating prompt: {e}")
                return f"Error generating prompt: {str(e)}"

        def autofill_from_codebook(code_name, instruction):
            """Automatically annotates text using the LLM"""
            if not code_name or not instruction:
                return "Please select a category and generate a prompt first"
            
            try:
                # Load codebook and find selected category
                codebook = annotator.load_codebook()
                selected_code = None
                for code in codebook:
                    if code['attribute'] == code_name:
                        selected_code = code
                        break
                        
                if not selected_code:
                    return "Selected category not found in codebook"
                    
                clean_name = clean_column_name(code_name)
                output_column = f"autofill_{clean_name}"
                
                # Check if category type is categorical or freetext
                is_categorical = selected_code.get('type', 'categorical') == 'categorical'
                
                if is_categorical:
                    # Get valid values for categorical type
                    valid_values = get_category_values(code_name)
                    if not valid_values:
                        return "No valid values found for the selected category"
                    
                    # Use constrained LLM call
                    from fannotate.lm import batch_process_transcripts
                    df, process_status = batch_process_transcripts(
                        annotator.df,
                        instruction,
                        'text',
                        output_column,
                        valid_values
                    )
                    values_str = ", ".join(valid_values)
                    status_msg = f"Processing with LLM constrained to values: [{values_str}]"
                    
                else:
                    # Use unconstrained LLM call for freetext
                    from fannotate.lm import batch_process_transcripts
                    df, process_status = batch_process_transcripts(
                        annotator.df,
                        instruction,
                        'text', 
                        output_column,
                        None  # No value constraints for freetext
                    )
                    status_msg = "Processing with unconstrained LLM for free text response"
                    
                if df is not None:
                    annotator.df = df
                    annotator.backup_df()
                    return f"{status_msg}\n\nAuto-fill completed. Results stored in column: {output_column}"
                else:
                    return f"{status_msg}\n\nError during auto-fill: {process_status}"
                    
            except Exception as e:
                print(f"Error in auto-fill process: {e}")
                return f"Error during auto-fill: {str(e)}"

        # Connect event handlers
        llm_reload_btn.click(
            fn=reload_llm_categories,
            outputs=[llm_code_select]
        )

        llm_code_select.change(
            fn=generate_prompt,
            inputs=[llm_code_select],
            outputs=[llm_instruction]
        )

        auto_fill_btn.click(
            fn=autofill_from_codebook,
            inputs=[llm_code_select, llm_instruction],
            outputs=[progress_bar]
        )

        return {
            'llm_code_select': llm_code_select,
            'llm_instruction': llm_instruction,
            'progress_bar': progress_bar
        } 