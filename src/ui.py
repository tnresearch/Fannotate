import gradio as gr
import pandas as pd
import json
from annotator import TranscriptionAnnotator
from lm import batch_process_transcripts
from gradio_rich_textbox import RichTextbox
from sklearn.metrics import accuracy_score, f1_score

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
            ###################################
            # Upload Tab
            ###################################
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
                        load_data_btn = gr.Button("Create annotation table", variant="primary")
                
                with gr.Row():
                    gr.Markdown("## Data preview")
                with gr.Row():
                    gr.Markdown("This shows the top 5 rows of your data, and a shortened version of the text to be annotated.")
                with gr.Row():
                    preview_df = gr.DataFrame(interactive=False, visible=True,
                                            row_count=(5, "fixed")  # Show 5 rows before scrolling)
                    )
            
            ###################################
            # Settings Tab
            ###################################
            with gr.Tab("⚙️ Settings"):
                with gr.Row():
                    gr.Markdown("## LLM Settings")
                with gr.Row():
                    gr.Markdown("Specify the endpoint, API-key (optional) and Model name of the LLM to use.")
                
                with gr.Row():
                    llm_url = gr.Textbox(label="LLM Endpoint URL", 
                                        value="http://192.168.50.155:8000/v1/")
                    llm_api_key = gr.Textbox(label="API Key", 
                                            value="token-abc123")
                    llm_model = gr.Textbox(label="Model Name", 
                                        value="google/gemma-2-2b-it")
                    llm_framework = gr.Dropdown(
                                        label="LLM Framework",
                                        choices=["vLLM", "OpenAI"],
                                        value="vLLM",
                                        interactive=True
                                    )
                
                with gr.Row():
                    apply_llm_settings_btn = gr.Button("Apply LLM Settings", variant="primary")
                    settings_status = gr.Textbox(label="Settings Status", interactive=False)
                

                with gr.Row():
                    gr.Markdown("<br><br>")
                #     gr.Markdown("Fannotate relies on a OpenAI-like API endpoint. It is recommended to serve the LLM with vLLM which supports all the functionality in Fannotate.")

                with gr.Row():
                    gr.Markdown("## LLM Settings Help")

                with gr.Row():
                    gr.Markdown("""### Endpoint URL: 
                                Fannotate relies on a OpenAI-like API endpoint. It is recommended to serve the LLM with vLLM which supports all the functionality in Fannotate.""")
                with gr.Row():
                    gr.Markdown("""
                                    - Frigg is found at: http://172.16.16.48:8000/v1/
                                    - ITX is found at: http://192.168.50.155:8000/v1/
                                    - OpenAI is found at: https://api.openai.com/v1/
                                <br>
                                """)
                with gr.Row():
                    gr.Markdown("""### Model Name:
                                        It is highly suggested to use *permissively licensed* LLMs that allow distillation/creation of training data for training of competing models (No OpenAI models can be used legally for this, due to <a href="https://openai.com/policies/row-terms-of-use/">OpenAI TOS</a>. """)
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            gr.Markdown("""### Gemma Apache 2.0 License:

                                        Requires you to include a copy of the license, document any changes made, retain all copyright, patent and attribution notices. 
                                        Full license: https://github.com/google-deepmind/gemma/blob/main/LICENSE""", container=True)
                                        
                        
                        with gr.Row():
                            gr.Markdown("""Suggested models:
                                            - ```google/Gemma-2-2B-it```
                                            - ```google/Gemma-2-9B-it```
                                            - ```google/Gemma-2-27B-it```
                                        """, container=True)
                
                
                    with gr.Column(scale=1):
                        with gr.Row():
                            gr.Markdown("""
                                        ### Meta Llama 3 Community License:

                                        Requires you to acknowledge "Built with Meta Llama 3"" in the documentation, and name any derivative model as 'llama-3*', and obtain additional licensing if services exceed 700 million monthly users. 
                                        Full license: https://www.llama.com/llama3/license/""", container=True)
                                        
                        
                        with gr.Row():
                            gr.Markdown("""Suggested models:                                        
                                        - ```meta-llama/Llama-3.1-8B-Instruct```
                                        - ```meta-llama/Llama-3.1-70B-Instruct```
                                        - ```meta-llama/Llama-3.1-405B-Instruct```
                                        """, container=True)
                        

            
            ###################################
            # Codebook Tab
            ###################################
            with gr.Tab("📓 Codebook"):
                with gr.Row():
                    gr.Markdown("## Annotation codebook")
                
                #with gr.Group():  
                with gr.Row():
                    gr.Markdown("Upload existing codebook or initialize a new one.")
                with gr.Row():
                    codebook_upload = gr.File(label="Upload Codebook (Optional)")
                    new_codebook_btn = gr.Button("New Codebook")
                
                with gr.Row():
                    gr.Markdown("## Current codebook")
                with gr.Row():
                    codes_display = gr.JSON(label="Current Codebook")                    
                with gr.Row():
                    gr.Markdown("## (1) Add New Attribute")

                with gr.Group():                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            attribute_name = gr.Textbox(
                                label="Attribute Name",
                                placeholder="Enter the name of the attribute"
                            )
                            attribute_description = gr.TextArea(
                                label="Attribute Description",
                                placeholder="Describe what this attribute represents"
                            )
                        with gr.Column(scale=1):
                            attribute_type = gr.Radio(
                                label="Attribute Type",
                                choices=["categorical", "freetext"],
                                value="categorical"
                            )                            
                    
                    with gr.Row():
                            instruction_start = gr.TextArea(
                                label="Instruction Start",
                                placeholder="Enter the initial instruction for annotators"
                            )
                            instruction_end = gr.TextArea(
                                label="Instruction End",
                                placeholder="Enter the final instruction for annotators"
                            )
                    with gr.Row():
                        add_attribute_btn = gr.Button("Add Attribute", variant="secondary")#, variant="primary")

                with gr.Row():
                    gr.Markdown("## (2) Add Category to Attribute")


                with gr.Group():      
                    with gr.Row():
                        with gr.Column(scale=1):
                            attribute_select = gr.Dropdown(
                                label="Select Attribute",
                                choices=[],
                                interactive=True,
                                allow_custom_value=False,
                                #placeholder="Choose an attribute to add categories to"
                            )
                            reload_attributes_btn = gr.Button("🔄 Reload Attributes")
                            category_name = gr.Textbox(
                                label="Category Name",
                                #placeholder="Enter the name of the category"
                            )
                            with gr.Row():
                                category_icon = gr.Textbox(
                                    label="Category Icon (Emoji)",
                                    placeholder="Enter an emoji icon"
                                )
                        with gr.Column(scale=1):
                            category_description = gr.TextArea(
                                label="Category Description",
                                #placeholder="Describe what this category represents"
                            )

                    with gr.Row():
                        add_category_btn = gr.Button("Add Category", variant="secondary")#, variant="primary")


            
            ###################################
            # Auto-fill Tab
            ###################################
            with gr.Tab("🤖 Auto-fill"):
                with gr.Row():
                    gr.Markdown("## Auto annotation")
                    
                with gr.Row():
                    gr.Markdown("Automated annotation of the text using the codebook.")
                    
                with gr.Row():
                    # Modify the dropdown to trigger prompt generation on change
                    llm_code_select = gr.Dropdown(
                        label="Select Category to Auto-fill", 
                        choices=[], 
                        interactive=True, 
                        allow_custom_value=True
                    )
                    llm_reload_btn = gr.Button("Reload Categories")
                    
                with gr.Row():
                    # Remove the generate prompt button since it's no longer needed
                    auto_fill_btn = gr.Button("Auto-fill from Codebook", variant="primary")
                    
                with gr.Row():
                    llm_instruction = gr.TextArea(
                        label="Codebook instruction for LLM", 
                        placeholder="Full prompt.", 
                        interactive=True
                    )
                    progress_bar = gr.Textbox(label="Progress", interactive=False)

            
            ###################################
            # Review Tab
            ###################################
            with gr.Tab("✏️ Review"):
                # Keep original code_select for compatibility
                code_select = gr.Dropdown(label="Category", choices=[], visible=False, interactive=True)
                value_select = gr.Radio(label="Value", choices=[], visible=False, interactive=True)
                with gr.Row():
                    gr.Markdown("## Annotation review")
                
                with gr.Row():
                    gr.Markdown("Manual review of the annotated data.")
                
                
                with gr.Row():
                    num_categories = gr.Dropdown(
                        label="Number of Categories to Annotate",
                        choices=["1", "2", "3", "4"],
                        value="1",
                        interactive=True
                    )
                
                # Dynamic category columns container
                category_columns = gr.Row(visible=True)
                
                with category_columns:
                    with gr.Column(visible=True) as col1:
                        code_select1 = gr.Dropdown(label="(1) Select Category", choices=[], interactive=True)
                        value_select1 = gr.Radio(label="(2) Select Value", choices=[], interactive=True)
                    
                    with gr.Column(visible=False) as col2:
                        code_select2 = gr.Dropdown(label="(1) Select Category", choices=[], interactive=True)
                        value_select2 = gr.Radio(label="(2) Select Value", choices=[], interactive=True)
                    
                    with gr.Column(visible=False) as col3:
                        code_select3 = gr.Dropdown(label="(1) Select Category", choices=[], interactive=True)
                        value_select3 = gr.Radio(label="(2) Select Value", choices=[], interactive=True)
                    
                    with gr.Column(visible=False) as col4:
                        code_select4 = gr.Dropdown(label="(1) Select Category", choices=[], interactive=True)
                        value_select4 = gr.Radio(label="(2) Select Value", choices=[], interactive=True)
                
                with gr.Row():
                    reload_codebook_btn = gr.Button("🔄 Reload Categories", variant="secondary")
                    annotate_next_btn = gr.Button("Annotate and continue to next!", variant="primary", size="lg")

                with gr.Row():
                    prev_btn = gr.Button("Previous")
                    next_btn = gr.Button("Next")
                
                with gr.Row():
                    gr.Markdown("## LLM suggestions")
                
                with gr.Row():
                    with gr.Column():
                        categorical_summary = RichTextbox(
                            label="Categorical Auto-fill Annotations:",
                            interactive=False
                        )
                    with gr.Column():
                        freetext_summary = RichTextbox(
                            label="Free-text Auto-fill Annotations:",
                            interactive=False
                        )
                
                with gr.Row():
                    annotation_status = gr.Textbox(label="Annotation Status", interactive=False)
                
                with gr.Row():
                    gr.Markdown("## Text content")
                
                with gr.Row():
                    current_index_display = gr.Markdown("**Current Index:** 0")
                
                with gr.Row():
                    transcript_box = RichTextbox(label="Text Content", interactive=False)

            
            ###################################
            # Status Tab
            ###################################
            with gr.Tab("📊 Status"):
                with gr.Row():
                    gr.Markdown("## Annotation Progress")
                
                with gr.Row():
                    review_progress = gr.Markdown("Loading progress...")
                
                with gr.Row():
                    gr.Markdown("## Agreement Analysis")
                
                with gr.Row():
                    with gr.Column():
                        category_select = gr.Dropdown(
                            label="Select Category to Analyze",
                            choices=[],
                            interactive=True
                        )
                    with gr.Column():
                        refresh_codebook_btn_3 = gr.Button("🔄 Refresh Categories", variant="secondary")
                        refresh_stats_btn = gr.Button("Refresh Statistics", variant="primary")
                
                with gr.Row():
                    metrics_display = RichTextbox(
                        label="Classification Metrics",
                        interactive=False
                    )


            
            ###################################
            # Download Tab
            ###################################
            with gr.Tab("💾 Download", id="download_tab") as download_tab:
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


            ###################################
            # About Tab
            ###################################
            with gr.Tab("ℹ️ About"):
                #try:
                gr.Markdown("""## Overview""")
                with gr.Row():
                    gr.Markdown("""Fannotate is a tool for *faster* text annotation aided by LLMs. Central to Fannotate is the *codebook* (annotation guidelines) which help the LLM do an initial guess at the categories and labels. Fannotate lets you create any attributes to the text you'd like, which can then help you annotate the text faster, without having to read the whole transcript.""")
                gr.Image("../bin/concept.png", 
                        label=None,  # Removes the label
                        show_label=False,  # Ensures no label space is reserved
                        container=False,  # Removes the container box
                        show_download_button=False,  # Removes the download button
                        interactive=False,  # Prevents user interaction
                        height="auto",  # Adjusts height automatically
                        width="400px"#"100%"  # Takes full width of parent container
                        )
                
                
                with gr.Row():
                    gr.Markdown("""## Procedure""")
                
                with gr.Row():
                    gr.Markdown("""The intended workflow is based on 1) Creating a codebook, 2) having a LLM auto-annotate the text based on the descriptions in the codebook, and 3) Manual verification of the annotations created. The main difference in quality is determined by the rigorousness of step 3.""")
                gr.Image("../bin/procedure.png", 
                            label=None,  # Removes the label
                            show_label=False,  # Ensures no label space is reserved
                            container=False,  # Removes the container box
                            show_download_button=False,  # Removes the download button
                            interactive=False,  # Prevents user interaction
                            height="auto",  # Adjusts height automatically
                            width="700px"#"100%"  # Takes full width of parent container
                            )
                    
                with gr.Row():
                    gr.Markdown("## User guide")
                with gr.Row():
                    gr.Markdown("The Fannotate user guide <a href='https://github.com/tnresearch/Fannotate/blob/main/userguide.md'>can be found here</a>")
                #except FileNotFoundError:
                #    gr.Markdown("Instructions file not found. Please create instructions.md")

        
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
                gr.Dropdown(choices=codes),  # code_select
                gr.Dropdown(choices=codes),  # code_select1
                gr.Dropdown(choices=codes),  # code_select2
                gr.Dropdown(choices=codes),  # code_select3
                gr.Dropdown(choices=codes),  # code_select4
                gr.Radio(choices=[]),       # value_select
                process_df_for_display(None)
            )
        
        file_upload.change(
            fn=handle_file_upload,
            inputs=[file_upload, codebook_upload],
            outputs=[
                upload_status, sheet_select, codes_display,
                code_select, code_select1, code_select2, 
                code_select3, code_select4, value_select, 
                preview_df
            ]
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

        ################################################
        # Codebook
        ################################################

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


        def update_attribute_choices():
            """Updates the attribute dropdown with current codebook attributes"""
            try:
                codebook = annotator.load_codebook()
                attributes = [code["attribute"] for code in codebook]
                return gr.Dropdown(choices=attributes)
            except Exception as e:
                print(f"Error updating attribute choices: {e}")
                return gr.Dropdown(choices=[])


        def add_attribute_to_codebook(name, description, attr_type, instr_start, instr_end):
            """
            Purpose: Adds a new attribute to the existing codebook with basic configuration.
            """
            try:
                if not name:
                    return ("Error: Attribute name is required", None, 
                        name, description, attr_type, instr_start, instr_end)
                    
                # Load existing codebook
                if not annotator.codebook_path.exists():
                    annotator.create_new_codebook()
                    
                with open(annotator.codebook_path, 'r') as f:
                    codebook = json.load(f)
                    
                # Check if attribute already exists
                if any(code['attribute'] == name for code in codebook['codes']):
                    return (f"Error: Attribute '{name}' already exists", None,
                        name, description, attr_type, instr_start, instr_end)
                    
                # Create new attribute
                new_attribute = {
                    "attribute": name,
                    "description": description,
                    "type": attr_type,
                    "instruction_start": instr_start,
                    "instruction_end": instr_end,
                    "categories": []
                }
                
                # Add to codebook
                codebook['codes'].append(new_attribute)
                
                # Save updated codebook
                with open(annotator.codebook_path, 'w') as f:
                    json.dump(codebook, f, indent=4)
                    
                # Return success message and clear all inputs
                return (f"Successfully added attribute: {name}", codebook,
                        "", "", "categorical", "", "")
                
            except Exception as e:
                return (f"Error adding attribute: {str(e)}", None,
                        name, description, attr_type, instr_start, instr_end)

        def add_category_to_attribute(attribute_name, category, description, icon):
            """Adds a new category with an emoji icon to an existing attribute"""
            try:
                if not all([attribute_name, category, description]):
                    return ("Error: All fields are required", None,
                        attribute_name, category, description, icon)
                    
                with open(annotator.codebook_path, 'r') as f:
                    codebook = json.load(f)
                    
                for code in codebook['codes']:
                    if code['attribute'] == attribute_name:
                        if any(cat['category'] == category for cat in code['categories']):
                            return (f"Error: Category '{category}' already exists", None,
                                attribute_name, category, description, icon)
                                
                        code['categories'].append({
                            "category": category,
                            "description": description,
                            "icon": icon or ""  # Store empty string if no icon provided
                        })
                        
                with open(annotator.codebook_path, 'w') as f:
                    json.dump(codebook, f, indent=4)
                    
                return (f"Successfully added category '{category}'", codebook,
                        attribute_name, "", "", "")
                        
            except Exception as e:
                return (f"Error adding category: {str(e)}", None,
                        attribute_name, category, description, icon)

        # Add the click handler
        add_attribute_btn.click(
            fn=add_attribute_to_codebook,
            inputs=[
                attribute_name,
                attribute_description, 
                attribute_type,
                instruction_start,
                instruction_end
            ],
            outputs=[
                upload_status,
                codes_display,
                attribute_name,          # Clear name
                attribute_description,   # Clear description
                attribute_type,         # Reset to default
                instruction_start,      # Clear start instruction
                instruction_end         # Clear end instruction
            ]
        )

        add_category_btn.click(
            fn=add_category_to_attribute,
            inputs=[
                attribute_select,
                category_name,
                category_description,
                category_icon
            ],
            outputs=[
                upload_status,
                codes_display,
                attribute_select,       # Keep selected attribute
                category_name,         # Clear category name
                category_description,   # Clear category description
                category_icon
            ]
        )

        reload_attributes_btn.click(
            fn=update_attribute_choices,
            outputs=[attribute_select]
        )

        # Update attribute choices when codebook changes
        new_codebook_btn.click(fn=update_attribute_choices, outputs=[attribute_select])
        codebook_upload.change(fn=update_attribute_choices, outputs=[attribute_select])
        add_attribute_btn.click(fn=update_attribute_choices, outputs=[attribute_select])

        ################################################
        # Settings
        ################################################

        def apply_llm_settings(framework, url, api_key, model):
            from lm import update_llm_config
            update_llm_config(framework, url, api_key, model)
            return f"LLM settings applied successfully. Framework: {framework}, Endpoint: {url}, Model: {model}"
        
        apply_llm_settings_btn.click(
            fn=apply_llm_settings,
            inputs=[llm_framework, llm_url, llm_api_key, llm_model],
            outputs=[settings_status]
        )

        def process_df_for_display(df, top_n=5):
            """
            Purpose: Formats a DataFrame for display in the UI by truncating long text fields for better readability. Used whenever a DataFrame needs to be shown in the interface, particularly after file uploads or data processing.
            Inputs: df - The DataFrame to be processed for display, number of rows to display
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
                
                return df_display.head(top_n) #reduce the number of rows displayed
            except Exception as e:
                print(f"Error processing DataFrame: {e}")
                return None

        def update_value_choices(code_name):
            """Updates the radio button choices based on selected category"""
            if not code_name:
                return gr.Radio(choices=[])
            
            try:
                # Load codebook to get category type
                codebook = annotator.load_codebook()
                selected_code = None
                for code in codebook:
                    if code['attribute'] == code_name:
                        selected_code = code
                        break
                        
                if selected_code:
                    # Get values and their icons
                    choices = []
                    for category in selected_code['categories']:
                        icon = category.get('icon', '')
                        label = f"{icon} {category['category']}" if icon else category['category']
                        choices.append(label)
                    
                    return gr.Radio(
                        choices=choices,
                        label="Select Value",
                        value=None  # Reset selection when category changes
                    )
                return gr.Radio(choices=[])
                
            except Exception as e:
                print(f"Error updating value choices: {e}")
                return gr.Radio(choices=[])
        
        sheet_select.change(fn=lambda x: gr.Dropdown(choices=annotator.get_columns(x)), 
                            inputs=[sheet_select], 
                            outputs=[column_select])
        
        # Add event handler to sync primary code_select with code_select1
        def sync_code_selects(choices):
            return [gr.Dropdown(choices=choices) for _ in range(5)]  # For code_select and code_select1-4
        
        code_select.change(
            fn=sync_code_selects,
            inputs=[code_select],
            outputs=[code_select, code_select1, code_select2, code_select3, code_select4]
        )
        

        ################################################
        # Auto fill
        ################################################

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
            
        auto_fill_btn.click(fn=autofill_from_codebook, 
                            inputs=[llm_code_select, llm_instruction], 
                            outputs=[progress_bar])


        def create_prompt_from_json(json_data):
            """
            Converts a JSON codebook entry into a formatted prompt string and appends text if provided.
            
            Args:
                json_data (dict): The JSON codebook entry containing attribute, categories, and instructions
                text (str): Optional text to append to the prompt
                
            Returns:
                str: A formatted prompt string
            """
            try:
                # Start with the instruction header
                prompt = json_data.get('instruction_start', '')
                #prompt += "\n\n"
                
                # Add each category and its description
                for category in json_data.get('categories', []):
                    prompt += f"- {category['category']}: {category['description']}\n\n"
                    
                # Add the instruction ending
                prompt += json_data.get('instruction_end', '')
                prompt += "\n\nText: "
                
                return prompt
                
            except Exception as e:
                print(f"Error creating prompt: {str(e)}")
                return None

        def generate_prompt(code_name):
            """
            Creates a structured prompt for the LLM based on the selected category and current text.
            
            Args:
                code_name (str): The category name selected from the codebook
                
            Returns:
                str: A formatted prompt string containing the category details and current text
            """
            if not code_name:
                return "Please select a category first"
            
            try:
                # Load codebook and find selected category
                codebook = annotator.load_codebook()
                selected_code = None
                clean_name = clean_column_name(code_name)
                
                for code in codebook:
                    if clean_column_name(code['attribute']) == clean_name:
                        selected_code = code
                        break
                        
                if not selected_code:
                    return f"Selected category '{code_name}' not found in codebook"
                
                # Generate prompt using the helper function
                prompt = create_prompt_from_json(selected_code)
                
                if prompt is None:
                    return "Error generating prompt"
                    
                return prompt
                
            except Exception as e:
                print(f"Error generating prompt: {e}")
                return f"Error generating prompt: {str(e)}"
            
        
        llm_code_select.change(
            fn=generate_prompt,
            inputs=[llm_code_select],
            outputs=[llm_instruction]
        )


        # def refresh_annotation_dropdowns():
        #     """
        #     Purpose: Updates all dropdown menus that contain category choices from the codebook. Used whenever the codebook is modified to ensure all dropdowns reflect current categories.
        #     Inputs: None
        #     Outputs: Two Gradio Dropdown components - one for category selection and one for value selection
        #     """
        #     try:
        #         categories = [code["attribute"] for code in annotator.load_codebook()]
        #         return (
        #             gr.Dropdown(choices=categories, value=None, allow_custom_value=True),
        #             gr.Dropdown(choices=[], value=None)#, allow_custom_value=True)
        #         )
        #     except Exception as e:
        #         print(f"Error refreshing dropdowns: {e}")
        #         return gr.Dropdown(), gr.Dropdown()
            
        # reload_codebook_btn_2.click(fn=refresh_annotation_dropdowns, 
        #                             outputs=[code_select, value_select])
        
        
        
        ################################################
        # Review tab
        ################################################

            
        def get_autofill_summary(index):
            """Get separate summaries for categorical and free-text annotations"""
            try:
                if annotator.df is None or index >= len(annotator.df):
                    return "", ""
                    
                with open(annotator.codebook_path, 'r') as f:
                    codebook = json.load(f)
                    
                categorical_summary = []
                freetext_summary = []
                
                for column in annotator.df.columns:
                    if column.startswith('autofill_'):
                        value = annotator.df.iloc[index][column]
                        if pd.notna(value):
                            clean_col = column.replace('autofill_', '')
                            
                            # Find attribute type from codebook
                            for code in codebook['codes']:
                                if code['attribute'] == clean_col:
                                    if code['type'] == 'categorical':
                                        for cat in code['categories']:
                                            if cat['category'] == value:
                                                icon = cat.get('icon', '')
                                                categorical_summary.append(f"[b][u]{clean_col}[/u][/b]: {icon} {value}<br>")
                                                break
                                    else:
                                        # Format free-text with BBCode for bold and underline
                                        freetext_summary.append(f"[b][u]{clean_col}[/u][/b]: {value}<br><br>")
                                    break
                                    
                return (
                    "\n".join(categorical_summary) if categorical_summary else "No categorical annotations",
                    "\n".join(freetext_summary) if freetext_summary else "No free-text annotations"
                )
                
            except Exception as e:
                print(f"Error getting autofill summary: {e}")
                return "Error loading categorical annotations", "Error loading free-text annotations"
                

        def navigate_transcripts(direction):
            if annotator.df is None or annotator.selected_column is None:
                return None, "**Current Index:** 0", "", ""
            
            try:
                if direction == "next":
                    annotator.current_index = min(annotator.current_index + 1, len(annotator.df) - 1)
                else:
                    annotator.current_index = max(annotator.current_index - 1, 0)
                    
                text = annotator.df.iloc[annotator.current_index]['text']
                index_display = f"**Current Index:** {annotator.current_index}"
                categorical, freetext = get_autofill_summary(annotator.current_index)
                
                return text, index_display, categorical, freetext
                
            except Exception as e:
                print(f"Error navigating transcripts: {e}")
                return None, "**Current Index:** 0", "", ""
        

        def refresh_all_dropdowns():
            try:
                categories = [code["attribute"] for code in annotator.load_codebook()]
                return [gr.Dropdown(choices=categories) for _ in range(4)]
            except Exception as e:
                print(f"Error refreshing dropdowns: {e}")
                return [gr.Dropdown(choices=[]) for _ in range(4)]

        reload_codebook_btn.click(
            fn=refresh_all_dropdowns,
            outputs=[code_select1, code_select2, code_select3, code_select4]
        )

        # Event handler for category selection
        def update_value_choices(code_name):
            if not code_name:
                return gr.Radio(choices=[])
            try:
                codebook = annotator.load_codebook()
                selected_code = None
                for code in codebook:
                    if code['attribute'] == code_name:
                        selected_code = code
                        break
                
                if selected_code:
                    choices = []
                    for category in selected_code['categories']:
                        icon = category.get('icon', '')
                        label = f"{icon} {category['category']}" if icon else category['category']
                        choices.append(label)
                    return gr.Radio(choices=choices)
                return gr.Radio(choices=[])
            except Exception as e:
                print(f"Error updating value choices: {e}")
                return gr.Radio(choices=[])

        # Connect each code_select dropdown to its corresponding value_select
        code_select1.change(
            fn=update_value_choices,
            inputs=[code_select1],
            outputs=[value_select1]
        )
        code_select2.change(
            fn=update_value_choices,
            inputs=[code_select2],
            outputs=[value_select2]
        )
        code_select3.change(
            fn=update_value_choices,
            inputs=[code_select3],
            outputs=[value_select3]
        )
        code_select4.change(
            fn=update_value_choices,
            inputs=[code_select4],
            outputs=[value_select4]
        )


        ##### multi label
        # Event handlers for dynamic category selection
        def update_category_columns(num_cats):
            columns = [col1, col2, col3, col4]
            visibilities = [False] * 4
            for i in range(int(num_cats)):
                visibilities[i] = True
            return [gr.Column(visible=v) for v in visibilities]
        
        num_categories.change(
            fn=update_category_columns,
            inputs=[num_categories],
            outputs=[col1, col2, col3, col4]
        )

        # Event handler for category selection
        def update_value_choices_multi(code_name, dropdown_index):
            if not code_name:
                return gr.Radio(choices=[])
            try:
                codebook = annotator.load_codebook()
                selected_code = None
                for code in codebook:
                    if code['attribute'] == code_name:
                        selected_code = code
                        break
                
                if selected_code:
                    choices = []
                    for category in selected_code['categories']:
                        icon = category.get('icon', '')
                        label = f"{icon} {category['category']}" if icon else category['category']
                        choices.append(label)
                    return gr.Radio(
                        choices=choices,
                        label="Select Value",
                        value=None
                    )
                return gr.Radio(choices=[])
            except Exception as e:
                print(f"Error updating value choices: {e}")
                return gr.Radio(choices=[])

        # Connect each code_select dropdown to its corresponding value_select
        code_select1.change(
            fn=lambda x: update_value_choices_multi(x, 1),
            inputs=[code_select1],
            outputs=[value_select1]
        )
        code_select2.change(
            fn=lambda x: update_value_choices_multi(x, 2),
            inputs=[code_select2],
            outputs=[value_select2]
        )
        code_select3.change(
            fn=lambda x: update_value_choices_multi(x, 3),
            inputs=[code_select3],
            outputs=[value_select3]
        )
        code_select4.change(
            fn=lambda x: update_value_choices_multi(x, 4),
            inputs=[code_select4],
            outputs=[value_select4]
        )

        def save_multiple_annotations(code1, value1, code2, value2, code3, value3, code4, value4, num_cats):
            status_messages = []
            for i, (code, value) in enumerate([(code1, value1), (code2, value2), 
                                            (code3, value3), (code4, value4)], 1):
                if i <= int(num_cats):
                    if code and value:
                        status, _ = annotator.save_annotation(code, value)
                        status_messages.append(status)
            
            # Navigate to next transcript after saving
            text, index = annotator.navigate_transcripts("next")
            return ("\n".join(status_messages), 
                    text, 
                    f"**Current Index:** {index}", 
                    *get_autofill_summary(index))
        
        annotate_next_btn.click(
            fn=save_multiple_annotations,
            inputs=[
                code_select1, value_select1,
                code_select2, value_select2,
                code_select3, value_select3,
                code_select4, value_select4,
                num_categories
            ],
            outputs=[
                annotation_status,
                transcript_box,
                current_index_display,
                categorical_summary,
                freetext_summary
            ]
        )
            
        
        prev_btn.click(
            fn=lambda: navigate_transcripts("prev"),
            outputs=[transcript_box, current_index_display, categorical_summary, freetext_summary]
        )

        next_btn.click(
            fn=lambda: navigate_transcripts("next"),
            outputs=[transcript_box, current_index_display, categorical_summary, freetext_summary]
        )


        ##########################
        # Status tab
        ##########################
        def get_review_progress():
            """Calculate and display review progress"""
            try:
                if annotator.df is None:
                    return "No data loaded"
                
                total_rows = len(annotator.df)
                reviewed = annotator.df['is_reviewed'].sum()
                progress_pct = (reviewed / total_rows) * 100
                
                if reviewed == total_rows:
                    status_emoji = "🏆"
                    status_msg = "All rows reviewed!"
                else:
                    status_emoji = "👎"
                    status_msg = f"Progress: {reviewed}/{total_rows} rows reviewed"
                
                return f"## {status_emoji} {status_msg} ({progress_pct:.1f}%)"
            except Exception as e:
                return f"Error calculating progress: {str(e)}"

        def calculate_metrics(category):
            """Calculate classification metrics for selected category"""
            try:
                if annotator.df is None:
                    return "[b][u]Error[/u][/b]: No data loaded"
                    
                # Check if both user and autofill columns exist
                user_col = f"user_{category}"
                auto_col = f"autofill_{category}"
                
                if user_col not in annotator.df.columns or auto_col not in annotator.df.columns:
                    return "[b][u]Error[/u][/b]: Category not fully annotated by both user and LLM"
                
                # Get non-null values where both annotations exist
                mask = annotator.df[user_col].notna() & annotator.df[auto_col].notna()
                if not mask.any():
                    return "[b][u]Error[/u][/b]: No complete annotations found"
                    
                y_true = annotator.df[user_col][mask]
                y_pred = annotator.df[auto_col][mask]
                
                # Calculate metrics
                metrics = [
                    f"[b][u]Accuracy[/u][/b]: {accuracy_score(y_true, y_pred):.3f}",
                    f"[b][u]Macro F1[/u][/b]: {f1_score(y_true, y_pred, average='macro'):.3f}",
                    f"[b][u]Weighted F1[/u][/b]: {f1_score(y_true, y_pred, average='weighted'):.3f}",
                    f"[b][u]Samples Compared[/u][/b]: {len(y_true)}",
                    f"[b][u]Agreement Rate[/u][/b]: {(y_true == y_pred).mean():.3f}"
                ]
                
                return "<br><br>".join(metrics)
                
            except Exception as e:
                return f"[b][u]Error[/u][/b]: {str(e)}"

        def update_category_choices():
            """Update the category dropdown with available categories"""
            try:
                categories = [code["attribute"] for code in annotator.load_codebook()]
                return gr.Dropdown(choices=categories)
            except Exception as e:
                return gr.Dropdown(choices=[])

        # Add the event handlers
        refresh_stats_btn.click(
            fn=calculate_metrics,
            inputs=[category_select],
            outputs=[metrics_display]
        )


        def refresh_status_categories():
            """Updates the category dropdown in the Status tab"""
            try:
                categories = [code["attribute"] for code in annotator.load_codebook()]
                return gr.Dropdown(choices=categories)
            except Exception as e:
                print(f"Error refreshing status categories: {e}")
                return gr.Dropdown(choices=[])

        # Add the click handler
        refresh_codebook_btn_3.click(
            fn=refresh_status_categories,
            outputs=[category_select]
        )

        # Update progress when tab is selected
        demo.load(
            fn=get_review_progress,
            outputs=[review_progress]
        )

        # Update category choices when codebook changes
        demo.load(
            fn=update_category_choices,
            outputs=[category_select]
        )



        ##########################
        # Download tab
        ##########################

        download_btn.click(fn=annotator.save_excel, 
                           outputs=[download_output, download_status])
        
        codebook_download_btn.click(fn=annotator.save_codebook,
                        outputs=[codebook_output, download_status])
        
        return demo