import gradio as gr
from ..utils.display import clean_column_name
from ...constants import MODEL_CHOICES, BASE_URLS

def create_settings_tab(annotator):
    """Creates and returns the settings tab interface"""
    with gr.Tab("⚙️ Settings"):
        with gr.Row():
            gr.Markdown("## LLM Settings")

        
        def update_model_choices(framework_select):
            return gr.update(choices=MODEL_CHOICES[framework_select], value=None)
            
        with gr.Row():
            with gr.Column():
                framework_select = gr.Dropdown(
                    choices=["vLLM", "OpenAI", "TN-GenAI-V1"],
                    value="vLLM",
                    label="LLM Framework",
                    interactive=True
                )
                
                model_name = gr.Dropdown(
                    choices = MODEL_CHOICES["vLLM"],
                    label = "Model choices",
                    interactive = True
                )

                framework_select.change(fn=update_model_choices, inputs=[framework_select], outputs=[model_name])
                
            with gr.Column():
                max_tokens = gr.Number(
                    value=500,
                    label="Max Tokens",
                    interactive=True,
                    minimum=1,
                    maximum=2000,
                    step=1
                )

                max_transcript_length = gr.Number(
                    value=500,
                    label="Max Transcript Length (characters)",
                    info="Maximum number of characters to include from each transcript",
                    interactive=True,
                    minimum=100,
                    maximum=10000,
                    step=100
                )

        with gr.Row(visible=True) as default_settings:
            temperature = gr.Slider(
                value=0.0,
                minimum=0.0,
                maximum=2.0,
                step=0.01,
                label="Temperature",
                interactive=True
            )

            api_key = gr.Textbox(
                value="token-abc123",
                label="API Key",
                interactive=True,
                type="password"
            )

        # Add PrivateGPT specific settings
        with gr.Row(visible=False) as privategpt_settings:
            with gr.Column():
                chat_id = gr.Textbox(
                    value="",
                    label="Chat ID (optional)",
                    interactive=True
                )
            with gr.Column():
                history_size = gr.Number(
                    value=10,
                    label="History Size",
                    interactive=True,
                    minimum=1,
                    maximum=100
                )
            """with gr.Column():
                agent_id = gr.Textbox(
                    value="",
                    label="Agent ID",
                    interactive=True
                )
            """

        with gr.Row():
            apply_llm_settings_btn = gr.Button("Apply LLM Settings", variant="primary")
            settings_status = gr.Textbox(label="Settings Status", interactive=False)

        # Help sections
        with gr.Row():
            gr.Markdown("<br><br>")
        
        with gr.Row():
            gr.Markdown("## LLM Settings Help")

        # with gr.Group():
        copntainer_onoff=False
        with gr.Row():
            gr.Markdown("""**LLM Framework:** 
                        - This setting lets you choose between vLLM (Open-source models) and OpenAI/Azure hosted models.""", container=copntainer_onoff)
        
        with gr.Row():
            gr.Markdown("""**Endpoint URL:** 
                        - This setting lets you specify the URL of the LLM endpoint. If you are using vLLM, this will be the IP address of the vLLM server. If you are using Azure/OpenAI, this will be the API endpoint URL.""", container=copntainer_onoff)
        
        with gr.Row():
            gr.Markdown("""**Model Name:** 
                        - This setting lets you specify which model to use. Models not on the list can also be used by manually typing in the name. It is suggested to use *permissively licensed* LLMs that allow distillation/creation of data for training of language models.""", container=copntainer_onoff)
        
        with gr.Row():
            gr.Markdown("""**API Key:** 
                        - This setting lets you specify the API key to use for the LLM. If you are using OpenAI or Azure, this will be your OpenAI API key/token. If you are using vLLM, leave this field as is.""", container=copntainer_onoff)
        
        with gr.Row():
            gr.Markdown("""**Max Tokens:** 
                        - The maximum number of tokens (words/subwords) that the model will generate in its response. Higher values allow for longer responses but use more computational resources. For most annotation tasks, 500 tokens is sufficient.""", container=copntainer_onoff)
        
        with gr.Row():
            gr.Markdown("""**Temperature:** 
                        - Controls the randomness in the model's responses. A value of 0.0 makes the model more deterministic, always choosing the most likely next token. Higher values (up to 2.0) make the output more random and 'creative'. For annotation tasks, lower values (0.0-0.3) are recommended for consistency.""", container=copntainer_onoff)

        ############################################################
        # Event handlers
        ############################################################

        def apply_settings(framework, model, max_tokens_val, temp_val, 
                         chat_id_val, history_size_val, max_transcript_length_val):
            try:
                from fannotate.lm import update_llm_config
                update_llm_config(
                    framework=framework,
                    base_url=BASE_URLS[framework],
                    api_key=None,
                    model=model,
                    max_tokens=int(max_tokens_val),
                    temperature=float(temp_val),
                    chat_id=chat_id_val,
                    history_size=int(history_size_val) if history_size_val else None,
                    max_transcript_length=int(max_transcript_length_val)
                )
                return "Settings applied successfully"
            except Exception as e:
                return f"Error applying settings: {str(e)}"

        # Show/hide PrivateGPT settings based on framework selection
        def update_tn_genai_settings(framework):
            return gr.Row(visible=(framework == "TN-GenAI-V1"))

        def update_default_settings(framework):
            return gr.Row(visible=(framework != "TN-GenAI-V1"))

        framework_select.change(
            fn=update_tn_genai_settings,
            inputs=[framework_select],
            outputs=[privategpt_settings]
        )
        framework_select.change(
            fn=update_default_settings,
            inputs=[framework_select],
            outputs=[default_settings]
        )

        # Connect event handlers
        apply_llm_settings_btn.click(
            fn=apply_settings,
            inputs=[
                framework_select,
                #api_key,
                model_name,
                max_tokens,
                temperature,
                chat_id,
                history_size,
                max_transcript_length
            ],
            outputs=[settings_status]
        )

        return {
            'framework_select': framework_select,
            #'base_url': base_url,
            #'api_key': api_key,
            'model_name': model_name,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'chat_id': chat_id,
            'history_size': history_size,
            'max_transcript_length': max_transcript_length,
            'settings_status': settings_status
        } 
