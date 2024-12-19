import gradio as gr
import json
from ..utils.display import clean_column_name
from .codebook_handlers import (
    add_attribute_to_codebook,
    add_category_to_attribute,
    update_attribute_choices
)

def create_codebook_tab(annotator):
    """Creates and returns the codebook tab interface"""
    with gr.Tab("📓 Codebook"):
        with gr.Row():
            gr.Markdown("## Annotation codebook")
        
        with gr.Row():
            gr.Markdown("Upload existing codebook.")
        with gr.Row():
            codebook_upload = gr.File(label="Upload Codebook")
        
        with gr.Row():
            gr.Markdown("## Current codebook")
        with gr.Row():
            codes_display = gr.JSON(label="Current Codebook")                    

        ############################################################
        # Event handlers
        ############################################################

        def handle_codebook_upload(codebook_file):
            """Handle codebook file upload"""
            try:
                status = annotator.upload_codebook(codebook_file)
                current_codebook = annotator.load_codebook()
                codes = [code["attribute"] for code in current_codebook]
                return (
                    status,
                    current_codebook,
                    gr.Dropdown(choices=codes),
                    gr.Dropdown(choices=codes),
                    gr.Dropdown(choices=codes),
                )
            except Exception as e:
                return (
                    f"Error uploading codebook: {str(e)}",
                    [],
                    gr.Dropdown(choices=[]),
                    gr.Dropdown(choices=[]),
                    gr.Dropdown(choices=[])
                )

        # Connect event handlers
        codebook_upload.change(
            fn=handle_codebook_upload,
            inputs=[codebook_upload],
            outputs=[codes_display]
        )

        # Return all components that need to be accessed from other tabs
        return {
            'codes_display': codes_display,
            'codebook_upload': codebook_upload
        }