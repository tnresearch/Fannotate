import gradio as gr
from ..utils.display import process_df_for_display

def create_upload_tab(annotator):
    """Creates and returns the upload tab interface"""
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
                                    row_count=(5, "fixed"))

        ############################################################
        # Event handlers
        ############################################################

        def handle_data_upload(file):
            """Handle data file upload"""
            try:
                status, sheets, _ = annotator.upload_file(file, None)
                return (
                    status,
                    gr.Dropdown(choices=sheets),
                    process_df_for_display(None)
                )
            except Exception as e:
                return (
                    f"Error uploading file: {str(e)}",
                    gr.Dropdown(choices=[]),
                    None
                )

        # Connect event handlers
        file_upload.change(
            fn=handle_data_upload,
            inputs=[file_upload],
            outputs=[upload_status, sheet_select, preview_df]
        )

        sheet_select.change(
            fn=lambda x: gr.Dropdown(choices=annotator.get_columns(x)),
            inputs=[sheet_select],
            outputs=[column_select]
        )

        return {
            'sheet_select': sheet_select,
            'column_select': column_select,
            'load_data_btn': load_data_btn,
            'upload_status': upload_status,
            'preview_df': preview_df
        } 