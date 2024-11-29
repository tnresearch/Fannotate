import gradio as gr

def create_download_tab(annotator):
    """Creates and returns the download tab interface"""
    with gr.Tab("💾 Download", id="download_tab"):
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
            
        with gr.Row():
            download_status = gr.Textbox(label="Status", interactive=False)

        ############################################################
        # Event handlers
        ############################################################

        download_btn.click(
            fn=annotator.save_excel,
            outputs=[download_output, download_status]
        )
        
        codebook_download_btn.click(
            fn=annotator.save_codebook,
            outputs=[codebook_output, download_status]
        )

        return {
            'download_output': download_output,
            'codebook_output': codebook_output,
            'download_status': download_status
        } 