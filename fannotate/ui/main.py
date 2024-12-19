import gradio as gr
from fannotate.annotator import TranscriptionAnnotator
from .tabs.upload import create_upload_tab
from .tabs.settings import create_settings_tab
from .tabs.codebook import create_codebook_tab
from .tabs.autofill import create_autofill_tab
from .tabs.review import create_review_tab
from .tabs.status import create_status_tab
from .tabs.download import create_download_tab
from .utils.display import process_df_for_display

def create_ui():
    """Creates and returns the main Fannotate UI"""
    theme = gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="stone",
    )
    demo = gr.Blocks(theme=theme)
    #demo = gr.Blocks(theme='SebastianBravo/simci_css')
    with demo:
        # Add header markdown above tabs
        gr.Markdown("# 📝 Fannotate v0.1")
        
        annotator = TranscriptionAnnotator()
        
        # Create tabs - reordered to put settings after upload
        with gr.Tabs():
            upload_components = create_upload_tab(annotator)
            settings_components = create_settings_tab(annotator)
            codebook_components = create_codebook_tab(annotator)
            autofill_components = create_autofill_tab(annotator)
            review_components = create_review_tab(annotator, demo)
            status_components = create_status_tab(annotator)
            download_components = create_download_tab(annotator)
            
            # Connect cross-tab interactions
            def load_data(sheet, column):
                """Load data with selected sheet and column"""
                try:
                    status, preview = annotator.load_settings(sheet, column)
                    current_codebook = annotator.load_codebook()
                    codes = [code["attribute"] for code in current_codebook]
                    
                    # Update slider maximum when data is loaded
                    max_index = len(preview) - 1 if preview is not None else 100
                    
                    return (
                        status,
                        process_df_for_display(preview),
                        review_components['code_select'],
                        review_components['value_select'],
                        autofill_components['llm_code_select'],
                        current_codebook,
                        gr.Slider(maximum=max_index)  # Update slider maximum
                    )
                except Exception as e:
                    return (
                        f"Error loading data: {str(e)}",
                        None,
                        gr.Dropdown(choices=[]),
                        gr.Dropdown(choices=[]),
                        gr.Dropdown(choices=[]),
                        [],
                        gr.Slider(maximum=100)  # Default slider maximum
                    )

            # Connect load data button
            upload_components['load_data_btn'].click(
                fn=load_data,
                inputs=[
                    upload_components['sheet_select'],
                    upload_components['column_select']
                ],
                outputs=[
                    upload_components['upload_status'],
                    upload_components['preview_df'],
                    review_components['code_select'],
                    review_components['value_select'],
                    autofill_components['llm_code_select'],
                    codebook_components['codes_display'],
                    review_components['index_slider']  # Add slider to outputs
                ]
            )

    return demo 