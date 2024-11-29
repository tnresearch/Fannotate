import gradio as gr
from gradio_rich_textbox import RichTextbox
from ..utils.display import clean_column_name
from .review_handlers import (
    update_value_choices_multi,
    get_autofill_summary,
    navigate_transcripts,
    save_multiple_annotations
)

def create_review_tab(annotator, demo=None):
    """Creates and returns the review tab interface"""
    with gr.Tab("✏️ Review", id="review_tab") as review_tab:
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
                value_select1_radio = gr.Radio(label="(2) Select Value", choices=[], interactive=True, visible=False)
                value_select1_text = gr.Textbox(label="(2) Enter Value", interactive=True, visible=False)
            
            with gr.Column(visible=False) as col2:
                code_select2 = gr.Dropdown(label="(1) Select Category", choices=[], interactive=True)
                value_select2_radio = gr.Radio(label="(2) Select Value", choices=[], interactive=True, visible=False)
                value_select2_text = gr.Textbox(label="(2) Enter Value", interactive=True, visible=False)
            
            with gr.Column(visible=False) as col3:
                code_select3 = gr.Dropdown(label="(1) Select Category", choices=[], interactive=True)
                value_select3_radio = gr.Radio(label="(2) Select Value", choices=[], interactive=True, visible=False)
                value_select3_text = gr.Textbox(label="(2) Enter Value", interactive=True, visible=False)
            
            with gr.Column(visible=False) as col4:
                code_select4 = gr.Dropdown(label="(1) Select Category", choices=[], interactive=True)
                value_select4_radio = gr.Radio(label="(2) Select Value", choices=[], interactive=True, visible=False)
                value_select4_text = gr.Textbox(label="(2) Enter Value", interactive=True, visible=False)
        
        with gr.Row():
            with gr.Column():
                index_slider = gr.Slider(
                    minimum=0,
                    maximum=100,  # Default value, will be updated
                    step=1,
                    value=0,
                    label="Navigate to Index",
                    interactive=True
                )
            with gr.Column():
                reload_codebook_btn = gr.Button("🔄 Refresh Categories", variant="secondary")
                annotate_next_btn = gr.Button("Annotate and continue to next!", variant="primary")#, size="lg")

        with gr.Row():
            gr.Markdown("## 🤖 LLM suggestions")
        
        # Then use it with elem_classes
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    categorical_summary = RichTextbox(
                        label="Categorical Auto-fill Annotations:",
                        interactive=False
                    )
                with gr.Column():
                    freetext_summary = RichTextbox(
                        label="Free-text Auto-fill Annotations:",
                        interactive=True
                    )
        
        with gr.Row():
            #annotation_status = gr.Textbox(label="Annotation Status", interactive=False)
            annotation_status = gr.Markdown("**Annotation Status:** Not annotated")
        
        with gr.Row():
            gr.Markdown("## Original text content")
        
        with gr.Row():
            current_index_display = gr.Markdown("**Current Index:** 0")
        
        with gr.Row():
            transcript_box = RichTextbox(label="Text content", interactive=False)
        

        ############################################################
        # Event handlers for dynamic category selection
        ############################################################
        
        def update_category_columns(num_cats):
            """Updates visibility of category columns based on selection"""
            columns = [col1, col2, col3, col4]
            visibilities = [False] * 4
            for i in range(int(num_cats)):
                visibilities[i] = True
            return [gr.Column(visible=v) for v in visibilities]

        def refresh_all_dropdowns():
            """Refreshes all category dropdowns with current codebook values"""
            try:
                categories = [code["attribute"] for code in annotator.load_codebook()]
                return [gr.Dropdown(choices=categories) for _ in range(4)]
            except Exception as e:
                print(f"Error refreshing dropdowns: {e}")
                return [gr.Dropdown(choices=[]) for _ in range(4)]

        # Connect event handlers
        num_categories.change(
            fn=update_category_columns,
            inputs=[num_categories],
            outputs=[col1, col2, col3, col4]
        )

        reload_codebook_btn.click(
            fn=refresh_all_dropdowns,
            outputs=[code_select1, code_select2, code_select3, code_select4]
        )

        ############################################################
        # Event handlers
        ############################################################
        
        def jump_to_index(index):
            """Navigate to specific index using slider"""
            if annotator.df is None or annotator.selected_column is None:
                return None, "**Current Index:** 0", "", ""
            try:
                annotator.current_index = int(index)
                text = annotator.df.iloc[annotator.current_index]['text']
                categorical, freetext = get_autofill_summary(annotator, annotator.current_index)
                return text, f"**Current Index:** {annotator.current_index}", categorical, freetext
            except Exception as e:
                print(f"Error jumping to index: {e}")
                return None, "**Current Index:** 0", "", ""

        # Connect slider event
        index_slider.change(
            fn=jump_to_index,
            inputs=[index_slider],
            outputs=[
                transcript_box,
                current_index_display,
                categorical_summary,
                freetext_summary
            ]
        )

        # Connect annotation handlers
        annotate_next_btn.click(
            fn=lambda *args: save_multiple_annotations(annotator, *args),
            inputs=[
                code_select1, value_select1_radio, value_select1_text,
                code_select2, value_select2_radio, value_select2_text,
                code_select3, value_select3_radio, value_select3_text,
                code_select4, value_select4_radio, value_select4_text,
                num_categories
            ],
            outputs=[
                annotation_status,
                transcript_box,
                current_index_display,
                categorical_summary,
                freetext_summary,
                index_slider
            ]
        )

        # Connect value selection handlers for each category
        for code_select_n, radio_n, text_n in [
            (code_select1, value_select1_radio, value_select1_text),
            (code_select2, value_select2_radio, value_select2_text),
            (code_select3, value_select3_radio, value_select3_text),
            (code_select4, value_select4_radio, value_select4_text),
        ]:
            code_select_n.change(
                fn=lambda x: update_value_choices_multi(annotator, x),
                inputs=[code_select_n],
                outputs=[radio_n, text_n]
            )

        # Add a function to get initial data
        def get_initial_data():
            """Get initial data for the first entry"""
            if annotator.df is None or annotator.selected_column is None:
                return None, "**Current Index:** 0", "", ""
            try:
                text = annotator.df.iloc[0]['text']
                categorical, freetext = get_autofill_summary(annotator, 0)
                return text, "**Current Index:** 0", categorical, freetext
            except Exception as e:
                print(f"Error loading initial data: {e}")
                return None, "**Current Index:** 0", "", ""

        # Add tab selection event
        review_tab.select(
            fn=get_initial_data,
            outputs=[
                transcript_box,
                current_index_display,
                categorical_summary,
                freetext_summary
            ]
        )

        return {
            'code_select': code_select,
            'value_select': value_select,
            'transcript_box': transcript_box,
            'categorical_summary': categorical_summary,
            'freetext_summary': freetext_summary,
            'current_index_display': current_index_display,
            'annotation_status': annotation_status,
            'index_slider': index_slider
        } 