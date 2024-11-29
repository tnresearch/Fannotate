import gradio as gr
import pandas as pd

def update_value_choices_multi(annotator, code_name):
    """Updates the value selection component based on the attribute type"""
    if not code_name:
        return [
            gr.Radio(choices=[], visible=False),
            gr.Textbox(value="", visible=False)
        ]
    try:
        codebook = annotator.load_codebook()
        selected_code = None
        for code in codebook:
            if code['attribute'] == code_name:
                selected_code = code
                break
        
        if selected_code:
            # Check the attribute type
            is_categorical = selected_code.get('type', 'categorical') == 'categorical'
            
            if is_categorical:
                choices = []
                for category in selected_code['categories']:
                    icon = category.get('icon', '')
                    label = f"{icon} {category['category']}" if icon else category['category']
                    choices.append(label)
                return [
                    gr.Radio(choices=choices, label="Select Value", visible=True),
                    gr.Textbox(value="", visible=False)
                ]
            else:
                # For freetext type
                return [
                    gr.Radio(choices=[], visible=False),
                    gr.Textbox(label="Enter Value", visible=True)
                ]
        return [
            gr.Radio(choices=[], visible=False),
            gr.Textbox(value="", visible=False)
        ]
    except Exception as e:
        print(f"Error updating value choices: {e}")
        return [
            gr.Radio(choices=[], visible=False),
            gr.Textbox(value="", visible=False)
        ]

def get_autofill_summary(annotator, index):
    """Get separate summaries for categorical and free-text annotations"""
    try:
        if annotator.df is None or index >= len(annotator.df):
            return "", ""
            
        codebook = annotator.load_codebook()
            
        categorical_summary = []
        freetext_summary = []
        
        for column in annotator.df.columns:
            if column.startswith('autofill_'):
                value = annotator.df.iloc[index][column]
                if pd.notna(value):
                    clean_col = column.replace('autofill_', '')
                    
                    # Find attribute type from codebook
                    for code in codebook:
                        if code['attribute'] == clean_col:
                            if code['type'] == 'categorical':
                                for cat in code['categories']:
                                    if cat['category'] == value:
                                        icon = cat.get('icon', '')
                                        categorical_summary.append(f"[b][u]{clean_col}[/u][/b]: {icon} {value}<br>")
                                        break
                            else:
                                freetext_summary.append(f"[b][u]{clean_col}[/u][/b]: {value}<br><br>")
                            break
                    
        return (
            "\n".join(categorical_summary) if categorical_summary else "No categorical annotations",
            "\n".join(freetext_summary) if freetext_summary else "No free-text annotations"
        )
        
    except Exception as e:
        print(f"Error getting autofill summary: {e}")
        return "Error loading categorical annotations", "Error loading free-text annotations"

def navigate_transcripts(annotator, direction):
    """Navigate through transcripts and return updated values including current index"""
    if annotator.df is None or annotator.selected_column is None:
        return None, "**Current Index:** 0", "", "", 0
    try:
        if direction == "next":
            annotator.current_index = min(annotator.current_index + 1, len(annotator.df) - 1)
        else:
            annotator.current_index = max(annotator.current_index - 1, 0)
            
        text = annotator.df.iloc[annotator.current_index]['text']
        categorical, freetext = get_autofill_summary(annotator, annotator.current_index)
        
        return (
            text, 
            f"**Current Index:** {annotator.current_index}", 
            categorical, 
            freetext,
            annotator.current_index  # Return current index for slider
        )
    except Exception as e:
        print(f"Error navigating transcripts: {e}")
        return None, "**Current Index:** 0", "", "", 0

def save_multiple_annotations(annotator, code1, value1_radio, value1_text, 
                            code2, value2_radio, value2_text,
                            code3, value3_radio, value3_text,
                            code4, value4_radio, value4_text, 
                            num_cats):
    """Save multiple annotations and navigate to next transcript"""
    current_index = annotator.current_index  # Store current index before navigation
    status_messages = []
    
    for i, (code, radio_val, text_val) in enumerate([
        (code1, value1_radio, value1_text),
        (code2, value2_radio, value2_text),
        (code3, value3_radio, value3_text),
        (code4, value4_radio, value4_text)
    ], 1):
        if i <= int(num_cats):
            if code:
                # Determine if this is a categorical or freetext attribute
                codebook = annotator.load_codebook()
                attr_type = 'categorical'  # default
                for c in codebook:
                    if c['attribute'] == code:
                        attr_type = c.get('type', 'categorical')
                        break
                
                # For categorical type, use radio_val
                # For freetext type, use the entire text_val without any processing
                if attr_type == 'categorical':
                    value = radio_val
                else:
                    value = text_val  # Use raw text input without any processing

                if value:  # Check if there's any input
                    status, _ = annotator.save_annotation(code, value)
                    # Add index number to status message
                    status_messages.append(f"[Index {current_index}] {status}")

    # Navigate to next transcript after saving
    text, index = annotator.navigate_transcripts("next")
    categorical, freetext = get_autofill_summary(annotator, index)
    
    # Format status messages with double line breaks
    formatted_status = "**Annotation Status:**\n\n" + "\n\n".join(status_messages) if status_messages else "No annotations added"
    
    # Return updated values including the new slider value
    return (
        formatted_status, 
        text, 
        f"**Current Index:** {annotator.current_index}",
        categorical,
        freetext,
        gr.Slider(value=annotator.current_index)  # Update slider value
    ) 