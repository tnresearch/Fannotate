def add_attribute_to_codebook(annotator, name, description, attr_type, instruction):
    """Adds a new attribute to the existing codebook"""
    try:
        if not name:
            return ("Error: Attribute name is required", None, 
                name, description, attr_type, instruction)
            
        # Load existing codebook
        if not annotator.codebook_path.exists():
            annotator.create_new_codebook()
            
        with open(annotator.codebook_path, 'r') as f:
            codebook = json.load(f)
            
        # Check if attribute already exists
        if any(code['attribute'] == name for code in codebook['codes']):
            return (f"Error: Attribute '{name}' already exists", None,
                name, description, attr_type, instruction)
            
        # Create new attribute
        new_attribute = {
            "attribute": name,
            "description": description,
            "type": attr_type,
            "instruction": instruction,
            "categories": []
        }
        
        # Add to codebook
        codebook['codes'].append(new_attribute)
        
        # Save updated codebook
        with open(annotator.codebook_path, 'w') as f:
            json.dump(codebook, f, indent=4)
            
        # Return success message and clear all inputs
        return (f"Successfully added attribute: {name}", codebook,
                "", "", "categorical", "")
        
    except Exception as e:
        return (f"Error adding attribute: {str(e)}", None,
                name, description, attr_type, instruction)

def add_category_to_attribute(annotator, attribute_name, category, description, icon):
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

def update_attribute_choices(annotator):
    """Updates the attribute dropdown with current codebook attributes"""
    try:
        codebook = annotator.load_codebook()
        attributes = [code["attribute"] for code in codebook]
        return gr.Dropdown(choices=attributes)
    except Exception as e:
        print(f"Error updating attribute choices: {e}")
        return gr.Dropdown(choices=[]) 