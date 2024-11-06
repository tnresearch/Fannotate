# Fannotate User Guide

This guide will help you get started with the annotation process.

### 1. Upload Data
- Upload your Excel file containing the text to annotate
- Select the appropriate sheet and column containing the text
- Click "Create annotation table" to prepare the data

### 2. Configure Settings
Set up connection to the Language Model (LLM). The default endpoint is: http://172.16.16.48:8000/v1/ which requires no API key, however, the correct model needs to be specified. Remember that OpenAI models do not permit use for data annotation.

### 3. Manage Codebook
- Upload an existing codebook OR
- Create a new codebook with your annotation categories
- Each category should have predefined valid values

### 4. Auto-annotation
- Select a category to auto-annotate
- Generate a prompt for the LLM
- Run auto-annotation to get initial labels
- Review the results in the Review tab

### 5. Custom Annotation
- Create custom annotation categories
- Write specific prompts for the LLM
- Generate additional annotations for comparison

### 6. Review & Edit
- Go through annotations one by one
- Approve or correct auto-generated labels
- Add manual annotations as needed
- Track progress with the status indicators

### 7. Export Results
- Download the complete annotated dataset
- Save your codebook for future use

## Tips
- Always backup your codebook before making major changes
- Review auto-annotations carefully
- Use custom annotations to validate auto-annotations
- Save progress regularly by downloading intermediate results