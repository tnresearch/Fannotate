from openai import OpenAI

class LLMConfig:
    def __init__(self):
        self.base_url = "http://192.168.50.155:8000/v1/"
        self.api_key = "token-abc123"
        self.model = "google/gemma-2-2b-it"
        self.framework = "vLLM"

config = LLMConfig()

def update_llm_config(framework=None, base_url=None, api_key=None, model=None):
    if framework:
        config.framework = framework
    if base_url:
        config.base_url = base_url
    if api_key:
        config.api_key = api_key
    if model:
        config.model = model

# def query_llm(instruction):
#     try:
#         client = OpenAI(
#             base_url=config.base_url,
#             api_key=config.api_key,
#         )
        
#         completion = client.chat.completions.create(
#             model=config.model,
#             seed=1337,
#             temperature=0.0,
#             messages=[
#                 {"role": "user", "content": instruction}
#             ]
#         )
#         return completion.choices[0].message.content
#     except Exception as e:
#         return f"Error querying LLM: {str(e)}"

# def query_constrained_llm(instruction, values):
#     try:
#         client = OpenAI(
#             base_url=config.base_url,
#             api_key=config.api_key,
#         )
        
#         completion = client.chat.completions.create(
#             model=config.model,
#             messages=[
#                 {"role": "user", "content": instruction}
#             ],
#             extra_body={
#                 "guided_choice": values
#             }
#         )
#         return completion.choices[0].message.content
#     except Exception as e:
#         return f"Error querying LLM: {str(e)}"

def query_llm(instruction):
    try:
        client = OpenAI(
            base_url=config.base_url if config.framework == "vLLM" else "https://api.openai.com/v1/",
            api_key=config.api_key
        )

        completion = client.chat.completions.create(
            model=config.model,
            seed=1337,
            temperature=0.0,
            messages=[
                {"role": "user", "content": instruction}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error querying LLM: {str(e)}"

def query_constrained_llm(instruction, values):
    try:
        client = OpenAI(
            base_url=config.base_url if config.framework == "vLLM" else "https://api.openai.com/v1/",
            api_key=config.api_key
        )

        if config.framework == "vLLM":
            completion = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "user", "content": instruction}
                ],
                extra_body={
                    "guided_choice": values
                }
            )
        else:
            # For OpenAI, use a different approach for constrained generation
            values_str = ", ".join(values)
            modified_instruction = f"{instruction}\n\nPlease choose one of the following options: {values_str}"
            completion = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "user", "content": modified_instruction}
                ]
            )

        return completion.choices[0].message.content
    except Exception as e:
        return f"Error querying LLM: {str(e)}"

def batch_process_transcripts(df, instruction, column_name, output_column, values=None):
    try:
        client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        
        # Convert comma-separated string to list and strip whitespace
        if values:
            if values and isinstance(values, str):
                values = [v.strip() for v in values.split(',')]
        
        results = []
        for idx, row in df.iterrows():
            # Use 'text' column instead of the original column name
            transcript = str(row['text'])[:500]
            prompt = f"{instruction}\n\nTranscript: {transcript}"
            
            if values:
                completion = client.chat.completions.create(
                    model=config.model,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"guided_choice": values}
                )
            else:
                completion = client.chat.completions.create(
                    model=config.model,
                    messages=[{"role": "user", "content": prompt}]
                )
            
            results.append(completion.choices[0].message.content)
        
        df[output_column] = results
        return df, "Processing completed successfully"
    except Exception as e:
        return None, f"Error in batch processing: {str(e)}"