from openai import OpenAI
import json

class LLMConfig:
    def __init__(self):
        self.base_url = "http://192.168.50.155:8000/v1/"
        self.api_key = "token-abc123"
        self.model = "google/gemma-2-2b-it"
        self.framework = "vLLM"  # Default framework
        self.max_tokens = 500    # Default max tokens
        self.temperature = 0.0   # Default temperature

    def update_config(self, framework=None, base_url=None, api_key=None, model=None, max_tokens=None, temperature=None):
        if framework:
            self.framework = framework
        if base_url:
            self.base_url = base_url
        if api_key:
            self.api_key = api_key
        if model:
            self.model = model
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature

config = LLMConfig()

def update_llm_config(framework=None, base_url=None, api_key=None, model=None, max_tokens=None, temperature=None):
    config.update_config(
        framework=framework,
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )

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
        print(f"Current framework: {config.framework}")  # Debug line
        client = OpenAI(
            base_url=config.base_url if config.framework == "vLLM" else "https://api.openai.com/v1/",
            api_key=config.api_key
        )
        
        if config.framework == "vLLM":
            print("Using vLLM framework")  # Debug line
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
            print("Using OpenAI framework")  # Debug line
            values_str = ", ".join(values)
            modified_instruction = f"{instruction}\n\nYou must choose exactly one of these options: {values_str}"
            
            # Remove temperature for OpenAI calls
            completion = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": "You must respond with exactly one of the allowed values, nothing else."},
                    {"role": "user", "content": modified_instruction}
                ],
                max_tokens=50
            )
            
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error querying LLM: {str(e)}"

def batch_process_transcripts(df, instruction, column_name, output_column, values=None):
    try:
        client = OpenAI(
            base_url=config.base_url if config.framework == "vLLM" else "https://api.openai.com/v1/",
            api_key=config.api_key
        )
        
        # Convert comma-separated string to list and strip whitespace
        if values and isinstance(values, str):
            values = [v.strip() for v in values.split(',')]
        
        results = []
        for idx, row in df.iterrows():
            transcript = str(row['text'])[:500]
            # Replace <<text>> tag with actual transcript
            prompt = instruction.replace('<<text>>', transcript)
            
            if values:
                if config.framework == "vLLM":
                    completion = client.chat.completions.create(
                        model=config.model,
                        messages=[{"role": "user", "content": prompt}],
                        extra_body={"guided_choice": values}
                    )
                else:
                    # For OpenAI models
                    values_str = ", ".join(values)
                    modified_prompt = f"{prompt}\n\nPlease choose exactly one of these options: {values_str}"
                    completion = client.chat.completions.create(
                        model=config.model,
                        messages=[
                            {"role": "system", "content": "You must respond with exactly one of the allowed values, nothing else."},
                            {"role": "user", "content": modified_prompt}
                        ],
                        max_tokens=50
                    )
            else:
                # For unconstrained responses
                completion = client.chat.completions.create(
                    model=config.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500  # Add reasonable max_tokens limit for unconstrained responses
                )
            
            results.append(completion.choices[0].message.content)
            
        # Update DataFrame with results
        df[output_column] = results
        return df, "Processing completed successfully"
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return None, f"Error in batch processing: {str(e)}"