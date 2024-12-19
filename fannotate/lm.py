from openai import OpenAI
import json
import os
import requests
from requests.auth import HTTPBasicAuth
from constants import TN_GENAI_BASE_URL, TN_GENAI_TOKEN_URL

class LLMConfig:
    def __init__(self):
        self.base_url = "http://192.168.50.155:8000/v1/"
        self.api_key = "token-abc123"
        self.model = "google/gemma-2-2b-it"
        self.framework = "vLLM"  # Default framework
        self.max_tokens = 500    # Default max tokens
        self.temperature = 0.0   # Default temperature
        self.max_transcript_length = 500  # Default max transcript length in characters
        # PrivateGPT specific configs
        self.chat_id = None      # For PrivateGPT chat history
        self.history_size = 10   # Default history size for PrivateGPT

        # For private in-house Telenor gen-ai solution...
        self.token_url = TN_GENAI_TOKEN_URL
        self.api_client_id = os.getenv("GENAI_API_CLIENT_KEY")
        self.api_client_secret = os.getenv("GENAI_API_CLIENT_SECRET")
        self.genai_api_base_url = TN_GENAI_BASE_URL

    def update_config(self, framework=None, base_url=None, api_key=None, model=None, 
                     max_tokens=None, temperature=None, chat_id=None, history_size=None, 
                     agent_id=None, max_transcript_length=None):
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
        if chat_id is not None:
            self.chat_id = chat_id
        if history_size is not None:
            self.history_size = history_size
        if max_transcript_length is not None:
            self.max_transcript_length = max_transcript_length

config = LLMConfig()

def update_llm_config(
    framework=None,
    base_url=None,
    api_key=None,
    model=None,
    max_tokens=None,
    temperature=None,
    chat_id=None,
    history_size=None,
    max_transcript_length=None
):
    config.update_config(
        framework=framework,
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        chat_id=chat_id,
        history_size=history_size,
        max_transcript_length=max_transcript_length
    )

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
        
        if config.framework == "TN-GenAI-V1":
            # PrivateGPT specific implementation

            token = requests.post(
                config.token_url,
                data={
                    "grant_type": "client_credentials"
                },
                auth = HTTPBasicAuth(config.api_client_id, config.api_client_secret)
            ).json()["access_token"]
            
            values_str = ", ".join(values)
            modified_instruction = f"{instruction}\n\nYou must choose exactly one of these options: {values_str}"
            
            payload = {
                "message": modified_instruction,
                #"history_size": config.history_size,
                "model": config.model,
                #"seed": 1337
            }
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{config.genai_api_base_url}/chat",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            print(response.request)
            
            return data["response"]
            
        elif config.framework == "vLLM":
            client = OpenAI(
                base_url=config.base_url if config.framework == "vLLM" else "https://api.openai.com/v1/",
                api_key=config.api_key
            )
            
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


# Troubleshooting method
def format_prepped_request(prepped, encoding=None):
    # prepped has .method, .path_url, .headers and .body attribute to view the request
    encoding = encoding or requests.utils.get_encoding_from_headers(prepped.headers)
    body = prepped.body.decode(encoding) if encoding else '<binary data>' 
    headers = '\n'.join(['{}: {}'.format(*hv) for hv in prepped.headers.items()])
    return f"""\
{prepped.method} {prepped.path_url} HTTP/1.1
{headers}

{body}"""

def batch_process_transcripts(df, instruction, column_name, output_column, values=None):
    prep_request = None
    try:
        if config.framework == "TN-GenAI-V1":

            token = requests.post(
                config.token_url,
                data={
                    "grant_type": "client_credentials"
                },
                auth = HTTPBasicAuth(config.api_client_id, config.api_client_secret)
            ).json()["access_token"]
            
            results = []
            for idx, row in df.iterrows():
                transcript = str(row['text'])[:config.max_transcript_length]
                prompt = instruction.replace('<<text>>', transcript)
                
                payload = {
                    #"chat_id": config.chat_id,
                    "message": prompt,
                    #"history_size": config.history_size,
                    "model": config.model,
                    #"seed": 1337
                }
                
                if values:
                    # Add constrained options to the prompt
                    values_str = ", ".join(values)
                    payload["message"] = f"{prompt}\n\nPlease choose exactly one of these options: {values_str}"
                
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                }
                
                response = requests.post(
                    f"{config.genai_api_base_url}/chat",
                    json=payload,
                    headers=headers
                )
                prep_request = format_prepped_request(response.request, "utf8")
                print(prep_request)
                response.raise_for_status()
                data = response.json()
                
                results.append(data["response"])
                
            df[output_column] = results
            return df, "Processing completed successfully"
            
        else:
            client = OpenAI(
                base_url=config.base_url if config.framework == "vLLM" else "https://api.openai.com/v1/",
                api_key=config.api_key
            )
            
            # Convert comma-separated string to list and strip whitespace
            if values and isinstance(values, str):
                values = [v.strip() for v in values.split(',')]
            
            results = []
            for idx, row in df.iterrows():
                transcript = str(row['text'])[:config.max_transcript_length]
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
        print(f"Error in batch processing: {e}. Request sent was:\n{prep_request}")
        return None, f"Error in batch processing: {str(e)} Request sent was:\n{prep_request}"

    return None, f"Error in batch processing: {str(e)}"
