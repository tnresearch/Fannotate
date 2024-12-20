import os

TN_GENAI_BASE_URL = os.getenv("GENAI_API_BASE_URL")
TN_GENAI_TOKEN_URL = os.getenv("GENAI_API_TOKEN_URL")

BASE_URLS = {
    "vLLM": "http://192.168.50.155:8000/v1/",
    "OpenAI": "https://api.openai.com/v1/",
    "TN-GenAI-V1": TN_GENAI_BASE_URL
}


MODEL_CHOICES = {
    "vLLM": [
        'gpt-4',
        'gpt-4-turbo',
        'gpt-3.5-turbo',
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-405B-Instruct",
    ],
    "OpenAI": [
        'gpt-4',
        'gpt-4-turbo',
        'gpt-3.5-turbo',
    ],
    "TN-GenAI-V1": [
        "gpt-4"
    ]
}
