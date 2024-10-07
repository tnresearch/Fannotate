import pandas as pd
from tqdm import tqdm
import logging
import torch
import time
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
#import outlines
#from outlines import models
import requests
import json

class PromptGenerator:
    """
    This class generates a prompt template using model and prompt specs, and is re-used for encoding the various tasks.
    It has some static elements that is re-used, while the task from the dataset is changing.
    This expected to be re-initialized between models, prompts and datasets, while remaining static across tasks.
    """
    def __init__(self, dataset_name: str, prompt_name:str, test_mode:str,  model_name: str, 
                 tokenizer: object, 
                 device: object):
        
        self.dataset_name = dataset_name
        self.prompt_name = prompt_name
        self.test_mode = test_mode
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device

        # load model attributes based on model name
        from utils.data import load_json
        self.model_attributes = load_json('configs/models.json')["models"][self.model_name]
        
        {
            "model_dest": "google/gemma-2-2b-it",
            "server":"transformers",
            "model_type": "instruction",
            "chat_format": "chatml",
            "instruction_role": "user",
            "quantized":False,
            "gated":False
        }
         # load prompt attributes based on prompt name
        from utils.data import load_json
        self.prompt_attributes = load_json('configs/prompts.json')[self.dataset_name][self.prompt_name]
        {
            "sysprompt": "### Transcript: Jeg skal gjerne si opp abonnementet mitt. #### Category: ",
            "model_type": "base",
            "prompt_type": "n_shot",
            "dataset_name": "Testdata"
        }

        logging.info(f"Loaded prompt attributes: {self.prompt_name}")

    def create_and_encode_prompt(self, task, refine=False):
        """Creates the prompt, based on the settings needed"""

        # Check if this is the first prompt in a chain or not
        if refine == False:
            # main instruction to be given
            sysprompt = self.prompt_attributes["sysprompt"]
        if refine == True:
            # follow-up on sysprompt and result
            part1 = self.prompt_attributes["refine_prompt"]["inst_1"]            
            # the refine instruction now becomes the main instruction, and task becomes previous answer (passed as task)
            sysprompt = part1
        
        # Conditional formatting and tokenization using transformers
        if  self.model_attributes["server"] == "transformers":
            # For constrained generation no encoding is needed 
            if self.test_mode == "constrained":
                if self.model_attributes["instruction_role"] == "system":
                    # default condition
                    conversation = [
                                    {"role": "system", "content": "### Instruction:\n" + sysprompt},
                                    {"role": "user", "content": "### Input:\n" + task },
                                    ]
                if self.model_attributes["instruction_role"] == "user":
                    # special condition for Gemma-2
                    conversation = [
                                    {"role": "user", "content": "### Instruction:\n" + sysprompt + "\n\n ### Input:\n" + task },
                                    ]
                
                # early norwegian models
                if self.model_attributes["chat_format"] == "chatml":
                    formatted_prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                    inputs = formatted_prompt # same prompt as unconstrained without tokenization

                # early norwegian models
                if self.model_attributes["chat_format"] == "alpaca":
                    formatted_prompt = "### Instruction:\n" + sysprompt + "\n### Input:\n" + task + "\n### Response:\n"
                    inputs = formatted_prompt # same prompt as unconstrained without tokenization

                # models without a specified chat format
                if self.model_attributes["chat_format"] == "unknown":
                    formatted_prompt = sysprompt + task
                    inputs = formatted_prompt # same prompt as unconstrained without tokenization

            if self.test_mode == "unconstrained":
                # prompts can either consist of instructions or a pattern (n_shot)
                if self.prompt_attributes["prompt_type"] == "zero-shot":
                    if self.model_attributes["chat_format"] == "chatml":
                        #specify conversation
                        if self.model_attributes["instruction_role"] == "system":
                            # default condition
                            conversation = [
                                            {"role": "system", "content": "### Instruction:\n" + sysprompt},
                                            {"role": "user", "content": "### Input:\n" + task },
                                            ]
                                        
                        if self.model_attributes["instruction_role"] == "user":
                            # special condition for Gemma-2
                            conversation = [
                                                {"role": "user", "content": "### Instruction:\n" + sysprompt + "\n\n ### Input:\n" + task },
                                            ]

                    # newer models
                    if self.model_attributes["chat_format"] == "chatml":
                        formatted_prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

                    # early norwegian models
                    if self.model_attributes["chat_format"] == "alpaca":
                        formatted_prompt = "### Instruction:\n" + sysprompt + "\n### Input:\n" + task + "\n### Response:\n"
                        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

                    # models without a specified chat format
                    if self.model_attributes["chat_format"] == "unknown":
                        formatted_prompt = sysprompt + task
                        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        # Conditional formatting and tokenization using transformers
        if self.model_attributes["server"] == "llama.cpp":
            # prompts can either consist of instructions or a pattern (n_shot)
            if self.prompt_attributes["prompt_type"] == "zero-shot":
                if self.model_attributes["chat_format"] == "chatml":
                    #specify conversation
                    if self.model_attributes["instruction_role"] == "system":
                        # default condition
                        conversation = [
                                        {"role": "system", "content": "### Instruction:\n" + sysprompt},
                                        {"role": "user", "content": "### Input:\n" + task },
                                        ]
                        inputs = conversation
                                     
                    if self.model_attributes["instruction_role"] == "user":
                        # special condition for Gemma-2
                        conversation = [
                                            {"role": "user", "content": "### Instruction:\n" + sysprompt + "\n\n ### Input:\n" + task },
                                        ]
                        inputs = conversation

                # early norwegian models
                if self.model_attributes["chat_format"] == "alpaca":
                    formatted_prompt = "### Instruction:\n" + sysprompt + "\n### Input:\n" + task + "\n### Response:\n"
                    inputs = formatted_prompt

                # models without a specified chat format
                if self.model_attributes["chat_format"] == "unknown":
                    formatted_prompt = sysprompt + task
                    inputs = formatted_prompt

        
        # OpenAI API-based approach is separated from the above logic
        if self.model_attributes["server"] == "openai":
            formatted_prompt = ""
        
        # update previous prompt to the prompt used above
        self.prev_prompt = sysprompt
        return inputs