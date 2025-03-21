"""
going to classify the data based on the activations when probing for a specific document or test using maths philosophy and physics
if it can be able to classify beyond the random for any (33%) then it shows that the selected nodes are actually useful
"""

# setup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datasets import load_dataset
from datasets import load_from_disk
from datasets import Dataset
import time
from tqdm import tqdm
import random
import string

class ActivationAnalyser:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.activations = {}
        self._register_hooks()
    
    def _activation_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def _register_hooks(self):
        for name, module in self.model.named_modules():
            # eval specific layers
            if "mlp." in name:
                module.register_forward_hook(self._activation_hook(name))  
    
    def analyze_text(self, data, top_k=1000, data_type = "test"):
        self.activations.clear()
        tally = {}
        results = {}

        # runs through all the training data
        for i, text in enumerate(tqdm(data, desc="Processing texts", unit="text")):
            if data_type == "train":
                question = text['train']['question']
                choices = text['train']['choices']
                prompt = (
                f"Question: {question}\n\nChoices:\n"
                f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n"
                f"Do not explain and give the answer with strictly one letter from options A, B, C, or D.\nAnswer:"
                )
            else:
                prompt = text["text"]
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=4096,
                )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # add the activations per text to the tally
            for name, activation in self.activations.items():
                # setting the value to 1 or 2 whether it's greater or smaller than 0.1
                activation = torch.where(
                    torch.abs(activation) <= 1,
                    torch.tensor(1.0, device=activation.device), 
                    torch.tensor(2.0, device=activation.device)
                )
                
                # Sum over dimensions 0 and 1
                current_sum = activation.sum(dim=(0, 1))
                
                # Update tally with running average
                if i == 0:
                    tally[name] = current_sum
                else:
                    tally[name] = (tally[name] * i + current_sum) / (i + 1)
                 
            # unload inputs and ouputs from gpu
            del inputs
            del outputs
            torch.cuda.empty_cache()


        for name, subtal in tally.items():
            top_values, top_indices = torch.topk(subtal, top_k)

            results[name] = {
                "indices": top_indices.tolist(),
                "values": top_values.tolist()
            }
            
        return results
