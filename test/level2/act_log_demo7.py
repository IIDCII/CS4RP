"""
act log 5 but getting the average based on the activations but based on all of the layers
"""

import os
# set the GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# setting for vllm inference so that it can run in parallel
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
    
    def analyze_text(self, data, top_k=1000):
        self.activations.clear()
        tally = {}
        results = {}

        # runs through all the training data
        for i, text in enumerate(tqdm(data, desc="Processing texts", unit="text"), ):

            inputs = self.tokenizer(
                text["text"],
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
                    torch.abs(activation) <= 0.1, 
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
    
    def visualize_activations(self, results, layer_name):
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(results[layer_name]["values"])), 
                results[layer_name]["values"])
        plt.title(f"Top Neuron Activations in {layer_name}")
        plt.xlabel("Neuron Index")
        plt.ylabel("Mean Activation")
        plt.show()

# loading the model
base_model_name = "Llama-3.1-8B-Instruct"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# loading the data
data_path = "data/Philosophy,1970-2022"
dataset = load_from_disk(data_path)
dataset = dataset[:1000]["text"]
dataset = [{"text": text} for text in dataset]
dataset = Dataset.from_list(dataset)

# active neuron eval
base_analyser = ActivationAnalyser(base_model, tokenizer)

# get the topk for that single node given maths
# k set to 1000
bf = base_analyser.analyze_text(dataset, top_k=1000)

# store all of the results
# make sure the file name is correct
with open('topk/base_philosophy.pkl', 'wb') as f:
    pickle.dump(bf, f)

with open('topk/base_philosophy.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

print ("process complete")