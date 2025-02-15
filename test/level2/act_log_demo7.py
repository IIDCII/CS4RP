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

class ActivationAnalyser:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.activations = {}
        self._register_hooks()
    
    def _activation_hook(self, name):
        def hook(module, input, output):
            if name not in self.activations:
                self.activations[name] = output.detach()
            self.activations[name] = self.activations[name] + output.detach()
        return hook
    
    def _register_hooks(self):
        for name, module in self.model.named_modules():
            # eval specific layers
            if "mlp." in name:
                module.register_forward_hook(self._activation_hook(name))  
    
    def analyze_text(self, data, top_k=1000):
        self.activations.clear()
        
        # runs through all the training data
        for i,text in enumerate(data):
            # print progress bar out of total
            print(f"Processing text {i+1}/{len(data)}")
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
                activation[abs(activation)%1 <= 0.15] = 0
                activation[activation != 0] += 1
                
            # unload inputs and ouputs from gpu
            del inputs
            del outputs
            torch.cuda.empty_cache()

        results = {}

        for name, activation in self.activations.items():
            # get the activation
            tally = activation.abs().mean(dim=(0, 1))

            top_values, top_indices = torch.topk(tally, top_k)
            
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
data_path = "data/Mathematics,1970-2002"
dataset = load_from_disk(data_path)
dataset = dataset[:10]["text"]
dataset = [{"text": text} for text in dataset]
dataset = Dataset.from_list(dataset)

# active neuron eval
base_analyser = ActivationAnalyser(base_model, tokenizer)

# get the topk for that single node given maths
# k set to 1000
bf = base_analyser.analyze_text(dataset, top_k=1000)

# store all of the results
# make sure the file name is correct
with open('topk/base_maths.pkl', 'wb') as f:
    pickle.dump(bf, f)

with open('topk/base_maths.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

print ("process complete")