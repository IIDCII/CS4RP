"""
Get the topk nodes for the first single point in the maths MMLU set
just realised that you should just take the k=1000 and then you can crop it later since all of it is in order
"""

import os
# set the GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
# setting for vllm inference so that it can run in parallel
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datasets import load_dataset

class ActivationAnalyzer:
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
    
    def analyze_text(self, prompts, top_k=1000, data_type = "test"):
        self.activations.clear()
        
        for text in prompts:
            # print progress
            print(f"Processing text {text}")

            # format the prompt
            if data_type == 'test':
                question = text['question']
                choices = text['choices']
            elif data_type == 'train':
                question = text['train']['question']
                choices = text['train']['choices']

            prompt = (
            f"Question: {question}\n\nChoices:\n"
            f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n"
            f"Do not explain and give the answer with strictly one letter from options A, B, C, or D.\nAnswer:"
            )

            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
        results = {}
        for name, activation in self.activations.items():
            # if abs(activation) <= 0.02 then set to 0, else set to 1
            activation[abs(activation) <= 0.02] = 0
            activation[activation != 0] = 1

            # get the tally
            tally = activation.abs().sum(dim=(0, 1))
            
            # Get top-k neurons (neurons with the highest tally)
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

# removes all values from dict1 that's in dict2 to isolate the the most used transferred
def remove_common_values(dict1, dict2):
    print ("total amount of weights: ", sum(len(d['indices']) for d in dict1.values()))
    removing = 0
    for name in dict1:
        if name in dict2:
            indices_to_remove = set(dict2[name]['indices']) & set(dict1[name]['indices'])
            removing += len(indices_to_remove)
            dict1[name]['values'] = [v for i, v in zip(dict1[name]['indices'], dict1[name]['values']) 
                                    if i not in indices_to_remove]
            dict1[name]['indices'] = [i for i in dict1[name]['indices'] if i not in indices_to_remove]
    
    print ("Removing ",removing, " from the total")
    print ("final number of weights: ", sum(len(d['indices']) for d in dict1.values()))
    return dict1

# merges all the values in dict1 and dict2
def merge_indices(dict1, dict2):
    result = dict1.copy()
    for name in dict2:
        if name in result:
            # Create dictionary with index:value pairs, dict2 values override dict1
            combined = dict(zip(result[name]['indices'], result[name]['values']))
            combined.update(dict(zip(dict2[name]['indices'], dict2[name]['values'])))
            # Sort and split back into separate lists
            sorted_items = sorted(combined.items())
            result[name]['indices'], result[name]['values'] = zip(*sorted_items)
        else:
            result[name] = dict2[name]
    print ("total amount of weights in dict1: ", sum(len(d['indices']) for d in dict1.values()))
    print ("total amount of weights in dict2: ", sum(len(d['indices']) for d in dict2.values()))
    print ("total amount of weights in returned: ", sum(len(d['indices']) for d in result.values()))
    return result

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
data_name = "cais/mmlu"
subset_name = "auxiliary_train"

dataset = load_dataset(data_name, subset_name, split = "train")
dataset = dataset.select(range(1000))

# active neuron eval
base_analyzer = ActivationAnalyzer(base_model, tokenizer)

# get the topk for that single node given maths
# k set to 1000
bf = base_analyzer.analyze_text(dataset, top_k=1000, data_type = 'train')

# store all of the results
# make sure the file name is correct
with open('topk/base_auxt.pkl', 'wb') as f:
    pickle.dump(bf, f)

with open('topk/base_auxt.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)


print ("process complete")