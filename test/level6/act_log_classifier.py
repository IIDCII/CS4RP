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
import pickle
from datasets import load_from_disk
from datasets import Dataset
from tqdm import tqdm
import random

# hooks the model to get the activations
class ActivationAnalyser:
    def __init__(self, model, tokenizer, act_log):
        self.tokenizer = tokenizer
        self.model = model
        self.activations = {}
        self._register_hooks()
        self.act_log = act_log
    
    def _activation_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def _register_hooks(self):
        for name, module in self.model.named_modules():
            # eval specific layers
            if "mlp." in name:
                module.register_forward_hook(self._activation_hook(name))  
    
    def classify(self, data): 
        self.activations.clear()
        tally = {}
        results = {}
        total = len(data)
        ans_correct = 0

        # runs through all the training data
        for i, text in enumerate(tqdm(data, desc="Processing texts", unit="text")):
            prompt = text[0]["text"]
            
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
                
            for name, subtal in tally.items():
                top_values, top_indices = torch.topk(subtal, 1000)

                results[name] = {
                    "indices": top_indices.tolist(),
                    "values": top_values.tolist()
                }
                 
            # compare the activations and select the check to see if it's right
            self.compare_activations(text[1])

            # unload inputs and ouputs from gpu
            del inputs
            del outputs
            torch.cuda.empty_cache()

            #clear all info from the following document
            self.activations.clear()
            tally = {}
            results = {}

    def compare_activations(self,y_true):
        result = 0
        ans = 0
        for i, act_log in enumerate(self.act_log):
            comp = self.compare(self.activations, act_log) 
            if comp > result:
                ans = i
            result = max(comp, result)
        
        print ("confidence on ",ans,": ", result)

    def compare(dict1, dict2):
        corr = 0
        total = 0
        
        for name in dict1:
            total += len(dict1[name]['indices'])
            corr += len(set(dict1[name]['indices']).intersection(set(dict2[name]['indices'])))

        return (corr/total) * 100


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

# make sure to turn these off since they will affect the results
with open('topk/base_auxt.pkl', 'rb') as f:
    topk_base_auxt = pickle.load(f)
with open('topk/base_maths.pkl', 'rb') as f:
    topk_base_maths = pickle.load(f)
with open('topk/base_physics.pkl', 'rb') as f:
    topk_base_physics = pickle.load(f)
with open('topk/base_philosophy.pkl', 'rb') as f:
    topk_base_philosophy = pickle.load(f)
with open('topk/base_rand.pkl', 'rb') as f:
    topk_base_rand = pickle.load(f)

# make sure that act logs and data paths are in the same order
act_logs = (topk_base_maths, topk_base_physics, topk_base_philosophy)

# loading the data
data = []
data_paths = ["data/Mathematics,1970-2002",
              "data/Physics,1970-1997",
              "data/Philosophy,1970-2022"]

for i, data_path in enumerate(data_paths):
    dataset = load_from_disk(data_path)
    dataset = dataset[1000:1010]["text"]
    dataset = [{"text": text} for text in dataset]
    dataset = Dataset.from_list(dataset)
    data.append((dataset, i))

# shuffle the data
data = random.sample(data, len(data))

for act_log in act_logs:
    base_analyser = ActivationAnalyser(base_model, tokenizer, act_log)
    accuracy = base_analyser.classify(data)

print ("\n accuracy: ", accuracy,"%")
print  ("process complete")