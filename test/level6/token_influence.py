"""
just going to see the correlation between each of the tokens and output the percentage for each one.
"""

# setup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets
from datasets import load_from_disk
from datasets import Dataset
from tqdm import tqdm
import random
import pickle
import torch

# hooks the model to get the activations
class ActivationAnalyser:
    def __init__(self, model, tokenizer, act_logs):
        self.tokenizer = tokenizer
        self.model = model
        self.activations = {}
        self._register_hooks()
        self.act_logs = act_logs
    
    def _activation_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def _register_hooks(self):
        for name, module in self.model.named_modules():
            # eval specific layers
            if "mlp." in name:
                module.register_forward_hook(self._activation_hook(name))  
    
    def classify(self, texts, topk_nodes): 
        self.activations.clear()

        # runs through all the training data
        for i, prompt in enumerate(texts):
            print(f"on text {i}")
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                )
            
            # add the activations per text to the tally
            for name, activation in self.activations.items():
                # setting the value to 1 or 2 whether it's greater or smaller than 0.1
                activation = torch.where(
                    torch.abs(activation) <= 1.2,
                    torch.tensor(1.0, device=activation.device), 
                    torch.tensor(2.0, device=activation.device)
                )
                
                # compare each of the token outputs to the dict
                for i in activation:
                    value = self.compare(topk_nodes[name]["indices"], i)
                    print (f"token {i} value:{value}")

            # unload inputs and ouputs from gpu
            del inputs
            torch.cuda.empty_cache()

    def compare(self, list1, list2):
        corr = 0
        total = 0
        
        total = len(list1)
        inter =  set(list1).intersection(set(list2))
        corr = len(inter)

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

topk_nodes = topk_base_maths

texts = ["","",""]
texts[0] = "There are no three positive integers $a, b, c$ that satisfy the equation $a^n + b^n = c^n$ for any integer value of $n$ greater than 2."
texts[1] = "The principle of superposition states that for linear systems, the net response caused by two or more stimuli is the sum of the responses that would have been caused by each stimulus individually."
texts[2] = "Existentialism emphasizes individual freedom, responsibility, and the subjective experience of existence."

base_analyser = ActivationAnalyser(base_model, tokenizer)
accuracy = base_analyser.get_ti(texts, topk_nodes)

print ("\n accuracy: ", accuracy,"%")
print  ("process complete")