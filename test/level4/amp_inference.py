# using this just for checking lvl 2 viability
# look at CS4RP scientific process ## lvl2 steps for more info

import os
from re import sub
# set the GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# setting for vllm inference so that it can run in parallel
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset, concatenate_datasets
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy


class NeuronManipulator:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.amplified_neurons = {}
        self.amp_factor = {}
        self._register_hooks()
    
    def _manipulation_hook(self, name):
        def hook(module, input, output):
            # If this layer has disabled neurons, zero them out
            if name in self.amplified_neurons:
                mask = torch.ones_like(output)
                for neuron_idx in self.amplified_neurons[name]:
                    mask[:, :, neuron_idx] = self.amp_factor.get(name)
                return output * mask
            return output
        return hook
    
    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if "mlp" in name or "attention" in name:
                module.register_forward_hook(self._manipulation_hook(name))
    
    def amplify_neurons(self, layer_name, neuron_indices, amp_factor = 1.15):
        """Disable specific neurons in a layer"""
        self.amplified_neurons[layer_name] = neuron_indices
        self.amp_factor[layer_name] = amp_factor
    
    def reset_all_neurons(self):
        """Reset all disabled neurons, re-enabling them"""
        self.amplified_neurons.clear()
    
    def MMLU(self, dataset, data_type = "test"):    
        correct = 0
        total = 0

        for prompt in dataset:

            print (f"prompt: {prompt}")

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

            # set the inputs to cuda
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    temperature = 0.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # print (self.tokenizer.decode(outputs[0][-1:], skip_special_tokens=True).strip())
            response = self.tokenizer.decode(outputs[0][-1:], skip_special_tokens=True).strip()

            print(f"response: {response}")


# removes all values from dict1 that's in dict2 to isolate the the most used transferred 
def remove_common_values(dict1, dict2):
    # Create a new dictionary to store the result
    result = {}
    
    for name in dict1:
        if name in dict2:
            # Find common indices to remove
            indices_to_remove = set(dict1[name]['indices']).intersection(set(dict2[name]['indices']))
            
            # Create new lists for indices and values, excluding common indices
            new_indices = [i for i in dict1[name]['indices'] if i not in indices_to_remove]
            new_values = [v for i, v in zip(dict1[name]['indices'], dict1[name]['values']) 
                          if i not in indices_to_remove]
            
            # Add the modified entry to the result dictionary
            result[name] = {
                'indices': new_indices,
                'values': new_values,
            }
        else:
            result[name] = dict1[name]
    
    return result

# adjusting the top k for freezing weights
def adjust_topk(data, topk: int, mink=0):
    # Create a new dictionary to store the result
    result = {}
    
    for name in data:
        # Create a new entry with adjusted indices
        result[name] = {
            'indices': data[name]['indices'][mink:topk],  # Slice the indices
            'values': data[name]['values'][mink:topk],    # Slice the values
        }
    
    return result

# analysis
def amplify(topk_act, amp_value = 1.15):
    count = 0
    for i in range(len(topk_act)):
        count += len(list(topk_act.items())[i][1]['indices'])
        neurons_to_amplify = list(topk_act.items())[i][1]['indices']
        layer_name = list(topk_act.items())[i][0]
        manipulator.amplify_neurons(layer_name, neurons_to_amplify, amp_value) 
    print ("total number of neurons amplified: ", count)



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

# this will act as the new model from this point
manipulator = NeuronManipulator(base_model,tokenizer)

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

# this will act as the new model from this point
manipulator = NeuronManipulator(base_model,tokenizer)

k = 100
mk = 0

# adjust the topk
topk_act = adjust_topk(topk_base_philosophy, k, mink = mk)
topk_sub = adjust_topk(topk_base_auxt, k, mink = mk)

topk_act = remove_common_values(topk_act,topk_sub)

amplify(topk_act, amp_value = 1.15)

query = ["Can you tell me about the city of Bath?"]

manipulator.MMLU(query, "test")

manipulator.reset_all_neurons()

print ("single process complete")