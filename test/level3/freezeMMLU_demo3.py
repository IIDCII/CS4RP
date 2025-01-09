# using this just for checking lvl 2 viability
# look at CS4RP scientific process ## lvl2 steps for more info

import os
# set the GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
# setting for vllm inference so that it can run in parallel
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('topk/base_auxt.pkl', 'rb') as f:
    topk_base_auxt = pickle.load(f)
with open('topk/base_hsm.pkl', 'rb') as f:
    topk_base_hsm = pickle.load(f)
with open('topk/base_hsp.pkl', 'rb') as f:
    topk_base_hsp = pickle.load(f)


class NeuronManipulator:
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model = model
        self.disabled_neurons = {}
        self._register_hooks()
    
    def _manipulation_hook(self, name):
        def hook(module, input, output):
            # If this layer has disabled neurons, zero them out
            if name in self.disabled_neurons:
                mask = torch.ones_like(output)
                for neuron_idx in self.disabled_neurons[name]:
                    mask[:, :, neuron_idx] = 0
                return output * mask
            return output
        return hook
    
    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if "mlp" in name or "attention" in name:
                module.register_forward_hook(self._manipulation_hook(name))
    
    def disable_neurons(self, layer_name, neuron_indices):
        """Disable specific neurons in a layer"""
        self.disabled_neurons[layer_name] = neuron_indices
    
    def enable_neurons(self, layer_name, neuron_indices=None):
        """Re-enable specific or all neurons in a layer"""
        if neuron_indices is None:
            self.disabled_neurons.pop(layer_name, None)
        else:
            self.disabled_neurons[layer_name] = [
                n for n in self.disabled_neurons.get(layer_name, [])
                if n not in neuron_indices
            ]
    
    def MMLU(self, dataset):    
        correct = 0
        total = 0

        for item in dataset:
            if total%50 == 0:
                print(correct, " correct out of ", total)
            question = item['question']
            choices = item['choices']
            answer = item['answer']

            prompt = (
                    f"Question: {question}\n\nChoices:\n"
                    f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n"
                    f"Do not explain and give the answer with strictly one letter from options A, B, C, or D.\nAnswer:"
                )


            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    temperature = 0.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bad_words_ids=[[self.tokenizer.encode(" ")[0]], [self.tokenizer.encode("The")[0]], [self.tokenizer.encode("Step")[0]], [self.tokenizer.encode("Let")[0]]]
                )
            
            # print (self.tokenizer.decode(outputs[0][-1:], skip_special_tokens=True).strip())
            pred = self.tokenizer.decode(outputs[0][-1:], skip_special_tokens=True).strip()
                    
            # Convert prediction to index (A=0, B=1, etc.)
            try:
                pred_idx = ord(pred.upper()) - ord('A')
                if pred_idx == answer:
                    correct += 1
            except:
                pass
            total += 1
        
        print ("MMLU score: ", (correct/total) * 100)


# loading the model
base_model_name = "Llama-3.1-8B-Instruct"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = base_model

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# this will act as the new model from this point
manipulator = NeuronManipulator(base_model,tokenizer)

# change this to alter the scope
topk_act = topk_base_auxt

print ("disabling neurons")
# disable the neurons
for i in range(len(topk_act)):
    neurons_to_disable = list(topk_act.items())[i][1]['indices']
    # just testing the base with nothing deleted
    # neurons_to_disable = []
    layer_name = list(topk_act.items())[i][0]
    manipulator.disable_neurons(layer_name, neurons_to_disable)
print ("disabling complete")

# loading the data
data_name = "cais/mmlu"
subset_name = "high_school_mathematics"
dataset = load_dataset(data_name, subset_name, split = "test")

manipulator.MMLU(dataset)