import os
# set the GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
# setting for vllm inference so that it can run in parallel
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('topk_act.pkl', 'rb') as f:
    topk_act = pickle.load(f)


class NeuronManipulator:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
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
    
    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

manipulator = NeuronManipulator(model_name = "Llama-3.1-8B-Instruct")

for i in range(len(topk_act)):
    neurons_to_disable = list(topk_act.items())[i][1]['indices']
    # just testing the base with nothing deleted
    # neurons_to_disable = []
    layer_name = list(topk_act.items())[i][0]
    manipulator.disable_neurons(layer_name, neurons_to_disable)

prompt = "Hello, how are you?"

# may have to change the following to vllm so that inference runs a lot faster than it currently does
output = manipulator.generate_text(prompt)
print ("generated the following with disabling the topk activated neurons:\n ", output)