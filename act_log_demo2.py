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

class ActivationAnalyzer:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.activations = {}
        self._register_hooks()
    
    def _activation_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if "mlp" in name or "attention" in name:
                module.register_forward_hook(self._activation_hook(name))
    
    def analyze_text(self, text, top_k=3):
        inputs = self.tokenizer(text, return_tensors="pt")
        self.activations.clear()
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        results = {}
        for name, activation in self.activations.items():
            # Calculate mean activation per neuron
            mean_activation = activation.abs().mean(dim=(0, 1))
            # Get top-k neurons
            top_values, top_indices = torch.topk(mean_activation, top_k)
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




analyzer = ActivationAnalyzer(model_name = "Llama-3.1-8B-Instruct")
text = "Hello, how are you?"
results = analyzer.analyze_text(text)

# Print top activated neurons for each layer
for layer_name, data in results.items():
    print(f"\nLayer: {layer_name}")
    print("Top activated neurons:")
    for idx, value in zip(data["indices"], data["values"]):
        print(f"Neuron {idx}: {value:.4f}")
    
    # Visualize activations for this layer
    analyzer.visualize_activations(results, layer_name)

# store all of the results

with open('topk_act.pkl', 'wb') as f:
    pickle.dump(results, f)

with open('topk_act.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)