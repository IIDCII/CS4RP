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
            if "mlp." in name or "q_proj" in name:
                module.register_forward_hook(self._activation_hook(name))
    
    def analyze_text(self, prompts, top_k=3):
        self.activations.clear()
        
        for text in prompts:
            inputs = self.tokenizer(text, return_tensors="pt")
            
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

# removes all values from dict1 that's in dict2 to isolate the the most used transferred 
def remove_common_values(dict1, dict2):
   total_removed = 0
   for name in dict1:
       if name in dict2:
           original_len = len(dict1[name])
           dict1[name] = [x for x in dict1[name] if x not in dict2[name]]
           removed = original_len - len(dict1[name])
           total_removed += removed
   print(f"Total values removed: {total_removed}")
   return dict1

# loading the model
base_model_name = "Llama-3.1-8B-Instruct-Math2"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# model = PeftModel.from_pretrained(base_model, altered)
# model = model.merge_and_unload()

model = base_model

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"



# loading the data
data_name = "cais/mmlu"
base_subset_name = "philosophy"
alt_subset_name = "high_school_mathematics"

base_dataset = load_dataset(data_name, base_subset_name, split = "test")
alt_dataset = load_dataset(data_name, base_subset_name, split = "test")




# active neuron eval
analyzer = ActivationAnalyzer(model, tokenizer)

# fined tuned and base knowledge
ftk = analyzer.analyze_text(base_dataset)
# fine tuned, base and transferred knowledge
tk = analyzer.analyze_text(alt_dataset)

# remove all of the neurons that show up in ftk from tk
results = remove_common_values(tk,ftk)


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