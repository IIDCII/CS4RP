import os
# set the GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
# setting for vllm inference so that it can run in parallel
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datasets import load_dataset
from typing import Dict, List


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
    
    def analyze_text(self, prompts, top_k=3, data_type = "test"):
        self.activations.clear()
        
        for text in prompts:
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

class TopKIsolation:
    def remove_common_values(self, dict1: Dict[str, Dict[str, List]], dict2: Dict[str, Dict[str, List]]) -> Dict[str, Dict[str, List]]:
        """
        Remove values from dict1 that are also present in dict2.
        
        :param dict1: First dictionary with indices and values
        :param dict2: Second dictionary to compare against
        :return: Modified dict1 with common values removed
        """
        total_initial_weights = sum(len(d['indices']) for d in dict1.values())
        print(f"Total initial weights: {total_initial_weights}")
        
        removing_count = 0
        for name in dict1:
            if name in dict2:
                # Find common indices
                common_indices = set(dict2[name]['indices']) & set(dict1[name]['indices'])
                removing_count += len(common_indices)
                
                # Filter out common indices
                dict1[name]['values'] = [
                    v for i, v in zip(dict1[name]['indices'], dict1[name]['values']) 
                    if i not in common_indices
                ]
                dict1[name]['indices'] = [
                    i for i in dict1[name]['indices'] 
                    if i not in common_indices
                ]
        
        total_final_weights = sum(len(d['indices']) for d in dict1.values())
        print(f"Removed {removing_count} weights")
        print(f"Final number of weights: {total_final_weights}")
        
        return dict1

    def merge_indices(self, dict1: Dict[str, Dict[str, List]], dict2: Dict[str, Dict[str, List]]) -> Dict[str, Dict[str, List]]:
        """
        Merge indices from dict1 and dict2, with dict2 values taking precedence.
        
        :param dict1: First dictionary to merge
        :param dict2: Second dictionary to merge
        :return: Merged dictionary
        """
        result = dict1.copy()
        
        for name in dict2:
            if name in result:
                # Combine indices and values, with dict2 taking precedence
                combined = dict(zip(result[name]['indices'], result[name]['values']))
                combined.update(dict(zip(dict2[name]['indices'], dict2[name]['values'])))
                
                # Sort and split back into separate lists
                sorted_items = sorted(combined.items())
                result[name]['indices'], result[name]['values'] = zip(*sorted_items)
            else:
                # If name not in result, add entire entry from dict2
                result[name] = dict2[name]
        
        # Logging weight information
        self._log_weight_info(dict1, dict2, result)
        
        return result
    
    def _log_weight_info(self, dict1: Dict[str, Dict[str, List]], dict2: Dict[str, Dict[str, List]], result: Dict[str, Dict[str, List]]) -> None:
        """
        Log weight information for evaluation purposes
        
        :param dict1: First input dictionary
        :param dict2: Second input dictionary
        :param result: Merged result dictionary
        """
        print(f"Total weights in dict1: {sum(len(d['indices']) for d in dict1.values())}")
        print(f"Total weights in dict2: {sum(len(d['indices']) for d in dict2.values())}")
        print(f"Total weights in result: {sum(len(d['indices']) for d in result.values())}")

class TopKLoad:
    def __init__(self, bf, ttf, tf, ttf_tf_bf):
        """
        Input args in the following order: bf, ttf, tf, ttf_tf_bf
        """
        # Ensure the directory exists
        os.makedirs('topk', exist_ok=True)
        
        # Dictionary to store file paths for easier management
        self.save_paths = {
            'bf': 'topk/topk_act_bf.pkl',
            'ttf': 'topk/topk_act_ttf.pkl',
            'tf': 'topk/topk_act_tf.pkl',
            'full': 'topk/topk_act_full.pkl'
        }
        
        # Dictionary of data to be saved
        self.data = {
            'bf': bf,
            'ttf': ttf,
            'tf': tf,
            'full': ttf_tf_bf
        }
        
        # Save all data
        self.save_all_data()
        
        # Load all data
        self.loaded_data = self.load_all_data()
    
    def save_all_data(self):
        """Save all data to pickle files."""
        for key, data in self.data.items():
            with open(self.save_paths[key], 'wb') as f:
                pickle.dump(data, f)
    
    def load_all_data(self):
        """Load all data from pickle files."""
        loaded_data = {}
        for key, path in self.save_paths.items():
            with open(path, 'rb') as f:
                loaded_data[key] = pickle.load(f)
        return loaded_data
    
    def get_loaded_data(self, key):
        """
        Retrieve specific loaded data.
        
        :param key: The key of the data to retrieve ('bf', 'ttf', 'tf', or 'full')
        :return: The loaded data for the specified key
        """
        return self.loaded_data.get(key)