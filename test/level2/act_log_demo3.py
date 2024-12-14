import os
# set the GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
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
altered = "llama-3.1-8B-Instruct-Math2"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# merging the model
model = PeftModel.from_pretrained(base_model, altered)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# loading the data
data_name = "cais/mmlu"
aux_name = "auxiliary_train"
ft_name = "philosophy"
fta_name = "high_school_mathematics"

aux_dataset = load_dataset(data_name, aux_name, split = "train")
aux_dataset = aux_dataset.select(range(200))

base_dataset = load_dataset(data_name, ft_name, split = "test")
alt_dataset = load_dataset(data_name, fta_name, split = "test")


# active neuron eval
base_analyzer = ActivationAnalyzer(base_model, tokenizer)
ft_analyzer = ActivationAnalyzer(model, tokenizer)

# bf -> base fundamental activations
# tf -> task fundamental activations
# ttf -> task transfer fundamental activations
bf = base_analyzer.analyze_text(aux_dataset, top_k=3, data_type = 'train')
# change back to ft to get the ft phil model
tf_bf = base_analyzer.analyze_text(base_dataset, top_k=3)
ttf_tf_bf = base_analyzer.analyze_text(alt_dataset, top_k=3)

# isolating the fundamentals
ttf = remove_common_values(ttf_tf_bf, tf_bf)
tf = remove_common_values(tf_bf, bf)

# store all of the results
with open('topk_act_bf.pkl', 'wb') as f:
    pickle.dump(bf, f)

with open('topk_act_bf.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)


with open('topk_act_ttf.pkl', 'wb') as f:
    pickle.dump(ttf, f)

with open('topk_act_ttf.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)


with open('topk_act_tf.pkl', 'wb') as f:
    pickle.dump(tf, f)

with open('topk_act_tf.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)


with open('topk_act_full.pkl', 'wb') as f:
    pickle.dump(ttf_tf_bf, f)

with open('topk_act_full.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)