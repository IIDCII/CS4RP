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
    
    def classify(self, data): 
        self.activations.clear()
        tally = {}
        results = {}
        total = len(data)
        ans_correct = 0

        # runs through all the training data
        for i, text in enumerate(data):
            prompt = text["text"]
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=1024,
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
                tally[name] = current_sum
                
            for name, subtal in tally.items():
                top_values, top_indices = torch.topk(subtal, 1000)

                results[name] = {
                    "indices": top_indices.tolist(),
                    "values": top_values.tolist()
                }
            
            self.activations.clear()
            tally = {}  

            # compare the activations and select the check to see if it's right
            ans_correct += self.compare_activations(text["class"], results)

            # unload inputs and ouputs from gpu
            results = {}
            del inputs
            del outputs
            torch.cuda.empty_cache()

        return (ans_correct / total) * 100 

    def compare_activations(self,y_true, results):
        result = 0
        ans = 0
        for i, act_log in enumerate(self.act_logs):
            comp = self.compare(results, act_log) 
            if comp > result:
                ans = i
                result = comp 
            print ("confidence on ", i,": ", comp)
        print ("y_true: ", y_true, " predicted: ", ans)

        if ans == y_true:
            return 1
        else:
            return 0

    def compare(self, dict1, dict2):
        corr = 0
        total = 0
        
        for name in dict1:
            if name in dict2:
                total += len(dict1[name]['indices'])
                inter =  set(dict1[name]['indices']).intersection(set(dict2[name]['indices']))
                corr += len(inter)

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
with open('topk/base_maths_sparse.pkl', 'rb') as f:
    topk_base_maths = pickle.load(f)
with open('topk/base_physics_sparse.pkl', 'rb') as f:
    topk_base_physics = pickle.load(f)
with open('topk/base_philosophy_sparse.pkl', 'rb') as f:
    topk_base_philosophy = pickle.load(f)
with open('topk/base_rand.pkl', 'rb') as f:
    topk_base_rand = pickle.load(f)

# make sure that act logs and data paths are in the same order
act_logs = (topk_base_maths, topk_base_philosophy, topk_base_physics)

# loading the data
data = []
data_class = []
docs = 10
data_paths = ["data/Mathematics,1970-2002",
              "data/Philosophy,1970-2022",
              "data/Physics,1970-1997",]

for i, data_path in enumerate(data_paths):
    dataset = load_from_disk(data_path)
    data = data + dataset[1000:1000 + docs]["text"]
    for _ in range(docs):
        data_class.append(i)

data = [{"text": text, "class": data_class[i]} for i, text in enumerate(data)]
data = Dataset.from_list(data)

base_analyser = ActivationAnalyser(base_model, tokenizer, act_logs)
accuracy = base_analyser.classify(data)

print ("\n accuracy: ", accuracy,"%")
print  ("process complete")