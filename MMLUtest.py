# important to run this first or else GPU allocation will not work
import os
# set the GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
# setting for vllm inference so that it can run in parallel
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# adding all of the neccessary imports
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments, pipeline, BitsAndBytesConfig
from huggingface_hub import login, snapshot_download, hf_hub_download
from datasets import load_dataset
from trl import SFTTrainer
import torch
import numpy as np
import math
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import torch.nn as nn
import pynvml
import matplotlib.pyplot as plt
from accelerate import init_empty_weights, infer_auto_device_map
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import gc
from vllm import LLM, SamplingParams
from typing import List, Optional
import torch.multiprocessing as mp

# Model and tokenizer names
base_model_name = "Llama-3.1-8B-Instruct"

altered = "llama-3.1-8B-Instruct-Math2"

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
subset_name = "high_school_mathematics"

dataset = load_dataset(data_name, subset_name, split = "test")


# model and tokenizer will be the merged version
# is this not the same tokenizer used in the first place

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


    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            temperature = 0.7,
            pad_token_id=tokenizer.eos_token_id,
            bad_words_ids=[[tokenizer.encode(" ")[0]], [tokenizer.encode("The")[0]], [tokenizer.encode("Step")[0]], [tokenizer.encode("Let")[0]]]
        )
    
    print (tokenizer.decode(outputs[0][-1:], skip_special_tokens=True).strip())
    pred = tokenizer.decode(outputs[0][-1:], skip_special_tokens=True).strip()
            
    # Convert prediction to index (A=0, B=1, etc.)
    try:
        pred_idx = ord(pred.upper()) - ord('A')
        if pred_idx == answer:
            correct += 1
    except:
        pass
    total += 1


print ("high school math MMLU score: ", (correct/total) * 100)