# important to run this first or else GPU allocation will not work
import os
# set the GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
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
base_model_name = "llama-3.1-8B-Instruct"
altered = "llama-3.1-8B-Instruct-Math"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, altered)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# loading the data
data_name = "cais/mmlu"
subset_name = "high_school_mathematics"

dataset = load_dataset(data_name, subset_name, split = "train")


# model and tokenizer will be the merged version
# is this not the same tokenizer used in the first place



def calculate_perplexity(model, tokenizer, text, max_length=300):
    """
    Calculate the perplexity of a text using a language model.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text to evaluate
        max_length: Maximum sequence length to process
        
    Returns:
        float: The perplexity score
    """
    # Encode the text
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    
    # Get input IDs and create target labels (shifted by 1)
    input_ids = encodings.input_ids
    target_ids = input_ids.clone()
    
    # Calculate loss with no gradient tracking
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
    
    # Calculate perplexity
    ppl = torch.exp(neg_log_likelihood)
    loss = neg_log_likelihood
    return ppl.item(), loss

def evaluate_dataset(model, tokenizer, texts):
    """
    Calculate average perplexity across multiple texts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of texts to evaluate
        
    Returns:
        float: Average perplexity across all texts
    """
    perplexities = []
    total_loss = []
    for text in texts:
        try:
            ppl_and_loss = calculate_perplexity(model, tokenizer, text)
            ppl = ppl_and_loss[0]
            loss = ppl_and_loss[1]

            perplexities.append(ppl)
            total_loss.append(loss)

        except Exception as e:
            print(f"Error processing text: {e}")
            continue
    
    return perplexities, total_loss

model = AutoModelForCausalLM.from_pretrained(
    model,
    torch_dtype=torch.float16,
    # need to 
    device_map="auto",
)


ppl_and_loss = evaluate_dataset(model, tokenizer, dataset)

ppl = ppl_and_loss[0]
loss = ppl_and_loss[1]