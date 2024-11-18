"""
before running, run nvidia-smi in the terminal to see what gpu's are free in your cluster
"""
# setting up script
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# imports
import torch
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model


# load model
base_model_name = "Llama-3.1-8B-Instruct"
new_model_name = "llama-3.1-8B-Instruct-Math2"

# load tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# init model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

print("loading data")
# load data
data_path = "data/Mathematics,1970-2002"
dataset = load_from_disk(data_path)
dataset = dataset[:500]["text"]
dataset = [{"text": text} for text in dataset]
dataset = Dataset.from_list(dataset)
print("finished loading")


# Print max sequence length in tokens
max_len = max(len(llama_tokenizer.encode(text["text"])) for text in dataset)
print(f"Max sequence length: {max_len} tokens")

# Check if any sequences exceed model's context window
context_window = base_model.config.max_position_embeddings
print(f"Sequences exceeding context window: {sum(1 for text in dataset if len(llama_tokenizer.encode(text['text'])) > context_window)}")

# Training Params
train_params = TrainingArguments(
    output_dir="./results_modified",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=50,
    learning_rate=2e-5,
    weight_decay=0.05,
    fp16=False,
    bf16=True,
    max_grad_norm=1.0,
    max_steps=-1,
    warmup_ratio=0.1,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
)

from peft import get_peft_model
# LoRA Config
# reduce rank r if you're running out of vram
peft_parameters = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, peft_parameters)
model.print_trainable_parameters()

# Trainer with LoRA configuration
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    max_seq_length= None,
    args=train_params,
)

# Training
fine_tuning.train()

# Save Model 
fine_tuning.model.save_pretrained(new_model_name)