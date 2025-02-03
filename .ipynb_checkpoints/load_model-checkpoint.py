"""
Req:
- permission from model authors
- hf access
- hf key
"""
# imports
from huggingface_hub import login, snapshot_download

# getting the authorisation from huggingface
access_key = open('hf_ak.txt','r').read()
login(token = access_key)

model_path = snapshot_download("meta-llama/Llama-3.1-8B-Instruct", local_dir="./Llama-3.1-8B-Instruct")