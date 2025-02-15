# imports
from datasets import load_dataset
import os

# load data from hf
# if the data is local then this step isn't needed
# make sure to select the correct split
data_name = "claran/modular-s2orc"
subset_name = "Physics,1970-1997"
save_dir = "data"

os.makedirs(save_dir, exist_ok=True)
dataset = load_dataset(data_name, subset_name, split="train")

dataset.save_to_disk(os.path.join(save_dir, subset_name))