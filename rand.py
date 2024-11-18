
from datasets import load_from_disk

# load data
data_path = "data/Mathematics,1970-2002"
dataset = load_from_disk(data_path)
dataset = dataset[:80]["text"]