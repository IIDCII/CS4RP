import pickle
from datasets import load_from_disk
from datasets import Dataset

# loading the data
data_path = "data/Mathematics,1970-2002"
dataset = load_from_disk(data_path)
dataset = dataset[:1000]["text"]
dataset = [{"text": text} for text in dataset]
dataset = Dataset.from_list(dataset)


print (dataset[0]["text"])