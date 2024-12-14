from datasets import load_dataset

# loading the data
data_name = "CohereForAI/Global-MMLU"
aux_name = "auxiliary_train"
ft_name = "philosophy"
fta_name = "high_school_mathematics"

dataset = load_dataset(data_name, "en",  split = "test")


print (dataset[0]["subject"])