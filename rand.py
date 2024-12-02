from datasets import load_dataset

# loading the data
data_name = "cais/mmlu"
aux_name = "auxiliary_train"
ft_name = "philosophy"
fta_name = "high_school_mathematics"

aux_dataset = load_dataset(data_name, aux_name, split = "train")
aux_dataset = aux_dataset.select(range(10))


for i in aux_dataset:
    print (i['train']['question'])
    break