# ibr, I'm only using this if jup or vsc is no longer available

import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import scipy
from skimage import filters
from skimage import data
import seaborn as sns
from collections import Counter
import copy

# loading all of the files

# Define the file paths
files = [
    '../../topk/base_auxt.pkl',
    '../../topk/base_hsm.pkl',
    '../../topk/base_hsm1.pkl',
    '../../topk/base_hsp.pkl',
    '../../topk/base_maths.pkl',
    '../../topk/base_physics.pkl',
    '../../topk/base_philosophy.pkl',
]

# Load all files into a single dict
topk = {}

for file_path in files: 
    with open(file_path, 'rb') as f:
        topk[file_path] = pickle.load(f)

# get the topk_act
def get_act_index(d,topk):
    act_index = []
    for layer_name, values in d.items():
        # removing down just because it has 4000 nodes per layer unlike the rest
        if "down" not in layer_name:
            for name, val in values.items():
                if name == "indices":
                    act_index.append(val[0:topk])
    return act_index


# get the topk_act values
def get_act_values(d,topk):
    act_index = []
    for layer_name, values in d.items():
        # removing down just because it has 4000 nodes per layer unlike the rest
        # if "down" not in layer_name:
        for name, val in values.items():
            if name == "values":
                act_index.append(val[0:topk])
    return act_index

# print the graph for the items
# need to make a heat color based on the order of it
def show_act_in_layers(data,color):
    for row_index, row in enumerate(data):
        x_values = [row_index] * len(row)  # X-axis is the row index
        y_values = row  # Y-axis is the row values
        plt.scatter(
            x_values,
            y_values,
            marker='.',
            s=20,
            color=color,
            label=f"Row {row_index}"
        )


# load all of the topk needed
maths = topk['../../topk/base_maths.pkl']
act_index_maths = get_act_index(maths,1000)
act_values_maths = get_act_values(maths,1000)

# show the graph for the maths
act_index_maths20 = get_act_index(maths,20)

show_act_in_layers(act_index_maths20,"green")
plt.xlabel("Layer")
plt.ylabel("Nodes")
plt.title("topk maths in each layer")

# save to folder ao
plt.savefig("ao/topk20_maths.png")

print ("process complete")
