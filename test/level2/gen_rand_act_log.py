import pickle
import random

topk = {}
topk_rand = {}

file_name = './topk/base_physics.pkl'

# using the follwing to use the names of the layers
with open(file_name, 'rb') as f:
    topk = pickle.load(f)

for name in topk:
    rand_range = 14000
    # rand range
    if "down" in name:
        rand_range = 4000
    rand_list = [random.randint(0, rand_range) for _ in range(1000)]
    topk_rand[name] = {
        'indices' : rand_list,
        'values' : topk[name]['values'],
    }

with open('topk/base_rand.pkl', 'wb') as f:
    pickle.dump(topk_rand, f)
