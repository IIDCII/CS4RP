import pickle

with open('topk_act.pkl', 'rb') as f:
    topk_act = pickle.load(f)

# first int is the one to iterate through
print (list(topk_act.items())[0][1]['indices'])

# why isn't it working here though 
for i in range(100):
    neurons_to_disable = list(topk_act.items())[i][1]['indices']
    
    layer_name = list(topk_act.items())[i][0]
    print (layer_name)

print (neurons_to_disable)
print (layer_name)