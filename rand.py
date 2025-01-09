import pickle

with open('topk/base_auxt.pkl', 'rb') as f:
    topk_base_auxt = pickle.load(f)
with open('topk/base_hsm.pkl', 'rb') as f:
    topk_base_hsm = pickle.load(f)
with open('topk/base_hsp.pkl', 'rb') as f:
    topk_base_hsp = pickle.load(f)

# adjusting the top k for freezing weights
def adjust_topk(data,topk: int):
    for name in data:
        for indices in name:
            indices = indices[:topk]
    return data