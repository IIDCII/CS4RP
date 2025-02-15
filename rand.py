import torch

# 2 tokens, 2 docs, 3 nodes
# should be size 1,2,3

t = torch.tensor([[[1.,2.,3.],
                   [4.,5.,6.],
                   [7.,8.,9.],
                   [10.,11.,12.]]])

print (t.shape)

mean = t.abs().mean(dim=(0, 1))

print (mean)