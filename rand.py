import torch

# Simulating a tensor of shape [batch_size=4, channels=3, height=5, width=5]
activation = torch.randn(4, 3, 5, 5)

# Compute mean absolute activation per neuron
mean_activation = activation.abs().mean(dim=(0, 1))


print("Shape of mean activation:", mean_activation)