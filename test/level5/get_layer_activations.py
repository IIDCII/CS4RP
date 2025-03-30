"""
Getting the layer activations for a layer (what the activations look like before and after).
Store all of the before and after activations for that layer in a 2D list.
This will be used as the training data for the sae.
"""

# setup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

