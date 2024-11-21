# setting up script
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

class ActivationExtractor:
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.activation_dict = defaultdict(list)
        self.hooks = []
        
    def _activation_hook(self, name: str):
        def hook(module, input, output):
            self.activation_dict[name].append(output.detach().cpu())
        return hook
    
    def register_hooks(self, target_modules: List[str] = None):
        """Register hooks for all or specific modules"""
        self.clear_hooks()  # Clear any existing hooks
        
        if target_modules is None:
            # Default to capturing all attention and MLP outputs
            target_modules = ['attention', 'mlp']
            
        for name, module in self.model.named_modules():
            if any(target in name.lower() for target in target_modules):
                hook = module.register_forward_hook(self._activation_hook(name))
                self.hooks.append(hook)
    
    def clear_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activation_dict.clear()
    
    def get_activations(self, text: str) -> Dict[str, torch.Tensor]:
        """Get activations for a given input text"""
        inputs = self.tokenizer(text, return_tensors="pt")
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process activations
        processed_activations = {}
        for name, activations in self.activation_dict.items():
            if activations:  # Check if we captured any activations
                # Average across batch dimension if present
                processed_activations[name] = torch.mean(torch.stack(activations), dim=0)
        
        self.activation_dict.clear()  # Clear for next run
        return processed_activations
    
    def compare_activations(self, text1: str, text2: str) -> Dict[str, float]:
        """Compare activations between two inputs"""
        act1 = self.get_activations(text1)
        act2 = self.get_activations(text2)
        
        similarities = {}
        for name in act1.keys():
            if name in act2:
                # Flatten activations and compute cosine similarity
                flat1 = act1[name].flatten()
                flat2 = act2[name].flatten()
                similarity = torch.cosine_similarity(flat1, flat2, dim=0)
                similarities[name] = similarity.item()
        
        return similarities
    
    def get_activation_statistics(self, text: str) -> Dict[str, Dict[str, float]]:
        """Get statistical measures of activations"""
        activations = self.get_activations(text)
        stats = {}
        
        for name, activation in activations.items():
            flat_activation = activation.flatten()
            stats[name] = {
                'mean': float(torch.mean(flat_activation)),
                'std': float(torch.std(flat_activation)),
                'max': float(torch.max(flat_activation)),
                'sparsity': float((flat_activation == 0).sum() / flat_activation.numel())
            }
            
        return stats

def analyze_reasoning_patterns(
    extractor: ActivationExtractor,
    base_problem: str,
    transfer_problem: str
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Analyze activation patterns between base domain and transfer domain problems
    """
    # Compare activations between problems
    similarities = extractor.compare_activations(base_problem, transfer_problem)
    
    # Get detailed statistics for each problem
    base_stats = extractor.get_activation_statistics(base_problem)
    transfer_stats = extractor.get_activation_statistics(transfer_problem)
    
    return similarities, {
        'base': base_stats,
        'transfer': transfer_stats
    }

# Example usage
if __name__ == "__main__":
    # Initialize extractor with your fine-tuned model
    extractor = ActivationExtractor("Llama-3.1-8B-Instruct")
    
    # Register hooks for specific layers you want to analyze
    extractor.register_hooks(['attention', 'mlp'])
    
    # Example problems demonstrating similar reasoning patterns in different domains
    math_problem = """
    Prove that if a < b and b < c, then a < c.
    Let's approach this step by step:
    1) First, we know a < b
    2) Second, we know b < c
    3) Therefore, by transitive property, a < c
    """
    
    philosophy_problem = """
    If all humans are mortal, and Socrates is human, is Socrates mortal?
    Let's approach this step by step:
    1) First, we know all humans are mortal
    2) Second, we know Socrates is human
    3) Therefore, by logical deduction, Socrates is mortal
    """
    
    # Analyze patterns
    similarities, statistics = analyze_reasoning_patterns(
        extractor,
        math_problem,
        philosophy_problem
    )
    
    # Clean up
    extractor.clear_hooks()