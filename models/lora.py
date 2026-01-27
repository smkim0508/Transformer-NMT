# LoRA (Low-Rank Adaptation) Layer to Fine-tune Model

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.parametrize as parametrize

class LoRAParametrization(nn.Module):
    """
    Paramaterization for LoRA (Low-rank Adaptation) Layer.
    - Implementation follows architecture described in section 4.1 of LoRA paper.
    """
    def __init__(self, device, feat_in, feat_out, rank, lora_alpha=1, enable_lora = True):
        super().__init__()
        # A initialized with random Gaussian, B with zeros
        # This ensures initial LoRA output is zero (no change to pretrained weights)
        self.loraA = nn.Parameter(torch.randn(rank, feat_out).to(device))
        self.loraB = nn.Parameter(torch.zeros(feat_in, rank).to(device))
        self.scale = lora_alpha / rank
        self.enabled = enable_lora # whether to enable LoRA or not

    def forward(self, original_weight):
        if self.enabled:
            return original_weight + (self.loraB @ self.loraA) * self.scale
        return original_weight

# helper to add LoRA parameterization
def linear_layer_parameterization(layer, device, rank=1, lora_alpha=1):
    # Only add the parameterization to the weight matrix, ignore the Bias
    # From section 4.2 of the paper:
    features_in, features_out = layer.weight.shape
    return LoRAParametrization(
        device=device,
        feat_in=features_in,
        feat_out=features_out,
        rank=rank,
        lora_alpha=lora_alpha,
        enable_lora=True
    )
