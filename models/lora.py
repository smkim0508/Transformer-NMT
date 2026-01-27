# LoRA (Low-Rank Adaptation) Layer to Fine-tune Model

import torch
import torch.nn as nn
from torch.nn import functional as F

class LoRALayer(nn.Module):
    def __init__(self, feat_in, feat_out, rank, lora_alpha=1, enable_lora = True):
        super().__init__()
        self.loraA = nn.Parameter(rank, feat_out)
        self.loraB = nn.Parameter(feat_in, rank)
        self.scale = lora_alpha / rank
        self.enable = enable_lora # whether to enable LoRA or not

    def forward(self, original_weight):
        if self.enable:
            return original_weight + (self.loraB @ self.loraA).view(original_weight.shape) * self.scale
        return original_weight
