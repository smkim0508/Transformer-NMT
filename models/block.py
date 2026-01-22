# block to orchestrate feedforward and self attention layers, as defined in attention paper

import torch
import torch.nn as nn
from torch.nn import functional as F

# models
from models.feed_forward import FeedForward
from models.self_attention import MultiHeadAttention

class Block(nn.Module):
    """
    Transformer block that layers communication -> computation
    """

    def __init__(self, n_embed, n_head, block_size):
        super().__init__()
        head_size = n_embed // n_head # dimensionality of each head
        self.sa = MultiHeadAttention(n_heads=n_head, head_size=head_size, n_embed=n_embed, block_size=block_size)
        self.ff = FeedForward(n_embed)
        
    def forward(self, x):
        # NOTE: enables residual pathway via addition
        x = x + self.sa(x) # original x addition is the residual path
        x = x + self.ff(x)
        return x
