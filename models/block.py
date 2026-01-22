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

    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        # NOTE: head_size is reduced by factor of n_head to account for concat -> output is still head_size
        head_size = n_embed // n_head # dimensionality of each head
        self.sa = MultiHeadAttention(n_heads=n_head, head_size=head_size, n_embed=n_embed, block_size=block_size, dropout=dropout)
        self.ff = FeedForward(n_embed, dropout=dropout)
        # layer normalizations
        self.ln1 = nn.LayerNorm(n_embed) # TODO: why n_embed?
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        # NOTE: enables residual pathway via addition
        # layer normalization occurs BEFORE transformation (attention or feedforward)
        x = x + self.sa(self.ln1(x)) # original x addition is the residual path
        x = x + self.ff(self.ln2(x))
        return x
