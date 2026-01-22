# the self-attention heads used in the transformer
import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """
    A single self-attention head for the transformer.
    """

    def __init__(self, n_embed, head_size, block_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # NOTE: tril is not a parameter of the model, so it needs to be registered as buffer
        self.tril: torch.Tensor # for Pylance
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute attention scores
        weight = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weight = F.softmax(weight, dim=-1) # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = weight @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
