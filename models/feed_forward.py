# a simple feed-forward layer as defined in attention paper

import torch
import torch.nn as nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    """
    Feed forward layer: simple linear layer followed by a ReLU (non-linearity)
    - Purpose is to give tokens more "time" to reflect upon communication established with multi-head attention.
    """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
    