# a simple feed-forward layer as defined in attention paper

import torch
import torch.nn as nn
from torch.nn import functional as F

class FeedForward(nn.Module):
    """
    Feed forward layer: simple linear layer followed by a ReLU (non-linearity)
    - Purpose is to give tokens more "time" to reflect upon communication established with multi-head attention.
    """
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            # NOTE: the inner layer of feed forward net is 4x larger than the input, as defined in attention paper
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), # project layer to take back to the residual pathway
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    