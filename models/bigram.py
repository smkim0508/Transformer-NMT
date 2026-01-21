# bigram language model, predicting the next character given a SINGLE character

import torch
import torch.nn as nn
from torch.nn import functional as F

# set manual seed
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    """
    Base class for bigram language model.
    """

    def __init__(self):
        super(BigramLanguageModel, self).__init__()
        pass

    def forward(self):
        pass
