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

    def __init__(self, vocab_size):
        """
        Components of this model.
        Sets up a token embedding table that is NxN where N = vocab size.
            - For each item in Vocab, we have a unique embeddings map
        """
        super(BigramLanguageModel, self).__init__()
        # each token directly reads off the logits for next token from look up table
        self.token_embeddings_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        """
        Connects the components.
        Finds the ith row from the embeddings table, computes CE loss and returns.
        """
        # idx and targets -> (B,T) tensor of integers
        logits = self.token_embeddings_table(idx) # (B,T,C)

        # NOTE: cross_entropy() expects tensor of dimension (B,C,T), so we need to format our logits as such
        # Therefore, instead of creating a messy logit, we just stretch the 1st dimension to effectively make logit (N, C) and target (N)
        B, T, C = logits.shape
        logits = logits.view(B*T, C) # stretches positional dimensions while preserving channel dim
        targets = targets.view(B*T) # also makes targets 1 dimensional to match logits

        loss = F.cross_entropy(logits, targets)

        return logits, loss
