# bigram language model, predicting the next character given a SINGLE character
# NOTE: this is a true bigram model - no attention, no context beyond the current token

import torch
import torch.nn as nn
from torch.nn import functional as F

# set manual seed
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    """
    True bigram language model.
    Each token directly predicts the next token using only its own embedding.
    No attention, no positional embeddings, no context window.
    """

    def __init__(self, vocab_size):
        """
        Simple bigram model components:
            - token_embedding: maps each token to a vector
            - lm_head: projects embedding to vocabulary logits
        """
        super().__init__()
        # each token embedding directly gives logits for predicting next token
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass: each token embedding is the logits for the next token.
        """
        # idx -> (B, T), logits -> (B, T, vocab_size)
        logits = self.token_embedding(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens autoregressively.
        For bigram, only the last token matters for prediction.

        idx: (B, T) starting context
        max_new_tokens: number of tokens to generate

        Returns idx: (B, T + max_new_tokens) extended sequence
        """
        for _ in range(max_new_tokens):
            # only need the last token for bigram
            idx_last = idx[:, -1:]  # (B, 1)
            # get predictions
            logits, _ = self(idx_last)
            # logits is (B, 1, vocab_size), take last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # convert to probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            # sample next token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append to sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
