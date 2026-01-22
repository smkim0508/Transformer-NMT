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

    def __init__(self, vocab_size, n_embed):
        """
        Components of this model.
        Sets up a token embedding table that is NxN where N = vocab size.
            - For each item in Vocab, we have a unique embeddings map
            - n_embed is the dimensionality of the embedding
        """
        super(BigramLanguageModel, self).__init__()
        # each token directly reads off the logits for next token from look up table
        self.token_embeddings_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) # output layer

    def forward(self, idx, targets = None):
        """
        Connects the components.
        Finds the ith row from the embeddings table, computes CE loss and returns.

        Target is optional when we want to generate from model -> in this case, loss is None.
        """
        # idx and targets -> (B,T) tensor of integers
        token_embeddings = self.token_embeddings_table(idx) # (B, T, n_embed)
        logits = self.lm_head(token_embeddings) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # NOTE: cross_entropy() expects tensor of dimension (B,C,T), so we need to format our logits as such
            # Therefore, instead of creating a messy logit, we just stretch the 1st dimension to effectively make logit (B, C) and target (B)
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # stretches positional dimensions while preserving channel dim
            targets = targets.view(B*T) # also makes targets 1 dimensional to match logits

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Logic for generating new tokens (characters) given some starting seed (idx).
        This makes the bigram language model a generative model.

        The purpose is to take a current context of chars (idx) in shape (B,T) and extend to (B,T+1), (B,T+2), ...
        - continues until max_new_tokens reached
        """
        # idx becomes the (B,T) array of indices in the current context
        # i.e. current context of characters in some batch
        for _ in range(max_new_tokens):
            # first fetch predictions, loss is not used
            logits, _ = self.forward(idx) # TODO: how could this reference target() w/ just self(idx)?
            # fetch only the last time step NOTE: this can be expanded to a more general history for increased context_size
            logits = logits[:, -1, :] # becomes (B,C) -> TODO: what about previously?
            # apply softmax for probabilities
            probs = F.softmax(logits, dim=1) # (B,C)
            # sample from distribution the best prediction TODO: why not argmax or similar?
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx
    