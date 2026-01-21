# scratch script to play around with self-attention logic
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 2 # batch, time, channels
# NOTE: we want tokens in the time dimension to communicate with past tokens, not future
# one of the simplest form of context communication is summing or averaging the current time's channels w/ the pasts
x = torch.randn(B,T,C)
print(x.shape)

# quick test with torch.mean()
a = torch.randn(2, 4) # t = 2, C = 4
print(f"a: {a}")
print(f"mean 0: {torch.mean(a, 0)}") # mean across dimension 0... aka across time -> leaves (C) tensor
print(f"mean 1: {torch.mean(a, 1)}") # mean across dimension 1... aka across channel -> leaves (t) tensor
print(f"all: {torch.mean(a)}") # reduces down to (1) tensor

# begin with averaging across time dimension for each channel value
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # becomes tensor of shape (t,C) where T dimension has been spliced upto t+1
        # NOTE: bag-of-words for batch b and time t = average channel vals across past times -> xbow[b, t] = C 1-dim vector
        xbow[b, t] = torch.mean(xprev, 0) # takes the average of previous channels

print(f"original x[0]:\n{x[0]}")
print(f"bow average x[0]:\n{xbow[0]}")