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

# NOTE: we should expect ith time idx of xbow[0] to be the average of 0 to ith time idx of x[0]
print(f"original x[0]:\n{x[0]}")
print(f"bow average x[0]:\n{xbow[0]}")

# using a mathematical trick, we can try to be very EFFICIENT with this average computation, instead of nested loops
# NOTE: with the use of lower-half triangular matrix of normalized row sum to 1, matrix multiplication produces a running average
# see example below for clarity
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True) # dividng by sum across dim 1 gives each row normalized to 1
b = torch.randint(low=0, high=10, size=(3, 2)).float()
c = a @ b # matrix multiply
print(f"a:\n{a}")
print(f"b:\n{b}")
print(f"c:\n{c}")