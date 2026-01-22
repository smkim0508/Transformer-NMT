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

# Below, we will try to reproduce the above results in more efficient ways, w/ math trick instead of nested loops
# NOTE: with the use of lower triangular matrix of normalized row sum to 1, matrix multiplication produces a running average
# see example below for clarity
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True) # dividng by sum across dim 1 gives each row normalized to 1
b = torch.randint(low=0, high=10, size=(3, 2)).float()
c = a @ b # matrix multiply
print(f"a:\n{a}")
print(f"b:\n{b}")
print(f"c:\n{c}")

# now actually make the previous computation efficient
# approach 1) lower triangular matrix then normalizing rows
w = torch.tril(torch.ones(T, T)) # the dimensions should be T
w = w / w.sum(1, keepdim=True)
print(f"weights:\n{w}")
# NOTE: since (T, T) @ (B, T, C) isn't compatible in dimensions, PyTorch inserts a third batch dim: (T, T) -> (B, T, T)
xbow2 = w @ x # this becomes a batched (T, T) @ (T, C) multiplication -> (B, T, C)
print(f"is xbow and xbow2 identical? {torch.allclose(xbow, xbow2)}") # this compares the two tensors, we expect TRUE if identical

# approach 2) back-filling the weight matrix w/ masked_fill() and softmax()
tril = torch.tril(torch.ones(T, T)) # create a lower triangular matrix
w2 = torch.zeros((T, T))
w2 = w2.masked_fill(tril == 0, float('-inf')) # find where values are 0, replace with -inf
w2 = F.softmax(w2, dim=1) # by taking softmax, we are calculating the same normalized row values (via exponentiating and normalizing each row)
xbow3 = w2 @ x # we should expect the same output
print(f"is xbow and xbow3 identical? {torch.allclose(xbow, xbow3)}")
# NOTE: we actually prefer approach 2 because it simulates self-attention more accurately: by back-filling -inf values with a seed, the model is holding attention to past tokens and interacting in different weights.
