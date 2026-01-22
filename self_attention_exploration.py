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
w2 = w2.masked_fill(tril == 0, float('-inf')) # find where values are 0 for lower triangular, replace with -inf
# NOTE: exponentiation of 0 is 1, and exponentiation of -inf is 0 -> leaves us with the same effect as starting w/ lower triangular 1's and normalizing.
w2 = F.softmax(w2, dim=1) # by taking softmax, we are calculating the same normalized row values (via exponentiating and normalizing each row)
xbow3 = w2 @ x # we should expect the same output
print(f"is xbow and xbow3 identical? {torch.allclose(xbow, xbow3)}")
# NOTE: we actually prefer approach 2 because it simulates self-attention more accurately: by back-filling -inf values with a seed, the model is holding attention to past tokens and interacting in different weights.

# now onto self-attention mechanism with single head
B,T,C = 4, 8, 32 # batch, time, channels
x = torch.randn(B,T,C)
head_size = 16 # the dimensions of the Query, Key, Value vectors
# NOTE: for now, the key, query, values are independent of each other when C is passed through
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x) # (B, T, head_size)
q = query(x) # (B, T, head_size)
v = value(x) # (B, T, head_size)
# NOTE: by taking the mat. mul. of key and query we define weight to have dot product of token i's key and token j's query at position (i, j)
weight = q @ k.transpose(-2, -1) * head_size**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T), NOTE: the batch size stays constant while T, H dims are transposed to MM
# we also divide by sqrt(head_size) to normalize the variance
print(f"self-attention weight variance: {weight.var()}")

# take the weight and mask with lower triangular matrix to "eliminate" attention to future tokens
tril = torch.tril(torch.ones(T,T))
# NOTE: instead of torch.zeros init, we keep the weight matrix and backfill the upper half with -inf and softmax.
weight = weight.masked_fill(tril == 0, float('-inf'))
weight = F.softmax(weight, dim=1)
# instead of taking the raw token x, we do mat. mul. with value to allow flexibility
# NOTE: the output[b, t, :] is the context-aware representation for token t in batch b, where weight's distribution contributes to the output
out = weight @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
print(f"self-attention out shape: {out.shape}\nExample (B=0, T=0):\n{out[0][0]}")
