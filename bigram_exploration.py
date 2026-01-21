# exploring bigram model, tensors, chunking, etc.

import torch
import torch.nn as nn
from torch.nn import functional as F

# load in the dataset
# NOTE: the current test input is derived from my README in another project
with open("data/test_input.txt", 'r', encoding="utf-8") as t:
    text = t.read()

# find the unique chars occuring in the data, to define Vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {len(chars)}, Vocab: {''.join(chars)}")

# define simple encoder-decoder to map Vocab
stoi = {ch:i for i, ch in enumerate(chars)} # already sorted
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s] # NOTE: review lambda logic!
decode = lambda l: ''.join([itos[i] for i in l])

test_text = "hello world!"
print(f"original: {test_text}, encoded: {encode(test_text)}, decoded: {decode(encode(test_text))}")

# wrap the text in torch tensor 
data = torch.tensor(encode(text), dtype=torch.long)
print(f"data shape: {data.shape}, type: {data.dtype}")
print(f"sample encoding: {data[:50]}") # view the sample encoding

# split data into train and val
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# use only block size to train
block_size = 8
# the target at idx t+1 in the training data is expected to be output of context t-8 to t
# NOTE: by training w/ samples where context length is < 8, we can produce better results for small input cases
# observe example w/ the first block_size items in train data
x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context}, target is: {target}")

# sampling random locations in dataset
torch.manual_seed(1337) # for reproducibility, remove for truly random
batch_size = 4
block_size = 8 # NOTE: set again for testing purposes

def get_batch(split):
    """
    batches random chunks of data from train/val data
    """
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    idx = torch.randint(high = len(data) - block_size, size = (batch_size,)) # sets output tensor to be batch_size
    print(f"random idx chosen: {idx}")
    # sets the sample context and targets
    # NOTE: the ith target in one batch of y tensor is the correct prediction for the accumulation of 0 to ith items in the corresponding batch of x tensor.
    # e.g. 3rd tagret in 1st batch of y: 11, which is the expected next char given 0-2 chars in 1st batch of x: 69, 75, 68 -> 11
    x = torch.stack([data[i : i+block_size] for i in idx])
    y = torch.stack([data[i+1 : i+block_size+1] for i in idx])
    return x, y

# view outputs
xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

# now we can observe the previous input-target relationship but for the entire batch
for b in range(batch_size): # batch dim
    for t in range(block_size): # time dim, aka the passage of diff chars in context
        # NOTE: for each batch, we can take upto ith in x's context = ith target in y
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context}, target is: {target}")
