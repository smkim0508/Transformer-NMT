# training the bigram language model
import torch
import torch.nn as nn
from torch.nn import functional as F

# models
from models.bigram import BigramLanguageModel

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_intervals = 300 # used for averaging loss during train
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu" # use GPU if available TODO: does tensor.to(device) not affect for CPU?
eval_iters = 200

torch.manual_seed(1337) # for reproducibility

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

# wrap the text in torch tensor and split into train/val
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# define helper functions below
def get_batch(split, batch_size, block_size):
    """
    Batches random chunks of data from train/val data
    NOTE: this essentially mimics torch's dataloader functionality
    """
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    idx = torch.randint(high = len(data) - block_size, size = (batch_size,)) # sets output tensor to be batch_size
    # print(f"random idx chosen: {idx}")

    # sets the sample context and targets
    # NOTE: the ith target in one batch of y tensor is the correct prediction for the accumulation of 0 to ith items in the corresponding batch of x tensor.
    # e.g. 3rd tagret in 1st batch of y: 11, which is the expected next char given 0-2 chars in 1st batch of x: 69, 75, 68 -> 11
    x = torch.stack([data[i : i+block_size] for i in idx])
    y = torch.stack([data[i+1 : i+block_size+1] for i in idx])
    x, y = x.to(device), y.to(device) # move tensors to GPU if available
    return x, y

@torch.no_grad() # PyTorch avoids computing/tracking gradient for efficiency, this is purely for logging purposes
def estimate_loss(model):
    """
    When called, estimates the current average loss by iterating through train/val data again.
    - NOTE: could be combined w/ the actual training loss update to remove redundancy.
    - model is passed in to set eval/train stages
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size=batch_size, block_size=block_size)
            logits, loss = model.forward(X, Y)
            losses[k] = loss.item() # accumulate loss
        out[split] = losses.mean() # average loss for each split
    model.train()
    return out
