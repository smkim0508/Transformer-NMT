# exploring bigram model, tensors, chunking, etc.
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.bigram import BigramLanguageModel
from tqdm import tqdm

# load in the dataset
# NOTE: the current test input is derived from my README in another project
with open("data/test_input.txt", 'r', encoding="utf-8") as t:
    text = t.read()

# set device to CPU strictly for exploration, needed for bigram model
device = "cpu"

# find the unique chars occuring in the data, to define Vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
n_embed = 32
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

def get_batch(split, batch_size, block_size):
    """
    batches random chunks of data from train/val data
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
    return x, y

# view outputs
xb, yb = get_batch('train', batch_size=batch_size, block_size=block_size)
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

# test the bigram model
model = BigramLanguageModel(
    vocab_size=vocab_size,
    n_embed=n_embed,
    block_size=block_size,
    n_head=4,
    n_layer=4,
    dropout=0.2,
    device=device
)
# pass xb as idx, xy as target
logits, loss = model(xb, yb) # TODO: verify how this refers to forward()
print(logits.shape) # we expect the shape to be (block_size * batch_size, vocab_size), since we stretched the dimensions in model definition
print(loss)
# NOTE: the CE loss expected without any training is -ln(1/vocab_size) = -ln(1/81) ~ 4.394
# TODO: verify the above statement and reason the discrepency in actual loss

# now experiment w/ token generation
idx = torch.zeros((1,1), dtype=torch.long) # test with a 1x1 tensor holding 0, to represent idx being 0 (the first item in Vocab)
# TODO: verify how idx being 1x1 tensor fits in to the above logic
# NOTE: take the 0th idx to fetch first batch of results, and convert to simple python list (from tensor) to feed into decoder
print(f"random tokens generated: {decode(model.generate(idx, max_new_tokens=100)[0].tolist())}") # the output here is expected to be non-sensical

# now we can experiment w/ training to see the difference

# optimizer set to Adam
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # TODO: diff between Adam and AdamW?

# train w/ larger batch size
batch_size = 32
steps = 10000
for step in tqdm(range(steps)):
    # sample a batch of data
    xb, yb = get_batch('train', batch_size=batch_size, block_size=block_size)
    # eval loss
    logits, loss = model.forward(xb, yb)
    if not loss:
        print(f"Error: loss must exist, please verify target is set")
        break
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"loss after {steps} steps: {loss.item() if loss else None}")
# now try to generate randomly again to compare differences
# NOTE: the output here should still be non-sensical but more "structured" than previous
print(f"tokens generated after preliminary training: {decode(model.generate(idx, max_new_tokens=200)[0].tolist())}")
