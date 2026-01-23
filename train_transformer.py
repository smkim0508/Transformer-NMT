# training the full transformer language model
# torch essentials
import torch
import torch.nn as nn
from torch.nn import functional as F

# training monitoring
from tqdm import tqdm

# models
from models.bigram import BigramLanguageModel
from models.transformer import TransformerLanguageModel

# hyperparameters
batch_size = 64 # number of independent sequences in parallel
block_size = 256 # max context window size for prediction
max_iters = 5000
eval_interval = 500 # used to average loss during train for logging
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu" # use GPU if available
eval_iters = 200
n_embed = 384 # dims for token embeddings
n_head = 6 # number of heads in multi-head attention
n_layer = 6 # number of transformer blocks
dropout = 0.2 # 20% of intermediate nodes are disabled at random
# NOTE: implies head_size = n_embed//n_head = 64//4 = 16

torch.manual_seed(1337) # for reproducibility

# load in the dataset
data_path = "data/input.txt"
# NOTE: the current test input is derived from my README in another project
with open(data_path, 'r', encoding="utf-8") as t:
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

# helper functions defined below
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
    model.eval() # set to eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size=batch_size, block_size=block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item() # accumulate loss
        out[split] = losses.mean() # average loss for each split
    model.train() # return back to training mode
    return out

def save_checkpoint(model, iter):
    """
    save model checkpoint to refernece in the future
    """
    # save model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_size': vocab_size,
        'n_embed': n_embed,
        'block_size': block_size,
        'n_head': n_head,
        'n_layer': n_layer,
        'dropout': dropout,
        'stoi': stoi,
        'itos': itos,
    }
    torch.save(checkpoint, f"checkpoints/transformer_model_{iter}.pt")
    # disabled print for now to not clutter terminal
    # print(f"Model checkpoint saved to checkpoints/transformer_model_{iter}.pt")

if __name__ == "__main__":
    print(f"Initializing model...")
    # actual training loop defined here
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        n_embed=n_embed,
        n_head=n_head,
        block_size=block_size,
        n_layer=n_layer,
        dropout=dropout,
        device=device
    )
    m = model.to(device) # use GPU if available
    print(f"Transformer model initialized on {device}!")

    # optimizer set to Adam
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    losses = []
    # training loop
    for iter in tqdm(range(max_iters), desc="Training Model"):
        # calculate the average loss every eval_interval
        if (iter+1) % eval_interval == 0:
            # NOTE: since we have a progress bar, avoid printing loss until the very end
            loss = (estimate_loss(m))
            loss.update({'iter': iter})
            losses.append(loss)
            # save checkpoint at this point
            save_checkpoint(model, iter+1)

        # sample batch data
        xb, yb = get_batch('train', batch_size=batch_size, block_size=block_size)

        # eval the loss and update weights
        logits, loss = m(xb, yb)
        if not loss:
            print(f"Error: loss must exist, please verify target is set")
            break
        optimizer.zero_grad(set_to_none=True) # init gradients
        loss.backward()
        optimizer.step() # steps down gradients

    for loss in losses:
        print(f"Step {loss['iter']}: train loss {loss['train']:.4f}, val loss {loss['val']:.4f}")
    
    # generate text from model
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    generated_text = decode(m.generate(context, max_new_tokens=500)[0].tolist())
    print(f"Generated text:\n{generated_text}")

    # save generated text to file
    with open('outputs/generated_text.txt', 'w', encoding='utf-8') as f:
        f.write(generated_text)
    print("Generated text saved to outputs/generated_text.txt")

    # do a final save for checkpoint
    save_checkpoint(model, "final")
    print("Final model checkpoint saved to checkpoints/transformer_model_final.pt")
