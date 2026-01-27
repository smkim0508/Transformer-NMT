# fine-tuning a pre-trained model w/ LoRA layer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from tqdm import tqdm

# LoRA
from models.lora import LoRAParametrization, linear_layer_parameterization
# Pre-trained model
from models.transformer import TransformerLanguageModel

# hyperparameters for fine-tuning
batch_size = 32
block_size = 256 # must match pretrained model
max_iters = 1000
eval_interval = 100
learning_rate = 1e-4 # lower LR for fine-tuning
eval_iters = 50
lora_rank = 4 # rank of LoRA matrices
lora_alpha = 1 # scaling factor

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# load fine-tuning dataset (different from pretraining data)
data_path = "data/fine_tuning_hamlet.txt"
with open(data_path, 'r', encoding="utf-8") as f:
    text = f.read()

# load checkpoint to get vocab mappings
checkpoint_path = "checkpoints/transformer_model_final.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

# use same vocab from pretrained model
stoi = checkpoint['stoi']
itos = checkpoint['itos']
vocab_size = checkpoint['vocab_size']
encode = lambda s: [stoi[ch] for ch in s if ch in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# prepare data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_split = train_data if split == "train" else val_data
    idx = torch.randint(high=len(data_split) - block_size, size=(batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in idx])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in idx])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def apply_lora_to_model(model, device, rank, alpha):
    """
    Apply LoRA to all Q, K, V projections in attention heads.
    This is where LoRA is most effective for language models.
    """
    for block in model.blocks:
        # Apply LoRA to each attention head's Q, K, V
        for head in block.sa.heads:
            parametrize.register_parametrization(
                head.query, "weight",
                linear_layer_parameterization(head.query, device, rank=rank, lora_alpha=alpha)
            )
            parametrize.register_parametrization(
                head.key, "weight",
                linear_layer_parameterization(head.key, device, rank=rank, lora_alpha=alpha)
            )
            parametrize.register_parametrization(
                head.value, "weight",
                linear_layer_parameterization(head.value, device, rank=rank, lora_alpha=alpha)
            )
    print(f"LoRA applied to all Q, K, V projections (rank={rank}, alpha={alpha})")

def freeze_base_weights(model):
    """Freeze all non-LoRA parameters."""
    for name, param in model.named_parameters():
        if 'lora' not in name.lower():
            param.requires_grad = False

def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

if __name__ == "__main__":
    print(f"Loading pretrained model from {checkpoint_path}...")

    # recreate model with saved hyperparameters
    model = TransformerLanguageModel(
        vocab_size=checkpoint['vocab_size'],
        n_embed=checkpoint['n_embed'],
        block_size=checkpoint['block_size'],
        n_layer=checkpoint['n_layer'],
        n_head=checkpoint['n_head'],
        dropout=checkpoint['dropout'],
        device=device
    )

    # load pretrained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("Pretrained weights loaded!")

    # apply LoRA
    apply_lora_to_model(model, device, rank=lora_rank, alpha=lora_alpha)

    # freeze base model weights (only train LoRA parameters)
    freeze_base_weights(model)

    # count parameters
    trainable, total = count_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # optimizer only for trainable params
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )

    # fine-tuning loop
    print(f"\nStarting fine-tuning for {max_iters} iterations...")
    for iter in tqdm(range(max_iters), desc="Fine-tuning"):
        if (iter + 1) % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"\nStep {iter+1}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # final evaluation
    final_losses = estimate_loss(model)
    print(f"\nFinal: train loss {final_losses['train']:.4f}, val loss {final_losses['val']:.4f}")

    # generate sample text
    print("\nGenerating sample text...")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = decode(model.generate(context, max_new_tokens=500)[0].tolist())
    print(generated)

    # save fine-tuned model (only LoRA weights for efficiency)
    lora_state_dict = {k: v for k, v in model.state_dict().items() if 'lora' in k.lower()}
    torch.save({
        'lora_state_dict': lora_state_dict,
        'full_state_dict': model.state_dict(),  # also save full for convenience
        'lora_rank': lora_rank,
        'lora_alpha': lora_alpha,
    }, 'checkpoints/transformer_lora_finetuned.pt')
    print("Fine-tuned model saved to checkpoints/transformer_lora_finetuned.pt")