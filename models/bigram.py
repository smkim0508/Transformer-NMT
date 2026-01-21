# bigram language model, predicting the next character given a SINGLE character
# begin with TinyShakespear Dataset

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
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])

test_text = "hello world!"
print(f"original: {test_text}, encoded: {encode(test_text)}, decoded: {decode(encode(test_text))}")