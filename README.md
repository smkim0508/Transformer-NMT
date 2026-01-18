# Transformer-NMT
Refining my PyTorch skills to build models from scratch [2] - Re-creating Transformer-based Neural Machine Translation model with NLTK library.
*This README will also contain my personal notes as I review key concepts and practice with examples.*
- This project is also to be used as custom tutoring material in the future

## NOTES:
### NLTK Library
- Allows conveinent, easy parsing of natural language data, used for various NLP tasks and applied models like LLMs.
- Commonly use `punkt`: pretrained, unsupervised model to parse text into tokens based on heuristics.
    - Other tokenizer models can be accessed via the nltk library as well.
- `nltk.download()` downloads sub modules provided by the nltk library; empty argument to view GUI or place package names when instantiating.
    - Use this to download punkt at the beginning of program: `nltk.download('punkt')`.
- `nltk.tokenize.word_tokenize()` splits text into meaningful tokens.
- `nltk.tokenize.sent_tokenize()` splits text by sentences.
- `nltk.probability.FreqDist(tokens)` to turn tokens into frequency distribution.
    - Pair with matlab pyplot to diplay graph.
### NLP
- Stop words are a set of words filtered out during an NLP task to imporve accuracy and remove noise/uncertainty.
    - No universal set of stop words for all tasks
    - Early NLP task/models focused more on topic not deep position/contextual information, so stop words included pronouns like "he"/"she", but these can be important in more complex tasks.

### TODO:
- NLTK library exploration
- transformer architecture building
- NMT re-creation