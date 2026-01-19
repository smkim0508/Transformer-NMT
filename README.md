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
- **Lemmatization** uses collected dictionary (e.g. WordNet) to intelligently reduce words to its base root (e.g. studies -> study, has -> have), whereas **Stemming** uses rule-based truncation logic that is often more harsh and non-intelligent (e.g. studies -> studi, has -> ha).
- Sentiment analysis can be done with static, non-ML based approach like VADER lexicon.
    - VADER is a dictionary of human-rated words, and heuristic rules determine the sentiment of given text.
    - Signals like emphasis, negation, intensity modifier (e.g. very), contribute to the scores.
    - Compound score is often most useful; even if ratio of scores where pos is 1.0 and rest are 0.0, the compound can hint to just how positive the given text was.
- Synonyms / Antonyms can be found using synsets. A synset is a similar concept to the original word.
    - The list of synsets for a given word in the dictionary is ordered by the frequency of the word's usage in the dictionary (e.g. "good" -> "good" as the first synset, but "automobile" -> "car" as it's first synset)
    - `synset.lemmas()` provides all synonym words for a synset.
    - Therefore, by collecting all lemmas for each synset of a given word (double loop), we can effectively collect all synonyms of a word, but this may contain duplicates.
    - Likewise, take `synset.antonyms()` if exists to collect all antonyms.

### TODO:
- NLTK library exploration
- transformer architecture building
- NMT re-creation