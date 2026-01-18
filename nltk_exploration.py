# exploring the nltk library
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# download punkt tokenizer
# NOTE: this can be a one-time download and commented thereafter
nltk.download('punkt')
nltk.download('punkt_tab')

text = "Hello, world. Don't skip this part! This is an example sentence for the NLTK world."

# tokenization
tokens = word_tokenize(text)
print(f"word tokenization: {tokens}")

# sentence tokenization
sent_tokens = sent_tokenize(text)
print(f"sentence tokenization: {sent_tokens}")

# explore tokenization capabilities for tricky text examples.
tricky_text = "is this a sentence? my name is 'John Doe'. and My email is 'johndoe@.com'.. . .. . ."
tricky_tokens = word_tokenize(tricky_text)
print(f"tricky tokens: {tricky_tokens}")
tricky_sent_tokens = sent_tokenize(tricky_text)
print(f"tricky sent tokens: {tricky_sent_tokens}")
