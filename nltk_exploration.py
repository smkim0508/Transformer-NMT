# exploring the nltk library
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt
from nltk.probability import FreqDist

# download punkt tokenizer
# NOTE: this can be a one-time download and commented thereafter
nltk.download('punkt')
nltk.download('punkt_tab')

text = "Hello, world. Don't skip this part! This is an example sentence for the NLTK world."

# 1) tokenization & distribution graphs
# word tokenization
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

# see frequency distribution and plot w/ matplotlib
fd = FreqDist(tokens)
fd.plot(20, cumulative=False, title="freq. distribution for normal text tokens") # cumulative = True shows cumulative distribution
plt.show() # display graph with matplotlib, fd.plot() is integrated

# 2) stop words
# download package for stopwords
nltk.download('stopwords')

# observe stopwords in english provided by NLTK library
# NOTE: use other libraries for different set of stop words
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
print(f"stop words: {stopwords}")

# NOTE: by using stopwords as a set(), we can easily work with tokens to remove stop words
stopwords_set = set(stopwords)
filtered_tokens = [token for token in tokens if token not in stopwords_set]

# display the filtered tokens and the original for comparison
print(f"filtered tokens: {filtered_tokens}")
print(f"original tokens: {tokens}")
print(f"words filtered out: {set(tokens) - set(filtered_tokens)}")
