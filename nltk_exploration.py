# exploring the nltk library
# word tokenizer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt
# frequency distribution
from nltk.probability import FreqDist
# stop words
from nltk.corpus import stopwords
# lemmatization
from nltk.stem import WordNetLemmatizer, PorterStemmer
# sentiment analyzer
from nltk.sentiment import SentimentIntensityAnalyzer

# download punkt tokenizer
# NOTE: this can be a one-time download and commented thereafter
nltk.download('punkt')
nltk.download('punkt_tab')

text = "Hello, world. Don't skip this part! This is an example sentence for the NLTK world."

# 1) tokenization & distribution graphs
print(f"1) Tokenization Exploration\n")
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
print(f"\n2) Stop Words Exploration\n")
# download package for stopwords
nltk.download('stopwords')

# observe stopwords in english provided by NLTK library
# NOTE: use other libraries for different set of stop words
stopwords = stopwords.words('english')
print(f"stop words: {stopwords}")

# NOTE: by using stopwords as a set(), we can easily work with tokens to remove stop words
stopwords_set = set(stopwords)
filtered_tokens = [token for token in tokens if token not in stopwords_set]

# display the filtered tokens and the original for comparison
print(f"filtered tokens: {filtered_tokens}")
print(f"original tokens: {tokens}")
print(f"words filtered out: {set(tokens) - set(filtered_tokens)}")

# 3) lemmatization + stemming
print(f"\n3) Lemmazation + Stemming Exploration\n")
# download package for lemmatization
nltk.download('wordnet')

# test words to compare lemmatization vs stemming
test_words = ["better", "studies", "doing", "do", "going", "go", "gone", "have", "had", "has", "having", "I", "you", "coding"]
# init lemmatizer + stemmer
lemmatizer = WordNetLemmatizer() # uses intelligent WordNet dictionary-based lemmatization
stemmer = PorterStemmer() # uses rule-based, faster stemming

# compare lemmatization and stemming
print(f"Comparing lemmatization and stemming for the test words.")
print(f"{'Original':<15} | {'Lemmatization':<15} | {'Stemming':<15}")
print("-" * 50)
for word in test_words:
    # NOTE: lemmatization can take args for verb form, noun form, etc
    # if lemmatizer can't find word in the dictionary corpus for the given part of speech, it will return the original word
    print(f"{word:<15} | {lemmatizer.lemmatize(word, pos='v'):<15} | {stemmer.stem(word):<15}")

# 4) sentiment analysis
print(f"\n4) Sentiment Analysis Exploration\n")
# download package for sentiment analysis
nltk.download('vader_lexicon')
# TODO: complete this section