import re
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

IMPORTANT_TAGS = set(["b", "strong", "h1", "h2", "h3"])

# Download and fetch stopwords list
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Initialize Porter Stemmer
stemmer = PorterStemmer()

# Helper function to tokenize and stem text
def tokenize_and_stem(text):
    tokens = word_tokenize(text.lower())  # Tokenization
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and len(token) >= 3]  # Stemming
    return stemmed_tokens

# Function to calculate tf-idf score
def calculate_tf_idf(term_freq, total_docs, doc_freq):
    tf = 1 + math.log(term_freq, 10) if term_freq > 0 else 0
    idf = math.log(total_docs / (doc_freq + 1), 10)
    return tf * idf