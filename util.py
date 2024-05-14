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
    valid_tokens = []
    for token in tokens:
        stemmed_token = stemmer.stem(token)
        if token not in stop_words and stemmed_token not in stop_words and len(stemmed_token) >= 3:
            valid_tokens.append(stemmed_token)  # Stemming

    return valid_tokens