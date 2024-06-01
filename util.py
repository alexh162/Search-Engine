import nltk
import time
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from collections import defaultdict


# TODO
IMPORTANT_TAGS = set(["title", "b", "strong", "h1", "h2", "h3"])

# Initialize Porter Stemmer
stemmer = PorterStemmer()

def extract_important_text(soup):
    important_texts = []
    for tag in IMPORTANT_TAGS:
        elements = soup.find_all(tag)
        for element in elements:
            important_texts.append(element.get_text())
    return important_texts

# Tokenize and stem text
def tokenize_and_stem(text):
    tokens = word_tokenize(text.lower())  # Tokenization
    valid_tokens = []
    for token in tokens:
        stemmed_token = stemmer.stem(token)
        valid_tokens.append(stemmed_token)  # Stemming

    return valid_tokens

# Extract text from JSON content
def extract_text_from_json(json_content):
    raw_text = ""
    if "content" in json_content:
        raw_text = json_content["content"]
    soup = BeautifulSoup(raw_text, "html.parser")
    return soup

def compute_word_freq(tokens):
    freq_map = defaultdict(int)
    for token in tokens:
        freq_map[token] += 1
    return dict(freq_map)

def query_gram(tokens):
    ngrams = []

    ngrams.extend(tokens)
    
    # 2-grams
    for i in range(len(tokens) - 1):
        ngrams.append(tokens[i] + ' ' + tokens[i + 1])
    
    # 3-grams
    for i in range(len(tokens) - 2):
        ngrams.append(tokens[i] + ' ' + tokens[i + 1] + ' ' + tokens[i + 2])
    
    return ngrams

def current_milli_time():
    return round(time.time() * 1000)