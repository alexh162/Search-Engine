import os
import json
import pickle
import math
from bs4 import BeautifulSoup
from collections import defaultdict
from util import tokenize_and_stem

# Constants
# DATA_DIR = "./DEV/"
DATA_DIR = "./ANALYST/"
INDEX_DIR = "index_files"
PARTIAL_INDEX_PREFIX = "partial_index"
NUM_OF_DOC = 55393
ALPHA = set("qwertyuiopasdfghjklzxcvbnm")

# Extract text from JSON content
def extract_text_from_json(json_content):
    raw_text = ""
    if "content" in json_content:
        raw_text = json_content["content"]
    soup = BeautifulSoup(raw_text, "html.parser")
    text = soup.get_text()
    return text

# Store partial index on disk
def store_partial_index(index, index_num):
    index_file = f"{PARTIAL_INDEX_PREFIX}_{index_num}.pkl"
    with open(os.path.join(INDEX_DIR, index_file), 'wb') as file:
        pickle.dump(index, file)

def separate_tokens_by_first_letter(tokens_dict):
    # Separate tokens by first letter
    letter_files = defaultdict(dict)
    non_alpha_file = defaultdict(list)

    for token, postings in tokens_dict.items():
        first_letter = token[0].lower()
        if first_letter in ALPHA:
            letter_files[first_letter][token] = postings
        else:
            non_alpha_file[token] = postings

    # Save letter-specific files
    for letter, tokens in letter_files.items():
        with open(os.path.join(INDEX_DIR, f"{letter}_tokens.pkl"), "wb") as f:
            pickle.dump(tokens, f)

    # Save non-alphabetic tokens file
    with open(os.path.join(INDEX_DIR, "non_alpha_tokens.pkl"), "wb") as f:
        pickle.dump(non_alpha_file, f)

def replace_tf_with_tfidf(inverted_index):
    for postings in inverted_index.values():
        doc_freq = len(postings)
        idf = math.log(NUM_OF_DOC / doc_freq)
        for i, post in enumerate(postings):
            post_list = list(post)
            post_list[2] = post_list[2] * idf
            postings[i] = tuple(post_list)

def sort_by_tfidf(inverted_index):
    for postings in inverted_index.values():
        postings.sort(key=lambda x: x[2], reverse=True)

# Merge partial indexes into final index
def merge_partial_indexes(index_files):
    merged_index = defaultdict(list)
    for index_file in index_files:
        with open(index_file, 'rb') as file:
            partial_index = pickle.load(file)
            for token, postings in partial_index.items():
                merged_index[token].extend(postings)
    return merged_index

def build_inverted_index_from_json(file_paths):
    inverted_index = defaultdict(list)
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            json_content = json.load(json_file)
            text = extract_text_from_json(json_content)
            tokens = tokenize_and_stem(text)
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            for token, freq in term_freq.items():
                inverted_index[token].append((json_content["url"], freq, freq/len(tokens)))
    return inverted_index

def store_inverted_index(inverted_index):
    with open('index.pkl', 'wb') as file:
        pickle.dump(inverted_index, file)

# Main function to run indexing and merging
def main():
    file_paths = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file_name in files:
            if file_name.endswith(".json"):
                file_paths.append(os.path.join(root, file_name))

    total_files = len(file_paths)
    batch_size = 500  # Number of files to process in each batch
    num_batches = total_files // batch_size + (1 if total_files % batch_size != 0 else 0)

    for i in range(num_batches):
        batch_files = file_paths[i * batch_size: (i + 1) * batch_size]
        partial_index = build_inverted_index_from_json(batch_files)
        store_partial_index(partial_index, i)

    # List all partial index files
    partial_index_files = [os.path.join(INDEX_DIR, f) for f in os.listdir(INDEX_DIR) if f.startswith(PARTIAL_INDEX_PREFIX)]
    merged_index = merge_partial_indexes(partial_index_files)
    replace_tf_with_tfidf(merged_index)
    sort_by_tfidf(merged_index)
    separate_tokens_by_first_letter(merged_index)

    # Store merged index on disk
    with open(os.path.join(INDEX_DIR, 'merged_index.pkl'), 'wb') as merged_file:
        pickle.dump(merged_index, merged_file)

if __name__ == "__main__":
    main()
