import os
import json
import time
import pickle
from bs4 import BeautifulSoup
from collections import defaultdict
from util import tokenize_and_stem
from util import calculate_tf_idf

# Constants
DATA_DIR = "./DEV/"
# DATA_DIR = "./ANALYST/"
INDEX_DIR = "index_files"
PARTIAL_INDEX_PREFIX = "partial_index"

# Extract text from JSON content
def extract_text_from_json(json_content):
    text = ""
    if "content" in json_content:
        text = json_content["content"]
    return text

# Store partial index on disk
def store_partial_index(index, index_num):
    index_file = f"{PARTIAL_INDEX_PREFIX}_{index_num}.pkl"
    with open(os.path.join(INDEX_DIR, index_file), 'wb') as file:
        pickle.dump(index, file)

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
        print(f'Currently parsing {file_path}')
        with open(file_path, 'r', encoding='utf-8') as json_file:
            json_content = json.load(json_file)
            text = extract_text_from_json(json_content)
            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text()
            tokens = tokenize_and_stem(text)
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            for token, freq in term_freq.items():
                inverted_index[token].append((json_content["url"], freq))
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

    # Store merged index on disk
    with open(os.path.join(INDEX_DIR, 'merged_index.pkl'), 'wb') as merged_file:
        pickle.dump(merged_index, merged_file)

if __name__ == "__main__":
    main()
