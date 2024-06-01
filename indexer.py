import os
import json
import pickle
import math
from bs4 import BeautifulSoup
from collections import defaultdict
from util import tokenize_and_stem
from util import extract_important_text
from util import extract_text_from_json
from urllib.parse import urldefrag

DATA_DIR = "./DEV/"
# DATA_DIR = "./ANALYST/"
INDEX_DIR = "index_files"
PARTIAL_INDEX_PREFIX = "partial_index"
ALPHA = set("qwertyuiopasdfghjklzxcvbnm")
ID_URL = dict()
ID_SIZE = dict()
ID_PATH = dict()
VISITED = set()
DUPLICATES = set()
ANCHORS = dict()
id_count = 0

TAG_SCORES = {
    'title': .6,
    'h1': .4,
    'h2': .3,
    'h3': .3,
    'strong': .2,
    'b': .1,
    None: 1  # Default score for text with no tag
}

# Store partial index on disk
def store_partial_index(index, index_num):
    index_file = f"{PARTIAL_INDEX_PREFIX}_{index_num}.pkl"
    with open(os.path.join(INDEX_DIR, index_file), 'wb') as file:
        pickle.dump(index, file)

# Iterate through partial indexes and separate them into indexes based on its first letter
def merge_and_separate_indexes(index_files):
    # Open txt file for each letter
    letter_files = {letter: open(os.path.join(INDEX_DIR, f"{letter}_tokens.txt"), "a") for letter in ALPHA}

    # Open file for token that don't start with alphabetic characters
    non_alpha_file = open(os.path.join(INDEX_DIR, "non_alpha_tokens.txt"), "a")

    # Dict that maps document ID to their position in the index
    token_line_map = defaultdict(int)

    # Iterate through all partial indexes
    for index_file in index_files:
        print(f"Merging {index_file}")
        with open(index_file, 'rb') as file:
            partial_index = pickle.load(file)
            for token, postings in partial_index.items():
                if token:
                    first_letter = token[0].lower()
                    current_file = non_alpha_file
                    
                    if first_letter in ALPHA:
                        current_file = letter_files[first_letter]
                    token_line_map[token] = current_file.tell()
                    current_file.write(f"{postings}\n")

    # Close all the files
    for file in letter_files.values():
        file.close()
    non_alpha_file.close()

    # Save the token to line number mapping
    with open('token_line.pkl', 'wb') as file:
        pickle.dump(token_line_map, file)

# Sort the postings by their score
def sort_by_score(inverted_index):
    for postings in inverted_index.values():
        postings.sort(key=lambda x: x[1], reverse=True)

# Return list of tuples as (token, tag)
def extract_text_with_tags(soup):
    global id_count
    tokens_with_tags = []
    for tag, score in TAG_SCORES.items():
        if tag:
            for element in soup.find_all(tag):
                text = element.get_text()
                tokens_with_tags.extend([(token, tag) for token in tokenize_and_stem(text)])
        else:
            # Extract text not within specific tags
            all_text = soup.get_text()
            tokens_with_tags.extend([(token, None) for token in tokenize_and_stem(all_text)])
            ID_SIZE[id_count] = len(tokens_with_tags)
    return tokens_with_tags

# Traverse over json files to build inverted index
def build_inverted_index_from_json(file_paths):
    global id_count, ID_URL
    inverted_index = defaultdict(list)
    for file_path in file_paths:
        # EXTRA CREDIT: Remove Near Duplicates
        if file_path in DUPLICATES:
            continue

        with open(file_path, 'r', encoding='utf-8') as json_file:
            json_content = json.load(json_file)
            # Remove fragment
            uri_no_fragment = urldefrag(json_content["url"])[0]
            if uri_no_fragment in VISITED:
                continue
            VISITED.add(uri_no_fragment)

            soup = extract_text_from_json(json_content)
            tokens_with_tags = extract_text_with_tags(soup)
            term_freq = defaultdict(int)
            term_scores = defaultdict(float)
            term_positions = defaultdict(list)

            # Build dict that maps from document ID to URL
            doc_id = id_count
            ID_URL[doc_id] = json_content["url"]
            ID_PATH[doc_id] = file_path
            id_count += 1

            for position, (token, tag) in enumerate(tokens_with_tags):
                if tag == None:
                    term_freq[token] += 1
                    term_positions[token].append(position)
                term_scores[token] += TAG_SCORES[tag]

            # EXTRA CREDIT: Index Anchor Words for the Target Pages
            anchor_map = dict()
            if json_content["url"] in ANCHORS:
                anchor_map = ANCHORS[json_content["url"]]
            
            for token, freq in anchor_map.items():
                term_freq[token] += freq
                term_scores[token] += freq

            # Isolate tokens
            tokens = [token for token, tag in tokens_with_tags if tag == None]
            
            # EXTRA CREDIT: Process 2-grams
            for i in range(len(tokens) - 1):
                two_gram = tokens[i] + ' ' + tokens[i + 1]
                term_freq[two_gram] += 1
                term_scores[two_gram] += 1

            # EXTRA CREDIT: Process 3-grams
            for i in range(len(tokens) - 2):
                three_gram = tokens[i] + ' ' + tokens[i + 1] + ' ' + tokens[i + 2]
                term_freq[three_gram] += 1
                term_scores[three_gram] += 1

            # Populate inverted index
            for token, freq in term_freq.items():
                score = term_scores[token]
                positions = term_positions[token]
                inverted_index[token].append((doc_id, freq, score, positions))
    return inverted_index

# Store dict that maps document ID to it's URL
def store_id_url():
    global ID_URL
    with open('id_url.pkl', 'wb') as file:
        pickle.dump(ID_URL, file)

# Store dict that maps document ID to it's number of tokens (for TDIDF calculations)
def store_id_size():
    global ID_SIZE, VISITED
    ID_SIZE["num_of_docs"] = len(VISITED)
    with open('id_size.pkl', 'wb') as file:
        pickle.dump(ID_SIZE, file)

# Store dict that maps document ID to it's path in /DEV
def store_id_path():
    global ID_PATH
    with open('id_path.pkl', 'wb') as file:
        pickle.dump(ID_PATH, file)

# Sorts postings by thier score
def sort_postings_by_score(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    sorted_lines = []
    for line in lines:
        postings = eval(line)
        sorted_postings = sorted(postings, key=lambda x: x[2], reverse=True)
        sorted_line = f"{sorted_postings}\n"
        sorted_lines.append(sorted_line)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(sorted_lines)

def sort_all_inverted_indexes(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('_tokens.txt'):
            file_path = os.path.join(directory_path, filename)
            sort_postings_by_score(file_path)

# Load map with duplicate pages
def load_duplicates():
    global DUPLICATES
    with open("duplicates.pkl", "rb") as f:
        DUPLICATES = pickle.load(f)

# Load map with anchor text and the link
def load_anchors():
    global ANCHORS
    with open("anchors.pkl", "rb") as f:
        ANCHORS = pickle.load(f)

# Main function to run indexing and merging
def main():
    load_duplicates()
    load_anchors()
    file_paths = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file_name in files:
            if file_name.endswith(".json"):
                file_paths.append(os.path.join(root, file_name))

    total_files = len(file_paths)
    batch_size = 5000  # Number of files to process in each batch
    num_batches = total_files // batch_size + (1 if total_files % batch_size != 0 else 0)

    for i in range(num_batches):
        print(f"Processing batch {i+1} of {num_batches}")
        batch_files = file_paths[i * batch_size: (i + 1) * batch_size]
        partial_index = build_inverted_index_from_json(batch_files)
        store_partial_index(partial_index, i)

    # List all partial index files
    partial_index_files = [os.path.join(INDEX_DIR, f) for f in os.listdir(INDEX_DIR) if f.startswith(PARTIAL_INDEX_PREFIX)]

    merged_index = merge_and_separate_indexes(partial_index_files)

    print("Sorting inverted index by score")
    sort_all_inverted_indexes(INDEX_DIR)
    
    # sort_by_score(merged_index)
    store_id_url()
    store_id_size()
    store_id_path()
    print("Indexing Complete")

if __name__ == "__main__":
    main()
