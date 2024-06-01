import os
import json
import pickle
from util import tokenize_and_stem
from util import compute_word_freq
from util import extract_text_from_json
from bs4 import BeautifulSoup

INDEX_DIR = "./DEV/"
DUPLICATES = set()

def load_duplicates():
    try:
        with open("duplicates.pkl", "rb") as f:
            return pickle.load(f)
    except:
        print("Duplicates not found")
        return set()

def extract_anchors(soup):
    # Parse the HTML and extract anchor texts and their URLs
    anchors = [(a.get_text(), a.get('href')) for a in soup.find_all('a') if a.get('href')]
    return anchors

def traverse_and_process():
    result_map = {}
    curr = 0
    for root, dirs, files in os.walk(INDEX_DIR):
        for file_name in files:
            curr += 1
            print(f'Parsing file #{curr}')
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)
                if file_path not in DUPLICATES:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        soup = extract_text_from_json(data)
                        anchors = extract_anchors(soup)
                        for anchor in anchors:
                            tokens = tokenize_and_stem(anchor[0])
                            result_map[anchor[1]] = compute_word_freq(tokens)
    return result_map

def store_anchor_map(anchor_map):
    with open('anchors.pkl', 'wb') as file:
        pickle.dump(anchor_map, file)

DUPLICATES = load_duplicates()
result = traverse_and_process()
store_anchor_map(result)