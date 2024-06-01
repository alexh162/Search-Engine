import hashlib
import os
import json
import pickle
from util import tokenize_and_stem
from util import extract_text_from_json


INDEX_DIR = "./DEV/"

def hash_token(token):
    # Generate a hash for the token
    hash_value = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
    # Convert the hash to a binary string
    bit_hash = bin(hash_value)[2:].zfill(128)  # Ensure it's 128 bits
    return bit_hash

def distance(hash1, hash2):
    x = hash1 ^ hash2
    dist = 0
    while x:
        dist += 1
        x &= x - 1
    return dist

def simhash(tokens):
    vector = [0] * 128  # Initialize vector 

    for token in tokens:
        bit_hash = hash_token(token)
        for i, bit in enumerate(bit_hash):
            if bit == '1':
                vector[i] += 1
            else:
                vector[i] -= 1

    # Construct the simhash value
    simhash_value = 0
    for i, value in enumerate(vector):
        if value > 0:
            simhash_value |= (1 << (127 - i))
    return simhash_value

def store_duplicates(dup_set):
    with open('duplicates.pkl', 'wb') as file:
        pickle.dump(dup_set, file)

def find_duplicates(folder_path, threshold=3):
    simhash_dict = {}
    duplicates = set()
    curr = 0

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            curr += 1
            print(curr)
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)

                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    soup = extract_text_from_json(data)
                    content = soup.get_text()
                    tokens = tokenize_and_stem(content)
                    simhash_value = simhash(tokens)

                    dup_found = False
                    for existing_hash in simhash_dict.values():
                        if distance(simhash_value, existing_hash) <= threshold:
                            dup_found = True
                            duplicates.add(file_path)
                            break
                    
                    if not dup_found:
                        simhash_dict[file_path] = simhash_value
    return duplicates

if __name__ == "__main__":
    dups = find_duplicates(INDEX_DIR)
    store_duplicates(dups)