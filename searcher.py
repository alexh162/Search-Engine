import pickle
from util import tokenize_and_stem
from util import current_milli_time
import os
import math
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from util import extract_text_from_json
from util import query_gram
from util import compute_word_freq

ALPHA = set("qwertyuiopasdfghjklzxcvbnm")
INDEX_DIR = "index_files"

class Searcher:

    def __init__(self):
        self.token_line = {} # Map token to its position in file
        self.id_size = {} # Maps document ID to the number of tokens in text
        self.id_url = {} # Maps document ID to its URL
        self.id_path = {} # Maps document ID to its path in DEV folder
        self.initialize_maps()
        self.num_of_docs = self.id_size["num_of_docs"]
    
    def initialize_maps(self):
        self.token_line = self.load_pickle_file("token_line.pkl")
        self.id_size = self.load_pickle_file("id_size.pkl")
        self.id_url = self.load_pickle_file("id_url.pkl")
        self.id_path = self.load_pickle_file("id_path.pkl")
    
    def count_files_in_directory(directory):
        file_count = 0
        for root, dirs, files in os.walk(directory):
            file_count += len(files)
        return file_count
    
    # Loads a pickle file given the file name
    def load_pickle_file(self, filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"{filename} file not found.")
            return {}

    # Returns a list of the posting given a token
    def get_postings(self, token):
        # Tokens are split into indexes based on their first character
        first_letter = token[0].lower()
        if first_letter not in ALPHA:
            first_letter = 'non_alpha'

        file_name = f"{first_letter.lower()}_tokens.txt"
        file_path = os.path.join(INDEX_DIR, file_name)

        try:
            if token in self.token_line:
                line_number = self.token_line[token]
                with open(file_path, 'r') as file:
                    file.seek(line_number)
                    line = file.readline()

                    if line:
                        postings = eval(line)
                        return postings
            return []
        except FileNotFoundError:
            print(f"Inverted index file for '{token}' not found.")
            return {}

    def normalize(self, vector):
        vector = np.array(vector, dtype=float)
        length = np.nansum(vector**2)**0.5
        vector = vector/length
        return vector

    def search(self, query, top_n=5):
        query_terms = tokenize_and_stem(query)

        # EXTRA CREDIT: 2-grams & 3-grams
        query_terms_with_grams = query_gram(query_terms)

        query_dict = compute_word_freq(query_terms_with_grams)
        query_list = list(query_dict.keys())


        query_weights = dict()
        doc_scores = {}

        # Create tfidf vector for query
        query_len = 0
        for i in range(len(query_list)):
            term = query_list[i]
            freq = query_dict[term]
            
            # Calculate TF
            query_weights[term] = dict()
            tf = freq/len(query_terms)
            # Calculate IDF
            query_weights[term]["post"] = self.get_postings(term)
            post_size = 1
            if term in self.token_line:
                post_size = len(query_weights[term]["post"])
            query_weights[term]["idf"] = math.log10(self.num_of_docs/post_size)
            
            query_weights[term]['tfidf'] = tf * query_weights[term]["idf"]
            query_len += query_weights[term]['tfidf'] ** 2
        query_len = math.sqrt(query_len)
        # Normalize query
        for term in query_weights.keys():
            query_weights[term]["weight"] = query_weights[term]["tfidf"]/query_len


        # Document vector
        doc_weights = dict()
        for i in range(len(query_list)):
            term = query_list[i]
            posting = query_weights[term]["post"]

            for id, freq, _, _ in posting:
                if id not in doc_weights:
                    doc_weights[id] = [0 for _ in query_list]
                tf = freq/self.id_size[id]
                doc_weights[id][i] = tf * query_weights[term]["idf"]
        

        # Normalize the document vector
        normed = {document: self.normalize(doc_weights[document]) for document in doc_weights}
        
        # Calculate mean TF-IDF score for each document
        mean_scores = {document: np.mean([doc_weights[document]]) for document in doc_weights}

        # Select top documents based on mean TF-IDF scores
        top_docs = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)[:100]
        top_docs = [doc[0] for doc in top_docs]

        # Calculate cosine similarity for the top documents only
        query_weight_vector = [query_weights[term]["weight"] for term in query_list]
        scores = {document: cosine_similarity([query_weight_vector], [normed[document]])[0][0] for document in top_docs}

        # Sort documents based on similarity scores
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Return top results
        top_results = [doc for doc, _ in sorted_docs]
        return top_results

    def and_search(self, query):
        query_terms = query.split(" AND ")  # Split the query on "AND"
        ranked_docs = set(self.search(query_terms[0]))
        for term in query_terms[1:]:
            ranked_docs = ranked_docs.intersection(self.search(term))
        return ranked_docs
    
    def id_to_url(self, ids):
        return [self.id_url[id] for id in ids]

    def id_to_path(self, ids):
        return [self.id_path[id] for id in ids]
    
    def id_to_content(self, ids):
        paths = [self.id_path[id] for id in ids]
        content = []
        for path in paths:
            with open(path, 'r', encoding='utf-8') as json_file:
                json_content = json.load(json_file)
                soup = extract_text_from_json(json_content)
                content.append(soup.get_text())
        return content

    def run(self):
        self.search("computer")
        while True:
            query = input("Enter your search query: ")
            start_time = current_milli_time()

            if " AND " in query:
                ranked_docs = self.and_search(query)
            else:
                ranked_docs = self.search(query)
            
            print(f'Search Time: {current_milli_time() - start_time} milliseconds')
            urls = self.id_to_url(ranked_docs)
            paths = self.id_to_path(ranked_docs)
            # print(paths)

            if ranked_docs:
                for line, url in enumerate(urls, start=1):
                    print(f"{line}: {url}")
            else:
                print("No results found for the query.")

if __name__ == "__main__":
    searcher = Searcher()
    searcher.run()