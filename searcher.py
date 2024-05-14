import pickle
from util import tokenize_and_stem
import time
import os

ALPHA = set("qwertyuiopasdfghjklzxcvbnm")
INDEX_DIR = "index_files"


def current_milli_time():
    return round(time.time() * 1000)

def load_inverted_index(token):
    file_name = f"{token.lower()}_tokens.pkl"
    try:
        with open(os.path.join(INDEX_DIR, file_name), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Inverted index file for '{token}' not found.")
        return {}

def search(query, top_n=10):
    query_terms = tokenize_and_stem(query)
    relevant_docs = {}

    for term in query_terms:
        index = load_inverted_index(term[0])  # Load the inverted index based on the first character of the term
        if term in index:
            relevant_docs[term] = index[term][:top_n]

    # Flatten the relevant_docs dictionary and sort by TF-IDF score
    flattened_docs = [doc for docs_list in relevant_docs.values() for doc in docs_list]
    flattened_docs.sort(key=lambda x: x[2], reverse=True)  # Sort by TF-IDF score in descending order

    # Return top n relevant documents
    return flattened_docs[:top_n]

def main():
    while True:
        query = input("Enter your search query: ")
        start_time = current_milli_time()

        if " AND " in query:
            query_terms = query.split(" AND ")  # Split the query on "AND"
            ranked_docs = set([posting[0] for posting in search(query_terms[0])])
            for term in query_terms[1:]:
                ranked_docs = ranked_docs.intersection([posting[0] for posting in search(term)])
        else:
            ranked_docs = [posting[0] for posting in search(query)]
        
        print(f'Search Time: {current_milli_time() - start_time} milliseconds')
        if ranked_docs:
            for line, doc in enumerate(ranked_docs, start=1):
                print(f"{line}: {doc}")
        else:
            print("No results found for the query.")

if __name__ == "__main__":
    main()
