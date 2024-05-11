import pickle

# Fetch the set of visited URLs
def fetch_inverted_index():
    v = set()
    try:
        f = open('./index_files/merged_index.pkl', 'rb')
        v = pickle.load(f)
        f.close()
    except Exception as e:
        print(e)
        pass
    return v


index = fetch_inverted_index()
print(len(fetch_inverted_index()))
for key, value in index.items():
    print(f"Key: {key}, Value: {len(value)}")