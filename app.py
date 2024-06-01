from flask import Flask, render_template, request
from openai import OpenAI
import os
import sys
import json
from dotenv import load_dotenv
from util import current_milli_time
from searcher import Searcher

app = Flask(__name__)

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_KEY'))
searcher = Searcher()

# EXTRA CREDIT: OpenAI API Summarization
def get_summary(text):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a search engine. Summarize this text from a website in two sentences"},
            {"role": "user", "content": text}
        ]
    )
    response = json.loads(completion.model_dump_json())
    return response['choices'][0]['message']['content']

@app.route('/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        searcher.search("test")
        query = request.form['query']

        start_time = current_milli_time()
        results = searcher.search(query)
        search_time = current_milli_time() - start_time
        
        urls = searcher.id_to_url(results)
        content = searcher.id_to_content(results)

        start_time = current_milli_time()
        # EXTRA CREDIT: OpenAI API Summarization
        summaries = [get_summary(c) for c in content]
        # summaries = ["" for c in content]
        summarize_time = current_milli_time() - start_time

        results = list(zip(urls, summaries))
        return render_template('index.html', query=query, search_time=search_time, summarize_time=summarize_time, results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
