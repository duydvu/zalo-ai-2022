import json
import re
import string
from collections import defaultdict

ROOT_DIR = '/code'

with open(f'{ROOT_DIR}/tai_data/vietnamese-stopwords-dash.md', 'r') as f:
    STOPWORDS = f.read().split('\n')


def remove_stopwords(text):
    for w in STOPWORDS:
        text = re.sub(fr'\b{w}\b', '', text)
    return text


inverted_title = defaultdict(dict)

with open(f'{ROOT_DIR}/wikipedia_20220620_cleaned/wikipedia_20220620_cleaned.jsonl') as f:
    data_str = f.readline()
    i = 0
    while data_str:
        data_js = json.loads(data_str)
        text = re.sub(fr'([{string.punctuation}\\])', ' ', re.split(r'\.\s', data_js['text'])[0])

        for token in text.lower().split():
            if token in STOPWORDS:
                continue
            if token in data_js['title']:
                inverted_title[token][i] = 2
            else:
                inverted_title[token][i] = 1

        data_str = f.readline()
        i += 1

with open(f'{ROOT_DIR}/tai_data/inverted_title_v15.json', 'w') as f:
    f.write(json.dumps(inverted_title))
