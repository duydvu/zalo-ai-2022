import re
import pandas as pd
import numpy as np
import json
import pickle
from tqdm import tqdm
from vncorenlp import VnCoreNLP

# Read wiki data
wiki_data = []
with open('./wikipedia_20220620_cleaned/wikipedia_20220620_cleaned.jsonl', 'r') as f:
    for line in tqdm(f):
        wiki_data.append(json.loads(line))
print(len(wiki_data))

# Tokenization
model = VnCoreNLP('./VnCoreNLP-1.1.1.jar', annotators='wseg')
with open('./data/paragraphs_vncorenlp.jsonl', 'w') as f:
    for idx, item in tqdm(enumerate(wiki_data), total=len(wiki_data)):
        contents = []
        for p in item['text'].split('\n\n'):
            contents.extend([' '.join(sent) for sent in model.tokenize(p)])
        line = json.dumps({'id': str(idx), 'contents': contents}, ensure_ascii=False)
        f.write(f'{line}\n')


def clean(text):
    text = text.lower()
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

sentence_docs = []
with open('./data/paragraphs_vncorenlp.jsonl', 'r') as f:
    for l in tqdm(f):
        sentence_docs.append(json.loads(l))


print(len(wiki_data) == len(sentence_docs))

# Generate paragraphs from sentences
paragraph_docs = []
max_len = 256 # QA model's max length
window = 128
for idx, doc in tqdm(enumerate(sentence_docs), total=len(sentence_docs)):
    for i in range(0, len(doc['contents']), 2):
        par_tokens = (' '.join(doc['contents'][i:i+4])).split()
        for j in range(0, max(len(par_tokens) - max_len, 0) + window, window):
            p = ' '.join(par_tokens[j:j + max_len])
            cleaned_p = clean(p)
            paragraph_docs.append({
                'text': p,
                'cleaned_text': cleaned_p,
                'id': idx,
            })

# Store original texts for QA model
with open('./data/paragraphs2.jsonl', 'w') as f:
    for idx, doc in tqdm(enumerate(paragraph_docs)):
        contents = doc['text'].replace('_', ' ')
        contents = re.sub('\s+', ' ', contents).strip()
        line = json.dumps({'id': str(idx), 'contents': contents}, ensure_ascii=False)
        f.write(f'{line}\n')

# For indexing unigram model
with open('./data/pyserini2/paragraphs.jsonl', 'w') as f:
    for idx, doc in tqdm(enumerate(paragraph_docs)):
        contents = doc['cleaned_text'].replace('_', ' ')
        contents = re.sub('\s+', ' ', contents).strip()
        line = json.dumps({'id': str(idx), 'contents': contents}, ensure_ascii=False)
        f.write(f'{line}\n')

# For indexing n-grams model
with open('./data/pyserini_tokenized2/paragraphs.jsonl', 'w') as f:
    for idx, doc in tqdm(enumerate(paragraph_docs)):
        line = json.dumps({'id': str(idx), 'contents': doc['cleaned_text']}, ensure_ascii=False)
        f.write(f'{line}\n')
