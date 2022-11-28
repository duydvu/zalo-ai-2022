import json
import glob
import os
from time import time

from .document_retriever import retrieve_documents, load_model as load_retriver_model

load_retriver_model()

input_data = []
for f in glob.glob('/data/*.json'):
    tmp = json.load(open(f, 'r'))
    input_data.extend(tmp['data'])


def predict(item):
    question = item['question']

    contexts = retrieve_documents(question, k=20)

    # predict


results = []
for item in input_data:
    answer = predict(item)
    results.append({
        'id': item['id'],
        'question': item['question'],
        'answer': answer
    })

output_dir = '/result'
os.makedirs(output_dir, exist_ok=True)
json.dump({ 'data': results }, open(f'{output_dir}/submission.json', 'w'))
