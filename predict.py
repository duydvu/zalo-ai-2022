import json
import glob
import os
from time import time

from .document_retriever import retrieve_documents, load_model as load_retriver_model, vncorenlp_model

import pandas as pd
import string
import json
from tqdm.notebook import tqdm
from vncorenlp import VnCoreNLP

import nltk
nltk.download('punkt')
from collections import Counter
import string
from transformers import pipeline


import linking_wiki_entity
from utils import word_normalizer, word_tokenizer, strip_context


ROOT_DIR = '/code'


load_retriver_model()
linking_wiki_entity.load_title()

# Load model
model_checkpoint = f"{ROOT_DIR}/model/vi-mrc-large"
nlp = pipeline('question-answering', model=model_checkpoint, tokenizer=model_checkpoint, device=0)





def predict(item):

    docs = retrieve_documents(item['question'], k=50)
    ner = lambda t: [ [(v["form"],v["posTag"]) for v in s] for s in vncorenlp_model.annotate(t)["sentences"] ]

    inputs = []
    for doc in docs:
        text = ''
        ners = ner(item['question'])[0]
        lst_1, lst_2 = [], []
        for word in ners:
            if word[1] == 'Np'  or word[1] == 'M' or word[1] == 'Nu':
                lst_1.append(word[0].replace('_', ' '))
            if word[1] == 'A' or word[1] == 'V' or  word[1] == 'N':
                lst_2.append(word[0].replace('_', ' '))
        if len(lst_1) == 0:
            if any(word.lower() in doc.lower() for word in lst_2):
                text = doc
        elif len(lst_2) == 0:
            if all(word.lower() in doc.lower() for word in lst_1):
                text = doc
        else:
            if all(word.lower() in doc.lower() for word in lst_1) and any(word.lower() in doc.lower() for word in lst_2):
                text = doc
        if text == '':
            continue
        
        text = word_normalizer(text)
        context = strip_context(text)
        context = ' '.join(word_tokenizer(context))
        question = ' '.join(word_tokenizer(item['question']))

        inputs.append({
            'question': question,
            'context': context
        })


    if len(inputs) == 0:
        short_candidate = None
    
    else:
        extracted_answer = nlp(inputs, batch_size=len(inputs), truncation=True)
        
        if not isinstance(extracted_answer, list):
            extracted_answer = [extracted_answer]

        top_extracted_answer = [item for item in extracted_answer if len(item["answer"]) <= 30 and item['answer'] != string.punctuation and item['score'] >= 0.5]
        top_extracted_answer = sorted(top_extracted_answer, key=lambda item: item['score'], reverse=True)[:10]
        predicted_answer = Counter([item['answer'] for item in top_extracted_answer]).most_common(1)
        
        if len(predicted_answer) == 0:
            short_candidate = None
        else:
            short_candidate = predicted_answer[0][0]


    result = linking_wiki_entity.extract_answer(item['question'], short_candidate)

    return result


list_df = []
for f in glob.glob('/data/*.json'):

    private_test = json.load(open(f, 'r'))
    df = pd.DataFrame.from_dict(private_test)

    lst_predict = []
    for index, row in tqdm(df.iterrows()):
        lst_predict.append(predict(row))
    df['answer'] = lst_predict

    list_df.append(df)


df = pd.concat(list_df)

results = df.to_dict('record')

output_dir = '/result'
os.makedirs(output_dir, exist_ok=True)
json.dump({ 'data': results }, open(f'{output_dir}/submission.json', 'w'))
    


