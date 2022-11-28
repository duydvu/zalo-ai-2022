import json
import glob
import os
from time import time

from .document_retriever import retrieve_documents, load_model as load_retriver_model

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
from test import utils, document_retrieval, entity_linking


PATH_ROOT = 'datasets'


load_retriver_model()

# Load model
model_checkpoint = "model/vi-mrc-large"
nlp = pipeline('question-answering', model=model_checkpoint, tokenizer=model_checkpoint, device=0)
model_retrieve = VnCoreNLP("VnCoreNLP/VnCoreNLP-1.1.1.jar",annotators="wseg,pos,parse")





def predict(item):

    docs = document_retrieval.retrieve_documents(item['question'])
    ner = lambda t: [ [(v["form"],v["posTag"]) for v in s] for s in model_retrieve.annotate(t)["sentences"] ]

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
        
        text = utils.word_normalizer(text)
        context = utils.strip_context(text)
        context = ' '.join(utils.word_tokenizer(context))
        question = ' '.join(utils.word_tokenizer(item['question']))

        inputs.append({
            'question': question,
            'context': context
        })


    if len(inputs) == 0:
        short_candidate = "null"
    
    else:
        extracted_answer = nlp(inputs, batch_size=len(inputs), truncation=True)
        
        if not isinstance(extracted_answer, list):
            extracted_answer = [extracted_answer]

        top_extracted_answer = [item for item in extracted_answer if len(item["answer"]) <= 30 and item['answer'] != string.punctuation and item['score'] >= 0.5]
        top_extracted_answer = sorted(top_extracted_answer, key=lambda item: item['score'], reverse=True)[:10]
        predicted_answer = Counter([item['answer'] for item in top_extracted_answer]).most_common(1)
        
        if len(predicted_answer) == 0:
            short_candidate = "null"
        else:
            short_candidate = predicted_answer[0][0]


    result = entity_linking.linking_wiki_entity(item['question'], short_candidate)

    return result

    


