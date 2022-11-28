

import json
import re
from tqdm.notebook import tqdm

from pyserini.search.lucene import LuceneSearcher
from vncorenlp import VnCoreNLP

import nltk
nltk.download('punkt')


PATH_ROOT = 'datasets'


# model_retrieve = VnCoreNLP('/code/zalo-ai-2022/VnCoreNLP-1.1.1.jar', annotators='wseg')
model_retrieve = VnCoreNLP("VnCoreNLP/VnCoreNLP-1.1.1.jar",annotators="wseg,pos,parse")

paragraphs = []
with open(f'/code/zalo-ai-2022/data/paragraphs2.jsonl', 'r') as f:
    for line in tqdm(f):
        paragraphs.append(json.loads(line))
print(len(paragraphs))

uni_searcher = LuceneSearcher('/code/zalo-ai-2022/indexes/paragraphs2')
uni_searcher.set_language('vi')


def retrieve_documents_unigram(question):
    question = question.lower()
    question = re.sub('\W+', ' ', question)
    question = re.sub('\s+', ' ', question)
    question = question.strip()
    hits = uni_searcher.search(question, k=25)
    return hits

ngram_searcher = LuceneSearcher('/code/zalo-ai-2022/indexes/paragraphs_tokenized2')
ngram_searcher.set_language('vi')

def retrieve_documents_ngram(question):
    question = ' '.join([tok for sent in model_retrieve.tokenize(question) for tok in sent]).lower()
    question = re.sub('\W+', ' ', question)
    question = re.sub('\s+', ' ', question)
    question = question.strip()
    hits = ngram_searcher.search(question, k=25)
    return hits

def retrieve_documents(question):
    uni_res = retrieve_documents_unigram(question)
    ngram_res = retrieve_documents_ngram(question)
    res = {}
    for item in [*uni_res, *ngram_res]:
        if item.docid in res:
            res[item.docid] = max(item.score, res[item.docid])
        else:
            res[item.docid] = item.score
    res = list(res.items())
    res = sorted(res, key=lambda i: i[1], reverse=True)
    return [paragraphs[int(item[0])]['contents'] for item in res]


