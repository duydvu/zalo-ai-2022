from pyserini.search.lucene import LuceneSearcher
import re
import json
from tqdm.notebook import tqdm
from vncorenlp import VnCoreNLP

ROOT_DIR = '/code/zalo-ai-2022'

vncorenlp_model = VnCoreNLP(f'{ROOT_DIR}/VnCoreNLP-1.1.1.jar', annotators='wseg')

paragraphs = []
with open(f'{ROOT_DIR}/data/paragraphs2.jsonl', 'r') as f:
    for line in tqdm(f):
        paragraphs.append(json.loads(line))
print(len(paragraphs))

uni_searcher = LuceneSearcher(f'{ROOT_DIR}/indexes/paragraphs2')
uni_searcher.set_language('vi')
ngram_searcher = LuceneSearcher(f'{ROOT_DIR}/indexes/paragraphs_tokenized2')
ngram_searcher.set_language('vi')


def clean(text: str):
    text = text.lower()
    text = re.sub('\W+', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()


def retrieve_documents_unigram(question, k):
    question = clean(question)
    return uni_searcher.search(question, k=k)


def retrieve_documents_ngram(question, k):
    question = ' '.join([tok for sent in vncorenlp_model.tokenize(question) for tok in sent])
    question = clean(question)
    return ngram_searcher.search(question, k=k)


def retrieve_documents(question, k=20):
    uni_res = retrieve_documents_unigram(question, k=int(k / 2))
    ngram_res = retrieve_documents_ngram(question, k=int(k / 2))
    res = {}
    for item in [*uni_res, *ngram_res]:
        if item.docid in res:
            res[item.docid] = max(item.score, res[item.docid])
        else:
            res[item.docid] = item.score
    res = list(res.items())
    res = sorted(res, key=lambda i: i[1], reverse=True)
    return [paragraphs[int(item[0])]['contents'] for item in res]
