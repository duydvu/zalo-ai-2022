
from glob import glob
import pandas as pd

import re
import nltk
nltk.download('punkt')
from nltk import word_tokenize as lib_tokenizer
from cleantext import clean
import string



ROOT_DIR = '/code'


with open(f'{ROOT_DIR}/vietnamese-stopwords-dash.md', 'r') as f:
    STOPWORDS = f.read().split('\n')

dict_map = dict({})

def word_tokenizer(text):
    global dict_map
    words = text.split()
    words_norm = []
    for w in words:
        if dict_map.get(w, None) is None:
            dict_map[w] = ' '.join(lib_tokenizer(w)).replace('``', '"').replace("''", '"')
        words_norm.append(dict_map[w])
    return words_norm

def word_normalizer(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'ynm\w*', ' ', re.sub(r"'RE:(?=[^']*')", ' ', text))
    text = clean(text, no_emoji=True, to_ascii=False, no_line_breaks=True, lower=False)

    EMOJI_PATTERNS = [
        '[:=]\)+',
        '[:=]\(+',
        '<3',
        ':[<>v3]',
    ]
    for emoji_pattern in EMOJI_PATTERNS:
        text = re.sub(emoji_pattern, ' ', text)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', ' ', text)
    
    text = text.replace("BULLET : : : :", "")
    text = text.replace("=", "")
    text = text.replace("/", "")
    text = text.replace("`", "")
    
    return text

def strip_answer_string(text):
    text = text.strip()
    while text[-1] in '.,/><;:\'"[]{}+=-_)(*&^!~`':
        if text[0] != '(' and text[-1] == ')' and '(' in text:
            break
        if text[-1] == '"' and text[0] != '"' and text.count('"') > 1:
            break
        text = text[:-1].strip()
    while text[0] in '.,/><;:\'"[]{}+=-_)(*&^!~`':
        if text[0] == '"' and text[-1] != '"' and text.count('"') > 1:
            break
        text = text[1:].strip()
    text = text.strip()
    return text


def strip_context(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

    
def remove_stopwords(text):
    res = []
    for w in text:
        if w not in STOPWORDS:
            res.append(w)
    return ' '.join(res)