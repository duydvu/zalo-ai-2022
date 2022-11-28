import pandas as pd
import json
from glob import glob
import nltk
nltk.download('punkt')
import numpy as np
from tqdm import tqdm
import re
from nltk import word_tokenize as lib_tokenizer
from utils import word_tokenize, strip_answer_string, strip_context
import codecs
import string

ORI_DATASET_PATH = "original_datasets"
PRE_DATASET_PATH = "preprocessed_datasets"
MODEL_PATH = "model"


list_df = []

for file_name in glob(f"{ORI_DATASET_PATH}/*"):
    
    if file_name == f'{ORI_DATASET_PATH}/train_translated_squad.json':
        continue

    json_file = json.load(open(file_name))
    df = pd.DataFrame.from_dict(json_file['data'])
    
    if file_name != f'{ORI_DATASET_PATH}/zac2022_train_merged_final.json':
        pos = 0
        new_df = pd.DataFrame(columns=['id', 'question', 'context', 'text', 'answer_start'])
        for index, data in df.iterrows():
            for paragraph in data['paragraphs']:
                for qas in paragraph['qas']:
                    new_df.at[pos, 'id'] = qas['id']
                    new_df.at[pos, 'question'] = qas['question']
                    new_df.at[pos, 'context'] = paragraph['context']
                    if len(qas['answers']) > 0:
                        answer = qas['answers'][0]
                        new_df.at[pos, 'answer_start'] = answer['answer_start']
                        new_df.at[pos, 'text'] = answer['text']
                       
                    pos += 1    
    else:
        new_df = df.loc[df['category'] != 'PARTIAL_ANNOTATION']
        new_df = new_df.drop(['answer', 'category', 'is_long_answer'], axis=1)
        new_df.rename(columns = {'text':'context', 'short_candidate_start':'answer_start', 'short_candidate':'text'}, inplace = True)
        
    list_df.append(new_df)



json_file = json.load(codecs.open(f'{ORI_DATASET_PATH}/train_translated_squad.json', 'r', 'utf-8-sig'))
df = pd.DataFrame.from_dict(json_file['data'])

pos = 0
new_df = pd.DataFrame(columns=['id', 'question', 'context', 'text', 'answer_start'])

for paragraph in df['paragraphs'].iloc[0]:
    for qas in paragraph['qas']:
        new_df.at[pos, 'id'] = qas['id']
        new_df.at[pos, 'question'] = qas['question']
        new_df.at[pos, 'context'] = paragraph['context']
        if len(qas['answers']) > 0:
            answer = qas['answers'][0]
            new_df.at[pos, 'answer_start'] = answer['answer_start']
            new_df.at[pos, 'text'] = answer['text']
        pos += 1

list_df.append(new_df)

df = pd.concat(list_df)

norm_samples = []

for _, row in df.iterrows():
    index = row['id']
    context_raw = row['context']
    question = row['question']
    answer_raw = row['text']
    
    lst_text, lst_answer_start = [], []
    

    if isinstance(answer_raw, str):
        
        if re.sub(r'[^\w\s]', '', answer_raw) == "":
            continue

        answer_index_raw = int(row['answer_start'])
        if context_raw[answer_index_raw: answer_index_raw + len(answer_raw)] == answer_raw:
            context_prev = strip_context(context_raw[:answer_index_raw])
            answer = strip_answer_string(answer_raw)
            context_next = strip_context(context_raw[answer_index_raw + len(answer):])

            context_prev = ' '.join(word_tokenize(context_prev))
            context_next = ' '.join(word_tokenize(context_next))
            answer = ' '.join(word_tokenize(answer))
            question = ' '.join(word_tokenize(question))

            context = "{} {} {}".format(context_prev, answer, context_next).strip()

            lst_text.append(answer)
            lst_answer_start.append(len("{} {}".format(context_prev, answer).strip()) - len(answer))

            norm_samples.append({
                "id": index,
                "context": context,
                "question": question,
                "answers": {
                    "text": lst_text,
                    "answer_start": lst_answer_start
                }
            })
            
        else:
            print(row)
       
    else:
        context_raw = ' '.join(word_tokenize(context_raw))
        question = ' '.join(word_tokenize(question))
            
        norm_samples.append({
            "id": index,
            "context": context_raw,
            "question": question,
            "answers": {
                "text": lst_text,
                "answer_start": lst_answer_start
            }
        })
    
with open(f'{PRE_DATASET_PATH}/final_dataset_squadv2.jsonl', 'w', encoding='utf-8') as file:
    for item in norm_samples:
        file.write("{}\n".format(json.dumps(item, ensure_ascii=False)))

print("Total: {} samples".format(len(norm_samples)))