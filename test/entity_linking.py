

from collections import defaultdict
import json
import re
import nltk
nltk.download('punkt')


PATH_ROOT = 'datasets'

date_entity = '(ngày)|(tháng)|(năm)'
time_long_entity = '(thời gian)|(thời kỳ)|(thời điểm)|(giai đoạn)'
time_short_entity = '(thời)|(khi)|(lúc)'
quantity_entity = '(số)|(giờ)|(phút)|(giây)|(thế kỷ)|(thiên niên kỷ)'

quantity_question_entity = '(mấy)|(bao nhiêu)'
wiki_question_entity = '(ai)|(ở đâu)|(gì)|(vì sao)|(tại sao)'

with open('/code/zalo-ai-2022/wikipedia_20220620_cleaned/wikipedia_20220620_all_titles.txt') as f:
  TITLE = f.read().split('\n')
lower_title = [t.lower() for t in TITLE]


with open(f'{PATH_ROOT}/inverted_title.json', 'r') as f:
    inverted_title = defaultdict(dict, json.load(f))

def predict_type(question):
  question = question.lower()

  if re.search(fr'({date_entity}) (.*là)?({quantity_question_entity}|(nào))', question):
    return 1
  if re.search(fr'^({date_entity})', question) \
    and not re.search(fr'({quantity_question_entity}|{wiki_question_entity}|(nào))', question):
    return 1

  if re.search(fr'({quantity_question_entity})', question):
    return 2
  if re.search(fr'^({quantity_entity})', question) \
    and not re.search(fr'({quantity_question_entity}|{wiki_question_entity}|(nào))', question):
    return 2

  if re.search(fr'({time_long_entity}) (.*là)?({quantity_question_entity}|((lịch sử )?nào))', question):
    return 3
  if re.search(fr'^({time_long_entity})', question) \
    and not re.search(fr'({quantity_question_entity}|{wiki_question_entity}|(nào))', question):
    return 3
  if re.search(fr'({time_short_entity}) nào', question) or re.search(fr'(bao giờ)', question):
    return 3

  return 0

def value(r):
    if (r == 'I'):
        return 1
    if (r == 'V'):
        return 5
    if (r == 'X'):
        return 10
    if (r == 'L'):
        return 50
    if (r == 'C'):
        return 100
    if (r == 'D'):
        return 500
    if (r == 'M'):
        return 1000
    return -1
 
def romanToDecimal(str):
    res = 0
    i = 0
    while (i < len(str)):
        s1 = value(str[i])
        if (i + 1 < len(str)):
            s2 = value(str[i + 1])
            if (s1 >= s2):
                res = res + s1
                i = i + 1
            else:
                res = res + s2 - s1
                i = i + 2
        else:
            res = res + s1
            i = i + 1
    return res

def linking_wiki_entity(question, candidate):
  answer_type = predict_type(question)

  if answer_type==3:
    if re.search(r'\d{3,4}', candidate):
      answer_type = 1
    elif re.search(fr'{quantity_entity}', candidate):
      answer_type = 2

  num = re.findall(r'\d+', candidate)
  if answer_type==1 and re.search(r'\d{3,4}', candidate):
    if len(num)==1:
      return f'năm {num[0]}'
    elif len(num)==2:
      return f'tháng {num[0]} năm {num[1]}'
    elif len(num)==3:
      return f'ngày {num[0]} tháng {num[1]} năm {num[2]}'

  elif answer_type==2:
    if len(num)==1:
      return f'{num[0]}'
    roman = re.search(r'\bM{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b', candidate)
    if re.search(r'thế kỷ', candidate) and roman:
      return f'{romanToDecimal(roman.group())}'

  candidate = re.sub(r'^(Ngày|ngày)', '', candidate).strip()
  if candidate.lower() in lower_title:
    return f'wiki/{"_".join(candidate.split())}'

  title_candidates = defaultdict(lambda: 0)
  for word in candidate.split():
    for title, weight in inverted_title[word].items():
      title_candidates[title] += weight
  return f'wiki/{max(title_candidates, key=title_candidates.get)}' if title_candidates else 'null'