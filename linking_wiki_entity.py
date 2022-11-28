import json
import re
import string
from collections import defaultdict

# from py_vncore import model
from roman import romanToDecimal

ROOT_DIR = '/code'
TITLE = None
lower_title = None
inverted_title = None


date_entity = '(ngày)|(tháng)|(năm)'
time_long_entity = '(thời gian)|(thời kỳ)|(thời điểm)|(giai đoạn)'
time_short_entity = '(thời)|(khi)|(lúc)'
quantity_entity = '(số)|(giờ)|(phút)|(giây)|(thế kỷ)|(thiên niên kỷ)'

quantity_question_entity = '(mấy)|(bao nhiêu)'
wiki_question_entity = '(ai)|(ở đâu)|(gì)|(vì sao)|(tại sao)'


def load_title():
    global TITLE, lower_title, inverted_title
    with open(f'{ROOT_DIR}/wikipedia_20220620_cleaned/wikipedia_20220620_all_titles.txt') as f:
        TITLE = f.read().split('\n')
    lower_title = [
        re.sub(r'\s+', ' ', re.sub(fr'([{string.punctuation}\\])', ' ', t)).strip().lower() for t in TITLE
    ]
    with open(f'{ROOT_DIR}/tai_data/inverted_title_v3.json', 'r') as f:
        inverted_title = defaultdict(dict, json.load(f))


def predict_type(question):
    question = question.lower()

    if re.search(fr'({date_entity}) (.*là )?({quantity_question_entity}|(nào))', question):
        return 1
    if re.search(fr'^({date_entity})', question) \
            and not re.search(fr'({quantity_question_entity}|{wiki_question_entity}|(nào))', question):
        return 1

    if re.search(fr'({quantity_question_entity})', question):
        return 2
    if re.search(fr'^({quantity_entity})', question) \
            and not re.search(fr'({quantity_question_entity}|{wiki_question_entity}|(nào))', question):
        return 2

    if re.search(fr'({time_long_entity}) (.*là )?({quantity_question_entity}|((lịch sử )?nào))', question):
        return 3
    if re.search(fr'^({time_long_entity})', question) \
            and not re.search(fr'({quantity_question_entity}|{wiki_question_entity}|(nào))', question):
        return 3
    if re.search(fr'({time_short_entity}) nào', question) or re.search(fr'(bao giờ)', question):
        return 3

    return 0


def extract_answer(question, candidate):
    answer_type = predict_type(question)

    if answer_type == 3:
        if re.search(r'\d{3,4}', candidate):
            answer_type = 1
        elif re.search(fr'{quantity_entity}', candidate.lower()):
            answer_type = 2

    num = re.findall(r'\d+', candidate)
    if answer_type == 1:
        if re.search(r'\d{3,4}', candidate):
            if len(num) == 1:
                return f'năm {num[0]}'
            elif len(num) == 2:
                if re.search(fr'(năm) ({quantity_question_entity}|(nào))', question.lower()):
                    return f'năm {num[-1]}'
                return f'tháng {num[0]} năm {num[1]}'
            elif len(num) == 3:
                if re.search(fr'(?<!tháng )(năm) ({quantity_question_entity}|(nào))', question.lower()):
                    return f'năm {num[-1]}'
                if re.search(fr'(?<!ngày )(tháng năm) ({quantity_question_entity}|(nào))', question.lower()):
                    return f'tháng {num[-2]} năm {num[-1]}'
                return f'ngày {num[0]} tháng {num[1]} năm {num[2]}'
        elif re.search(r'\d{8}', candidate):
            return f'ngày {num[0][:2]} tháng {num[0][2:4]} năm {num[0][4:]}'
        elif re.search(r'\d{1,2}', candidate):
            if re.search(fr'(?<!tháng )(năm) ({quantity_question_entity}|(nào))', question.lower()) and len(num) == 1:
                return f'năm {num[0]}'
            if re.search(fr'(?<!ngày )(tháng năm) ({quantity_question_entity}|(nào))', question.lower()) and len(
                    num) == 2:
                return f'tháng {num[0]} năm {num[1]}'
            if re.search(fr'(ngày tháng năm) ({quantity_question_entity}|(nào))', question.lower()) and len(num) == 3:
                return f'ngày {num[0]} tháng {num[1]} năm {num[2]}'

    elif answer_type == 2:
        if len(num) == 1:
            return f'{num[0]}'
        if re.search(fr'({quantity_entity}) (.*là )?(thứ )?({quantity_question_entity}|(nào))', question.lower()) \
                and re.search(fr'({quantity_entity}) (thứ )?\d+', candidate.lower()):
            return f'{num[0]}'
        if re.search(r'\d+([.,]\d+)+', candidate):
            return re.sub(r'[.,]', '', re.search(r'\d+([.,]\d+)+', candidate).group())
        roman = re.search(r'\bM{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b', candidate)
        if re.search(r'thế kỷ', candidate.lower()) and roman:
            return f'{romanToDecimal(roman.group())}'
        return None

    title_candidates = defaultdict(lambda: 0)

    candidate = re.sub(r'^(Ngày|ngày)', '', candidate).strip()
    if re.sub(r'\s+', ' ', re.sub(fr'([{string.punctuation}\\])', ' ', candidate)).strip().lower() in lower_title:
        return f'wiki/{"_".join(candidate.split())}'

    for word in candidate.lower().split():
        for title_id, weight in inverted_title[word].items():
            title_candidates[title_id] += weight

    for title_id in title_candidates.keys():
        for word in TITLE[int(title_id)].lower().split():
            if word not in candidate.lower():
                title_candidates[title_id] -= 1

    return f'wiki/{"_".join(TITLE[int(max(title_candidates, key=title_candidates.get))].split())}' if title_candidates else None
