import json

import pandas as pd

from extract_answer import extract_answer
from to_json import convert_to_json

with open('tai_data/ver4.txt', 'r') as f:
    ans = f.read().split('\n')

test_df = pd.read_csv('submission.csv')
test_df['new_info'] = ans

print('predicting...')
test_df['predicted_wiki'] = test_df.apply(lambda _: extract_answer(_.question, _.new_info), axis=1)
print('predicted')

data_list = []
_ = test_df.apply(lambda _: convert_to_json(_, data_list), axis=1)
with open('tai_data/tainlq_submission.json', 'w') as f:
    f.write(json.dumps({"data": data_list}, ensure_ascii=False))
