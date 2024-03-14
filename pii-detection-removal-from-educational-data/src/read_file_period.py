import polars as pl
import numpy as np
import sys

from datasets import Dataset, DatasetDict

train_df = pl.read_json("data/train.json").to_pandas()

# 3175
# print(train_df['full_text'][3175])
# sys.exit()
# tokens = train_df['tokens'][3175]
# for i, token in enumerate(tokens):
#     print(f"{token}")
#     if i > 200: break
# sys.exit()

"""
max_tokens_list = []
for i, tokens in enumerate(train_df['tokens']):
    token_count_list = []
    token_count = 0
    for token in tokens:
        if token in ['.', ';', '\n\n']:
            token_count_list.append(token_count)
            token_count = 0
        else:
            token_count += 1      
    token_count_list.append(token_count)
    max_tokens_list.append(max(token_count_list))

# print(np.argmax(max_tokens_list))
# print(max(max_tokens_list))
# print(len(train_df['full_text'][np.argmax(max_tokens_list)]))
# print(train_df['full_text'][np.argmax(max_tokens_list)])
# print(train_df['tokens'][np.argmax(max_tokens_list)][:1000])

"""


for i, (labels, tokens) in enumerate(zip(train_df['labels'], train_df['tokens'])):
    has_addr = False
    for label, token in zip(labels, tokens):
        if label == 'B-STREET_ADDRESSl' or label == 'I-STREET_ADDRESS':
            print(f"-{token}-")
            has_addr = True
    if has_addr:
        print(train_df['full_text'][i])
        print(f"----{i}----")