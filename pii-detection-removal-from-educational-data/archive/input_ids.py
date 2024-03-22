from transformers import AutoTokenizer, BertTokenizer
import polars as pl
import numpy as np
import sys
from datasets import Dataset, DatasetDict

train_df = pl.read_json("data/train.json").to_pandas()
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

# docs have address label
docs = [1103, 1887]
for doc in docs:
    print(train_df['full_text'][doc])
    # print(train_df['tokens'][doc])
    tokens = train_df['tokens'][doc]
    # print(type(tokens))
    # print(tokens)
    # tokenized_data = tokenizer(
    #     tokens, truncation=True, is_split_into_words=True
    # )
    tokenized_data = tokenizer(
        list(tokens), truncation=True, is_split_into_words=True
    )
    input_ids = tokenized_data['input_ids']
    for id in input_ids: print(id, tokenizer.decode(id))