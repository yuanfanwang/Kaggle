import polars as pl
import pandas
import sys

from datasets import Dataset, DatasetDict


train_df = pl.read_json("data/train.json")
labels_list = train_df['labels'].to_list()
mask = [any(label != 'O' for label in labels) for labels in labels_list]
new_df = train_df.filter(mask)
print(len(new_df))

# train_df = pl.read_json("data/train.json").to_pandas()
# count = 0
# for i, labels in enumerate(train_df['labels']):
#     has_pii = False
#     for label in labels:
#         if label != 'O':
#             has_pii = True
#             break
#     if has_pii: count += 1
# 
# print(count)