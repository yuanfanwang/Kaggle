from transformers import AutoTokenizer, BertTokenizer
import polars as pl
import numpy as np
import sys
from datasets import Dataset, DatasetDict

train_df = pl.read_json("data/train.json").to_pandas()
train_df = train_df[:100]
train_dataset = Dataset.from_pandas(train_df)

def multiple(examples):
    print(type(examples))
    new_example1 = examples.copy()
    new_example2 = examples.copy()
    return [new_example1, new_example2]

print(len(train_dataset['tokens']))
train_dataset = train_dataset.map(
    multiple,
    batched=True,
)

print(len(train_dataset['tokens']))
