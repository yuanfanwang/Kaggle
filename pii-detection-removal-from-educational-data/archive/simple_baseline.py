import transformers
from transformers import AutoTokenizer,           \
                         AutoModel,               \
                         AutoModelForSequenceClassification, \
                         DataCollatorWithPadding, \
                         Trainer,                 \
                         TrainingArguments

from transformers.modeling_outputs import ModelOutput

from datasets import Dataset

import polars as pl
import numpy as np
import torch
import torch.nn as nn


##### Config
model_name = "microsoft/deberta-v3-base"
np.object = object


##### Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


##### DataSet
"""
train_json_file_path = "data/train.json"
test_json_file_path = "data/test.json"
train_data = pl.read_json(train_json_file_path).to_pandas()
train_data = Dataset.from_pandas(train_data)
train_data = train_data.map(lambda example: tokenizer(example['full_text'], padding="max_length", truncation=True, max_length=768, return_tensors="pt"))
train_data = train_data.remove_columns(["document", "full_text", "tokens", "trailing_whitespace"])
# dummy labels
def dummy_labels(example):
    example['labels'] = torch.tensor([1])
    example['attention_mask'] = torch.tensor(example['attention_mask']).squeeze()
    example['input_ids'] = torch.tensor(example['input_ids']).squeeze()
    example['token_type_ids'] = torch.tensor(example['token_type_ids']).squeeze()
    return example
train_data = train_data.map(dummy_labels)
train_data.save_to_disk("/home/competitions/pii-detection-removal-from-educational-data/data/dataset/simple_baseline_train")

test_data = pl.read_json(test_json_file_path).to_pandas()
test_data = Dataset.from_pandas(test_data)
test_data = test_data.map(lambda example: tokenizer(example['full_text'], padding="max_length", truncation=True, max_length=768, return_tensors="pt"))
test_data = test_data.remove_columns(["document", "full_text", "tokens", "trailing_whitespace"])
test_data.save_to_disk("/home/competitions/pii-detection-removal-from-educational-data/data/dataset/simple_baseline_test")
"""
# TODO: not use 'pt' format in the map function to make easier to create columns
train_data = Dataset.load_from_disk("/home/competitions/pii-detection-removal-from-educational-data/data/dataset/simple_baseline_train")
test_data = Dataset.load_from_disk("/home/competitions/pii-detection-removal-from-educational-data/data/dataset/simple_baseline_test")
# train_data = train_data.with_format("torch")
# test_data = train_data.with_format("torch")

print(train_data[0])

##### DataCollator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


##### Model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


##### Compute metrics


##### TrainingArguments
training_arguments = TrainingArguments(
    output_dir="simple_baseline",
    report_to="none",
)


##### Trainer
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
)
trainer.train()