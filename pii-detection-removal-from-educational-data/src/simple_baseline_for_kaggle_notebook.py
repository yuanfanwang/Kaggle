from transformers import AutoTokenizer,                      \
                         AutoModelForTokenClassification, \
                         DataCollatorWithPadding,            \
                         Trainer,                            \
                         TrainingArguments

from datasets import Dataset

import polars as pl
import numpy as np
import torch
import torch.nn as nn


##### Config
Local = False
model_name = "microsoft/deberta-v3-small"
np.object = object

if Local:
    data_path = "data/"
else:
    data_path = "/kaggle/input/pii-detection-removal-from-educational-data/"

##### Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)


##### DataSet
def dummy_labels(example):
    example['labels'] = 1
    return example

train_json_file_path = data_path + "train.json"
train_data = pl.read_json(train_json_file_path).to_pandas()
train_data = Dataset.from_pandas(train_data)
train_data = train_data.map(lambda example: tokenizer(example['full_text'], padding="max_length", truncation=True, max_length=768))
train_data = train_data.remove_columns(["document", "full_text", "tokens", "trailing_whitespace"])
train_data = train_data.map(dummy_labels)

# print(train_data[0])
test_json_file_path = data_path + "test.json"
test_data = pl.read_json(test_json_file_path).to_pandas()
test_data = Dataset.from_pandas(test_data)
test_data = test_data.map(lambda example: tokenizer(example['full_text'], padding="max_length", truncation=True, max_length=768))
test_data = test_data.remove_columns(["document", "full_text", "tokens", "trailing_whitespace"])


##### Model
# AutoModelForTokenClassification
# https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#transformers.AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


##### DataCollator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


##### TrainingArguments
training_arguments = TrainingArguments(
    output_dir="simple_baseline",
    num_train_epochs=1,
    per_device_train_batch_size=4,  # T4x2 should be half? if not, is there any con to use the save batch size as P100
    per_device_eval_batch_size=4,   # T4x2 should be half? if not, is there any con to use the save batch size as P100
    report_to="none",
)


##### Trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_arguments,
    train_dataset=train_data,
)

trainer.train()