import sys

import numpy as np
import polars as pl

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer

Local = True
model_checkpoint = "microsoft/deberta-v3-base"
np.object = object


if Local:
    data_path = "data/"
else:
    data_path = "/kaggle/input/pii-detection-removal-from-educational-data/"

label_names = [
    "O",
    "B-NAME_STUDENT",
    "I-NAME_STUDENT",
    "B-EMAIL",
    "I-EMAIL",
    "B-USERNAME",
    "I-USERNAME",
    "B-ID_NUM",
    "I-ID_NUM",
    "B-PHONE_NUM",
    "I-PHONE_NUM",
    "B-URL_PERSONAL",
    "I-URL_PERSONAL",
    "B-STREET_ADDRESS",
    "I-STREET_ADDRESS",
]

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {label: i for i, label in enumerate(label_names)}

def make_dataset():
    train_data = pl.read_json(data_path + "train.json").to_pandas()
    train_dataset = Dataset.from_pandas(train_data)
    test_data = pl.read_json(data_path + "test.json").to_pandas()
    test_dataset = Dataset.from_pandas(test_data)
    datasets = DatasetDict({"train": train_dataset, "test": test_dataset})
    return datasets

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    if "labels" not in examples:
        return tokenized_inputs

    all_labels = examples["labels"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        labels = [label2id[label] for label in labels]
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
datasets = make_dataset()

remove_columns = datasets["train"].column_names
remove_columns.remove('labels')

tokenized_datasets = datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=remove_columns,
)

