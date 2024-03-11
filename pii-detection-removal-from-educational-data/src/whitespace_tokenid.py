"""
!pip install "/kaggle/input/seqeval/seqeval-1.2.2-py3-none-any.whl"
"""
import os
import random
import sys
import torch

import numpy as np
import pandas as pd
import polars as pl

from datasets import Dataset, DatasetDict
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer


torch.cuda.empty_cache()
np.object = object
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CFG:
    sample_data_size = 5
    batch_size = 1
    token_max_length = 1024
    epoch = 3
    fold = 4


Local = True
if Local:
    data_path = "data/"
    model_checkpoint = "microsoft/deberta-v3-base"
else:
    data_path = "/kaggle/input/pii-detection-removal-from-educational-data/"
    model_checkpoint = "/kaggle/input/debertav3base"

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
id2label = {str(i): label for i, label in enumerate(label_names)}
label2id = {label: i for i, label in enumerate(label_names)}


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
    return new_labels


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=CFG.token_max_length
    )
    word_ids = [tokenized_inputs.word_ids(i) for i in range(len(tokenized_inputs["input_ids"]))]
    tokenized_inputs["word_ids"] = word_ids

    # return when it is test dataset
    if "labels" not in examples: return tokenized_inputs

    # modify labels for train data
    all_labels = examples["labels"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        labels = [label2id[label] for label in labels]
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def add_fold_column(examples):
    # return when it is test dataset
    if 'labels' not in examples: return examples
    # just want to know the length of the example to add fold column
    eg_len = len(examples['input_ids'])
    folds = np.random.randint(0, CFG.fold, eg_len)
    examples['fold'] = folds
    return examples


def add_training_whitespace_to_tokens(examples):
    tokens_batch = examples['tokens']
    whitespaces_batch = examples['trailing_whitespace']    
    new_tokens_batch = []
    for tokens, whitespaces in zip(tokens_batch, whitespaces_batch):
        new_tokens = []
        for token, whitespace in zip(tokens, whitespaces):
            new_tokens.append(token + (' ' * whitespace))
        new_tokens_batch.append(new_tokens)
    examples['tokens'] = new_tokens_batch
    return examples


def make_dataset():
    train_df = pl.read_json(data_path + "train.json").to_pandas()
    train_df = train_df[:CFG.sample_data_size]
    train_dataset = Dataset.from_pandas(train_df)
    test_df = pl.read_json(data_path + "test.json").to_pandas()
    test_dataset = Dataset.from_pandas(test_df)
    datasets = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    datasets = datasets.map(
        add_training_whitespace_to_tokens,
        batched=True,
    )

    datasets = datasets.map(
        tokenize_and_align_labels,
        batched=True,
    )

    datasets = datasets.map(
        add_fold_column,
        batched=True,
    )

    return datasets


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenized_datasets = make_dataset()
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
