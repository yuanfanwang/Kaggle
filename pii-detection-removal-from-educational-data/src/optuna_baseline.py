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
from seqeval.metrics import precision_score, recall_score, accuracy_score, f1_score


class CFG:
    local = False
    sample_data_size = None
    batch_size = 1
    token_max_length = 1024
    epoch = 3
    fold = 4
    use_optuna = True

# set random seed
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# initialize
seed_everything(seed=42)
torch.cuda.empty_cache()
np.object = object
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if CFG.use_optuna:
    import optuna

if CFG.local:
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


def add_fold_column(example):
    # return when it is test dataset
    if 'labels' not in example: return example
    # just want to know the length of the example to add fold column
    eg_len = len(example['input_ids'])
    folds = np.random.randint(0, CFG.fold, eg_len)
    example['fold'] = folds
    return example


def split_below_max_length(df: pl.DataFrame):
    is_train = 'labels' in df
    max_length = CFG.token_max_length - 2  # [CLS], [SEP]
    splitable_input_ids = [323, 2600]  # ['.', ';'] -> [323, 2600]       
    new_df = {
        'document': [],
        'input_ids': [],
        'token_type_ids': [],
        'attention_mask': [],
        'word_ids': [],
    }
    if is_train:
        new_df['fold'] = []
        new_df['labels'] = []

    def append_new_row(document, input_ids, token_type_ids, attention_mask, word_ids, labels, fold):
        new_df['document'].append(document)             #  None
        new_df['input_ids'].append(input_ids)           # [1, 2]
        new_df['token_type_ids'].append(token_type_ids) # [0, 0]
        new_df['attention_mask'].append(attention_mask) # [1, 1]     
        new_df['word_ids'].append(word_ids)             # [None, None]
        if is_train:
            new_df['labels'].append(labels)             # [-100, -100]
            new_df['fold'].append(fold)                 #  None

    for row in df.iter_rows(named=True):
        document       = row['document']
        input_ids      = row['input_ids'][1:-1]
        token_type_ids = row['token_type_ids'][1:-1]
        attention_mask = row['attention_mask'][1:-1]
        word_ids       = row['word_ids'][1:-1]
        labels         = row['labels'][1:-1] if is_train else None
        fold           = row['fold'] if is_train else None

        splitable_idx = [0]
        for i, input_id in enumerate(input_ids):
            if input_id in splitable_input_ids:
                splitable_idx.append(i+1)
        splitable_idx.append(len(input_ids))

        start_id = 0
        for i in range(len(splitable_idx)-1):
            if splitable_idx[i+1] - start_id > max_length:
                end_id = splitable_idx[i]
                append_new_row(
                    document,
                    [1]    + input_ids[start_id:end_id]      + [2],
                    [0]    + token_type_ids[start_id:end_id] + [0],
                    [1]    + attention_mask[start_id:end_id] + [1],
                    [None] + word_ids[start_id:end_id]       + [None],
                    None if not is_train else ([-100] + labels[start_id:end_id] + [-100]),
                    fold
                )
                start_id = end_id

        if start_id != len(input_ids):
            append_new_row(
                document,
                [1]    + input_ids[start_id:]      + [2],
                [0]    + token_type_ids[start_id:] + [0],
                [1]    + attention_mask[start_id:] + [1],
                [None] + word_ids[start_id:]       + [None],
                None if not is_train else ([-100] + labels[start_id:] + [-100]),
                fold
            )

    new_df = pl.DataFrame(new_df)
    return new_df


def make_train_dataset():
    train_df = pl.read_json(data_path + "train.json") \
                 .filter(pl.col('labels').map_elements(lambda x: not all(label == 'O' for label in x))) \
                 .to_pandas()
    if CFG.sample_data_size: train_df = train_df[:CFG.sample_data_size]
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
    )
    train_dataset = train_dataset.map(
        add_fold_column,
        batched=True,
    )
    train_dataset = pl.DataFrame(train_dataset.to_pandas())
    train_df = split_below_max_length(train_dataset)
    train_df = train_df.to_pandas()
    train_dataset = Dataset.from_pandas(train_df)

    return train_dataset


def make_test_dataset():
    test_df = pl.read_json(data_path + "test.json").to_pandas()
    test_dataset = Dataset.from_pandas(test_df)
    test_dataset = test_dataset.map(
        tokenize_and_align_labels,
        batched=True,
    )

    test_dataset = pl.DataFrame(test_dataset.to_pandas())
    test_df = split_below_max_length(test_dataset)
    test_df = test_df.to_pandas()
    test_dataset = Dataset.from_pandas(test_df)

    return test_dataset


def make_dataset():
    train_dataset = make_train_dataset()
    test_dataset = make_test_dataset()
    datasets = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    return datasets


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenized_datasets = make_dataset()
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


def f_score(precision, recall, beta=1):
    epsilon = 1e-7
    return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + epsilon)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "f5": f_score(precision_score(true_labels, true_predictions), recall_score(true_labels, true_predictions), 5),
        "accuracy": accuracy_score(true_labels, true_predictions),
    }

def learning(hyperparams):
    if hyperparams is not None:
        print(f"\n\
                learning_rate:    {hyperparams['learning_rate']},\
                num_train_epochs: {hyperparams['num_train_epochs']}")

    best_f5_score = -1.0
    best_trainer = None

    gkf = GroupKFold(n_splits=CFG.fold)
    gkf_dataset = gkf.split(X=tokenized_datasets['train'],
                            y=tokenized_datasets['train']['labels'],
                            groups=tokenized_datasets['train']['fold'])
    avg_f5_score = 0.0
    for i, (train_index, valid_index) in enumerate(gkf_dataset):
        print(f"\nFold {i}")
        train_dataset = tokenized_datasets['train'].select(train_index)    
        valid_dataset = tokenized_datasets['train'].select(valid_index)

        model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint, id2label=id2label, label2id=label2id).to(device)

        learning_rate = 0.00960811179292991 if not CFG.use_optuna else hyperparams['learning_rate']
        num_train_epochs = 8 if not CFG.use_optuna else hyperparams['num_train_epochs']
        args = TrainingArguments(
            disable_tqdm=False,
            output_dir=f"bert-finetune-ner_{i}", 
            evaluation_strategy="steps", # "epoch", "no"
            eval_steps=1000,
            fp16=True,
            save_strategy="no",
            per_device_train_batch_size=CFG.batch_size,  # 1 is not out of memory
            per_device_eval_batch_size=CFG.batch_size,   # 1 is not out of memory
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            # load_best_model_at_end=True,
            weight_decay=0.01,
            push_to_hub=False,
            report_to="none",
            log_level="error",
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )

        # train model
        trainer.train()

        # evaluate model with valid dataset to get the best model
        eval_result = trainer.evaluate()
        # print(f"1 - (f5): {1 - eval_result['eval_f5']}")
        avg_f5_score += 1 - eval_result['eval_f5']

    avg_f5_score /= CFG.fold
    print(f"avg_f5_score: {avg_f5_score}")
    return avg_f5_score

def objective(trial):
    hyperparams = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-7, 1e-2),
        'num_train_epochs': trial.suggest_int('num_train_epochs', 1, 10),
    }
    eval_result = learning(hyperparams)
    return eval_result

def run_optuna():
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    best_value = study.best_value

    print("best_params: ", best_params)
    print("best_value: ", best_value)

if CFG.use_optuna:
    print("Use optuna")
    run_optuna()
else:
    print("Not use optuna")
    learning(None)