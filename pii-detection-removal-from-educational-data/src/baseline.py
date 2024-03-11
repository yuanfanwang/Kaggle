"""
!pip install "/kaggle/input/seqeval/seqeval-1.2.2-py3-none-any.whl"
"""

import sys
import torch

import numpy as np
import pandas as pd
import polars as pl

from datasets import Dataset, DatasetDict, load_metric
from huggingface_hub import notebook_login
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
from seqeval.metrics import precision_score, recall_score, accuracy_score, f1_score


# set random seed
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(seed=42)


""" memo
# input_ids 1, 2 が trainer.evaluate() でどのように評価されるか
# whitespace, " ", "\n\n" などの対応
# seperate a long token
"""

torch.cuda.empty_cache()
np.object = object
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Local = False
if Local:
    data_path = "data/"
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

class CFG:
    sample_data_size = 1000
    batch_size = 1
    token_max_length = 1024
    epoch = 3
    fold = 4

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


def make_dataset():
    train_df = pl.read_json(data_path + "train.json").to_pandas()
    # train_df = train_df[:CFG.sample_data_size]
    train_dataset = Dataset.from_pandas(train_df)
    test_df = pl.read_json(data_path + "test.json").to_pandas()
    test_dataset = Dataset.from_pandas(test_df)
    datasets = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

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


best_f5_score = -1.0
best_trainer = None
gkf = GroupKFold(n_splits=CFG.fold)
gkf_dataset = gkf.split(X=tokenized_datasets['train'],
                        y=tokenized_datasets['train']['labels'],
                        groups=tokenized_datasets['train']['fold'])
for i, (train_index, valid_index) in enumerate(gkf_dataset):
    print(f"\nFold {i}")
    train_dataset = tokenized_datasets['train'].select(train_index)    
    valid_dataset = tokenized_datasets['train'].select(valid_index)

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, id2label=id2label, label2id=label2id).to(device)

    args = TrainingArguments(
        disable_tqdm=False,
        output_dir=f"bert-finetune-ner_{i}",
        # evaluation_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=1000,
        # fp16=True,
        save_strategy="no",
        per_device_train_batch_size=CFG.batch_size,  # 1 is not out of memory
        per_device_eval_batch_size=CFG.batch_size,   # 1 is not out of memory
        learning_rate=2e-5,
        num_train_epochs=CFG.epoch,
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
    print(f"f5: {eval_result['eval_f5']}")

    # TODO: for ansamble (but this process takes about 20min if the size of the test dataset is 20,000.)
    # prediction = trainer.predict(tokenized_datasets['test'])
    # label_predictions = np.argmax(prediction.predictions, axis=-1)

    if best_f5_score < eval_result['eval_f5']:
        best_f5_score = eval_result['eval_f5']
        best_trainer = trainer


# predict labels 
test_datasets     = tokenized_datasets['test']
prediction        = best_trainer.predict(test_datasets)  # PredictionOutput Object
label_predictions = np.argmax(prediction.predictions, axis=-1)  # (data_size, token max length)

# create submission file
row_id_sub, document_sub, token_sub, label_sub = [], [], [], []
test_df = pd.DataFrame(test_datasets)
for i, labels in enumerate(label_predictions):
    word_ids = test_df.at[i, 'word_ids']
    document = test_df.at[i, 'document']
    current_id = None
    for j, (pii_idx, word_id) in enumerate(zip(labels, word_ids)):
        if word_id == None: continue
        if current_id != word_id:
            current_id = word_id
            if pii_idx != 0:
                row_id_sub.append(len(row_id_sub))
                document_sub.append(document)
                token_sub.append(word_id)
                label_sub.append(label_names[pii_idx])

submission = {
    'row_id': row_id_sub,
    'document': document_sub,
    'token': token_sub,
    'label': label_sub
}

submission = pd.DataFrame(submission)
print(submission.head())
submission.to_csv("submission.csv", index=False)