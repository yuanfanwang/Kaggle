import sys
import torch

import numpy as np
import polars as pl
from datasets import Dataset, DatasetDict, load_metric
from huggingface_hub import notebook_login
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer

"""
!pip install seqeval
"""

torch.cuda.empty_cache()
np.object = object
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Local = True
if Local:
    data_path = "data/"
else:
    data_path = "/kaggle/input/pii-detection-removal-from-educational-data/"

model_checkpoint = "microsoft/deberta-v3-base"
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
    data_size = 100
    batch_size = 1
    token_max_length = 16

def make_dataset():
    train_data = pl.read_json(data_path + "train.json").to_pandas()
    train_data = train_data[:CFG.data_size]
    train_dataset = Dataset.from_pandas(train_data)
    valid_dataset = train_dataset
    test_data = pl.read_json(data_path + "test.json").to_pandas()
    test_dataset = Dataset.from_pandas(test_data)
    datasets = DatasetDict({
        "train": train_dataset,
        "validation": valid_dataset,
        # "test": test_dataset
    })
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
    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=CFG.token_max_length
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

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
metric = load_metric("seqeval")

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
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "f5": f_score(all_metrics["overall_precision"], all_metrics["overall_recall"], 5),
        "accuracy": all_metrics["overall_accuracy"],
    }

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
model = model.to(device)

args = TrainingArguments(
    output_dir="bert-finetune-ner",
    evaluation_strategy="epoch",
    # evaluation_strategy="steps",
    # eval_steps=10,
    save_strategy="epoch",
    per_device_train_batch_size=CFG.batch_size,  # 1 is not out of memory
    per_device_eval_batch_size=CFG.batch_size,   # 1 is not out of memory
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    report_to="none",
    log_level="error",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()

# how to make F beta score as the loss function
# Training Loss はなんの関数を使って計算されたのか