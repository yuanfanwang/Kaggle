import json
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import polars as pl
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, AutoModelForSequenceClassification
from datasets import Dataset
from torch.utils.data import Dataset as tds
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from transformers.modeling_outputs import ModelOutput

np.object = object

class Env:
    local = True

cache_dataset = True

if Env.local:
    train_json_file_path = "./data/train.json"
    test_json_file_path  = "./data/test.json"
else:
    train_json_file_path = "kaggle/train.json"
    test_json_file_path = "kaggle/test.json"


pii_to_num = {
    "O": 0,
    "B-NAME_STUDENT": 1,
    "I-NAME_STUDENT": 2,
    "B-EMAIL": 3,
    "I-EMAIL": 4,
    "B-USERNAME": 5,
    "I-USERNAME": 6,
    "B-ID_NUM": 7,
    "I-ID_NUM": 8,
    "B-PHONE_NUM": 9,
    "I-PHONE_NUM": 10,
    "B-URL_PERSONAL": 11,
    "I-URL_PERSONAL": 12,
    "B-STREET_ADDRESS": 13,
    "I-STREET_ADDRESS": 14,
}

num_to_pii = {
    1 : "O",
    2 : "B-NAME_STUDENT",
    3 : "I-NAME_STUDENT",
    4 : "B-EMAIL",
    5 : "I-EMAIL",
    6 : "B-USERNAME",
    7 : "I-USERNAME",
    8 : "B-ID_NUM",
    9 : "I-ID_NUM",
    10: "B-PHONE_NUM",
    11: "I-PHONE_NUM",
    12: "B-URL_PERSONAL",
    13: "I-URL_PERSONAL",
    14: "B-STREET_ADDRESS",
    15: "I-STREET_ADDRESS",
}

config = {
    'model': 'microsoft/deberta-v3-base'
}


####################################
# Tokenizer
####################################
class CustomTokenizer:
    def __init__(self, max_length=20):
        self.tokenizer = AutoTokenizer.from_pretrained(config['model'])
        self.max_length = max_length
    
    def remove_special_tokens(self, tokens):
        tokens.pop(0)
        tokens.pop(-1)
        return tokens

    def add_special_tokens(self, tokens):
        tokens.insert(0, 1)
        tokens.append(2)
        return tokens 

    # TODO: corresponding to max length
    def __call__(self, data):
        model_ids = []
        model_labels = []
        label_conversion_key = []

        for i, whitespace in enumerate(data['trailing_whitespace']):
            if whitespace:
                data['tokens'][i+1] += ' '

        for token, label in zip(data.get('tokens', []), data.get('labels', [])):
            input = self.tokenizer(token, padding="max_length", truncation=True)
            input_ids = input['input_ids']
            input_ids = self.remove_special_tokens(input_ids)

            model_ids += input_ids
            if 'labels' in data:
                model_labels += [pii_to_num[label]] * len(input_ids)
            label_conversion_key.append(len(input_ids))

            # trancate
            if len(model_ids) > self.max_length:
                max_sz = self.max_length-2
                model_ids = model_ids[:max_sz]
                model_labels = model_labels[:max_sz]
                label_conversion_key = label_conversion_key[:max_sz]
                break

        # TODO: model_label doens't have lables for special tokens at the beginnig and the end
        model_ids = self.add_special_tokens(model_ids)
        data["labels"] = model_labels
        # TODO: attention_mask should be 1 for only the size of the tokens
        additional = {
            'input_ids' : model_ids,
            'token_type_ids': [0] * len(model_ids),
            'attention_mask': [1] * len(model_ids),
            'label_conversion_key': label_conversion_key
        }
        return {**data, **additional}


def make_submission_labels(label_conversion_key, output_labels):
    submission_labels = []
    prev_idx = 0
    for l in label_conversion_key:
        token_label = output_labels[prev_idx: prev_idx + l]
        # use first label for now
        submission_labels.append(token_label[0])
        prev_idx += l

    return submission_labels


####################################
# Datasets 
####################################
tokenizer = CustomTokenizer()
if cache_dataset:
    train_dataset = Dataset.load_from_disk("/home/competitions/pii-detection-removal-from-educational-data/data/dataset/tokenized_train_20")
    test_dataset = Dataset.load_from_disk("/home/competitions/pii-detection-removal-from-educational-data/data/dataset/tokenized_test_20")
    # train_dataset = train_dataset.remove_columns(["labels"])
    # test_dataset = test_dataset.remove_columns(["labels"])
    # print(train_dataset[:5])
    # print(test_dataset[:5])
else:
    train_data = pl.read_json(train_json_file_path).to_pandas()   # document,
                                                                  # full_text,
                                                                  # tokens,
                                                                  # trailing_whitespace,
                                                                  # labels
    train_dataset = Dataset.from_pandas(train_data)
    train_dataset = train_dataset.map(tokenizer.__call__)
    train_dataset = train_dataset.remove_columns(["document", "full_text", "tokens", "trailing_whitespace", "label_conversion_key"])
    train_dataset.save_to_disk("/home/competitions/pii-detection-removal-from-educational-data/data/dataset/tokenized_train_20")

    test_data  = pl.read_json(test_json_file_path).to_pandas()
    test_dataset = Dataset.from_pandas(test_data)
    test_dataset = test_dataset.map(tokenizer.__call__)
    test_dataset = test_dataset.remove_columns(["document", "full_text", "tokens", "trailing_whitespace", "label_conversion_key"])
    test_dataset.save_to_disk("/home/competitions/pii-detection-removal-from-educational-data/data/dataset/tokenized_test_20")

train_dataset = train_dataset.with_format("torch")
test_dataset = test_dataset.with_format("torch")

####################################
# Model
####################################
class FbetaLoss(nn.Module):
    def __init__(self, beta=5):
        super().__init__()
        self.beta = beta
    
    # micro F beta
    def forward(self, targets, labels):
        if targets.shape != labels.shape:
            raise ValueError("targets and labels should have the same shape")

        true_positive = 1e-9
        false_positive = 1e-9
        false_negative = 1e-9
        true_negative = 1e-9
        for label, target in zip(labels[0], targets[0]):
            if label != 0 and target != 0:
                true_positive += 1
            elif label == 0 and target != 0:
                false_positive += 1
            elif label != 0 and target == 0:
                false_negative += 1
            elif label == 0 and target == 0:
                true_negative += 1

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        Fbeta = (1 + self.beta**2) * (precision * recall) / (self.beta**2 * precision + recall)
        return Fbeta

class CustomModel(nn.Module):
    def __init__(self, pretrained_model, loss_function, label_size):
        super().__init__()
        self.debartav3base = pretrained_model
        self.hidden_size = self.debartav3base.config.hidden_size
        self.liner = nn.Linear(self.hidden_size, label_size)
        self.loss_function = loss_function

    # witout labels
    def forward(self, input_data):
        print('labels' in input_data)
        labels = input_data.pop("labels")
        print('labels' in input_data)
        outputs = self.debartav3base(**input_data)
        targets = outputs.last_hidden_state[:, 0, :]  # shape: [1, 20, 768] -> [1, 768]
        targets = self.liner(targets)
        targets = torch.round(targets)

        loss=None
        if labels is not None:
            loss = self.loss_function(targets, labels)
        else:
            print("Error: labels not in input_data or loss_function is None")
        
        return ModelOutput(
            logits=targets,
            loss=loss,
            last_hidden_state=outputs.last_hidden_state,
            attentions=None,
            hidden_states=None
        )

"""
class CustomModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.liner = nn.Linear(768, 1)
 
    def forward(self, inputs):
        labels = inputs.pop("labels")
        outputs = self.model(**inputs)
        targets = outputs.last_hidden_state[:, 0, :]
        targets = self.liner(targets)
        loss = nn.L1Loss(targets, labels)

        return ModelOutput(
            loss=loss,
            logits=targets,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""


class CustomDataCollator(DataCollatorWithPadding):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config['model'])


model = AutoModelForSequenceClassification.from_pretrained(config['model'])
fbeta5_loss = FbetaLoss()
custom_model = CustomModel(model, fbeta5_loss, 18)
custom_data_collator = CustomDataCollator()
print(custom_model(train_dataset[:1]))
sys.exit()
training_args = TrainingArguments(
    output_dir="trained_model",
    evaluation_strategy="epoch",
    report_to="none",  # for wandb
)

trainer = Trainer(
    model=custom_model,
    args=training_args,
    train_dataset=train_dataset[:50],
    eval_dataset=test_dataset[:10],
    data_collator=custom_data_collator
)

####################################
# Train 
####################################
trainer.train()
# trainer.predict(test_dataset[:10])

"""
####################################
# Evaluation 
####################################
eval_results = trainer.evaluate()
print(eval_results)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_text = "Text"
input_data = tokenizer(input_text, return_tensors='pt').to(device)
outputs = model(**input_data)
outputs.logits.argmax(-1).item()
"""