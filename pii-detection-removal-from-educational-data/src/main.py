""" install offline
!pip install "/kaggle/input/seqeval/seqeval-1.2.2-py3-none-any.whl"
"""

""" beneficial links
https://www.kaggle.com/code/samsay23/pii-notebook-generalization-with-score-0-966
https://www.kaggle.com/code/sohaibahmed9920/pii-data-prep-eda-fix-punctuation-orig-ext/notebook
"""
import copy
import optuna
import os
import random
import sys
import torch

import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import polars            as pl
import torch.nn          as nn

from datasets                import (Dataset, DatasetDict)
from seqeval.metrics         import (accuracy_score, f1_score, precision_score, recall_score)
from sklearn.model_selection import GroupKFold
from tqdm                    import tqdm
from transformers            import (AutoModelForTokenClassification, AutoTokenizer, AutoConfig,
                                     DataCollatorForTokenClassification, Trainer, TrainingArguments,
                                     TrainerCallback, TrainerState, TrainerControl)
from typing                  import Dict


####################### . Config . #######################
class CFG:
    ## variable
    train_with_only_gen_data = True
    sample_data_size = None
    use_optuna = False
    train_kwargs = {
        "evaluation_strategy": "steps",
        "eval_steps": 500,
        "logging_steps": 250,
        "save_strategy": "no",
        "learning_rate": 1e-5,
        "num_train_epochs": 10,
        "weight_decay": 0.01,
    }
    optuna_kwargs = {
        "evaluation_strategy": "epoch",
        "logging_steps": 250,
        "save_strategy": "no",
        # "learning_rate": 
        # "num_train_epochs":
        # "weight_decay":
        "n_trials": 20,
    }
    # parameters are defined in optuna_hp_space and model_init as well

    ## constant
    local = False
    batch_size = 1
    token_max_length = 1024
    fold = 4
    downsampling = True


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(seed=42)
torch.cuda.empty_cache()
np.object = object
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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











####################### . Functions . #######################

class SoftF5Loss(nn.Module):
    def __init__(self, smooth=1e-16):
        super(SoftF5Loss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, labels):
        # logits:  [token_len, num_labels]
        # labels:  [token_len]
        # loss  :  []

        # adjust labels size to logits size [token_len] -> [token_len, num_labels]
        one_hot_labels = torch.zeros_like(logits)
        indices = [i for i, label in enumerate(labels) if label != -100]
        one_hot_labels[indices, labels[indices]] = 1  # TODO: if is it correct to use indices?

        # TODO: need to normalize probs?. Why I think so is that
        #       it is intuitive that each of the 15 labels has a probability, and summing them all together yields 1.
        probs = torch.sigmoid(logits) # TODO: sigmoid の曲率を上げる

        # remove first and end to avoid special token -100
        probs = probs[indices]
        one_hot_labels = one_hot_labels[indices]

        # Calculate F5
        tp = torch.sum(probs * one_hot_labels, dim=0)
        fp = torch.sum(probs * (1 - one_hot_labels), dim=0)
        fn = torch.sum((1 - probs) * one_hot_labels, dim=0)
        # tn = torch.sum((1 - probs) * (1 - one_hot_labels), dim=0)

        # label 0   と判定するのは基本的にpositiveなのでtpかfnが大きい
        # label 0以外と判定するのは基本的にnegativeなのでfpかtnが大きい
        # よって 以下の式より label 0 は soft_f5 が大きく label 0以外は小さい
        # これはデータの不足を意味しているはず, label 0 以外 の fp と tn の大きさが同じということは学習が初期段階だから
        # または sigmoid の曲率が不足しているからだと考えるので、両者を試す
        beta = 5
        soft_f5 = ((1 + beta**2)*tp) / ((1 + beta**2)*tp + (beta**2)*fn + fp + self.smooth)
        cost = 1 - soft_f5 # subtract from 1 to get cost

        # TODO: mean or sum?
        return cost.mean()


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss_fct = SoftF5Loss()
        # loss_fct = nn.CrossEntropyLoss()
        labels = inputs.pop("labels").view(-1)
        logits = outputs.get('logits').view(-1, self.model.config.num_labels)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


class LossCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.current_epoch = 0
        # depends on logging_steps
        self.eval_f5 = []
        self.logging_epoch = []
        # depends on eval_steps
        self.training_loss = []
        self.validation_loss = []
        self.eval_epoch = []

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if state.is_local_process_zero:
            if ('loss' in logs) and ('epoch' in logs):
                self.training_loss.append(logs['loss'])
                self.logging_epoch.append(logs['epoch'])
            if ('eval_f5' in logs) and ('eval_loss' in logs) and ('epoch' in logs):
                self.eval_f5.append(logs['eval_f5'])
                self.validation_loss.append(logs['eval_loss'])
                self.eval_epoch.append(logs['epoch'])
                if self.current_epoch < int(logs['epoch'] / 1):
                    self.current_epoch = int(logs['epoch'] / 1)
                    self.plot_all()

    def clear(self):
        self.current_epoch = -1
        self.eval_f5 = []
        self.logging_epoch = []
        self.training_loss = []
        self.validation_loss = []
        self.eval_epoch = []
        pass
        
    def plot_all(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.logging_epoch, self.training_loss, label='Training Loss', color='blue')
        plt.plot(self.eval_epoch, self.validation_loss, label='Validation Loss', color='orange')
        plt.plot(self.eval_epoch, self.eval_f5, label='F5', color='green')
        plt.title('Training and Evaluation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.show()


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
    train_df = pl.read_json(data_path + "train.json")
    if not CFG.local:
        faker_df = pl.read_json("/kaggle/input/pii-dd-mistral-generated/mixtral-8x7b-v1.json")
        new_series = pl.Series([int(f"10000{i}") for i in range(len(faker_df))])
        faker_df.replace("document", new_series)
        faker_df = faker_df.select(["document", "full_text", "tokens", "trailing_whitespace", "labels"])
        train_df = pl.concat([train_df, faker_df])
    if CFG.downsampling:
        train_df = train_df.filter(pl.col('labels').map_elements(lambda x: not all(label == 'O' for label in x)))
    train_df = train_df.to_pandas()
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


def f_score(precision, recall, beta=1):
    epsilon = 1e-16
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


def optuna_hp_space(trial):
    # OPTUNA: learning_rate, num_train_epochs, weight_decay
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-3), # 2e-5  as default
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),   # 3     as default
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1),   # 0.01  as default
        # "optimizer": "AdamW",
    }


def model_init(trial):
    config = AutoConfig.from_pretrained(model_checkpoint, id2label=id2label, label2id=label2id)
    config.hidden_dropout_prob = 0.2
    # OPTUNA: hidden_dropout_prob, attention_probs_dropout_prob
    if trial is not None:
        config.hidden_dropout_prob = trial.suggest_float("hidden_dropout_prob", 1e-2, 1.0)                   # 0.1 as default
        config.attention_probs_dropout_prob = trial.suggest_float("attention_probs_dropout_prob", 1e-2, 1.0) # 0.1 as default
    return AutoModelForTokenClassification.from_pretrained(model_checkpoint, config=config).to(device)


def compute_objective(metrics: Dict[str, float]) -> float:
    metrics = copy.deepcopy(metrics)
    loss = metrics['eval_f5']
    return loss


def create_trainer(train_dataset, valid_dataset, **kwargs):
    args = TrainingArguments(
        ## variable
        output_dir=f"bert_fold{kwargs.get('fold', '')}", 
        evaluation_strategy=kwargs.get('evaluation_strategy', 'epoch'),
        eval_steps=kwargs.get('eval_steps', None),
        logging_steps=kwargs.get('logging_steps', 500),
        save_strategy=kwargs.get('save_strategy', 'no'),
        learning_rate=kwargs.get('learning_rate', 2e-5),
        num_train_epochs=kwargs.get('num_train_epochs', 3),
        weight_decay=kwargs.get('weight_decay', 0.01),
        ## constant
        disable_tqdm=False,
        fp16=True,
        push_to_hub=False,
        report_to="none",
        log_level="error",
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size,
        # load_best_model_at_end=True,
    )

    plot_loss_callback.clear()
    trainer = CustomTrainer(
        model_init=model_init,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[plot_loss_callback],
    )

    return trainer







####################### . Model . #######################
tokenizer          = AutoTokenizer.from_pretrained(model_checkpoint)
tokenized_datasets = make_dataset()
data_collator      = DataCollatorForTokenClassification(tokenizer=tokenizer)
plot_loss_callback = LossCallback()







####################### . Training . #######################
def train_model_simple():
    all_train_dataset = tokenized_datasets['train']
    if CFG.train_with_only_gen_data:
        train_index = [i for i in range(len(all_train_dataset)) if all_train_dataset['document'][i] >= 10000]
        valid_index = [i for i in range(len(all_train_dataset)) if all_train_dataset['document'][i] < 10000]
    else:
        dataset_index = [i for i in range(len(all_train_dataset))]
        train_index = dataset_index[:int(len(all_train_dataset)*0.8)]
        valid_index = dataset_index[int(len(all_train_dataset)*0.8):]
    train_dataset = all_train_dataset.select(train_index)
    valid_dataset = all_train_dataset.select(valid_index)

    trainer = create_trainer(train_dataset, valid_dataset, **CFG.train_kwargs)
    trainer.train()

    return trainer


def train_model_kfold():
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

        CFG.train_kwargs['fold'] = i
        trainer = create_trainer(train_dataset, valid_dataset, **CFG.train_kwargs)
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

    return best_trainer


def train_model_optuna():

    all_train_dataset = tokenized_datasets['train']
    dataset_index = [i for i in range(len(all_train_dataset))]
    train_index = dataset_index[:int(len(all_train_dataset)*0.8)]
    valid_index = dataset_index[int(len(all_train_dataset)*0.8):]
    train_dataset = all_train_dataset.select(train_index)
    valid_dataset = all_train_dataset.select(valid_index)

    trainer = create_trainer(train_dataset, valid_dataset, **CFG.optuna_kwargs)

    best_trails = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=CFG.optuna_kwargs['n_trials'],
        compute_objective=compute_objective
    )

    # https://rightcode.co.jp/blog/information-technology/torch-optim-optimizer-compare-and-verify-update-process-and-performance-of-optimization-methods
    # classes = ['SGD', 'Adagrad', 'RMSprop', 'Adadelta', 'Adam', 'AdamW']
    # weight decay is the parameter for AdamW
    # when sould I use suggest_categorical to choose optimizer?
    print(best_trails)















####################### . Main . #######################

if CFG.use_optuna:
    train_model_optuna()
else:
    # trainer = train_model_kfold()
    trainer = train_model_simple()

    ####################### . Submission  . #######################

    # predict labels 
    test_datasets     = tokenized_datasets['test']
    prediction        = trainer.predict(test_datasets)  # PredictionOutput Object
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
                    token_sub.append(int(word_id))
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