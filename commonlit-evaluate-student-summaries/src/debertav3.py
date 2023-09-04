import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import logging
import os
import shutil
import json
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from datasets import Dataset,load_dataset, load_from_disk
from transformers import TrainingArguments, Trainer
from datasets import load_metric, disable_progress_bar
from sklearn.metrics import mean_squared_error
import torch
from sklearn.model_selection import KFold, GroupKFold

warnings.simplefilter('ignore')
logging.disable(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
disable_progress_bar()


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
    torch.backends.cudnn.benchmard = True

# seed_everything(42)

class CFG:
    model_name                   = "debertav3base"
    learning_rate                = 1.5e-5
    weight_decay                 = 0.02
    hidden_dropout_prob          = 0.0
    attention_probs_dropout_prob = 0.0
    num_train_epochs             = 5
    n_splits                     = 4
    batch_size                   = 12
    random_seed                  = 42
    save_steps                   = 100
    max_length                   = 512

# DATA_DIR = "/kaggle/input/commonlit-evaluate-student-summaries/"
DATA_DIR = "data/"

prompts_train     = pd.read_csv(DATA_DIR + "prompts_train.csv")
prompts_test      = pd.read_csv(DATA_DIR + "prompts_test.csv")
summaries_train   = pd.read_csv(DATA_DIR + "summaries_train.csv")
summaries_test    = pd.read_csv(DATA_DIR + "summaries_test.csv")
sample_submission = pd.read_csv(DATA_DIR + "sample_submission.csv")
train = summaries_train.merge(prompts_train, how="left", on="prompt_id")
test  = summaries_test.merge(prompts_test, how="left", on="prompt_id")

gkf = GroupKFold(n_splits=CFG.n_splits)

for i, (_, val_index) in enumerate(gkf.split(train, groups=train['prompt_id'])):
    train.loc[val_index, 'fold'] = int(i)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}

def compute_mcrmes(eval_pred):
    preds, labels = eval_pred

    col_rmse = np.sqrt(np.mean((preds - labels) ** 2, axis=0))
    mcrmse = np.mean(col_rmse)

    return {
        "content_rmse": col_rmse[0],
        "wording_rmse": col_rmse[1],
        "mcrmse": mcrmse,
    }

def compt_score(content_true, content_pred, wording_true, wording_pred):
    content_score = mean_squared_error(content_true, content_pred)**(1/2)
    wordiing_score = mean_squared_error(wording_true, wording_pred)**(1/2)

    return (content_score + wordiing_score) / 2


class ContentScoreRegressor:
    def __init__(self,
                 model_name: str,
                 model_dir: str,
                 target: str,
                 hidden_dropout_prob: float,
                 attention_probs_dropout_prob: float,
                 max_length: int,
                 ):
        self.text_cols = ["text"]
        self.target = target
        self.target_cols = [target]

        self.model_name = model_name
        self.model_dir = model_dir
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(f"/kaggle/input/{model_name}")
        self.model_config = AutoConfig.from_pretrained(f"/kaggle/input/{model_name}")

        self.model_config.update({
            "hidden_dropout_prob": hidden_dropout_prob,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "num_labels": 1,
            "problem_type": "regression",
        })

        seed_everything(seed=42)

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer
        )

    def tokenize_function(self, examples: pd.DataFrame):
        labels = [examples[self.target]]
        tokenized = self.tokenizer(examples["text"],
                                   padding=False,
                                   truncation=True,
                                   max_length=self.max_length)
        return {
            **tokenized,
            "labels": labels
        }

    def tokenize_function_test(self, examples: pd.DataFrame):
        tokenized = self.tokenizer(examples["text"],
                                   padding=False,
                                   truncation=True,
                                   max_length=self.max_length)
        return tokenized
    
    def train(self,
              fold: int,
              train_df: pd.DataFrame,
              valid_df: pd.DataFrame,
              batch_size: int,
              learning_rate: float,
              weight_decay: float,
              num_train_epochs: int,
              save_steps: int,
        ) -> None:
        """ fine-tuning """

        train_df = train_df[self.text_cols + self.target_cols]
        valid_df = valid_df[self.text_cols + self.target_cols]
              
        model_content = AutoModelForSequenceClassification.from_pretrained(
            f"/kaggle/input/{self.model_name}",
            config=self.model_config
        )

        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        val_dataset = Dataset.from_pandas(valid_df, preserve_index=False)

        train_tokenized_datasets = train_dataset.map(self.tokenize_function, batched=False)
        val_tokenized_datasets = val_dataset.map(self.tokenize_function, batched=False)

        model_fold_dir = os.path.join(self.model_dir, str(fold))

        training_args = TrainingArguments(
            output_dir=model_fold_dir,
            load_best_model_at_end=True, # select best model
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=8,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            report_to='none',
            greater_is_better=False,
            save_strategy="steps",
            evaluation_strategy="steps",
            eval_steps=save_steps,
            save_steps=save_steps,
            metric_for_best_model="rmse",
            save_total_limit=1
        )

        trainer = Trainer(
            model=model_content,
            args=training_args,
            train_dataset=train_tokenized_datasets,
            eval_dataset=val_tokenized_datasets,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            data_collator=self.data_collator
        )

        trainer.train()

        model_content.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained
    
    def predict(self,
                test_df: pd.DataFrame,
                fold: int
        ):
        """ predict content score """
        test_ = test_df[self.text_cols]

        test_dataset = Dataset.from_pandas(test_, preserve_index=False)
        test_tokenized_dataset = test_dataset.map(self.tokenize_function_test, batched=False)

        model_content = AutoModelForSequenceClassification.from_pretrained(f"{self.model_dir}")
        model_content.eval()

        model_fold_dir = os.path.join(self.model_dir, str(fold))

        test_args = TrainingArguments(
            output_dir=model_fold_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=4,
            dataloader_drop_last=False,
        )

        # init trainer
        infer_content = Trainer(
                      model = model_content, 
                      tokenizer=self.tokenizer,
                      data_collator=self.data_collator,
                      args = test_args)

        preds = infer_content.predict(test_tokenized_dataset)[0]

        return preds


def train_by_fold(
        train_df: pd.DataFrame,
        model_name: str,
        target:str,
        save_each_model: bool,
        n_splits: int,
        batch_size: int,
        learning_rate: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        weight_decay: float,
        num_train_epochs: int,
        save_steps: int,
        max_length:int
    ):

    # delete old model files
    if os.path.exists(model_name):
        shutil.rmtree(model_name)
    
    os.mkdir(model_name)
        
    for fold in range(CFG.n_splits):
        print(f"fold {fold}:")
        
        train_data = train_df[train_df["fold"] != fold]
        valid_data = train_df[train_df["fold"] == fold]
        
        if save_each_model == True:
            model_dir =  f"{target}/{model_name}/fold_{fold}"
        else: 
            model_dir =  f"{model_name}/fold_{fold}"

        csr = ContentScoreRegressor(
            model_name=model_name,
            target=target,
            model_dir = model_dir, 
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
           )
        
        csr.train(
            fold=fold,
            train_df=train_data,
            valid_df=valid_data, 
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps,
        )

def validate(
    train_df: pd.DataFrame,
    target:str,
    save_each_model: bool,
    model_name: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    max_length : int
    ) -> pd.DataFrame:
    """predict oof data"""
    for fold in range(CFG.n_splits):
        print(f"fold {fold}:")
        
        valid_data = train_df[train_df["fold"] == fold]
        
        if save_each_model == True:
            model_dir =  f"{target}/{model_name}/fold_{fold}"
        else: 
            model_dir =  f"{model_name}/fold_{fold}"
        
        csr = ContentScoreRegressor(
            model_name=model_name,
            target=target,
            model_dir = model_dir, # モデル・foldごとにモデルファイルの保存先のdirを分ける
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
           )
        
        pred = csr.predict(
            test_df=valid_data, 
            fold=fold
        )
        
        train_df.loc[valid_data.index, f"{target}_pred"] = pred

    return train_df
    
def predict(
    test_df: pd.DataFrame,
    target:str,
    save_each_model: bool,
    model_name: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    max_length : int
    ):
    """predict using mean folds"""

    for fold in range(CFG.n_splits):
        print(f"fold {fold}:")
        
        if save_each_model == True:
            model_dir =  f"{target}/{model_name}/fold_{fold}"
        else: 
            model_dir =  f"{model_name}/fold_{fold}"

        csr = ContentScoreRegressor(
            model_name=model_name,
            target=target,
            model_dir = model_dir, 
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
           )
        
        pred = csr.predict(
            test_df=test_df, 
            fold=fold
        )
        
        test_df[f"{target}_pred_{fold}"] = pred
    
    test_df[f"{target}"] = test_df[[f"{target}_pred_{fold}" for fold in range(CFG.n_splits)]].mean(axis=1)

    return test_df


for target in ["content", "wording"]:
    train_by_fold(
        train,
        model_name=CFG.model_name,
        save_each_model=False,
        target=target,
        learning_rate=CFG.learning_rate,
        hidden_dropout_prob=CFG.hidden_dropout_prob,
        attention_probs_dropout_prob=CFG.attention_probs_dropout_prob,
        weight_decay=CFG.weight_decay,
        num_train_epochs=CFG.num_train_epochs,
        n_splits=CFG.n_splits,
        batch_size=CFG.batch_size,
        save_steps=CFG.save_steps,
        max_length=CFG.max_length
    )
    
    
    train = validate(
        train,
        target=target,
        save_each_model=False,
        model_name=CFG.model_name,
        hidden_dropout_prob=CFG.hidden_dropout_prob,
        attention_probs_dropout_prob=CFG.attention_probs_dropout_prob,
        max_length=CFG.max_length
    )

    rmse = mean_squared_error(train[target], train[f"{target}_pred"], squared=False)
    print(f"cv {target} rmse: {rmse}")

    test = predict(
        test,
        target=target,
        save_each_model=False,
        model_name=CFG.model_name,
        hidden_dropout_prob=CFG.hidden_dropout_prob,
        attention_probs_dropout_prob=CFG.attention_probs_dropout_prob,
        max_length=CFG.max_length
    )

mcrmse = compt_score(content_true=train["content"], 
            content_pred=train["content_pred"], 
            wording_true=train["wording"],
            wording_pred=train["wording_pred"], 
           )
print(f"cv mcrmse: {mcrmse}")
print(test)
print(sample_submission)

print(test[["student_id", "content", "wording"]].to_csv("submission.csv", index=False))