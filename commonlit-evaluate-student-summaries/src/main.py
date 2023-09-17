from typing import Any
import numpy as np
import pandas as pd
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
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from collections import Counter
import spacy
import re
#from autocorrect import Speller
from spellchecker import SpellChecker
import lightgbm as lgb

from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor


torch.cuda.empty_cache()
warnings.simplefilter("ignore")
logging.disable(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
disable_progress_bar()
tqdm.pandas()


class CFG:
    model_name="debertav3base"
    learning_rate=1.5e-5
    weight_decay=0.02
    hidden_dropout_prob=0.005
    attention_probs_dropout_prob=0.005
    num_train_epochs=5
    n_splits=4
    batch_size=10  # TODO: default: 12
    random_seed=42
    save_steps=100
    max_length=512

    
def seed_everything(seed: int):
    import random, os
    import numpy as np
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

seed_everything(CFG.random_seed)


DATA_DIR = "/kaggle/input/commonlit-evaluate-student-summaries/"

prompts_train = pd.read_csv(DATA_DIR + "prompts_train.csv")
prompts_test = pd.read_csv(DATA_DIR + "prompts_test.csv")
summaries_train = pd.read_csv(DATA_DIR + "summaries_train.csv")
summaries_test = pd.read_csv(DATA_DIR + "summaries_test.csv")
sample_submission = pd.read_csv(DATA_DIR + "sample_submission.csv")


class Tokenizer:
    def __init__(self,
                 model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(f"/kaggle/input/{model_name}")
        self.STOP_WARDS = set(stopwords.words('english'))
        self.spacy_ner_model = spacy.load('en_core_web_sm')
        self.speller = SpellChecker()

    def encode(self,
               text: str):
        return self.tokenizer.encode(text)

    def convert_ids_to_tokens(self,
                              text):
        return self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(text),
            skip_special_tokens=True
        )


class TextPreprocessor:
    def __init__(self,
                 tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.STOP_WARDS = set(stopwords.words('english'))
        self.spacy_ner_model = spacy.load('en_core_web_sm')
        self.speller = SpellChecker()

    def lower(self, text: str):
        res = text.lower()
        return res 

    def add_space_between_usual_symbol(self, text: str):
        res = text.replace('"', ' " ')
        res = res.replace(';', ' ; ')
        res = res.replace('.', ' . ')
        res = res.replace(',', ' , ')
        return res

    def remove_unusual_symbol(self, text: str):
        not_allowed_symbol = r'[^a-zA-Z0-9\s\"\'\-\;\.\,]'
        res = re.sub(not_allowed_symbol, ' ', text)
        return res

    def remove_blank(self, text: str):
        res = re.sub(r'\s{2,}', ' ', text)
        return res

    def delete_space_at_the_end(self, text: str):
        res = text.rstrip()
        if not res.endswith(' .'):
            res += ' .'
        return res

    def remove_stop_ward(self, text: str):
        pass

    def correct_spell(self, text: str):
        words = text.split(' ')
        #corrected_words = []
        #for word in words:
        #    corrected_word = self.speller.correction(word)
        #    if corrected_word == None:
        #        corrected_word = word
        #    corrected_words.append(corrected_word)

        res = ' '.join(words)
        return res
    
    # TODO: prompt_question prompt_title
    def __call__(self,
                 df: pd.DataFrame,
                 col: str) -> pd.DataFrame:
        df[col] = df[col].progress_apply(
            lambda x: self.lower(x))
        df[col] = df[col].progress_apply(
            lambda x: self.add_space_between_usual_symbol(x))
        df[col] = df[col].progress_apply(
            lambda x: self.remove_unusual_symbol(x))
        df[col] = df[col].progress_apply(
            lambda x: self.remove_blank(x))
        df[col] = df[col].progress_apply(
            lambda x: self.delete_space_at_the_end(x))
        # df[col] = df[col].apply(lambda x: self.remove_stop_ward(x))
        # df[col] = df[col].progress_apply(lambda x: self.correct_spell(x))
        return df


class FeatureExtractor:
    def __init__(self,
                 tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __call__(self):
        pass


class Summarizer:
    def __init__(self):
        self.auto_abstractor = AutoAbstractor()
        self.auto_abstractor.tokenizable_doc = SimpleTokenizer()
        self.auto_abstractor.delimiter_list = [".", "\n"]
        self.abstractable_doc = TopNRankAbstractor()

    # TODO result_dict has many keys
    def __call__(self, text: str) -> Any:
        result_dict = self.auto_abstractor.summarize(text, self.abstractable_doc)
        res = ""
        for sentence in result_dict["summarize_result"]:
            res += sentence.strip() + ' '
        return res.strip()


class ContentFeatureExtractor(FeatureExtractor):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer)
        self.summarizer = Summarizer()
    

    def __call__(self,
                 prompts: pd.DataFrame,
                 summaries: pd.DataFrame) -> pd.DataFrame:

        # TODO: should be removed symbols
        prompts["prompt_legth"] = prompts["prompt_text"].progress_apply(
            lambda x: len(self.tokenizer.encode(x))
        )

        # prompts["prompt_tokens"] = prompts["prompt_text"].progress_apply(
        #     lambda x: self.tokenizer.convert_ids_to_tokens(x)
        # )

        prompts["prompt_sum_text"] = prompts["prompt_text"].progress_apply(
            lambda x: self.summarizer(x)
        )

        # TODO: should be removed symbols
        summaries["summary_length"] = summaries["text"].progress_apply(
            lambda x: len(self.tokenizer.encode(x))
        )

        # summaries["summary_tokens"] = summaries["text"].progress_apply(
        #     lambda x: self.tokenizer.convert_ids_to_tokens(x)
        # )

        input_df = summaries.merge(prompts, how="left", on="prompt_id")

        return input_df
        # return input_df.drop(columns=["summary_tokens", "prompt_tokens"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}


class ContentScoreRegressor:
    def __init__(self,
                 model_name: str,
                 model_dir: str,
                 target: str,
                 hidden_dropout_prob: float,
                 attention_probs_dropout_prob: float,
                 max_length: int):
 
        self.input_cols = ["prompt_question", "prompt_title", "prompt_sum_text", "text"]

        self.target = target
        self.target_cols = [target]

        self.model_name = model_name
        self.model_dir = model_dir
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(f"/kaggle/input/{model_name}")
        self.model_config = AutoConfig.from_pretrained(f"/kaggle/input/{model_name}")
        self.model_config.update({
            "hedden_dropout_prob": hidden_dropout_prob,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "num_labels": 1,
            "problem_type": "regression"
        })

        seed_everything(CFG.random_seed)

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer
        )

    def tokenize_function(self, examples: pd.DataFrame):
        labels = [examples[self.target]]
        prompt_title = examples["prompt_title"]
        prompt_sum_text = examples["prompt_sum_text"]
        prompt_question = examples["prompt_question"]
        text = examples["text"]
        # TODO: can it be like this? [prompt_title, prompt_sum_text, prompt_question], text
        tokenized = self.tokenizer(prompt_title + prompt_sum_text + prompt_question, text,
                                   padding="max_length",
                                   truncation=True,
                                   max_length=self.max_length)
        token = {
            **tokenized,
            "labels": labels
        }
        # print(token)
        return token

    def train(self,
              fold: int,
              train_df: pd.DataFrame,
              valid_df: pd.DataFrame,
              batch_size: int,
              learning_rate: float,
              weight_decay: float,
              num_train_epochs: int,
              save_steps: int) -> None:

        input_train_df = train_df[self.input_cols + self.target_cols]
        input_valid_df = valid_df[self.input_cols + self.target_cols]

        model_content = AutoModelForSequenceClassification.from_pretrained(
            f"/kaggle/input/{self.model_name}",
            config=self.model_config
        )

        train_dataset = Dataset.from_pandas(input_train_df, preserve_index=False)
        val_dataset = Dataset.from_pandas(input_valid_df, preserve_index=False)

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
        self.tokenizer.save_pretrained(self.model_dir)


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
        max_length:int):

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


def main():
    tokenizer = Tokenizer(CFG.model_name)
    text_preprocessor = TextPreprocessor(tokenizer)
    content_feature_extractor = ContentFeatureExtractor(tokenizer)

    prompts = text_preprocessor(prompts_train, "prompt_text")
    summaries = text_preprocessor(summaries_train, "text")
    train = content_feature_extractor(prompts, summaries)

    gkf = GroupKFold(n_splits=CFG.n_splits)
    for i, (_, val_index) in enumerate(gkf.split(train, groups=train["prompt_id"])):
        train.loc[val_index, "fold"] = i

    for target in ["content"]:
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

main()
