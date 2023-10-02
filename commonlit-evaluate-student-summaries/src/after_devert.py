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
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag 
from collections import Counter
# import spacy
import re
from autocorrect import Speller
from spellchecker import SpellChecker
from autocorrect import Speller
import lightgbm as lgb

from rake_nltk import Rake

stop_words_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                   'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
                   'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                   "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                   'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                   'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                   'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                   'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                   'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                   'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                   'through', 'during', 'before', 'after', 'above', 'below', 'to',
                   'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                   'again', 'further', 'then', 'once', 'here', 'there', 'when','where',
                   'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
                   'other', 'some', 'such', 'no', 'nor', 'not','only', 'own', 'same',
                   'so', 'than', 'too', 'very', 's', 't','can', 'will', 'just', 'don',
                   "don't", 'should', "should've",'now', 'd', 'll', 'm', 'o', 're',
                   've', 'y', 'ain', 'aren',"aren't", 'couldn', "couldn't", 'didn',
                   "didn't", 'doesn',"doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                   'haven', "haven't",'isn', "isn't", 'ma', 'mightn', "mightn't",
                   'mustn', "mustn't",'needn', "needn't", 'shan', "shan't", 'shouldn',
                   "shouldn't",'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
                   'wouldn', "wouldn't"]

torch.cuda.empty_cache()
warnings.simplefilter("ignore")
logging.disable(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
disable_progress_bar()
tqdm.pandas()

class CFG:
    model_name="debertav3base"
    learning_rate= {
        "content": 7.0e-7,
        "wording": 7.0e-7
    }
    # learning_rate= {
    #     "content": 7.0e-7,
    #     "wording": 5.0e-7
    # }
    weight_decay=0.02
    hidden_dropout_prob=0.1  # default: 0.005
    attention_probs_dropout_prob=0.1  # default: 0.005
    num_train_epochs=5
    n_splits=4
    batch_size=15  # TODO: default: 12
    random_seed=42
    save_steps=100
    max_length=400


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

    def encode(self,
               text: str):
        return self.tokenizer.encode(text)

    def convert_ids_to_tokens(self,
                              text: str):
        return self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(text),
            skip_special_tokens=True
        )

    def trimming(self, text: str) -> str:
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        prompt_max_length = min(len(text_ids), 120)
        trimed_text = self.tokenizer.decode(text_ids[:prompt_max_length])
        return trimed_text
    
class TextPreprocessor:
    def __init__(self,
                 tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        # self.spacy_ner_model = spacy.load('en_core_web_sm')
    
    def preprocess(self, text: str):
        text = text.lower()
        text = text.replace('.', ' . ')
        text = text.replace(',', ' , ')
        not_allowed_symbol = r'[^a-zA-Z0-9\s\'\-\.\,]'
        text = re.sub(not_allowed_symbol, ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = text.replace(' . ', '. ')
        text = text.replace(' , ', '. ')
        text = text.rstrip()
        if text[-1] != '.':
            text += '.'
        return text
    
    # TODO: prompt_question prompt_title
    def __call__(self,
                 df: pd.DataFrame,
                 col: str) -> pd.DataFrame:
        df[col] = df[col].progress_apply(
            lambda x: self.preprocess(x)
        )
        return df


class FeatureExtractor:
    def __init__(self,
                 tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def __call__(self):
        pass


class Summarizer:
    def __init__(self):
        self.rake = Rake()

    def __call__(self, text: str) -> str:
        sentences = text.split(".")
        self.rake.extract_keywords_from_text(text)
        self.rake.extract_keywords_from_sentences(sentences)
        ranked_phrases = self.rake.get_ranked_phrases()
        unique_phrase_set = set()
        ranked_phrases_text = ""
        for phrase in ranked_phrases:
            phrase.lower()
            if phrase not in unique_phrase_set:
                ranked_phrases_text += phrase + ", "
                unique_phrase_set.add(phrase)
        ranked_phrases_text = ranked_phrases_text.rstrip()
        return ranked_phrases_text


class ContentFeatureExtractor(FeatureExtractor):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer)
        self.summarizer = Summarizer() 
        self.speller = Speller()
        self.spellchecker = SpellChecker()
        self.stop_words = set(stop_words_list)
        self.duplicate_loss_weight = 2

    def sentence_ratio(self, text: str):
        text_length = len(text.split())
        sentence_count = text.count('.')
        if text_length == 0:
            return 0
        else:
            return sentence_count / text_length

    def spell_miss_ratio(self, text: str):
        text_length = len(text.split())
        filtered_text = text.replace('.', '')
        filtered_text = filtered_text.replace(',', '')
        wordlist = filtered_text.split()
        miss_count = len(list(self.spellchecker.unknown(wordlist)))
        if text_length == 0:
            return 0
        else:
            return miss_count / text_length
    
    def add_spelling_dictionary(self, text: str):
        """dictionary update for pyspell checker and autocorrect"""
        filtered_text = text.replace('.', '')
        filtered_text = filtered_text.replace(',', '')
        wordlist = filtered_text.split()
        self.spellchecker.word_frequency.load_words(wordlist)
        self.speller.nlp_data.update({token:1000 for token in wordlist})

    def corrected_text(self, text: str):
        modified_text = text.replace('.', '')
        modified_text = modified_text.replace(',', '')
        modified_text = self.speller(modified_text)
        return modified_text

    # TODAY: culculation time
    def word_overlap_ratio(self, row):
        def check_is_stop_word(word):
            return word in self.stop_words
        prompt_text = row["prompt_text"]
        prompt_text = prompt_text.replace('.', '')
        prompt_text = prompt_text.replace(',', '')
        prompt_words = prompt_text.split()

        summaries_text = row["corrected_text"]
        summaries_text = summaries_text.replace('.', '')
        summaries_text = summaries_text.replace(',', '')
        summary_words = summaries_text.split()

        text_length = len(summary_words)
        if self.stop_words:
            prompt_words = list(filter(check_is_stop_word, prompt_words))
            summary_words = list(filter(check_is_stop_word, summary_words))
        overlap_word_length = len(set(prompt_words).intersection(set(summary_words)))
        if text_length == 0:
            return 0
        else:
            return overlap_word_length / text_length

    # TODAY: calculation time
    def ngrams(self, token, n):
        ngrams = zip(*[token[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]
    
    def ngram_co_occurrence(self, row, n: int) -> int:
        prompt_text = row["prompt_text"]
        prompt_text = prompt_text.replace('.', '')
        prompt_text = prompt_text.replace(',', '')
        prompt_words = prompt_text.split()

        summaries_text = row["corrected_text"]
        summaries_text = summaries_text.replace('.', '')
        summaries_text = summaries_text.replace(',', '')
        summary_words = summaries_text.split()

        prompt_ngrams = set(self.ngrams(prompt_words, n))
        summary_ngrams = set(self.ngrams(summary_words, n))
        common_ngrams = prompt_ngrams.intersection(summary_ngrams)
        if len(summary_words) - n + 1 < 0:
            return 0
        else:
            return len(common_ngrams) / (len(summary_words) - n + 1)

    def content_feature(self,
                        prompt_df: pd.DataFrame,
                        input_df: pd.DataFrame):

        def token_dict(text: str) -> dict:
            words = text.replace('.', '')
            words = words.replace(',', '')
            tokens = word_tokenize(words)        
            tokens = pos_tag(tokens)
            filtered_tokens = dict()
            for word, tag in tokens:
                if word in self.stop_words:
                    continue
                token = (word, tag)
                if not token in filtered_tokens:
                    filtered_tokens[token] = 1
                else:
                    filtered_tokens[token] *= self.duplicate_loss_weight
            return filtered_tokens

        prompt_dict_list = dict()
        for _, row in prompt_df.iterrows():
            prompt_dict_list[row["prompt_id"]] = token_dict(row["prompt_text"])

        for i, row in input_df.iterrows():
            summaries_dict = token_dict(row["corrected_text"])
            prompt_id = row["prompt_id"]
            duplicate_loss = 0
            jj_count = 0
            nn_count = 0
            rb_count = 0

            for key, val in summaries_dict.items():
                word, tag = key
                if val == 1:
                    continue
                if ((not "JJ" in tag) and
                    (not "RB" in tag)):
                    continue

                duplicate_loss += val

            prompt_dict = prompt_dict_list[prompt_id]
            unique_words_in_summaries = list(set(summaries_dict.keys()) - set(prompt_dict.keys()))
            for word, tag in unique_words_in_summaries:
                if "JJ" in tag:
                    jj_count += 1
                if "NN" in tag:
                    nn_count += 1
                if "RB" in tag:
                    rb_count += 1
            # [wording] 2
            input_df.at[i, "jj_count"] = jj_count
            # [wording] 3
            input_df.at[i, "nn_count"] = nn_count
            # [wording] 4
            input_df.at[i, "rb_count"] = rb_count
            # [wording] 5
            input_df.at[i, "duplicate_loss"] = duplicate_loss
    
    def __call__(self,
                 prompts: pd.DataFrame,
                 summaries: pd.DataFrame) -> pd.DataFrame:
        
        prompts["prompt_text"].apply(
            lambda x: self.add_spelling_dictionary(x)
        )

        prompts["prompt_length"] = prompts["prompt_text"].progress_apply(
            lambda x: len(x.split())
        )

        prompts["prioritized_prompt_words"] = prompts["prompt_text"].progress_apply(
            lambda x: self.summarizer(x)
        )

        # [devert]
        prompts["trimed_and_prioritized_prompt_words"] = prompts["prioritized_prompt_words"].progress_apply(
            lambda x: self.tokenizer.trimming(x)
        )

        # [devert]
        summaries["corrected_text"] = summaries["text"].progress_apply(
            lambda x: self.corrected_text(x)
        )
        print("summaries corrected text head: ", summaries["corrected_text"].head())

        summaries["summary_length"] = summaries["text"].progress_apply(
            lambda x: len(x.split())
        )

        # [content] 1
        # [wording] 1
        summaries["spell_miss_ratio"] = summaries["text"].progress_apply(
            lambda x: self.spell_miss_ratio(x)
        )

        # [content] 2
        summaries["sentence_ratio"] = summaries["text"].progress_apply(
            lambda x: self.sentence_ratio(x)
        )

        input_df = summaries.merge(prompts, how="left", on="prompt_id")

        # [content] 3
        input_df['length_ratio'] = input_df['summary_length'] / input_df['prompt_length']

        # [content] 4 
        input_df['word_overlap_ratio'] = input_df.progress_apply(self.word_overlap_ratio, axis=1)

        # [content] 5 
        input_df['bigram_overlap_ratio'] = input_df.progress_apply(self.ngram_co_occurrence,args=(2,), axis=1)

        # [content] 6 
        input_df['trigram_overlap_ratio'] = input_df.progress_apply(self.ngram_co_occurrence,args=(3,), axis=1)

        # [wording] 2 ~ 5
        self.content_feature(prompts, input_df)  # sould be executed after creating "corrected_text"

        return input_df


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}

def compute_mcrmse(eval_pred):
    """
    Calculates mean columnwise root mean squared error
    https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview/evaluation
    """
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
    wording_score = mean_squared_error(wording_true, wording_pred)**(1/2)
    
    return (content_score + wording_score)/2


class ContentScoreRegressor:
    def __init__(self,
                 model_name: str,
                 model_dir: str,
                 target: str,
                 hidden_dropout_prob: float,
                 attention_probs_dropout_prob: float,
                 max_length: int):

        self.input_cols = ["trimed_and_prioritized_prompt_words", "corrected_text"]

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
            "problem_type": "regression"
        })

        seed_everything(CFG.random_seed)

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer
        )
    
    def tokenize_function(self, examples: pd.DataFrame):
        labels = [examples[self.target]]
        trimed_and_prioritized_text = examples["trimed_and_prioritized_prompt_words"]
        text = examples["corrected_text"]

        # TODO: add prompt_question for third feature of the token. [0,0,0,1,1,1,1,1,2,2,2,2,2] 
        tokenized = self.tokenizer(trimed_and_prioritized_text, text,
                                   padding="max_length",
                                   truncation=True,
                                   max_length=self.max_length)
        token = {
            **tokenized,
            "labels": labels
        }
        return token
    
    def tokenize_function_test(self, examples: pd.DataFrame):
        trimed_and_prioritized_text = examples["trimed_and_prioritized_prompt_words"]
        text = examples["corrected_text"]

        # TODO: add prompt_question for third feature of the token. [0,0,0,1,1,1,1,1,2,2,2,2,2] 
        tokenized = self.tokenizer(trimed_and_prioritized_text, text,
                                   padding="max_length",
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

    def predict(self,
                test_df: pd.DataFrame,
                fold: int):
        """predict content score"""

        input_test_df = test_df[self.input_cols]

        test_dataset = Dataset.from_pandas(input_test_df, preserve_index=False)
        test_tokenized_datasets = test_dataset.map(self.tokenize_function_test, batched=False)

        model_content = AutoModelForSequenceClassification.from_pretrained(f"{self.model_dir}")
        model_content.eval()

        model_fold_dir = os.path.join(self.model_dir, str(fold))

        test_args = TrainingArguments(
            output_dir=model_fold_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=4,
            dataloader_drop_last=False
        )

        infer_content = Trainer(
            model=model_content,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            args=test_args
        )

        preds = infer_content.predict(test_tokenized_datasets)[0]
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
            model_dir = f"{target}/{model_name}/fold_{fold}"
        else: 
            model_dir = f"{model_name}/fold_{fold}"

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
        target: str,
        save_each_model: bool,
        model_name: str,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        max_length: int) -> pd.DataFrame:
    """predict of data"""
    for fold in range(CFG.n_splits):
        print(f"fold {fold}:")

        valid_data = train_df[train_df["fold"] == fold]

        if save_each_model == True:
            model_dir = f"{target}/{model_name}/fold_{fold}"
        else:
            model_dir = f"{model_name}/fold_{fold}"

        csr = ContentScoreRegressor(
            model_name=model_name,
            model_dir = model_dir,
            target=target,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length
        )

        pred = csr.predict(
            test_df=valid_data,
            fold=fold
        )

        train_df.loc[valid_data.index, f"{target}_pred"] = pred
    
    return train_df

def predict(
    test_df: pd.DataFrame,
    target: str,
    save_each_model: bool,
    model_name: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    max_length: int):
    """predict of data"""

    for fold in range(CFG.n_splits):
        print(f"fold {fold}:")

        if save_each_model == True:
            model_dir = f"{target}/{model_name}/fold_{fold}"
        else:
            model_dir = f"{model_name}/fold_{fold}"
        
        csr = ContentScoreRegressor(
            model_name=model_name,
            target=target,
            model_dir=model_dir,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length
            )
        
        pred = csr.predict(
            test_df=test_df,
            fold=fold
        )
        test_df[f"{target}_pred_{fold}"] = pred
        print(f"test_df {fold}: ", test_df)


    test_df[f"{target}"] = test_df[[f"{target}_pred_{fold}" for fold in range(CFG.n_splits)]].mean(axis=1)
    print("test_df final: ", test_df)

    return test_df

def main():

    targets = ["content", "wording"]
    train = pd.read_csv("/kaggle/input/train.csv")
    test  = pd.read_csv("/kaggle/input/test.csv")

    ## lgbm preprocess
    common_drop_columns = ["fold",
                           "student_id",
                           "prompt_id",
                           "text",
                           "prompt_question",
                           "prompt_title", 
                           "prompt_text",  # original
                           "prompt_length",
                           "prioritized_prompt_words",
                           "trimed_and_prioritized_prompt_words",
                           "corrected_text",
                           "summary_length"] + targets

    # TODO: fold should be added?
    additional_common_drop_columns = [f"content_pred_{i}" for i in range(CFG.n_splits)] + \
                                     [f"wording_pred_{i}" for i in range(CFG.n_splits)]

    content_drop_columns = ["jj_count",
                            "nn_count",
                            "rb_count",
                            "duplicate_loss"]

    wording_drop_columns = ["sentence_ratio",
                            "length_ratio",
                            "word_overlap_ratio",
                            "bigram_overlap_ratio",
                            "trigram_overlap_ratio"]

    # TODO: Redundant amounts of features would be allowed.
    lgbm_feature_drop_dict = {
        "content": common_drop_columns,
        "wording": common_drop_columns,
    }

    # lgbm_feature_drop_dict = {
    #     "content": common_drop_columns + content_drop_columns,
    #     "wording": common_drop_columns + wording_drop_columns,
    # }

    model_dict = {}
    for target in targets:
        models = []

        for fold in range(CFG.n_splits):
        
            X_train_cv = train[train["fold"] != fold].drop(columns=lgbm_feature_drop_dict[target])
            print("x_train_cv head: ", X_train_cv.head())
            y_train_cv = train[train["fold"] != fold][target]
            print("y_train_cv head: ", y_train_cv.head())

            X_eval_cv = train[train["fold"] == fold].drop(columns=lgbm_feature_drop_dict[target])
            y_eval_cv = train[train["fold"] == fold][target]

            dtrain = lgb.Dataset(X_train_cv, label=y_train_cv)
            dval = lgb.Dataset(X_eval_cv, label=y_eval_cv)

            params = {'boosting_type': 'gbdt',
                      'random_state': 42,
                      'objective': 'regression',
                      'metric': 'rmse',
                      'learning_rate': 0.05}

            evaluation_results = {}
            model = lgb.train(params,
                              num_boost_round=10000,
                                #categorical_feature = categorical_features,
                              valid_names=['train', 'valid'],
                              train_set=dtrain,
                              valid_sets=dval,
                              callbacks=[
                                  lgb.early_stopping(stopping_rounds=30, verbose=True),
                                  lgb.log_evaluation(100),
                                  lgb.callback.record_evaluation(evaluation_results)],
                              )
            models.append(model)

        model_dict[target] = models

    
    ## cv after lgbm
    rmses = []
    for target in targets:
        models = model_dict[target]

        preds = []
        trues = []

        for fold, model in enumerate(models):
            X_eval_cv = train[train["fold"] == fold].drop(columns=lgbm_feature_drop_dict[target])
            y_eval_cv = train[train["fold"] == fold][target]

            pred = model.predict(X_eval_cv)

            trues.extend(y_eval_cv)
            preds.extend(pred)

        rmse = np.sqrt(mean_squared_error(trues, preds))
        print(f"{target}_rmse : {rmse}")
        rmses = rmses + [rmse]

    print(f"mcrmse : {sum(rmses) / len(rmses)}")

    ## predict  
    pred_dict = {}
    for target in targets:
        model = model_dict[target]
        preds = []

        for fold, model in enumerate(models):
            drop_columns = lgbm_feature_drop_dict[target] + additional_common_drop_columns
            X_eval_cv = test.drop(columns=drop_columns) 
            print("predict: ", X_eval_cv)

            pred = model.predict(X_eval_cv)
            preds.append(pred)
        
        pred_dict[target] = preds
    

    for target in targets:
        preds = pred_dict[target]
        for i, pred in enumerate(preds):
            test[f"{target}_pred_{i}"] = pred

        test[target] = test[[f"{target}_pred_{fold}" for fold in range(CFG.n_splits)]].mean(axis=1)

    print(test)
    print(sample_submission)
    test[["student_id", "content", "wording"]].to_csv("submission.csv", index=False)

main()
