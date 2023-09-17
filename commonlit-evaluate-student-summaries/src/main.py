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
    batch_size=12
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

    def run(self,
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
 
    def run(self):
        pass


class ContentFeatureExtractor(FeatureExtractor):
    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer)

    def run(self,
            prompts: pd.DataFrame,
            summaries: pd.DataFrame) -> pd.DataFrame:

        prompts["prompt_legth"] = prompts["prompt_text"].progress_apply(
            lambda x: len(self.tokenizer.encode(x))
        )

        prompts["prompt_tokens"] = prompts["prompt_text"].progress_apply(
            lambda x: self.tokenizer.convert_ids_to_tokens(x)
        )

        summaries["summary_length"] = summaries["text"].progress_apply(
            lambda x: len(self.tokenizer.encode(x))
        )

        summaries["summary_tokens"] = summaries["text"].progress_apply(
            lambda x: self.tokenizer.convert_ids_to_tokens(x)
        )

        input_df = summaries.merge(prompts, how="left", on="prompt_id")

        return input_df.drop(columns=["summary_tokens", "prompt_tokens"])


def main():
    tokenizer = Tokenizer(CFG.model_name)
    text_preprocessor = TextPreprocessor(tokenizer)
    content_feature_extractor = ContentFeatureExtractor(tokenizer)

    prompts = text_preprocessor.run(prompts_train, "prompt_text")
    summaries = text_preprocessor.run(summaries_train, "text")
    train = content_feature_extractor.run(prompts, summaries)

    print(train)

main()