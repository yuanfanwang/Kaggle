import numpy as np
import pandas as pd
import warnings
import logging
import os
import shutil
import json
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollator
from datasets import load_metric, disable_progress_bar
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GroupkFold
from tqdm import tqdm
from collections import Counter
import spacy
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
from spellchecker import SpellChecker



warnings.simplefilter("ignore")
logging.disable(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
disable_progress_bar()
tqdm.pandas()

##################################################################################

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



##################################################################################

def seed_everything(seed: int):
    import random, os
    import numpy as np
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

seed_everything(CFG.random_seed)

##################################################################################

DATA_DIR = "kaggle/input/commonlit-evaluate-student-summaries"

prompts_train = pd.read_csv(DATA_DIR + "prompts_train.csv")
prompts_test = pd.read_csv(DATA_DIR + "prompts_test.csv")
summaries_train = pd.read_csv(DATA_DIR + "summaries_train.csv")
summaries_test = pd.read_csv(DATA_DIR + "summaries_test.csv")
sample_submission = pd.read_csv(DATA_DIR + "sample_submission.csv")

##################################################################################

class Tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(f"kaggle/input/{CFG.model_name}")
    STOP_WARDS = set(stopwords.words('english'))
    spacy_ner_model = spacy.load('en_core_web_sm')
    speller = SpellChecker()

class TextPreprocessor:
    def __init__(self,
                 model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(f"kaggle/input/")
        self.STOP_WARDS = set(stopwords.words('english'))
        self.spacy_ner_model = spacy.load('en_core_web_sm')
        self.speller = SpellChecker()

    def remove_unusual_symbol():
        pass

    def add_space_between_usual_symbol():
        pass

    def remove_stop_ward():
        pass

    def delete_blank_space():
        pass

    def correct_spell():
        pass

    def run():
        pass


class FeatureExtractor:
    def __init__(self):
        pass
    
    def run(self):
        pass


class ContentFeatureExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self,
                prompts: pd.DataFrame,
                summaries: pd.DataFrame,
                ) -> pd.DataFrame:
        
        prompts["prompt_legth"] = prompts["prompt_text"].apply(
            lambda x: len(self.tokenizer.encode(x))
        )

        



class WordingFeatureExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self, text: str):
        pass


