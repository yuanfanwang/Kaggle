import re
import torch

import numpy as np
import pandas as pd

from spellchecker import SpellChecker

prompts_test    = pd.read_csv('data/prompts_test.csv')
prompts_train   = pd.read_csv('data/prompts_train.csv')
summaries_train = pd.read_csv('data/summaries_train.csv')
summaries_test  = pd.read_csv('data/summaries_test.csv')

allowed_symbol = r'[^a-zA-Z0-9\s\"\'\-\;\.\,]'

for Text in summaries_train['text']:
    text = Text.lower()
    text = text.rstrip()

    text = text.replace('"', ' " ')
    text = text.replace(';', ' ; ')
    text = text.replace('.', ' . ')
    text = text.replace(',', ' , ')
    text = re.sub(allowed_symbol, ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)

    if not text.endswith(' . '):
        text += '. '
    
    # split text
    splited_text = text.split(' ')
    print(splited_text)
    break