import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize 
from nltk import pos_tag 

import pandas as pd

from autocorrect import Speller
from spellchecker import SpellChecker

speller = Speller(lang='en')
spellchecker = SpellChecker()

prompts_train = pd.read_csv("data/prompts_train.csv")
prompts_test = pd.read_csv("data/prompts_test.csv")
summaries_train = pd.read_csv("data/summaries_train.csv")
summaries_test = pd.read_csv("data/summaries_test.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")

summaries_train.at[0, "text"] = "He's a very nice guy."

for i, text in enumerate(summaries_train["text"]):
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
    # print(text)
    # print('\n')
    summaries_train.at[i, "text"] = text
    if i == 5:
        break


speller = Speller()
for i, text in enumerate(summaries_train["text"]):
    text = text.replace('.', '')
    text = text.replace(',', '')
    print(text)
    wordlist=text.split()
    amount_miss = len(list(spellchecker.unknown(wordlist)))
    print(list(spellchecker.unknown(wordlist)))
    print(amount_miss)
    print(speller(text))
    print('\n')
    if i == 5:
        break


# for i, text in enumerate(summaries_train["text"])
#     text = text.replace('.', '')
#     text = text.replace(',', '')
#     wordlist=text.split()
    