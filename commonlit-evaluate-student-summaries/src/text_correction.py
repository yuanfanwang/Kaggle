import re
import sys
import torch

import numpy as np
import pandas as pd

# from sklearn.model_selection import mean_squared_error


from spellchecker import SpellChecker

prompts_test    = pd.read_csv('data/prompts_test.csv')
prompts_train   = pd.read_csv('data/prompts_train.csv')
summaries_train = pd.read_csv('data/summaries_train.csv')
summaries_test  = pd.read_csv('data/summaries_test.csv')


# shuffle
# fold

spell = SpellChecker()

cnt = 0
allowed_symbol = r'[^a-zA-Z0-9\s\"\'\-\;\.\,]'
for Text in summaries_train['text']:
    word_count_before_shaping = len(Text.split(' '))

    print(Text)

    text = Text.lower()
    text = text.rstrip()
    text = text.replace('"', ' " ')
    text = text.replace(';', ' ; ')
    text = text.replace('.', ' . ')
    text = text.replace(',', ' , ')
    text = re.sub(allowed_symbol, ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.rstrip()
    if not text.endswith(' .'):
        text += ' .'

    words = text.split(' ')
    corrected_text = []
    for word in words:
        corrected_word = spell.correction(word)
        if corrected_word == None:
            corrected_word = word
            print("\nwrong\n")
            print(word)
        corrected_text.append(corrected_word)

    corrected_sentence = ' '.join(corrected_text)

    cnt += 1
    if cnt > 5:
        break

    print(text)
    print('\n')

    # added feature to the table
    # classifying this fucntion
