import numpy as np
import pandas as pd
import torch

import os

prompts_test    = pd.read_csv('data/prompts_test.csv')
prompts_train   = pd.read_csv('data/prompts_train.csv')
summaries_train = pd.read_csv('data/summaries_train.csv')
summaries_test  = pd.read_csv('data/summaries_test.csv')

"""
Content Model 
    Main idea
        title               -> how the summary is related to the title
        question            -> how the summary is reletad to the question
         
    Details                 -> word_to_vec
    Cohesion                -> count the conjection
                            -> Scores of differences between sentences and sentences


# Wording Model
    Voice
        objective word      -> calculate the ration of the number of the objective word in summary to the number of the total words in summary.
                            -> calculate the ration of the number of the unique objective word in summary to the number of the total unique words.
    Paraphrase
        paraphrase          -> word_to_vec
                            -> calculate the ratio of the number of the word in summary to the number of the word in the prompt
    Lanuage
        lexis               -> count the number of unique words
        syntax              -> count the number of unique grammar


# Feature not known which cagegory of each Model the belong to
    Length
        compression         -> compression ratio
    

# Also give a opposite score to each model. e.g. calculate the ration of the number unique objective word in summary to the number of the total unique objective word in prompt. The ratio is then produced as logloss.
"""


