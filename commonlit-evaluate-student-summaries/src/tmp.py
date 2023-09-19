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

document = """
Egyptian society was structured like a pyramid. At the top were the gods, such as Ra, Osiris, and Isis. Egyptians believed that the gods controlled the universe. Therefore, it was important to keep them happy. They could make the Nile overflow, cause famine, or even bring death. 
The Egyptians also elevated some human beings to gods. Their leaders, called pharaohs, were believed to be gods in human form. They had absolute power over their subjects. After pharaohs died, huge stone pyramids were built as their tombs. Pharaohs were buried in chambers within the pyramids. 
Because the people of Egypt believed that their pharaohs were gods, they entrusted their rulers with many responsibilities. Protection was at the top of the list. The pharaoh directed the army in case of a foreign threat or an internal conflict. All laws were enacted at the discretion of the pharaoh. Each farmer paid taxes in the form of grains, which were stored in the pharaoh’s warehouses. This grain was used to feed the people in the event of a famine. 
The Chain of Command 
No single person could manage all these duties without assistance. The pharaoh appointed a chief minister called a vizier as a supervisor. The vizier ensured that taxes were collected. 
Working with the vizier were scribes who kept government records. These high-level employees had mastered a rare skill in ancient Egypt — they could read and write. 
Noble Aims 
Right below the pharaoh in status were powerful nobles and priests. Only nobles could hold government posts; in these positions they profited from tributes paid to the pharaoh. Priests were responsible for pleasing the gods. 
Nobles enjoyed great status and also grew wealthy from donations to the gods. All Egyptians—from pharaohs to farmers—gave gifts to the gods. 
Soldier On 
Soldiers fought in wars or quelled domestic uprisings. During long periods of peace, soldiers also supervised the peasants, farmers, and slaves who were involved in building such structures as pyramids and palaces. 
Skilled workers such as physicians and craftsmen/women made up the middle class. Craftsmen made and sold jewelry, pottery, papyrus products, tools, and other useful things. 
Naturally, there were people needed to buy goods from artisans and traders. These were the merchants and storekeepers who sold these goods to the public. 
The Bottom of the Heap 
At the bottom of the social structure were slaves and farmers. Slavery became the fate of those captured as prisoners of war. In addition to being forced to work on building projects, slaves toiled at the discretion of the pharaoh or nobles. 
Farmers tended the fields, raised animals, kept canals and reservoirs in good order, worked in the stone quarries, and built the royal monuments. Farmers paid taxes that could amount to as much as 60% of their yearly harvest—that’s a lot of hay! 
Social mobility was not impossible. A small number of peasants and farmers moved up the economic ladder. Families saved money to send their sons to village schools to learn trades. These schools were run by priests or by artisans. Boys who learned to read and write could become scribes, then go on to gain employment in the government. It was possible for a boy born on a farm to work his way up into the higher ranks of the government. Bureaucracy proved lucrative.
"""

# Object of automatic summarization.
auto_abstractor = AutoAbstractor()
# Set tokenizer.
auto_abstractor.tokenizable_doc = SimpleTokenizer()
# Set delimiter for making a list of sentence.
auto_abstractor.delimiter_list = [".", "\n"]
# Object of abstracting and filtering document.
abstractable_doc = TopNRankAbstractor()
# Summarize document.
result_dict = auto_abstractor.summarize(document, abstractable_doc)

# Output result.
document2 = ""
for sentence in result_dict["summarize_result"]:
    document2 += sentence.strip() + " "

print(document2.strip())
