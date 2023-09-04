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

seed_everything(42)

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

