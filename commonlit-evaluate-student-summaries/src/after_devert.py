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

from collections import Counter
# import spacy
import re
import lightgbm as lgb

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


def main():

    targets = ["content", "wording"]
    train = pd.read_csv("/kaggle/input/for-lgbm/train.csv")
    test  = pd.read_csv("/kaggle/input/for-lgbm/test.csv")

    train["word_overlap_count"] = train["word_overlap_ratio"] * train["summary_length"]
    train["bigram_overlap_count"] = train["bigram_overlap_ratio"] * train["summary_length"]
    train["trigram_overlap_count"] = train["trigram_overlap_ratio"] * train["summary_length"]
    train["jjnnrb_count"] = train["jj_count"] + train["nn_count"] + train["rb_count"]
    train["jjnnrb_ratio"] = train["jjnnrb_count"] / train["summary_length"]
    train["duplicate_loss"] = train["duplicate_loss"] * -1
    train["spell_miss_count"] = train["spell_miss_ratio"] * train["summary_length"]

    test["word_overlap_count"] = test["word_overlap_ratio"] * test["summary_length"]
    test["bigram_overlap_count"] = test["bigram_overlap_ratio"] * test["summary_length"]
    test["trigram_overlap_count"] = test["trigram_overlap_ratio"] * test["summary_length"]
    test["jjnnrb_count"] = test["jj_count"] + test["nn_count"] + test["rb_count"]
    test["jjnnrb_ratio"] = test["jjnnrb_count"] / test["summary_length"]
    test["duplicate_loss"] = test["duplicate_loss"] * -1
    test["spell_miss_count"] = test["spell_miss_ratio"] * test["summary_length"]

    ## lgbm preprocess
    train_drop_columns = ["fold",
                          "student_id",
                          "prompt_id",
                          "text",
                          "prompt_question",
                          "prompt_title", 
                          "prompt_text",  # original
                          "prioritized_prompt_words",
                          "trimed_and_prioritized_prompt_words",
                          "corrected_text"] + targets

    test_drop_columns = ["student_id",
                         "prompt_id",
                         "text",
                         "prompt_question",
                         "prompt_title", 
                         "prompt_text",  # original
                         "prioritized_prompt_words",
                         "trimed_and_prioritized_prompt_words",
                         "corrected_text"] + \
                        [f"content_pred_{i}" for i in range(CFG.n_splits)] + \
                        [f"wording_pred_{i}" for i in range(CFG.n_splits)]

    content_drop_columns = ["jj_count",
                            "nn_count",
                            "rb_count",
                            "duplicate_loss"]

    wording_drop_columns = ["sentence_ratio",
                            "summary_length",
                            "length_ratio",
                            "sentence_ratio",
                            "length_ratio",
                            "word_overlap_ratio",
                            "word_overlap_count",
                            "bigram_overlap_ratio",
                            "bigram_overlap_count",
                            "trigram_overlap_ratio",
                            "trigram_overlap_count"]

    # TODO: Redundant amounts of features would be allowed.
    #lgbm_feature_drop_dict = {
    #    "content": train_drop_columns,
    #    "wording": train_drop_columns,
    #}


    model_dict = {}
    for target in targets:
        models = []

        for fold in range(CFG.n_splits):
            drop_col = train_drop_columns
            X_train_cv = train[train["fold"] != fold].drop(columns=drop_col)
            # print("x_train_cv head: ", X_train_cv.head())
            y_train_cv = train[train["fold"] != fold][target]
            # print("y_train_cv head: ", y_train_cv.head())

            X_eval_cv = train[train["fold"] == fold].drop(columns=drop_col)
            y_eval_cv = train[train["fold"] == fold][target]

            dtrain = lgb.Dataset(X_train_cv, label=y_train_cv)
            dval = lgb.Dataset(X_eval_cv, label=y_eval_cv)

            params = {'boosting_type': 'gbdt',
                      'random_state': 42,
                      'objective': 'regression',
                      'metric': 'rmse',
                      'learning_rate':8.0e-4}

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
            drop_col = train_drop_columns
            X_eval_cv = train[train["fold"] == fold].drop(columns=drop_col)
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
        models = model_dict[target]
        preds = []
        
        for fold, model in enumerate(models):
            drop_col = test_drop_columns
            X_eval_cv = test.drop(columns=drop_col) 
            # print("predict: ", X_eval_cv)

            pred = model.predict(X_eval_cv)
            preds.append(pred)
        
        pred_dict[target] = preds
    

    for target in targets:
        preds = pred_dict[target]
        for i, pred in enumerate(preds):
            test[f"{target}_pred_{i}"] = pred

        test[target] = test[[f"{target}_pred_{fold}" for fold in range(CFG.n_splits)]].mean(axis=1)

    # print(test)
    test[["student_id", "content", "wording"]].to_csv("submission.csv", index=False)

main()
