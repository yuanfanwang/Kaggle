import scipy
import pandas as pd

DATA_DIR = "data/"

prompts_train = pd.read_csv(DATA_DIR + "prompts_train.csv")
prompts_test = pd.read_csv(DATA_DIR + "prompts_test.csv")
summaries_train = pd.read_csv(DATA_DIR + "summaries_train.csv")
summaries_test = pd.read_csv(DATA_DIR + "summaries_test.csv")
sample_submission = pd.read_csv(DATA_DIR + "sample_submission.csv")

summaries_train["text_length"] = summaries_train["text"].apply(
    lambda x: len(x.split(' '))
)

summaries_train["text_length_norm"] = scipy.stats.zscore(summaries_train["text_length"])

print(summaries_train)