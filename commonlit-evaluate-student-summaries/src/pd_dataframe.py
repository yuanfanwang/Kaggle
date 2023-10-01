import pandas as pd

prompts_test    = pd.read_csv('data/prompts_test.csv')
prompts_train   = pd.read_csv('data/prompts_train.csv')
summaries_train = pd.read_csv('data/summaries_train.csv')
summaries_test  = pd.read_csv('data/summaries_test.csv')

def ref_pd(df: pd.DataFrame):
    df.at[0, "text"] = "wang wang"

print(summaries_train[summaries_train["prompt_id"] == "39c16e"][["student_id", "content", "wording"]])