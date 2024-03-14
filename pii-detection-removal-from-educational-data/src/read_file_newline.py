import polars as pl
import sys

from datasets import Dataset, DatasetDict

train_df = pl.read_json("data/train.json").to_pandas()
test_df = pl.read_json("data/test.json").to_pandas()
print("-------------------------")
print(len(train_df['tokens'][6317]))
print("-------------------------")
print(len(train_df['tokens'][6318]))
# 6318
# Assuming `train_df` is your DataFrame
text = train_df['full_text'][6318]
token = train_df['tokens'][6318]

# Open the file in write mode ('w')
with open('output.txt', 'w') as f:
    # Write the text to the file
    f.write(text)
with open('output_token.txt', 'w') as f:
    # Write the text to the file
    f.write(str(token))

sys.exit()
max_tokens_list = []
for i, tokens in enumerate(train_df['tokens']):
    token_count_list = []
    token_count = 0
    for token in tokens:
        if token == '\n\n':
            token_count_list.append(token_count)
            token_count = 0
        else:
            token_count += 1      
    token_count_list.append(token_count)
    max_tokens_list.append(max(token_count_list))
    if max(token_count_list) == 1397:
        #print(tokens)
        print("full_text")
        text = train_df['full_text'][i]
        print(text) 
        # print(len(tokens))
        # print(token_count_list)
        print(i)
    

print(max(max_tokens_list))