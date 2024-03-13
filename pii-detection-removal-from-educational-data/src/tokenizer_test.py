from transformers import AutoTokenizer, BertTokenizer

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

tokens = ['Visualization', '\t\r \xa0', 'Tool', 'insights', '.', '\n\n']
 
tokenized_data = tokenizer(
    tokens, truncation=True, is_split_into_words=True, max_length=100
)

input_ids = tokenized_data['input_ids']
print(tokenizer.decode(input_ids))

# if a text is too long, it should be divided by a period.
