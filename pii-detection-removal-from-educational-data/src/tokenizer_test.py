from transformers import AutoTokenizer, BertTokenizer

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

tokens = ['Visualization', 'Wang.', 'Yuanfan,', 'it', 'is', 'good', '.', 'Mr.', 'I', 'was', 'young', ',' '\n\n', 'Tool', 'insights', '.', '\n\n', 'great', ';', 'please' , '.']
 
tokenized_data = tokenizer(
    tokens, truncation=True, is_split_into_words=True, max_length=100
)

input_ids = tokenized_data['input_ids']
print(input_ids)
for id in input_ids: print(id, tokenizer.decode(id))
print(tokenizer.decode(input_ids))

# if a text is too long, it should be divided by a period.
