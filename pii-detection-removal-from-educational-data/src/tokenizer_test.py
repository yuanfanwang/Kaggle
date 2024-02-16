from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

sequence = 'A Titan RTX has 24GB of VRAM'
tokenized_sequence = tokenizer(text=sequence, max_length=5, truncation=True)
print(tokenized_sequence)

# if a text is too long, it should be divided by a period.
