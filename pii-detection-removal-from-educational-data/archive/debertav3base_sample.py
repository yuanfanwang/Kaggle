from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# モデルとトークナイザーの準備
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# クラス一覧
label_list = [
   "O",      
   "B-MISC",  
   "I-MISC", 
   "B-PER", 
   "I-PER",  
   "B-ORG", 
   "I-ORG", 
   "B-LOC", 
   "I-LOC" 
]

# テキスト
sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
          "close to the Manhattan Bridge."

# 前処理 (スペシャルトークンでトークンを取得)
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
inputs = tokenizer.encode(sequence, return_tensors="pt")

# 推論
outputs = model(inputs)[0]
predictions = torch.argmax(outputs, dim=2)

# 出力
print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())])


"""  GPT's suggestion
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset

# Define your labeled dataset
# For simplicity, let's assume we have tokenized sentences and corresponding labels
# in BIO format (Begin, Inside, Outside)
sentences = ["I live in New York .", "John works at Google ."]
labels = [["O", "O", "O", "B-Location", "I-Location", "O"], ["B-Person", "O", "O", "B-Organization", "O"]]

# Load the pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the input sentences and convert labels to numerical form
tokenized_texts = []
labels_ids = []

for sentence, label in zip(sentences, labels):
    # Tokenize the sentence
    tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=False)
    tokenized_texts.append(tokenized_sentence)
    
    # Convert labels to numerical form
    label_ids = [tokenizer.convert_tokens_to_ids(l) for l in label]
    labels_ids.append(label_ids)

# Pad sequences to the same length
max_len = max(len(tokenized_text) for tokenized_text in tokenized_texts)
padded_tokenized_texts = [tokenized_text + [tokenizer.pad_token_id] * (max_len - len(tokenized_text)) for tokenized_text in tokenized_texts]
padded_labels_ids = [label_id + [-100] * (max_len - len(label_id)) for label_id in labels_ids]  # Use -100 for padding in the loss function

# Convert lists to PyTorch tensors
input_ids = torch.tensor(padded_tokenized_texts)
labels = torch.tensor(padded_labels_ids)

# Create a TensorDataset
dataset = TensorDataset(input_ids, labels)

# Example usage of the dataset
for i in range(len(dataset)):
    input_ids, labels = dataset[i]
    print("Input IDs:", input_ids)
    print("Labels:", labels)
"""