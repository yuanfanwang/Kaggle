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