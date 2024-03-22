import sys
from datasets import Dataset
import tensorflow as tf
import numpy as np
import torch

dic = {
    'labels': torch.tensor([[0, 0], [0, 0], [0, 0]], dtype=torch.int32),
    'input_ids': torch.tensor([[2169, 12103], [270, 3513], [28310, 4593]], dtype=torch.int32),
    'token_type_ids': torch.tensor([[0, 0], [0, 0], [0, 0]], dtype=torch.int32),
    'attention_mask': torch.tensor([[1, 1], [1, 1], [1, 1]], dtype=torch.int32),
}

# monkey patch
np.object = object
    # alternatively
    # np.int = int
    # np.float = float
    # np.bool = bool

dataset = Dataset.from_dict(dic)
torch_dataset = dataset.with_format("torch")
print(tf.__version__)
print(np.__version__)
print(torch_dataset[0])
