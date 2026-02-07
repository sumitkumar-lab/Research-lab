import os
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import tiktoken

# Load dataset
dataset = load_dataset("roneneldan/TinyStories",
                       split="train[:10%]")

# # choose tokenizer
# tokenizer = tiktoken.get_encoding("gpt2")

# # Tokenize function
# def tokenize(example):
#     ids = tokenizer.encode(example["text"])
#     return ids

# token_ids = dataset.map(tokenize, remove_columns=['text'])
