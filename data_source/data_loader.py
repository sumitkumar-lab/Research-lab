import torch


with open("/content/the-verdict.txt", 'r', encoding='utf-8') as f:
    data = f.read()

if not data:
    raise ValueError("Input file is empty")

# encode with tiktoken gpt-2 bpe(byte pair encoding) -> it has 50257 vocab size lower model won't work.
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(data)
tokens = torch.tensor(tokens, dtype=torch.long)

# split into train_data and val_data
n = len(tokens)
train_ids = tokens[:int(n*0.9)]
val_ids = tokens[int(n*0.9):]

print(f"GPT-2 tokenizer vocab size: {enc.n_vocab}")
print(f"Max token ID in train: {train_ids.max().item()}")
print(f"Max token ID in val: {val_ids.max().item()}")

def get_batch(split):
    data_split = train_ids if split == "train" else val_ids
    idx = torch.randint(len(data_split)-seq_len, ("batch_size",))
    # idx : Random starting positions of [batch_size] -> 32
    x = torch.stack([data_split[i:i+seq_len] for i in idx])
    y = torch.stack([data_split[i+1:i+seq_len+1] for i in idx])
    x = x.to(device)
    y = y.to(device)
    return x, y