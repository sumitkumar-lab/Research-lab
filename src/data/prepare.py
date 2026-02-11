import os
import requests
import tiktoken
import numpy as np

# Configuration
# For research, we start with "TinyShakespeare" to debug quickly.
# Later, you swap this URL for "OpenWebText" or your own corpus.
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()

# 1. Dataset Split (Critical for Research Validity)
# We reserve 10% for validation. You must NEVER train on this.
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# 2. Tokenization
# We use GPT-2's tokenizer (Byte-Pair Encoding). 
# This is "Real World" tokenization, unlike character-level toy examples.
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_fast(train_data)
val_ids = enc.encode_fast(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# 3. Export to Binary (Memory Mapping Ready)
# We save as uint16 because GPT-2 vocab size (50257) fits in 2 bytes (0-65535).
# This cuts storage requirements in half compared to int32.
def export_bin(ids, filename):
    ids = np.array(ids, dtype=np.uint16)
    ids.tofile(os.path.join(os.path.dirname(__file__), filename))

export_bin(train_ids, 'train.bin')
export_bin(val_ids, 'val.bin')

print("Dataset preparation complete. Ready for mmap.")