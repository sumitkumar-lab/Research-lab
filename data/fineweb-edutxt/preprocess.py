import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

# Download dataset of specific configuration on your cache directory.
print("Downloading 10B Token Speedrun Data...")

# or load a subset with roughly 100B tokens, suitable for small- or medium-sized experiments
dataset = load_dataset("HuggingFaceFW/fineweb-edu", 
                       name="sample-10BT",
                       split="train[:20000]",
                       num_proc=64,
                       )  # set cache directory properly


save_path = r"D:\hf_cache\processed_data\Fineweb"
os.makedirs(save_path, exist_ok=True)

dataset.save_to_disk(save_path)

print("✅ Data Ready!")