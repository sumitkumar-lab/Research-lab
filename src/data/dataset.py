import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class MemMapDataset(Dataset):
    def __init__(self, data_dir, split, block_size):
        """
        Args:
            data_dir: Path to directory containing train.bin / val.bin
            split: 'train' or 'val'
            block_size: The context window length (e.g., 1024)
        """
        self.block_size = block_size
        filename = os.path.join(data_dir, f'{split}.bin')
        
        # Research Note: We use np.memmap to access the file on disk 
        # without loading it all into RAM.
        self.data = np.memmap(filename, dtype=np.uint16, mode='r')
        
        # The number of available examples is total_tokens - block_size
        self.length = len(self.data) - self.block_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # We grab a chunk of length block_size + 1
        # because x is input, y is target (shifted by 1)
        # x: [0, 1, 2] -> y: [1, 2, 3]
        
        # Validating index to prevent overflow
        if idx < 0 or idx >= self.length:
            raise IndexError
            
        chunk = torch.from_numpy((self.data[idx : idx + self.block_size + 1]).astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        
        return x, y

def create_dataloader(data_dir, split, block_size, batch_size, num_workers=0):
    """
    Factory function to create the PyTorch DataLoader.
    num_workers=0 is often faster for memory-mapped files due to OS-level caching.
    """
    dataset = MemMapDataset(data_dir, split, block_size)
    
    # We use a RandomSampler for training to decorrelate batches
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(split == 'train'), 
        num_workers=num_workers,
        pin_memory=True # Faster transfer to CUDA
    )
    return loader