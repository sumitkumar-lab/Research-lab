from dataclasses import dataclass
from typing import Optional

@dataclass
class GPTConfig:
    block_size: int = 1024       # Context window size
    vocab_size: int = 50257      # GPT-2 default
    n_layer: int = 12            # Depth
    n_head: int = 12             # Attention heads
    n_embd: int = 768            # Embedding dimension
    dropout: float = 0.0         # Dropout probability
    bias: bool = True            # True: bias in Linears/LayerNorms (like GPT-2). False: (like Llama)
    
    # Research specific hooks
    use_rms_norm: bool = False   # Switch between LayerNorm and RMSNorm
    rope_embeddings: bool = False # (Future) Switch for Rotary Embeddings