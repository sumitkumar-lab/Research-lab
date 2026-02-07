import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from tqdm import tqdm

import re
import random
import os
import time
import numpy as np


class BaseConfig(PretrainedConfig):
    model_type = "transformer"

    def __init__(
        self,
        vocab_size=50257,
        seq_len=128,
        emb_dim=516,
        n_heads=6,
        n_layers=12,
        # drop_rate=0.1,
        qkv_bias=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        # self.drop_rate = drop_rate
        self.qkv_bias = qkv_bias


# Create Multi-Head Attention class...
class MultiheadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()
        assert (emb_dim % n_heads == 0), \
            "emb_dim must be divisible by n_heads"

        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads # Reduce the projection dim to match desired output dim

        # Trainable weights
        self.W_query = nn.Linear(emb_dim, emb_dim)
        self.W_key = nn.Linear(emb_dim, emb_dim)
        self.W_value = nn.Linear(emb_dim, emb_dim)

        self.out_proj = nn.Linear(emb_dim, emb_dim)  # Linear layer to combine head outputs

    def forward(self, x):
        b, t, d = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, t, self.n_heads, self.head_dim)
        values = values.view(b, t, self.n_heads, self.head_dim)
        queries = queries.view(b, t, self.n_heads, self.head_dim)

        # Reshape batch -> Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        k = keys.transpose(1, 2)
        q = queries.transpose(1, 2)
        v = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = q @ k.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask = torch.triu(torch.ones(t, k.size(2), device=x.device), diagonal=1)
        attn_scores = attn_scores.masked_fill(mask.bool(), float('-inf'))

        attn_weights = torch.softmax(attn_scores / k.shape[-1]**0.5, dim=-1)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ v).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, t, self.emb_dim)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()
        self.attn = MultiheadAttention(emb_dim, n_heads)
        self.norm_1 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            nn.GELU(),
            nn.Linear(4*emb_dim, emb_dim)
        )
        self.norm_2 = nn.LayerNorm(emb_dim)


    def forward(self, x):
        """
        x = token_emb + pos_emb
        x = [b, t, emb_dim],  t = seq_len or context_len
        """
        attn_out = self.attn(x)

        x = x + attn_out   # Residual connection
        x = self.norm_1(x)
        x = x + self.ff(x)
        """
        x*W1+b1 -> GeLU -> x*W2+b2
        """
        x = self.norm_2(x)
        return x   # x -> [b, t, emb_dim] or [batch, context_len, emb_dim]


class Dillusion(PreTrainedModel):
    config_class = BaseConfig

    def __init__(self, cfg: BaseConfig):
        super().__init__(cfg)
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.emb_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.emb_dim, cfg.n_heads) for _ in range(cfg.n_layers)
            ])

        self.norm_f = nn.LayerNorm(cfg.emb_dim)
        self.head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)    # logits=h*W.T


    def forward(self, idx):  # input -> idx = [batch_size, sequence_length]
        b, t = idx.shape       # b -> batch, t -> seq_len
        # pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb[:, :t]

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        logits = self.head(x)   # y_pred or logits = softmax(x*W.T)
        
        return logits   # logits = [batch, context_len, vocab_size]
    

# vocab_size = enc.n_vocab
cfg = BaseConfig()
model = Dillusion(cfg)

x = torch.randint(0, cfg.vocab_size, (32, cfg.seq_len))
logits = model(x)
# logits.shape

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")