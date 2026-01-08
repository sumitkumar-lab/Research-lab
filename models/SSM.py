
# -------------------------------- Overfitting Test -------------------------------- #
# 0.52 / 68.75


from msilib.schema import _Validation_records
import torch
from torch import nn
import torch.nn.functional as F
import tiktoken
import matplotlib.pyplot as plt


class Config:
    def __init__(self,
                 vocab_size = 50,
                 emb_dim = 128,
                 seq_len = 256,
                 state_dim = 64,
                 n_head = 6,
                 n_layer = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.n_head = n_head
        self.n_layer = n_layer


class StateUpdate(nn.Module):
    def __init__(self, emb_dim, state_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.state_dim = state_dim
        self.state_flow = nn.Parameter(torch.eye(state_dim)*0.9)
        self.injection = nn.Parameter(torch.randn(state_dim, emb_dim))
        self.readout = nn.Parameter(torch.randn(emb_dim, state_dim))
        self.gate = nn.Linear(state_dim, state_dim)

    def forward(self, x):
        B, T, D = x.shape
        h = torch.zeros(B, self.state_dim)
        output = []

        for t in range(T):
            gate_input = self.gate(h) + x[:, t] @ self.injection.T
            g = torch.sigmoid(gate_input)

            h_candidate = h @ self.state_flow.T

            h = (1 - g) * h + g * h_candidate
            y = h @ self.readout.T
            output.append(y)

        return torch.stack(output, dim=1)
    
# state_out = StateUpdate(128, 64)
# x = torch.randn(32, 256, 128)
# out = state_out(x)
# print(out)
    
class StateBlock(nn.Module):
    def __init__(self, emb_dim, state_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.state = StateUpdate(emb_dim, state_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
                        nn.Linear(cfg.emb_dim, 4*cfg.emb_dim),
                        nn.GELU(),
                        nn.Linear(4*cfg.emb_dim, cfg.emb_dim)
        )
    
    def forward(self, x):
        x = self.norm1(x)
        x = x + self.state(x)
        
        x = self.mlp(x)
        out = x + self.norm2(x)
        return out


class SSM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.emb_dim = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.blocks = nn.ModuleList(
            [StateBlock(cfg.emb_dim, cfg.state_dim) for _ in range(cfg.n_layer)]
        )

        self.l_norm = nn.LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size)

    def forward(self, x):
        x = self.emb_dim(x)
        for block in self.blocks:
            h = block(x)
        
        x = self.l_norm(h)
        logits = self.out_head(x)
        return logits
    

cfg = Config()
model = SSM(cfg)

x = torch.randint(0, cfg.vocab_size, (32, cfg.seq_len))
out = model(x)
print(out)

print(sum(p.numel() for p in model.parameters()))





# # ------------------- Read Data ------------------- #

# with open("/tiny-llm-model/data_source/the-verdict.txt", "r", encoding="utf-8") as f:
#     text = f.read()

# # keep only the first 1000 characters
# text = text[:1000]

# # build vocabulary
# chars = sorted(list(set(text)))
# cfg.vocab_size = len(chars)

# stoi = {ch: i for i, ch in enumerate(chars)}
# itos = {i: ch for ch, i in stoi.items()}

# def encode(s): return [stoi[c] for c in s]
# def decode(ids): return ''.join([itos[i] for i in ids])

# data = torch.tensor(encode(text), dtype=torch.long)
# print("Dataset tokens:", len(data), "Vocab size:", cfg.vocab_size)


# tokenizer = tiktoken.get_encoding("gpt2")


# # # def get_batch(data, batch_size, seq_len):
# # #     idx = torch.randint(0, data.size(0)-seq_len, (batch_size,))
# # #     x = torch.stack([data[i:i+seq_len] for i in idx])
# # #     y = torch.stack([data[i+1:i+seq_len+1] for i in idx])
# # #     return x, y

# def get_batch(data, block_size, start_idx=0):
#     x = data[start_idx:start_idx+block_size].unsqueeze(0)
#     y = data[start_idx+1:start_idx+block_size+1].unsqueeze(0)
#     return x, y


# # batch_size=32

# model = SSM(cfg)
# print(f" Your Model Parameters are: {sum(p.numel() for p in model.parameters())}")
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

# # ----------------- Trining Loop ----------------- #

# for step in range(10000):
#     xb, yb = get_batch(data, block_size=32)

#     logits = model(xb)
#     loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), yb.view(-1))

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if step % 100 == 0:
#         print(f"Step {step:4d} | Loss: {loss.item():.4f}")


# @torch.no_grad()
# def evaluate_accuracy(model, data):
#     xb, yb = get_batch(data, block_size=32)
#     logits = model(xb)
#     preds = torch.argmax(logits, dim=-1)
#     acc = (preds == yb.view(-1)).float().mean().item()
#     return acc

# acc = evaluate_accuracy(model, data)
# print(f"Training accuracy: {acc * 100:.2f}%")


# # --- Step 6. Plot loss curve ---
# plt.figure(figsize=(6,4))
# plt.plot(loss, label="Training loss")
# plt.xlabel("Step")
# plt.ylabel("CrossEntropy Loss")
# plt.title("Tiny MLP LM Memorization Curve")
# plt.legend()
# plt.grid(True)
# plt.show()