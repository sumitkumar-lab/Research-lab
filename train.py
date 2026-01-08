"""
::: Designing trainign loop is itself a crazy task because every things related 
    to your model learnings depends on training loop if anything miscalculation 
    happense in your training loop directly affects your model parameters which
    eventually affect your time, cost, and hard-work.

    So, while designing trining loop we must follow 13-training engineering rule.
"""
import os
import json
import time
import math
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from datasets import DatasetDict
from datasets import load_from_disk

import matplotlib.pyplot as plt

from sympy import true
import torch
import torch.nn as nn
from torch import nn, optim
import torch.nn.functional as F

import wandb
import tiktoken
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from models.model_Anima_350M import Anima, AnimaConfig
from models.model_miniGPT_124M import miniGPT, miniGPTConfig
from torch.utils.checkpoint import checkpoint as activation_checkpoint # memory saving

# ---------------------
# Configuration set-up | Rule-13
# ---------------------
TrainConfig = {
    "batch_size" : 4,
    "learning_rate" : 3e-4,
    "max_iters" : 10000,

    "eval_interval" : 500,
    # "log_interval" : 10,

    "grad_clip" : 1.0,
    "weight_decay" : 0.1,
    "betas" : (0.9, 0.95),

    "resume_path" : "ckpt.pt",  # path to checkpoint
    "out_dir" : './checkpoints',
    "activation_checkpointing": True,

    "device" : 'cuda',
    "ddp": False,

    "max_steps": 10000,
    "accumulate_grad": 4,

    "dataset_name": "the-verdict"
}

# ----------- CUDA environment set-up ----------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# ---------------- Initialize distributed process group ---------------- #
def setup_distributed():
    # These are automatically set by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="nccl",       # GPU backend (use "gloo" for CPU-only)
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(local_rank)

    return local_rank, rank, world_size

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

# ---------------- Helper: Check if this is main process ----------------#
ModelConfig = miniGPTConfig()
model = miniGPT(ModelConfig)

if TrainConfig["ddp"]:
    local_rank, rank, world_size = setup_distributed()
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
else:
    model = model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=TrainConfig["learning_rate"],
                              betas=TrainConfig["betas"],
                              weight_decay=TrainConfig["weight_decay"]
                              )

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)    # optimizer=optimizer, learning_rate=3e-4, learning_rate_iter= , final_learning_rate= ,
# --------------------------------
# Optional: Mixed precision scaler
# --------------------------------
scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

# -----------
# DataLoading
# -----------
# x = torch.randint(0, ModelConfig.vocab_size, (TrainConfig["batch_size"], ModelConfig.seq_len))
# y = torch.randint(0, ModelConfig.vocab_size, (TrainConfig["batch_size"], ModelConfig.seq_len))
# xb = x.to(device)
# yb = y.to(device)

# splits = load_from_disk("D:/datasets/tinystories_512_splits")

# train_ds = splits["train"]
# val_ds = splits["validation"]
# test_ds = splits["test"]

with open("/content/the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()

enc = tiktoken.get_encoding("gpt2")
tokens = torch.tensor(enc.encode(text), dtype=torch.long)

n = len(tokens)
train_ids = tokens[: int(0.9 * n)]
val_ids = tokens[int(0.9 * n):]

def get_batch(split):
    data = train_ids if split == "train" else val_ids
    idx = torch.randint(len(data) - ModelConfig.seq_len, (TrainConfig["batch_size"],))
    x = torch.stack([data[i : i + ModelConfig.seq_len] for i in idx])
    y = torch.stack([data[i + 1 : i + ModelConfig.seq_len + 1] for i in idx])
    return x.to(device), y.to(device)

# ----------
# VALIDATION
# ----------
@torch.no_grad()
def evaluate(model, eval_iters=100):
    model.eval()
    losses = []
    correct = 0
    total = 0

    for _ in range(eval_iters):
        xb, yb = get_batch("val")
        logits = model(xb)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            yb.view(-1)
        )
        losses.append(loss.item())

        preds = logits.argmax(dim=-1)
        correct += (preds == yb).sum().item()
        total += yb.numel()

    model.train()
    avg_loss = sum(losses) / len(losses)
    return avg_loss, correct / total, math.exp(avg_loss)


metrics = defaultdict(list)

# ----- 
# WANDB
# -----
if is_main_process():
    wandb.init(
        project="miniGPT-training",
        name="miniGPT-124M-verdict",
        config=TrainConfig
    )
# --------------
# Resume Training
# --------------
# Training metrics png/pdf.....
OUT_DIR = "./training_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
# save_dir = os.path.join(OUT_DIR if "OUT_DIR" in globals() else "/content")
# os.makedirs(save_dir, exist_ok=True)
# resume_path = os.path.join(OUT_DIR, "ckpt.pt")
# resume_path = "ckpt.pt"
start_step = 0

if os.path.exists(TrainConfig["resume_path"]):
    print(f"Resuming from checkpoint: {TrainConfig["resume_path"]}")
    checkpoint = torch.load(TrainConfig["resume_path"], map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    start_step = checkpoint["step"] + 1  # continue from next step

    state = checkpoint["rng_state"]
    if isinstance(state, list):
        restored = [torch.ByteTensor(s) for s in state]
        torch.set_rng_state(restored)

    state = checkpoint["cuda_rng_state"]
    if isinstance(state, list):
    # convert list-of-lists back to ByteTensors
        restored = [s.to('cpu') for s in state]
        torch.cuda.set_rng_state_all(restored)

    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])

    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])

# -------------
# Training Loop
# -------------
model.train()

save_interval = 500
step = start_step

start_time = time.time()
tokens_per_step = TrainConfig["batch_size"] * ModelConfig.seq_len
accum_steps = TrainConfig["accumulate_grad"]

# optimizer.zero_grad(set_to_none=True)

try:
    while step < TrainConfig["max_steps"]:
        iter_start = time.time()
        # optimizer.zero_grad()

        for _ in range(accum_steps):
            xb, yb = get_batch("train")
            xb, yb = xb.to(device), yb.to(device)

            if scaler:
                with torch.cuda.amp.autocast():
                    if TrainConfig.get("activation_checkpointing", False):
                        logits = activation_checkpoint(model, xb)
                    else:
                        logits = model(xb)
                    loss = F.cross_entropy(logits.reshape(-1, ModelConfig['vocab_size']), yb.reshape(-1)) / accum_steps
                scaler.scale(loss).backward()
            else:
                if TrainConfig.get("activation_checkpointing", False):
                    logits = activation_checkpoint(model, xb)
                else:
                    logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), yb.view(-1)) / accum_steps
                loss.backward()
        
        if scaler:
            scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), TrainConfig["grad_clip"])

        # ---- Back Tracking ---->
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        # 3. Learning rate scheduler
        scheduler.step()

        iter_time = time.time() - iter_start
        tokens_per_sec = (tokens_per_step * accum_steps) / iter_time

        # <|---------Save checkpoint every N-step---------|>
        if step % TrainConfig["eval_interval"] == 0:  # step ==  TrainConfig["max_steps"] - 1 or  #step == max_step-1
            val_loss, val_acc, val_ppl = evaluate(model)
            elapsed = time.time() - start_time
            lr = scheduler.get_last_lr()[0]

            metrics["step"].append(step)
            metrics["train_loss"].append(loss.item() * accum_steps)
            metrics["val_loss"].append(val_loss)
            metrics["val_accuracy"].append(val_acc)
            metrics["val_perplexity"].append(val_ppl)
            metrics["learning_rate"].append(lr)
            metrics["tokens_per_sec"].append(tokens_per_sec)
            metrics["elapsed_time"].append(elapsed)

            if is_main_process():
                print(
                    f"Step {step} | "
                    f"Train {loss.item()*accum_steps:.4f} | "
                    f"Val {val_loss:.4f} | "
                    f"PPL {val_ppl:.2f} | "
                    f"Tok/s {tokens_per_sec:.0f}"
                )

                wandb.log({
                    "step": step,
                    "train_loss": loss.item() * accum_steps,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_perplexity": val_ppl,
                    "learning_rate": lr,
                    "tokens_per_sec": tokens_per_sec,
                    "elapsed_time": elapsed
                })

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate()
                }

                torch.save(checkpoint, TrainConfig["resume_path"])
                print(f"Checkpoint saved at step: {step}")
            
        step +=1

    total_time = time.time() - start_time

    def plot_training_metrics(metrics, out_dir=OUT_DIR):
        steps = metrics["step"]

        plt.figure(figsize=(18, 12))

        # ---------------- Loss ----------------
        plt.subplot(2, 3, 1)
        plt.plot(steps, metrics["train_loss"], label="Train Loss")
        plt.plot(steps, metrics["val_loss"], label="Val Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()

        # ---------------- Accuracy ----------------
        plt.subplot(2, 3, 2)
        plt.plot(steps, metrics["val_accuracy"])
        plt.xlabel("Steps")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")

        # ---------------- Perplexity ----------------
        plt.subplot(2, 3, 3)
        plt.plot(steps, metrics["val_perplexity"])
        plt.xlabel("Steps")
        plt.ylabel("Perplexity")
        plt.title("Validation Perplexity")

        # ---------------- Learning Rate ----------------
        plt.subplot(2, 3, 4)
        plt.plot(steps, metrics["learning_rate"])
        plt.xlabel("Steps")
        plt.ylabel("LR")
        plt.title("Learning Rate Schedule")

        # ---------------- Throughput ----------------
        plt.subplot(2, 3, 5)
        plt.plot(steps, metrics["tokens_per_sec"])
        plt.xlabel("Steps")
        plt.ylabel("Tokens/sec")
        plt.title("Training Throughput")

        # ---------------- Time ----------------
        plt.subplot(2, 3, 6)
        plt.plot(steps, metrics["elapsed_time"])
        plt.xlabel("Steps")
        plt.ylabel("Seconds")
        plt.title("Elapsed Training Time")

        plt.tight_layout()
        
        # ---- Save plots ----
        pdf_path = os.path.join(out_dir, "training_metrics.pdf")
        png_path = os.path.join(out_dir, "training_metrics.png")

        plt.savefig(pdf_path)
        plt.savefig(png_path, dpi=300)
        plt.close()

        print(f"Saved plots to:\n{pdf_path}\n{png_path}")

        # ---- Save raw metrics ----
        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved raw metrics to:\n{metrics_path}")
        # plt.show()

        # ---- Log to W&B ----
        if is_main_process() and wandb.run is not None:
            wandb.log({
                "training_plots": wandb.Image(png_path)
            })

    if is_main_process():
        plot_training_metrics(metrics)
        print(f"Total training time: {total_time/3600:.2f} hours")
        wandb.log({"total_training_time": total_time})
        wandb.finish()

    if TrainConfig["ddp"]:
        dist.destroy_process_group()

except KeyboardInterrupt:
    print("Checkpoints saved at last step...")   # Destroy all the ddp configuration, so to no error for the cumulative training or fine-tuning...

