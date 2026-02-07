from __future__ import annotations

import os
import itertools
import numpy as np
from typing import Union

from datasets import load_dataset, Dataset, IterableDataset
from datasets import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer

from configs.config_dataloader import LoaderConfig


# ------------------------------------------------
# Tokenizer
# ------------------------------------------------

def load_tokenizer(cfg: LoaderConfig) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name,
        use_fast=cfg.use_fast,
        trust_remote_code=cfg.trust_remote_code,
        cache_dir=cfg.hf_cache_dir,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# ------------------------------------------------
# Dataset loading
# ------------------------------------------------

def load_raw_dataset(cfg: LoaderConfig) -> Union[Dataset, IterableDataset]:
    ds = load_dataset(
        cfg.dataset_path,
        cfg.dataset_name,
        split=cfg.split,
        streaming=cfg.streaming,
        cache_dir=cfg.hf_cache_dir,
    )

    if cfg.max_samples is not None:
        if cfg.streaming:
            ds = ds.take(cfg.max_samples)
        else:
            ds = ds.select(range(min(len(ds), cfg.max_samples)))

    if cfg.text_column not in ds.column_names:
        raise ValueError(
            f"Text column '{cfg.text_column}' not found. "
            f"Available columns: {ds.column_names}"
        )

    return ds


# ------------------------------------------------
# Tokenization + chunking
# ------------------------------------------------

def tokenize_and_chunk(
    dataset: Union[Dataset, IterableDataset],
    tokenizer: PreTrainedTokenizer,
    cfg: LoaderConfig,
) -> Union[Dataset, IterableDataset]:

    is_streaming = isinstance(dataset, IterableDataset)

    def tokenize(batch):
        return tokenizer(
            batch[cfg.text_column],
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=None if is_streaming else cfg.num_proc,
    )

    block_size = cfg.seq_length

    def group_texts(batch):
        concat = {k: np.concatenate(batch[k]) for k in batch}
        total_len = len(concat["input_ids"])

        total_len = (total_len // block_size) * block_size
        if total_len == 0:
            return {}

        return {
            k: concat[k][:total_len].reshape(-1, block_size)
            for k in concat
        }

    return tokenized.map(
        group_texts,
        batched=True,
        num_proc=None if is_streaming else cfg.num_proc,
    )


# ------------------------------------------------
# Finalization
# ------------------------------------------------

def finalize_for_lm(
    dataset: Union[Dataset, IterableDataset],
    cfg: LoaderConfig,
) -> Dataset:

    def add_labels(batch):
        batch["labels"] = batch["input_ids"].copy()
        return batch

    dataset = dataset.map(add_labels, batched=True)

    if isinstance(dataset, IterableDataset):
        dataset = Dataset.from_generator(
            lambda: (x for x in dataset),
            features=dataset.features,
        )

    dataset.set_format(
        type="torch",
        columns=["input_ids", "labels"],
    )

    if cfg.save_to_disk is not None:
        os.makedirs(cfg.save_to_disk, exist_ok=True)
        dataset.save_to_disk(cfg.save_to_disk)

    return dataset


# ------------------------------------------------
# One public entry point
# ------------------------------------------------

def load_dataset_for_lm(cfg: LoaderConfig) -> Dataset:
    tokenizer = load_tokenizer(cfg)
    raw_ds = load_raw_dataset(cfg)
    lm_ds = tokenize_and_chunk(raw_ds, tokenizer, cfg)
    lm_ds = finalize_for_lm(lm_ds, cfg)
    return lm_ds


# Load it...

cfg_1 = LoaderConfig(
    dataset_path="roneneldan/TinyStories",
    text_column="text",
    tokenizer_name="gpt2",
    seq_length=512,
    save_to_disk="D:/datasets/tinystories_512",
)

# dataset = load_dataset_for_lm(cfg_1)

# # Load fineweb...
# cfg_2 = LoaderConfig(
#     dataset_path="HuggingFaceFW/fineweb",
#     dataset_name="sample-10BT",
#     streaming=True,
#     seq_length=2048,
#     max_samples=1_000_000,
# )

# dataset = load_dataset_for_lm(cfg_2)

def split_dataset(
    dataset,
    train_ratio=0.98,
    val_ratio=0.01,
    seed=42,
):
    assert train_ratio + val_ratio < 1.0

    # First split: train vs temp
    split_1 = dataset.train_test_split(
        test_size=1.0 - train_ratio,
        seed=seed,
        shuffle=True,
    )

    # Second split: val vs test
    temp = split_1["test"]
    val_test = temp.train_test_split(
        test_size=0.5,
        seed=seed,
        shuffle=True,
    )

    return DatasetDict(
        {
            "train": split_1["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )

dataset = load_dataset_for_lm(cfg_1)
splits = split_dataset(dataset)
splits.save_to_disk("D:/datasets/tinystories_512_splits")


# from datasets import load_from_disk

# splits = load_from_disk("D:/datasets/tinystories_512_splits")

# train_ds = splits["train"]
# val_ds = splits["validation"]
# test_ds = splits["test"]