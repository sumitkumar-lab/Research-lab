from dataclasses import dataclass
from typing import Optional


@dataclass
class LoaderConfig:
    # Dataset
    dataset_path: str                 # HF name or local path
    dataset_name: str = None  # HF subset (e.g. "default")
    split: str = "train"
    text_column: str = "text"

    # Tokenizer
    tokenizer_name: str = "gpt2"
    use_fast: bool = True
    trust_remote_code: bool = False

    # Processing
    seq_length: int = 1024
    num_proc: int = 4
    streaming: bool = False

    # Caching / saving
    hf_cache_dir: str = None
    save_to_disk: str = None  # e.g. "D:/datasets/fineweb_10B"

    # Debug / slicing
    max_samples: int = None  # for fast experiments
