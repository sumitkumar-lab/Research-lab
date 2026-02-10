# The Observatory: Personal AI Research Lab

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**The Observatory** is a minimalist, modular, and rigorous deep learning research framework designed from first principles.

Unlike production-oriented libraries that prioritize inference latency or ease of deployment, this lab prioritizes **observability**, **ablation**, and **scientific rigor**. It is a "Glass Box" environment where every variable—from initialization variance to gradient accumulation—is exposed, measurable, and controllable.

---

## 🔬 Core Philosophy

1.  **Code as Controlled Environment:** The codebase is a scientific instrument. Its primary purpose is to isolate variables to measure the impact of hypotheses.
2.  **No Magic:** We avoid "black box" abstractions. Layer norms, attention masks, and optimizer groups are implemented explicitly to allow for surgical modification.
3.  **Evaluation as First-Class Citizen:** We do not just measure "loss." We measure **Model FLOPs Utilization (MFU)**, **Gradient Norms**, and **Perplexity** to understand *learning dynamics*, not just outcomes.
4.  **Immutable Experiments:** Configurations are frozen at runtime. We do not "fix" running experiments; we iterate.

---

## 📂 Architecture

The project is structured to separate **Model Definition** (Math) from **Training Logic** (Environment).

```text
research_lab/
├── configs/                 # The Control Room (Hyperparameters)
│   ├── model/               # Architecture definitions (GPT-Nano, etc.)
│   └── trainer/             # Training loop settings
├── src/
│   ├── models/              # Pure mathematical definitions (Stateless)
│   │   ├── components/      # Atomic layers (Attention, MLP, Norms)
│   │   └── transformer.py   # The GPT Block composition
│   ├── training/            # The "Trainer" abstraction
│   │   └── trainer.py       # The step, forward, backward sequence
│   ├── data/                # Data Loading & Tokenization
│   │   ├── prepare.py       # Tokenization script (Text -> Binary)
│   │   └── dataset.py       # Memory-mapped dataset loader
│   └── evaluation/          # The Judge
│       └── metrics.py       # Perplexity, MFU, Generation
├── analysis/                # Post-mortem notebooks
│   └── notebooks/           # Jupyter notebooks for comparative analysis
├── experiments/             # OUTPUTS (Git-ignored)
│   └── [timestamp]_exp/     # Checkpoints, logs, and configs
└── train.py                 # The entry point