# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lab 2: LSTM Encoder-Decoder with Attention for Decimal-to-Roman Numeral Translation.

**Task**: Build and investigate seq2seq models with Luong attention for translating decimal numbers (e.g., "1999") to Roman numerals (e.g., "MCMXCIX").

## Project Structure

```
src/
├── __init__.py      # Package exports
├── data.py          # Dataset generation, vocabularies, dataloaders
├── models.py        # Encoder, Decoder, Seq2Seq model
├── attention.py     # Luong attention (dot, general, concat)
├── trainer.py       # Training loop, checkpointing, evaluation
├── metrics.py       # Accuracy, Levenshtein distance, error analysis
├── decoding.py      # Greedy, beam search, top-k, top-p decoding
├── visualization.py # Plotting functions for training curves, attention, errors
└── config.py        # Experiment configurations

experiments.ipynb    # Main notebook with all experiments
```

## Environment Setup

```bash
# Activate virtual environment
source venv/Scripts/activate  # Git Bash
venv\Scripts\activate.bat     # CMD
venv\Scripts\Activate.ps1     # PowerShell

# Run Jupyter notebook
jupyter notebook experiments.ipynb
```

## Key Commands

```python
# Quick model training
from src.data import create_datasets, create_dataloaders
from src.models import create_model
from src.trainer import Trainer, get_device

train_ds, val_ds, test_ds, src_vocab, tgt_vocab = create_datasets()
train_loader, val_loader, test_loader = create_dataloaders(train_ds, val_ds, test_ds)

model = create_model(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    hidden_size=256,
    bidirectional=True
)

trainer = Trainer(model, src_vocab, tgt_vocab, get_device())
trainer.fit(train_loader, val_loader, epochs=50)
```

## Model Configurations

Key parameters in `src/config.py`:
- `cell_type`: 'lstm' or 'gru'
- `bidirectional`: True/False for encoder
- `attention_method`: 'dot', 'general', 'concat'
- `embedding_type`: 'learned' or 'onehot'
- `hidden_size`: 64, 128, 256, 512
- `num_layers`: 1, 2, 3

## Experiments in Notebook

1. Dataset statistics visualization
2. Sequence length impact (number ranges 1-99, 1-999, 1-3999)
3. Architecture: hidden size, layers, embeddings
4. Cell types: LSTM vs GRU
5. Bidirectional vs unidirectional encoder
6. Attention methods comparison
7. Regularization: dropout, weight decay, label smoothing
8. Decoding strategies: greedy, beam search, top-k, top-p
9. Attention visualization
10. Error analysis by position and input length

## Notes

- Code runs on both CPU and GPU (auto-detected)
- Uses teacher forcing during training
- Metrics: sequence accuracy, character accuracy, Levenshtein distance
