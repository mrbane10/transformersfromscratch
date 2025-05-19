# Transformer from Scratch in PyTorch

This repository contains a PyTorch implementation of the Transformer architecture, built from the ground up without relying on high-level libraries. It includes:

- Input embedding and positional encoding
- Multi-head self-attention and cross-attention blocks
- Layer normalization and residual connections
- Encoder and decoder stacks
- Projection layer for output vocabulary prediction
- Tokenizer setup and dataset preparation (work in progress)

## Features

- Modular and extensible PyTorch modules for each Transformer component
- Custom implementation of multi-head attention and feed-forward networks
- Support for configurable model size, number of layers, heads, and dropout
- Positional encoding to capture sequence order
- Training utilities including tokenizer building and dataset loading (in progress)

## Files

- `model.py`: Core Transformer architecture components and model assembly.
- `train.py`: Tokenizer creation, dataset loading, and training logic (under development).

## Usage

1. Build the Transformer model using `build_transformer` with desired hyperparameters.
2. Prepare tokenizers and datasets for source and target languages.
3. Train the model using the training script once completed.

## Requirements

- Python 3.7+
- PyTorch
- `tokenizers` library
- `datasets` library (HuggingFace)

## Next Steps

- Complete training loop and evaluation
- Add inference pipeline with greedy/beam search decoding
- Support saving/loading model checkpoints
- Enhance tokenizer training and preprocessing

## License

MIT License

---

Feel free to contribute or open issues for bugs and feature requests!
