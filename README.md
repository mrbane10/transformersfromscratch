# English to Urdu Transformer from Scratch

A complete implementation of the Transformer architecture from scratch using PyTorch for English to Urdu neural machine translation. This project demonstrates a full end-to-end implementation including model architecture, training pipeline, evaluation metrics, and inference capabilities.

## 🎯 Project Overview

This project implements the original Transformer architecture as described in "Attention Is All You Need" (Vaswani et al., 2017) for English-to-Urdu translation. The model achieves competitive performance with approximately **64 million parameters**.

### Key Features

*   **Complete Transformer Implementation**: Built from scratch with all components (Multi-Head Attention, Positional Encoding, Feed-Forward Networks, etc.)
*   **Bilingual Translation**: English to Urdu translation with proper tokenization for both languages
*   **Comprehensive Evaluation**: Multiple metrics including BLEU, CER, and WER 
*   **Training Pipeline**: Full training loop with validation, checkpointing, and monitoring
*   **Inference Support**: Easy to use translation interface for new text

## 📊 Model Performance on the validation dataset

| Metric | Score |
| --- | --- |
| BLEU Score | 0.4391 |
| Character Error Rate (CER) | 0.3588 |
| Word Error Rate (WER) | 0.5303 |
| Parameters | ~64M |

## 🏗️ Architecture

The model implements the standard Transformer architecture with:

*   **Encoder-Decoder Structure**: 6 encoder and 6 decoder layers
*   **Multi-Head Attention**: 8 attention heads
*   **Model Dimension**: 512
*   **Feed-Forward Dimension**: 2048
*   **Maximum Sequence Length**: 410 tokens
*   **Vocabulary**: Dynamic based on training corpus

## 🚀 Quick Start

### Prerequisites

bash

    pip install datasets tokenizers
    pip install wandb torchmetrics nltk
    pip install tqdm pathlib

### Training

bash

    python train.py

### Inference

python

    from train import get_model, get_ds
    from config import get_config
    import torch
    
    # Load configuration and model
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load trained model
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    
    # Load weights
    model_path = "weights/tmodel_19.pt"
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    
    # Translate text
    def translate(text, model, tokenizer_src, tokenizer_tgt, config, device):
        # Implementation in train.py
        pass
    
    # Example usage
    english_text = "Hello, how are you?"
    urdu_translation = translate(english_text, model, tokenizer_src, tokenizer_tgt, config, device)
    print(f"English: {english_text}")
    print(f"Urdu: {urdu_translation}")

## 📁 Project Structure

    transformers-from-scratch/
    ├── train.py              # Main training script
    ├── model.py              # Transformer architecture implementation
    ├── config.py             # Configuration settings
    ├── dataset.py            # Dataset handling and preprocessing
    └── README.md             # This file

## 🔧 Configuration

Key configuration parameters in `config.py`:

python

    {
        'batch_size': 8,          # Training batch size
        'num_epochs': 20,         # Number of training epochs
        'lr': 1e-4,              # Learning rate
        'seq_len': 410,          # Maximum sequence length
        'd_model': 512,          # Model dimension
        'lang_src': 'en',        # Source language
        'lang_tgt': 'ur',        # Target language
        'model_folder': "weights", # Model save directory
        'preload': None,         # Checkpoint to resume from
    }

## 📈 Training Process

The training pipeline includes:

1.  **Data Loading**: Uses Helsinki-NLP/opus-100 dataset (English-Urdu pairs)
2.  **Tokenization**: WordLevel tokenizers with special tokens (\[SOS\], \[EOS\], \[PAD\], \[UNK\])
3.  **Training Loop**: Cross-entropy loss with label smoothing (0.1)
4.  **Validation**: Character/Word Error Rates and BLEU score evaluation
5.  **Monitoring**: Weights & Biases integration for experiment tracking
6.  **Checkpointing**: Model states saved after each epoch

### Training Features

*   **Adam Optimizer** with learning rate 1e-4
*   **Label Smoothing** (0.1) for better generalization
*   **Gradient Clipping** for training stability
*   **Learning Rate Scheduling** (optional)
*   **Early Stopping** based on validation metrics

## 🎯 Key Components

### 1\. Multi-Head Attention (`model.py`)

python

    class MultiHeadAttentionBlock(nn.Module):
        def __init__(self, d_model: int, h: int, dropout: float):
            # Implementation with scaled dot-product attention

### 2\. Transformer Architecture (`model.py`)

python

    class Transformer(nn.Module):
        def __init__(self, encoder, decoder, src_embedding, target_embedding, ...):
            # Complete transformer with encoder-decoder architecture

### 3\. Training Pipeline (`train.py`)

python

    def train_model(config):
        # Full training loop with validation and checkpointing

## 📊 Evaluation Metrics

The model is evaluated using multiple metrics:

*   **BLEU Score**: Measures n-gram overlap between predicted and reference translations
*   **Character Error Rate (CER)**: Character-level accuracy
*   **Word Error Rate (WER)**: Word-level accuracy
*   **Training Loss**: Cross-entropy loss during training

## 🔬 Technical Details

### Model Architecture

*   **Attention Mechanism**: Scaled dot-product attention with multiple heads
*   **Positional Encoding**: Sinusoidal encoding for sequence position information
*   **Layer Normalization**: Applied before each sub-layer (Pre-LN)
*   **Residual Connections**: Skip connections around each sub-layer
*   **Dropout**: Applied for regularization (0.1)

### Training Optimizations

*   **Xavier Uniform Initialization**: For stable training
*   **Gradient Accumulation**: For effective larger batch sizes
*   **Mixed Precision Training**: Optional for memory efficiency
*   **Causal Masking**: Prevents decoder from seeing future tokens

## 📋 Requirements

*   Python 3.7+
*   PyTorch 1.9+
*   transformers
*   datasets
*   tokenizers
*   wandb (for experiment tracking)
*   torchmetrics
*   nltk
*   tqdm

## 🚀 Getting Started

1.  **Clone the repository**:
    
    bash
    
        git clone https://github.com/yourusername/transformers-from-scratch.git
        cd transformers-from-scratch
    
2.  **Install dependencies**:
    
    bash
    
        pip install -r requirements.txt
    
3.  **Set up Weights & Biases** (optional):
    
    bash
    
        wandb login
    
4.  **Start training**:
    
    bash
    
        python train.py
    
5.  **Monitor training**:
    *   Check console output for training progress
    *   View metrics on Weights & Biases dashboard
    *   Model checkpoints saved in `weights/` directory

## 🎛️ Advanced Usage

### Custom Dataset

To use your own dataset, modify the `get_ds()` function in `train.py`:

python

    def get_ds(config):
        # Replace with your dataset loading logic
        ds_raw = load_your_dataset()
        # Rest of the preprocessing pipeline

### Hyperparameter Tuning

Modify `config.py` to experiment with different settings:

python

    def get_config():
        return {
            'batch_size': 16,      # Increase for faster training
            'd_model': 768,        # Larger model dimension
            'num_epochs': 30,      # More training epochs
            'lr': 5e-5,           # Different learning rate
        }


### Performance Monitoring

*   Training loss decreases from ~8.0 to ~1.8 over 20 epochs
*   Validation metrics improve consistently
*   BLEU score reaches 0.44, indicating good translation quality

## 🙏 Acknowledgments

*   Original Transformer paper: "Attention Is All You Need" by Vaswani et al.
*   Hugging Face for datasets and tokenizers
*   Weights & Biases for experiment tracking
*   Helsinki-NLP for the OPUS-100 dataset
*   Umar's Jamil's explanation on how to implement a transformer from scratch

## 📚 References

1.  Vaswani, A., et al. (2017). Attention is all you need. In Advances in neural information processing systems.
2.  Helsinki-NLP OPUS-100: [https://huggingface.co/datasets/Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
3.  The Annotated Transformer: [http://nlp.seas.harvard.edu/2018/04/03/attention.html](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
