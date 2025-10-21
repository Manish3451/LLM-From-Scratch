# Building a Large Language Model from Scratch

## Overview

This project implements a complete pipeline for building, training, and fine-tuning a GPT-2 style language model from scratch. The implementation follows a three-stage approach: building the foundational LLM architecture, pretraining on unlabeled data, and fine-tuning for specific downstream tasks including spam classification.

## Architecture

The model implements the standard GPT-2 transformer architecture with the following components:

### Core Components

**Token and Position Embeddings**
- Token embeddings convert discrete tokens into continuous vector representations
- Positional embeddings inject sequence order information into the model
- Both embeddings are learned during training

**Multi-Head Self-Attention Mechanism**
- Enables the model to attend to different positions in the input sequence
- Uses scaled dot-product attention with causal masking for autoregressive generation
- Multiple attention heads allow the model to capture diverse semantic relationships

**Feed-Forward Networks**
- Two-layer MLP with GELU activation applied after each attention block
- Expands the hidden dimension by 4x in the intermediate layer
- Provides non-linear transformations to increase model expressiveness

**Layer Normalization and Residual Connections**
- Pre-normalization applied before each sub-layer
- Residual connections around each sub-layer facilitate gradient flow
- Critical for training stability in deep networks

## Project Structure

```
LLM FROM SCRATCH/
├── __pycache__/
├── gpt2/
├── images/
│   ├── llm.png
│   └── Screenshot 2025-10-21 163332.png
├── llm/
├── sms_spam_collection/
├── .gitattributes
├── .gitignore
├── 4.66
├── gpt_download3.py
├── LICENSE
├── LLM.tokenizer.ipynb
├── loss-plot.pdf
├── model_and_optimizer.pth
├── model.pth
├── README.md
├── sms_spam_collection.zip
├── temperature-plot.pdf
├── test.csv
├── the-verdict.txt
├── train.csv
├── validation.csv
└── verdict.txt
```

## Three-Stage Training Pipeline

### Stage 1: Building the LLM

The first stage focuses on implementing the core architecture and understanding the fundamental mechanisms:

1. **Data Preparation & Sampling**: Tokenization and batching of training data
2. **Attention Mechanism**: Implementation of scaled dot-product attention with causal masking
3. **LLM Architecture**: Assembly of the complete transformer model with embedding layers, attention blocks, and feed-forward networks

### Stage 2: Pretraining the Foundation Model

The second stage creates a foundational language model through pretraining on unlabeled text:

4. **Pretraining**: Training the model on a large corpus using next-token prediction
5. **Training Loop**: Optimization using AdamW with learning rate scheduling
6. **Model Evaluation**: Validation on held-out data to monitor convergence

The model learns general language patterns, grammar, and semantic relationships during this phase.

### Stage 3: Fine-Tuning for Downstream Tasks

The final stage adapts the pretrained model for specific applications:

7. **Load Pretrained Weights**: Initialize with pretrained GPT-2 parameters
8. **Fine-Tuning for Classification**: Adapt the model for spam detection by adding a classification head
9. **Fine-Tuning for Chat**: Adapt the model for conversational AI using instruction-following datasets

## Training Progress

### Loss Curves

The training process shows healthy convergence with the training loss steadily decreasing over epochs:

![Training Progress](loss-plot.pdf)

**Key Observations:**
- Training loss decreases from approximately 9.5 to below 1.0 over 10 epochs
- Validation loss stabilizes around 6.5, indicating some overfitting in later epochs
- The gap between training and validation loss suggests the model is learning meaningful patterns
- Token efficiency improves as the model processes more data

## Implementation Details

### Tokenization

The project uses a custom byte-pair encoding (BPE) tokenizer compatible with GPT-2:

- Vocabulary size: 50,257 tokens
- Special tokens for padding, end-of-sequence, and unknown tokens
- Handles both encoding and decoding of text sequences

### Training Configuration

```python
# Model hyperparameters
vocab_size = 50257
context_length = 1024
embedding_dim = 768
num_heads = 12
num_layers = 12
dropout = 0.1

# Training hyperparameters
batch_size = 8
learning_rate = 5e-4
num_epochs = 10
weight_decay = 0.1
```

### Pretraining Dataset

The model was pretrained on `the-verdict.txt`, a curated text corpus suitable for demonstrating the language modeling objective.

### Fine-Tuning: Spam Classification

After pretraining, the model was fine-tuned on the SMS Spam Collection dataset:

- **Task**: Binary classification (spam vs. ham)
- **Dataset**: SMS messages labeled as spam or legitimate
- **Architecture**: GPT-2 backbone + classification head
- **Performance**: The fine-tuned model effectively distinguishes between spam and ham messages

## Loading Pretrained Weights

The project includes functionality to load pretrained GPT-2 weights from OpenAI:

```python
python gpt_download3.py
```

This script downloads the official GPT-2 weights and adapts them to the custom model architecture.

## Usage

### Training from Scratch

```python
# Initialize model
model = GPT2Model(config)

# Train the model
train_model(model, train_loader, val_loader, num_epochs=10)
```

### Fine-Tuning for Classification

```python
# Load pretrained weights
model.load_state_dict(torch.load('model.pth'))

# Add classification head
classifier = ClassificationModel(model, num_classes=2)

# Fine-tune on spam dataset
finetune_classifier(classifier, spam_train_loader, spam_val_loader)
```

### Generating Text

```python
# Generate text with the pretrained model
prompt = "Once upon a time"
generated_text = generate(model, tokenizer, prompt, max_tokens=100)
```

## Key Features

- **Complete implementation from scratch**: No reliance on high-level transformer libraries
- **Modular architecture**: Easy to modify and extend for different applications
- **Pretrained weight loading**: Compatible with OpenAI's GPT-2 checkpoints
- **Multi-task fine-tuning**: Supports both classification and generation tasks
- **Comprehensive training utilities**: Includes learning rate scheduling, gradient clipping, and checkpointing

## Results

The spam classification model achieves strong performance on the SMS Spam Collection dataset:

- **Training Loss**: Converges to < 1.0
- **Validation Performance**: Stable classification accuracy
- **Inference**: Fast prediction on new messages

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
tiktoken>=0.5.0
```

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

- OpenAI for the GPT-2 architecture and pretrained weights
- The transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017)
- SMS Spam Collection dataset for fine-tuning experiments

## References

- Vaswani, A., et al. (2017). "Attention Is All You Need"
- Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners"
- Brown, T., et al. (2020). "Language Models are Few-Shot Learners"

---

## Future Work

- Implement larger model variants (GPT-2 Medium, Large, XL)
- Add support for distributed training across multiple GPUs
- Extend fine-tuning to additional downstream tasks
- Implement more sophisticated sampling strategies for text generation
- Add support for instruction-following and RLHF
