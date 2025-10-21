Building a Large Language Model from Scratch

Overview

This project implements a complete pipeline for building, training, and fine-tuning a GPT-2 style language model from scratch using PyTorch. The implementation covers all fundamental stages, from data preparation and tokenization to building the transformer architecture, pretraining on unlabeled data, and fine-tuning for downstream classification tasks.

This work is based on the comprehensive concepts detailed in the book Build a Large Language Model (from Scratch) by Raschka, Mirjalili, and D'Souza.

<p align="center">
<img src="images/llm.png" alt="LLM Project Banner" width="800"/>
</p>

Project Structure

LLM FROM SCRATCH/
├── pycache/
├── gpt2/
├── images/
│   ├── llm.png
│   └── Screenshot 2025-10-21 163332.png
├── llm/
├── sms_spam_collection/
├── .gitattributes
├── .gitignore
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



Three-Stage Training Pipeline

The project follows a systematic three-stage approach to build and deploy a production-ready language model, leveraging the power of transfer learning.

<p align="center">
<img src="images/Screenshot%202025-10-21%20163332.png" alt="Three-Stage Training Pipeline" width="900"/>
</p>

Stage 1: Building the LLM

The first stage focuses on implementing the core architecture and understanding fundamental mechanisms:

1. Data Preparation & Sampling

Tokenization of raw text using OpenAI's tiktoken library (GPT-2's Byte-Pair Encoder).

Creation of input-target pairs using a sliding window approach.

Custom PyTorch DataLoader for efficient batch processing.

# Custom DataLoader creates batches for training the language model
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
)



2. Attention Mechanism

Implementation of causal multi-head self-attention.

Scaled dot-product attention with causal masking for autoregressive generation.

Multiple attention heads to capture diverse semantic relationships.

3. LLM Architecture

Assembly of the complete transformer model with embedding layers.

Stacked transformer blocks with attention and feed-forward networks.

Layer normalization and residual connections for training stability.

Stage 2: Pretraining the Foundation Model

The second stage creates a foundational language model through pretraining on unlabeled text:

4. Pretraining

Training on the the-verdict.txt corpus using next-token prediction.

The model learns general language patterns, grammar, and semantic relationships.

5. Training Loop

Optimization using AdamW optimizer with learning rate scheduling.

Cross-entropy loss calculation and gradient-based weight updates.

# Simplified training loop
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, ...):
    for epoch in range(num_epochs):
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()



6. Model Evaluation

Validation on held-out data to monitor convergence.

Regular checkpointing to save model progress.

Stage 3: Fine-Tuning for Downstream Tasks

The final stage adapts the pretrained model for specific applications:

7. Load Pretrained Weights

Initialize with official GPT-2 (124M) pretrained parameters.

Automatic download via the gpt_download3.py script.

8. Fine-Tuning for Classification

Adaptation for spam detection using the SMS Spam Collection dataset.

Replace language modeling head with a classification head.

Parameter-efficient fine-tuning by freezing most layers.

# Replace the output head for 2-class (spam/ham) classification
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"], 
    out_features=num_classes
)

# Unfreeze only the final transformer block for efficient training
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True



9. Fine-Tuning for Chat

Adaptation for conversational AI using instruction-following datasets.

Training the model to follow user instructions and maintain context.

Model Architecture

The model implements a GPT-2 style transformer architecture built from scratch using PyTorch. The architecture is defined by a configuration dictionary:

# GPT-2 Small (124M) configuration
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}



Core Components

Token and Position Embeddings

Token embeddings convert discrete tokens into continuous 768-dimensional vectors.

Positional embeddings encode sequence order information.

Both embeddings are learned during training.

Transformer Blocks
Each of the 12 transformer blocks contains:

Causal Multi-Head Self-Attention: Allows the model to weigh token importance while preventing information leakage from future tokens.

Feed-Forward Network (FFN): Two-layer MLP with GELU activation that expands the hidden dimension by 4x.

Layer Normalization: Pre-normalization applied before each sub-layer for stable training.

Residual Connections: Skip connections around each sub-layer to facilitate gradient flow in deep networks.

Results and Performance

Pretraining Loss

The training process demonstrates healthy convergence with steady loss reduction over epochs:

<p align="center">
<img src="loss-plot.png" alt="Training and Validation Loss Curves" width="650"/>
</p>

Key Observations:

Training loss decreases from approximately 9.5 to below 1.0 over 10 epochs.

Validation loss stabilizes around 6.5, indicating good generalization.

The model processes over 40,000 tokens efficiently.

Clear convergence pattern with diminishing returns after epoch 6.

Fine-Tuning Performance

The fine-tuned spam classification model achieves exceptional results:

Training Accuracy: 100.00%

Validation Accuracy: 97.50%

The model effectively distinguishes between spam and legitimate messages after just a few epochs of fine-tuning, demonstrating the power of transfer learning from the pretrained foundation model.

Text Generation with Temperature Control

Different decoding strategies control the randomness and creativity of generated text:

<p align="center">
<img src="temperature-plot.png" alt="Temperature Sampling Strategies" width="650"/>
</p>

Temperature scaling and top-k sampling allow fine-grained control over the balance between coherent, predictable outputs and creative, diverse generations.

Usage

Environment Setup

Install the required dependencies:

pip install torch tiktoken pandas matplotlib tensorflow tqdm



Download Pretrained Weights

The included script automatically downloads official GPT-2 (124M) weights:

python gpt_download3.py



Training from Scratch

Execute the notebook to walk through the complete pipeline:

jupyter notebook LLM.tokenizer.ipynb



The notebook covers:

Data preparation and tokenization

Model architecture implementation

Pretraining on unlabeled text

Fine-tuning for spam classification

Text generation with various decoding strategies

Fine-Tuning for Custom Tasks

Adapt the model for your specific use case:

# Load pretrained model
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load('model.pth'))

# Add custom classification head
num_classes = your_num_classes
model.out_head = torch.nn.Linear(
    in_features=GPT_CONFIG_124M["emb_dim"],
    out_features=num_classes
)

# Fine-tune on your dataset
train_classifier(model, your_train_loader, your_val_loader)



Key Features

Complete Implementation from Scratch: Full transparency into every component of the transformer architecture.

Modular and Extensible: Easy to modify for different model sizes and configurations.

Pretrained Weight Compatibility: Seamless loading of OpenAI's GPT-2 checkpoints.

Multi-Task Fine-Tuning: Supports both classification and text generation tasks.

Parameter-Efficient Training: Selective layer freezing for efficient fine-tuning.

Flexible Text Generation: Multiple decoding strategies including temperature scaling and top-k sampling.

Production-Ready Code: Includes checkpointing, validation, and comprehensive logging.

Technical Highlights

Component

Description

Benefit

Attention Mechanism

Causal masking and multi-head attention implemented.

Ensures autoregressive generation and captures diverse contextual relationships.

Training Optimization

AdamW optimizer with weight decay and learning rate scheduling.

Better generalization and stable convergence.

Memory Efficiency

Batch processing and gradient clipping implemented.

Allows training with configurable batch sizes and prevents exploding gradients.

Future Enhancements

Implementation of larger model variants (GPT-2 Medium, Large, XL).

Distributed training across multiple GPUs.

Advanced decoding strategies (nucleus sampling, beam search).

Integration with instruction-tuning datasets.

RLHF (Reinforcement Learning from Human Feedback) pipeline.

Model quantization for efficient deployment.

References

Raschka, S., Mirjalili, V., & D'Souza, D. (2024). Build a Large Language Model (from Scratch). Manning.

Vaswani, A., et al. (2017). "Attention Is All You Need"

Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners"

Brown, T., et al. (2020). "Language Models are Few-Shot Learners"

SMS Spam Collection dataset contributors

License

This project is licensed under the terms specified in the LICENSE file.

Acknowledgments

OpenAI for the GPT-2 architecture and pretrained weights.

Sebastian Raschka for the comprehensive book and educational materials.

The PyTorch team for the excellent deep learning framework.

SMS Spam Collection dataset contributors.
