GRPO Transformer: From Scratch AI Model
This project implements a Group Relative Policy Optimization (GRPO) based Transformer model from scratch using PyTorch. It provides functionalities for training a text generation model and then interacting with it in a conversational manner.

Features
Custom Tokenizer: A simple tokenizer designed for Turkish text, including special tokens for padding, unknown words, beginning of sequence, and end of sequence.
Transformer Architecture: Implementation of core Transformer components:
Multi-Head Attention with Xavier/Glorot initialization.
Feed-Forward Network with GELU activation.
Transformer Block incorporating attention, feed-forward, layer normalization, and dropout.
GPT-style Model: A generative pre-trained Transformer model that can learn language patterns.
GRPO Trainer: A custom training loop implementing Group Relative Policy Optimization with a self-certainty reward mechanism for fine-tuning the model.
Text Generation: A generate method with temperature, top-k, top-p, and repetition penalty for diverse and controlled text output.
Model Persistence: Functions to save and load the trained model and tokenizer.
Training Visualization: Plots to track training loss, reward, and perplexity over time.
Getting Started
Prerequisites
Python 3.x
PyTorch
NumPy
Matplotlib
You can install the necessary Python packages using pip:

Bash

pip install torch numpy matplotlib
Data Preparation
The model expects training data in a JSON file named data.json in the same directory as the script. The JSON file should contain a list of dictionaries, where each dictionary has a 'prompt' and a 'response' key:

JSON

[
  {
    "prompt": "Merhaba nasılsın?",
    "response": "İyiyim, sen nasılsın?"
  },
  {
    "prompt": "Bugün hava nasıl?",
    "response": "Güneşli ve sıcak."
  }
]
Usage
Run the main script from your terminal:

Bash

python your_script_name.py
You will be presented with a menu:

============================================================
GRPO TRANSFORMER - SIFIRDAN YAPAY ZEKA MODELI
Group Relative Policy Optimization with Self-Certainty
============================================================

Seçenekler:
1 - Train: Yapay zekayı data.json ile train et
2 - Test: Modeli test et (sohbet modu)
3 - Exit: Programdan çık

Seçiminizi yapın (1/2/3):
1 - Train: Yapay zekayı data.json ile train et
Select option 1 to start the training process. The script will:

Load data from data.json.
Build a vocabulary based on your training texts.
Initialize the GPT model with specified parameters (d_model=256, num_heads=4, num_layers=4, d_ff=1024, max_length=128).
Train the model using the GRPO algorithm for 20 epochs.
Save the trained model and tokenizer to grpo_model.pt.
Display plots of training loss, reward, and perplexity.
Training progress, including average loss, reward, and perplexity per epoch, will be printed to the console.

2 - Test: Modeli test et (sohbet modu)
Select option 2 to enter the testing mode. This allows you to interact with the trained model:

The script will attempt to load the grpo_model.pt file. If no trained model is found, it will prompt you to train it first.
Once loaded, you can type your prompts, and the AI will generate responses.
Type 'quit', 'exit', or 'q' to leave the testing mode.
3 - Exit: Programdan çık
Select option 3 to terminate the program.

Model Configuration
The train_model function defines the model's architecture. You can adjust these parameters before training:

d_model: Dimension of the model's embeddings and hidden states (default: 256).
num_heads: Number of attention heads in MultiHeadAttention (default: 4).
num_layers: Number of Transformer blocks (default: 4).
d_ff: Dimension of the feed-forward network's hidden layer (default: 1024).
max_length: Maximum sequence length for tokenization (default: 128).
learning_rate: Optimization learning rate (default: 3e-5).
batch_size: Number of samples processed per gradient update (default: 4).
accumulation_steps: Gradients are accumulated over this many batches before an optimization step (default: 16, resulting in an effective batch size of 64).
num_epochs: Number of full passes over the training dataset (default: 20).
Adjusting these parameters will impact the model's performance, training time, and memory consumption. Smaller values reduce training time and memory, but might lead to a less capable model.
