# GRPO Transformer: From Scratch AI Model

This project implements a **Group Relative Policy Optimization (GRPO)** based **Transformer model** entirely from scratch using **PyTorch**. It's designed for training a text generation model and then letting you interact with it in a conversational way.

---

## Features

* **Custom Tokenizer**: A simple tokenizer built for Turkish text, including special tokens for padding, unknown words, and sequence boundaries.
* **Transformer Architecture**: Includes all the core Transformer components:
    * **Multi-Head Attention** with proper weight initialization.
    * **Feed-Forward Network** with GELU activation.
    * **Transformer Block** combining attention, feed-forward layers, layer normalization, and dropout.
* **GPT-style Model**: A generative pre-trained Transformer model that can learn intricate language patterns.
* **GRPO Trainer**: A custom training loop leveraging Group Relative Policy Optimization with a self-certainty reward mechanism to fine-tune the model effectively.
* **Text Generation**: Features a versatile `generate` method with adjustable `temperature`, `top-k`, `top-p` filtering, and `repetition penalty` for controlled and diverse text output.
* **Model Persistence**: Easy-to-use functions for saving and loading both the trained model and its tokenizer.
* **Training Visualization**: Provides plots to track key training metrics such as **loss**, **reward**, and **perplexity** over time, helping you monitor progress.

---

## Getting Started

### Prerequisites

To get this project up and running, you'll need the following Python packages:

* Python 3.x
* PyTorch
* NumPy
* Matplotlib

You can install them quickly using pip:

```bash
pip install torch numpy matplotlib
