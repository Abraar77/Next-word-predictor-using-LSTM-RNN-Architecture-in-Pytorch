Next-Word Predictor Using LSTM in PyTorch

A powerful Next-Word Prediction model built from scratch using PyTorch and LSTM (Long Short-Term Memory) recurrent neural networks. This project demonstrates how to preprocess text data, train a sequence model on a GPU, and achieve high predictive accuracy — a great foundation for understanding language modeling and how modern LLMs work.

🚀 Project Overview

Natural language tasks like predicting the next word in a sentence are key building blocks for text generation, autocompletion, and even large language models. This repository shows how an LSTM-based model can be trained to learn patterns in text sequences and predict the most likely next word based on previous context.

In this project, we:

✅ Converted raw text into trainable sequences
✅ Built a PyTorch LSTM neural network
✅ Trained the model on GPU for faster performance
✅ Achieved ~98% accuracy on the prediction task
✅ Explored inference and next-word generation

🧠 Why LSTM?

LSTM networks are a type of Recurrent Neural Network (RNN) that can capture long-term dependencies in sequential data. They overcome many limitations of basic RNNs and are widely used in language modeling tasks like next-word prediction, machine translation, and text generation.

📦 Features

🧠 PyTorch implementation — clean, readable, and beginner-friendly

⚡ GPU training support — takes advantage of CUDA for fast learning

📈 High accuracy — impressive performance on next-word prediction

📝 Jupyter Notebook included — walk through all steps interactively

🐍 Complete training + inference pipeline

🛠️ How It Works

Text Preprocessing

Tokenize sentences

Turn words into integer sequences

Create input/output pairs for supervision

Model Architecture

Embedding layer

LSTM layers

Fully connected output layer

Softmax for word prediction

Training

Trained on text data using GPU (if available)

Loss & accuracy metrics logged

Inference

Feed a sequence of words

Model predicts the most probable next word
