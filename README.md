
# 🎬 IMDB Movie Review Sentiment Analysis

This project implements a sentiment analysis model for IMDB movie reviews using PyTorch. The model classifies movie reviews as positive or negative based on their content.

## 📑 Table of Contents

- [Features](#-features)
- [Results](#-results)
- [Model Architecture](#-model-architecture)
- [License](#-license)
- [Possible Improvements](#-possible-improvements)
- [Acknowledgments](#-acknowledgments)

## 🚀 Features

- Text preprocessing with lemmatization and stopword removal
- TF-IDF based feature extraction
- Neural network model for sentiment classification
- Model evaluation and inference scripts

## 📈 Results

The model achieves the following performance on the test set:
- Accuracy: ~85-88%
- F1-score: ~0.85-0.88

## 🤖 Model Architecture

- Input: TF-IDF features (10,000 most frequent words)
- Hidden layers:
  - Dense (256 units, ReLU)
  - Dropout (0.3)
  - Dense (128 units, ReLU)
  - Dropout (0.3)
- Output: 1 unit (Sigmoid)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🚀 Possible Improvements

Here are some potential enhancements for this project:

1. **Advanced Text Processing**
   - Implement more sophisticated text cleaning (URLs, special characters, etc.)
   - Experiment with different tokenization techniques
   - Handle emojis and emoticons more effectively

2. **Model Architecture**
   - Try different neural network architectures (LSTM, GRU, Transformer)
   - Experiment with pre-trained language models (BERT, RoBERTa)
   - Implement attention mechanisms

3. **Training Improvements**
   - Add learning rate scheduling
   - Implement early stopping
   - Experiment with different optimizers

## 🙏 Acknowledgments

- Dataset: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
