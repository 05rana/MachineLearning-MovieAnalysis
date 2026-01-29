# Predicting Movie Genre Popularity from Reviews

## Overview
This project applies machine learning techniques to predict movie genre popularity using popular user reviews, genre information, and release year. It analyzes large-scale movie data to understand how audience feedback influences genre-level ratings.

The project was developed as part of the IT461 Practical Machine Learning course.

## Objectives
- Analyze the predictive power of user reviews
- Combine textual and metadata features for popularity prediction
- Compare classical and deep learning models
- Identify trends in genre popularity over time

## Dataset
- Source: Letterboxd All Movie Data (Hugging Face)
- Total records: 847,000+ movies
- Features: Reviews, genre, release year, sentiment, ratings
- A balanced subset of 20,000 reviews was used for experiments

## Preprocessing
- Cleaned and normalized ratings
- Removed missing and outdated records
- Text cleaning and tokenization
- TF-IDF feature extraction
- Sentiment analysis using VADER
- One-hot encoding for genres
- Converted ratings into five balanced categories

## Models Used
- Logistic Regression (baseline)
- Multi-Layer Perceptron (Neural Network)
- LSTM (RNN)
- Baseline Dummy Classifier

## Methodology
- Feature engineering combining text, sentiment, and metadata
- Train-test split with multiple training windows (40%, 60%, 80%)
- Hyperparameter tuning for all models
- Evaluation using Accuracy, F1-Macro, and F1-Weighted

## Results
- Logistic Regression achieved the most consistent performance
- Neural Network performed competitively
- LSTM showed weaker results on this dataset
- Best accuracy ≈ 0.80
- Macro-AUC ≈ 0.95 for top models

## Technologies
- Python
- Jupyter Notebook
- Scikit-learn
- TensorFlow / Keras
- Pandas, NumPy
- Matplotlib, Seaborn
- Google Colab

## How to Run

Run the notebook:
jupyter notebook GenrePopularityPrediction_ML.ipynb

## Limitations
- High computational cost for large datasets
- Limited to first genre per movie
- Downsampling required for training

## Future Work
- Support multi-genre movies
- Use advanced NLP models
- Expand feature set
- Improve deep learning architecture

## Authors
- Rana Alharbi
- Aljawharah Alhowiedy
- Rana Albridi
- Mariam Al-Ahmed
- Haya Alibrahim

Supervised by: Dr. Abeer Aldayel
