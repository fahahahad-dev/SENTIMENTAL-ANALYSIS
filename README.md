# Fine-Tuning DistilBERT for Airline Sentiment Analysis

This repository contains a Jupyter Notebook (`AI_BERT.ipynb`) that demonstrates how to fine-tune a pre-trained DistilBERT model for sequence classification using TensorFlow. The project focuses on classifying the sentiment of airline-related tweets into three categories: Negative, Neutral, and Positive.

## Overview

The notebook walks through a complete Natural Language Processing (NLP) pipeline:
1. **Data Loading & Preprocessing:** Loading the Twitter US Airline Sentiment dataset, mapping sentiments to numerical labels, cleaning text (removing HTML tags, URLs, special characters), and removing stopwords.
2. **Exploratory Data Analysis (EDA):** Visualizing class distribution and generating word clouds to understand the dataset's vocabulary.
3. **Tokenization:** Encoding the cleaned text using the Hugging Face `DistilBertTokenizer`.
4. **Model Initialization:** Setting up a `TFDistilBertForSequenceClassification` model with TensorFlow.
5. **Class Weighting:** Computing class weights to handle imbalanced data.
6. **Fine-Tuning:** Training the model using the `Adam` optimizer and `SparseCategoricalCrossentropy` loss.
7. **Evaluation:** Generating a classification report (precision, recall, f1-score) and building an interactive prompt for real-time sentiment prediction.

## Prerequisites & Dependencies

To run this notebook, you need Python installed along with the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `plotly`
- `nltk`
- `tensorflow`
- `tensorflow-datasets`
- `scikit-learn`
- `tqdm`
- `transformers` (Hugging Face)

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib seaborn plotly nltk tensorflow tensorflow-datasets scikit-learn tqdm transformers
