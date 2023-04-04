# NLP-TextVectors-LM-Sentiment
Explore essential NLP concepts in this project, covering Latent Semantic Analysis with Singular Value Decomposition, N-gram Language Models, and Sentiment Analysis using Bidirectional LSTMs. Implemented in Python, it includes preprocessing, vector representation, language modeling, and deep learning techniques for sentiment classification.
This repository contains an NLP project that covers three main topics: Latent Semantic Analysis using Singular Value Decomposition, N-gram Language Models, and Sentiment Analysis using Bidirectional LSTMs.

Table of Contents
Installation
Project Overview
Latent Semantic Analysis (LSA) using Singular Value Decomposition (SVD)
N-gram Language Models
Sentiment Analysis using Bidirectional LSTM

Project Overview <a name="project-overview"></a>
Latent Semantic Analysis (LSA) using Singular Value Decomposition (SVD) <a name="lsa"></a>
In this part, we implement LSA using SVD to represent documents and terms in a reduced vector space. We use a Python library (Scipy) for matrix decomposition and the example dataset from R. A. Harshman (1990). Indexing by latent semantic analysis. Journal of the American society for information science.

Features
Preprocessing of documents
Assigning names to document names
Creating words to index mapping
Building a Document-Terms count matrix
Singular Value Decomposition
Visualizing documents in 2D space and printing coordinates
Visual representation of query/document and finding matching documents for given document
Theory questions answered
A) Give a short description of Left-eigen vectors, right-eigen vectors, and eigen-values matrix returned by Singular Value Decomposition of document-terms count matrix.
B) Visually represent the document "Graph and tree generation" in 2D space along with words and documents as given in the previous question.

N-gram Language Models <a name="n-gram"></a>
In this part, we train unigram, bigram, and trigram language models and evaluate their performance.

Task 1
Train unigram, bigram, and trigram models on given training files
Score on given test files for unigram, bigram, and trigram
Generate sentences from the trained model and compute perplexity
Task 2
Create training data for n > 3
Repeat Task 1 for the new training data
Exploration and Explanation
Experiment with n-gram models for n = [1,2,3..7] and explain the best choice of n that generates more meaningful sentences.
Sentiment Analysis using Bidirectional LSTM <a name="lstm"></a>
In this part, we build a bidirectional LSTM network to train and inference sentiment analysis on the IMDB dataset.

Features
Plot Positive vs. Negative reviews count
Clean the Reviews
Split the dataset and Encode Labels
Fit tokenizer on the training reviews
Convert reviews in the dataset to their index form
Pad the training and validation sequences (maxlen = 200)
Use GloVe vectors for embedding
Create the embedding matrix using the glove_dictionary
Complete the linear model in TensorFlow
Plot train loss vs. val loss, train auc vs. val auc, train recall vs. val recall, train
