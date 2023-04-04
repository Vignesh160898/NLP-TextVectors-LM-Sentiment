# NLP-TextVectors-LM-Sentiment
Explore essential NLP concepts in this project, covering Latent Semantic Analysis with Singular Value Decomposition, N-gram Language Models, and Sentiment Analysis using Bidirectional LSTMs. Implemented in Python, it includes preprocessing, vector representation, language modeling, and deep learning techniques for sentiment classification.
This repository contains an NLP project that covers three main topics: Latent Semantic Analysis using Singular Value Decomposition, N-gram Language Models, and Sentiment Analysis using Bidirectional LSTMs.

# NLP Techniques and Applications

This repository contains a Jupyter Notebook implementing various NLP techniques and applications. The project is divided into three parts: Latent Semantic Analysis (LSA), n-gram Language Models, and Sentiment Analysis using LSTM.

## Table of Contents
1. [Latent Semantic Analysis (LSA)](#lsa)
2. [n-gram Language Models](#ngram)
3. [Sentiment Analysis using LSTM](#lstm)

## Latent Semantic Analysis (LSA) <a name="lsa"></a>

The first part of the project demonstrates Latent Semantic Analysis (LSA) using Singular Value Decomposition (SVD) to represent text and documents in a distributed manner. The example dataset used is from R. A. Harshman's "Indexing by latent semantic analysis" paper (Journal of the American society for information science, 1990). 

The project includes:
- Preprocessing of documents
- Assigning names to document names
- Words to Index mapping
- Building a Document-Terms count matrix
- Singular Value Decomposition (SVD)
- Visualizing documents in 2D space
- Visual representation of query/document and finding matching documents

## n-Gram Language Models <a name="ngram"></a>

The second part of the project focuses on training n-gram language models. The tasks include:
1. Training unigram, bigram, and trigram models on given training files
2. Scoring on given test files for unigram, bigram, and trigram
3. Generating sentences from the trained model
4. Computing perplexity
5. Creating training data for n > 3, and repeating the above tasks

### Explore and Explain
The project experiments with n-gram models for n = [1,2,3..7] and provides an explanation for the best choice of n that generates more meaningful sentences.

## Sentiment Analysis using LSTM <a name="lstm"></a>

The third part of the project involves building a bidirectional LSTM network to train and inference sentiment analysis on the IMDB dataset. The project includes:

- Plotting Positive vs. Negative reviews count
- Cleaning the Reviews
- Splitting the dataset and Encoding Labels
- Using GloVe vectors for embedding
- Building and training the LSTM model using TensorFlow
- Plotting the performance metrics

## Dependencies
- Python
- TensorFlow
- Keras
- NumPy
- SciPy
- NLTK
- Matplotlib
- Pandas
