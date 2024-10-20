Project Description
In this project, we build a deep learning model for email classification to distinguish between spam and non-spam emails using natural language processing (NLP) techniques.

We preprocess the dataset by extracting email texts and labels, tokenizing sequences, and splitting the dataset into training, validation, and test sets.

We then train a Word2Vec model to generate word embeddings and construct a Long Short-Term Memory (LSTM) network using these embeddings for robust spam detection.

By the end of this project, you'll gain insights into spam email detection using deep learning and will have a functional model ready for deployment.

Key Steps Involved
Data Preprocessing:

Extract email texts and labels.
Split the dataset into training, validation, and test sets.
Tokenize and pad sequences for consistent input length.
Word2Vec Model:

Train a Word2Vec model to generate word embeddings for our dataset.
LSTM Model Construction:

Build and train an LSTM model for binary classification (spam or ham).
Model Evaluation:

Evaluate the model’s performance on the test set and generate a classification report.
Technologies and Libraries Used
Python
TensorFlow/Keras
Gensim (for Word2Vec)
scikit-learn (for data splitting and evaluation)
Dataset
The dataset used for this project contains labeled emails as spam or ham. It can be found in the Dataset.csv file.
