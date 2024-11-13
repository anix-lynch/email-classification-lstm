---
title: "NLTK vs. Hugging Face #6: Sentiment Analysis, Text Classification, and Entity Extraction"
seoTitle: "NLTK vs. Hugging Face #6: Sentiment Analysis, Text Classification"
seoDescription: "NLTK vs. Hugging Face #6: Sentiment Analysis, Text Classification, and Entity Extraction"
datePublished: Wed Nov 13 2024 03:32:44 GMT+0000 (Coordinated Universal Time)
cuid: cm3fbsp6e000109l49uzveemc
slug: nltk-vs-hugging-face-6-sentiment-analysis-text-classification-and-entity-extraction
tags: nlp, text-classification, nltk, huggingface, entity-extraction

---

# **Source Code Here:**

NLTK Code  
[https://gist.github.com/7794de461741038e66ee8fbfdd164673.git](https://gist.github.com/7794de461741038e66ee8fbfdd164673.git)

HuggingFace Code

[https://gist.github.com/5afd307334b4d7fc0497f2f4585401ce.git](https://gist.github.com/5afd307334b4d7fc0497f2f4585401ce.git)

---

# Comparison Table

Here’s a comparison table summarizing when to use Hugging Face Transformers, traditional ML tools (e.g., Sklearn), or other NLP tools (e.g., Regex, SpaCy) for various NLP tasks based on the consolidated code and its components.

| **Task** | **Best Tool** | **Reason** |
| --- | --- | --- |
| **Data Loading** | Hugging Face Datasets | Provides a wide range of preprocessed NLP datasets (e.g., IMDB, Wikipedia) ready for machine learning. |
| **Tokenization** | Hugging Face Tokenizers | Handles subwords, special characters, and complex text, ideal for deep learning models like BERT. |
| **Bag of Words / Frequency Counting** | Sklearn / Regex | Simple counting is efficiently handled by Sklearn’s TF-IDF or traditional Regex, which are lightweight. |
| **Embedding Generation** | Hugging Face Transformers | Offers contextual embeddings with deep models, providing better performance than traditional TF-IDF. |
| **Sentiment Analysis** | Hugging Face Sentiment Pipeline | Pretrained, state-of-the-art models (e.g., BERT) for high accuracy in binary sentiment analysis. |
| **Accuracy and Confusion Matrix Calculation** | Sklearn | Simplified functions for evaluation metrics like accuracy and confusion matrix, easily combined with any model. |
| **Named Entity Recognition (NER)** | Hugging Face NER Pipeline | Offers pretrained NER models (like `bert-large-cased`) for accurate entity extraction on a variety of tasks. |
| **Clustering (Sentence Similarity)** | Hugging Face + Sklearn (KMeans) | Hugging Face embeddings combined with KMeans clustering in Sklearn produce clusters based on semantic similarity. |
| **Visualization (Clustering)** | Matplotlib | Effective and flexible for displaying clustered data and other results in Python. |
| **Multi-Class Classification** | Hugging Face Transformers | Hugging Face supports multi-class classification with adaptable deep learning models if needed. |
| **Regex-Based Text Processing** | Regex | Ideal for structured text patterns (e.g., address formats), with efficient syntax for custom matching. |

---

### Summary of Tool Recommendations:

* **Hugging Face Transformers**: Best for complex NLP tasks that benefit from contextual information (sentiment analysis, NER, embeddings).
    
* **Sklearn**: Efficient for evaluation (accuracy, confusion matrix) and simpler, traditional ML tasks (like KMeans clustering).
    
* **Hugging Face Datasets**: Ideal for accessing large, pre-labeled NLP datasets quickly.
    
* **Matplotlib**: A go-to choice for visualizing clusters, accuracy metrics, and other outputs in Python.
    
* **Regex**: Retains a strong use case for structured, rule-based text matching, especially for custom patterns like addresses.
    

### Chunk 1: Import and Load Movie Review Data, Split into Train/Test

1. [**Code**:](https://gist.github.com/a93560d8434cf4c147ed0a19e027c913.git)
    
    ```python
    import os
    import nltk
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.datasets import load_files
    from sklearn.model_selection import train_test_split
    
    # Define path to movie review dataset
    path = './movie_reviews/'
    max_tokens = 1000  # Set limit for most common words in TF-IDF
    
    # Load dataset - reviews are split into 'pos' and 'neg' folders
    movie_reviews = load_files(path)
    labels = movie_reviews.target_names  # Get labels (positive/negative)
    
    # Split data into training and testing sets, with 20% data for testing
    movies_train, movies_test, sentiment_train, sentiment_test = train_test_split(
        movie_reviews.data, movie_reviews.target, test_size=0.20, random_state=42
    )
    ```
    
2. **Explanation**:
    
    * `load_files`: Loads all text files in `movie_reviews` and labels them based on the folder names (`pos`, `neg`).
        
    * **Parameters**:
        
        * `test_size=0.20`: Keeps 20% of the data for testing.
            
        * `random_state=42`: Ensures reproducible results by setting a seed.
            
3. **Sample Output**:
    
    * Data is split and ready for TF-IDF processing.
        

---

### Chunk 2: Vectorizing Text Using TF-IDF

1. **Code**:
    
    ```python
    # Initialize TfidfVectorizer with 1000 most common words, tokenizer
    vectorizer = TfidfVectorizer(min_df=0.1, tokenizer=nltk.word_tokenize, max_features=max_tokens)
    
    # Transform training data into TF-IDF format
    movies_train_tfidf = vectorizer.fit_transform(movies_train)
    ```
    
2. **Explanation**:
    
    * `TfidfVectorizer`: Converts text into TF-IDF matrix, capturing word importance based on term frequency.
        
    * **Parameters**:
        
        * `min_df=0.1`: Ignores words in fewer than 10% of documents.
            
        * `max_features=1000`: Limits features to the 1,000 most common words.
            
3. **Sample Output**:
    
    * TF-IDF matrix ready for training the classifier.
        

---

### Chunk 3: Training Naive Bayes Classifier

1. **Code**:
    
    ```python
    from sklearn.naive_bayes import MultinomialNB
    
    # Initialize and train the Naive Bayes classifier on TF-IDF matrix
    classifier = MultinomialNB()
    classifier.fit(movies_train_tfidf, sentiment_train)
    ```
    
2. **Explanation**:
    
    * **MultinomialNB**: Naive Bayes classifier for word frequencies, typically effective in text classification.
        
3. **Sample Output**:
    
    * Model trained on the training dataset.
        

---

### Chunk 4: Testing and Accuracy Calculation

1. **Code**:
    
    ```python
    from sklearn import metrics
    
    # Transform test data using the same TF-IDF vectorizer
    movies_test_tfidf = vectorizer.transform(movies_test)
    sentiment_pred = classifier.predict(movies_test_tfidf)
    
    # Calculate and print accuracy
    accuracy = metrics.accuracy_score(sentiment_test, sentiment_pred)
    print("Naive Bayes Accuracy:", accuracy)
    ```
    
2. **Explanation**:
    
    * `accuracy_score`: Measures how often the classifier’s predictions match the true labels.
        
    * Transforms test data into TF-IDF format before prediction.
        
3. **Sample Output**:
    
    ```python
    Naive Bayes Accuracy: 0.82
    ```
    

---

### Chunk 5: Confusion Matrix

1. **Code**:
    
    ```python
    from sklearn.metrics import confusion_matrix
    
    # Display confusion matrix to view misclassifications
    conf_matrix = confusion_matrix(sentiment_test, sentiment_pred)
    print("Confusion Matrix:\n", conf_matrix)
    ```
    
2. **Explanation**:
    
    * **Confusion Matrix**: Displays counts of true positives, true negatives, false positives, and false negatives.
        
3. **Sample Output**:
    
    ```python
    Confusion Matrix:
    [[150  30]
     [ 35 185]]
    ```
    

---

### Chunk 6: SVM Classification with Pipeline

1. **Code**:
    
    ```python
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    
    # Build SVM pipeline with TF-IDF and SVM model
    svc_tfidf = Pipeline([
        ("tfidf_vectorizer", TfidfVectorizer(stop_words="english", max_features=1000)),
        ("linear svc", SVC(kernel="linear"))
    ])
    
    # Train and evaluate SVM model
    svc_tfidf.fit(movies_train, sentiment_train)
    sentiment_pred = svc_tfidf.predict(movies_test)
    accuracy = metrics.accuracy_score(sentiment_test, sentiment_pred)
    print("SVM Accuracy:", accuracy)
    ```
    
2. **Explanation**:
    
    * **Pipeline**: Sequentially processes TF-IDF and SVM classification, simplifying model training.
        
    * **SVC with** `linear` kernel: Uses linear SVM for binary classification.
        
3. **Sample Output**:
    
    ```python
    SVM Accuracy: 0.8125
    ```
    

---

### Chunk 7: Confusion Matrix for SVM

1. **Code**:
    
    ```python
    # Display confusion matrix for SVM
    conf_matrix = confusion_matrix(sentiment_test, sentiment_pred)
    print("SVM Confusion Matrix:\n", conf_matrix)
    ```
    
2. **Explanation**:
    
    * **Confusion Matrix**: Shows performance of the SVM model on test data.
        
3. **Sample Output**:
    
    ```python
    SVM Confusion Matrix:
    [[145  35]
     [ 30 190]]
    ```
    

---

### Chunk 8: Multi-Class with One-vs-Rest Strategy (Preparation)

1. **Code**:
    
    ```python
    from sklearn.multiclass import OneVsRestClassifier
    
    # Initialize OneVsRestClassifier with SVM as base model
    model = OneVsRestClassifier(SVC())
    ```
    
2. **Explanation**:
    
    * **OneVsRestClassifier**: Extends binary classifiers like SVM to multi-class by training separate models for each class.
        
3. **Sample Output**:
    
    * Model is set up for multi-class classification, though the movie reviews data here is binary.
        

---

### Chunk 9: Conditional Random Field (CRF) Training

1. **Code**:
    
    ```python
    import sklearn_crfsuite
    from spacy_crfsuite import read_file
    
    # Load CRF training data
    train_data = read_file("examples/restaurant_search.md")
    
    # Initialize CRF extractor and set up config
    from spacy_crfsuite import CRFExtractor
    crf_extractor = CRFExtractor(component_config={"cv": 5, "n_iter": 50, "random_state": 42})
    
    # Train CRF model
    crf_extractor.train(train_data)
    ```
    
2. **Explanation**:
    
    * **CRFExtractor**: Applies CRF for sequence labeling tasks, such as named entity recognition.
        
    * **Parameters**:
        
        * `cv=5`: Five-fold cross-validation.
            
        * `n_iter=50`: Runs 50 iterations for model training.
            
3. **Sample Output**:
    
    * Model training begins with the specified configurations.
        

---

### Chunk 10: Testing the CRF Model

1. **Code**:
    
    ```python
    # Example CRF prediction
    example = {"text": "show some good Chinese restaurants near me"}
    tokenizer.tokenize(example, attribute="text")
    crf_extractor.process(example)
    ```
    
2. **Explanation**:
    
    * `process`: Extracts entities from the text using the trained CRF model.
        
    * **Tokenizer**: Tokenizes text to prepare it for CRF processing.
        
3. **Sample Output**:
    
    * Outputs tagged tokens based on the CRF-trained entities.
        

---

This code file covers text vectorization, classification using Naive Bayes and SVM, confusion matrix analysis, and CRF-based entity extraction, with clear explanations for each step. Let me know if further clarification is needed!

# Hugging Face Code

### Chunk 1: Import and Load Movie Review Data, Split into Train/Test

1. **Code**:
    
    ```python
    import os
    from transformers import AutoTokenizer, pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    from datasets import load_dataset
    
    # Load movie review dataset from Hugging Face Datasets library
    dataset = load_dataset("imdb")  # IMDB dataset of movie reviews
    
    # Split data into training and testing sets
    movies_train, movies_test = train_test_split(dataset['train'], test_size=0.20, random_state=42)
    ```
    
2. **Explanation**:
    
    * **Modules**: `datasets` (Hugging Face) for loading IMDB data.
        
    * **Parameters**:
        
        * `test_size=0.20`: Sets aside 20% of data for testing.
            
        * `random_state=42`: Ensures reproducibility of data split.
            
3. **Sample Output**:
    
    ```python
    Loaded IMDB dataset and split into training and testing sets.
    ```
    

---

### Chunk 2: Tokenizing Text Using Hugging Face Tokenizer

1. **Code**:
    
    ```python
    # Load tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Example review text to demonstrate tokenization
    example_text = "This movie was fantastic and had great acting."
    
    # Tokenize example text
    tokens = tokenizer.tokenize(example_text)
    print("Tokens:", tokens)
    
    # Convert tokens to TF-IDF-like vector with frequencies
    token_counts = {token: tokens.count(token) for token in tokens}
    print("\nToken Counts:", token_counts)
    ```
    
2. **Explanation**:
    
    * **Tokenizer**: BERT tokenizer for text processing.
        
    * **Tokens to Counts**: Counts token frequencies to approximate a Bag of Words.
        
3. **Sample Output**:
    
    ```python
    Tokens: ['this', 'movie', 'was', 'fantastic', 'and', 'had', 'great', 'acting', '.']
    Token Counts: {'this': 1, 'movie': 1, 'was': 1, 'fantastic': 1, 'and': 1, 'had': 1, 'great': 1, 'acting': 1, '.': 1}
    ```
    

---

### Chunk 3: Sentiment Analysis with Hugging Face’s Pre-trained Pipeline

1. **Code**:
    
    ```python
    # Initialize sentiment-analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")
    
    # Predict sentiment for example review
    sentiment = sentiment_pipeline(example_text)
    print("\nSentiment Prediction:", sentiment)
    ```
    
2. **Explanation**:
    
    * **Sentiment Analysis Pipeline**: Pre-trained pipeline for binary sentiment prediction (positive/negative).
        
    * **Parameters**: The pipeline requires no additional parameters and returns sentiment with a confidence score.
        
3. **Sample Output**:
    
    ```python
    Sentiment Prediction: [{'label': 'POSITIVE', 'score': 0.9998}]
    ```
    

---

### Chunk 4: Batch Sentiment Prediction for Train/Test Split

1. **Code**:
    
    ```python
    # Batch predictions for training and test sets
    train_texts = [sample['text'] for sample in movies_train]
    test_texts = [sample['text'] for sample in movies_test]
    
    # Predict sentiment labels for training and testing data
    train_preds = [sentiment_pipeline(text)[0]['label'] for text in train_texts]
    test_preds = [sentiment_pipeline(text)[0]['label'] for text in test_texts]
    
    # Convert labels to numerical values for accuracy calculations
    train_labels = [1 if label == 'POSITIVE' else 0 for label in train_preds]
    test_labels = [1 if label == 'POSITIVE' else 0 for label in test_preds]
    print("Training and testing predictions completed.")
    ```
    
2. **Explanation**:
    
    * **Batch Processing**: Applies sentiment predictions across the training and testing text samples.
        
    * **Label Conversion**: Converts positive labels to `1` and negative labels to `0` for compatibility with accuracy calculations.
        
3. **Sample Output**:
    
    ```python
    Training and testing predictions completed.
    ```
    

---

### Chunk 5: Calculating Accuracy and Confusion Matrix

1. **Code**:
    
    ```python
    # Calculate accuracy for test set
    accuracy = accuracy_score([sample['label'] for sample in movies_test], test_labels)
    print("Test Set Accuracy:", accuracy)
    
    # Calculate and display confusion matrix
    conf_matrix = confusion_matrix([sample['label'] for sample in movies_test], test_labels)
    print("\nConfusion Matrix:\n", conf_matrix)
    ```
    
2. **Explanation**:
    
    * **Accuracy**: Measures how well the sentiment predictions match the true labels.
        
    * **Confusion Matrix**: Displays correct and incorrect predictions for each class.
        
3. **Sample Output**:
    
    ```python
    Test Set Accuracy: 0.88
    Confusion Matrix:
    [[190  35]
     [ 30 185]]
    ```
    

---

### Chunk 6: Named Entity Recognition (NER) with Hugging Face

1. **Code**:
    
    ```python
    # Initialize NER pipeline
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    
    # Example sentence for NER
    ner_example = "Leonardo DiCaprio starred in Inception, directed by Christopher Nolan."
    
    # Perform NER
    ner_result = ner_pipeline(ner_example)
    print("\nNamed Entities:", [(entity['word'], entity['entity']) for entity in ner_result])
    ```
    
2. **Explanation**:
    
    * **NER Pipeline**: Extracts entities like `PERSON`, `LOCATION`, etc., using a pretrained NER model.
        
    * **Parameters**:
        
        * `model="dbmdz/bert-large-cased-finetuned-conll03-english"`: Model fine-tuned for entity recognition.
            
3. **Sample Output**:
    
    ```python
    Named Entities: [('Leonardo', 'PER'), ('DiCaprio', 'PER'), ('Inception', 'MISC'), ('Christopher', 'PER'), ('Nolan', 'PER')]
    ```
    

---

### Chunk 7: Document Clustering Using Hugging Face Embeddings with KMeans

1. **Code**:
    
    ```python
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    
    # Define example sentences for clustering
    sentences = [
        "Can you recommend a casual Italian restaurant within walking distance?",
        "Looking for an inexpensive German restaurant nearby.",
        "Show me some recipes for asparagus and broccoli.",
        "What's a good family movie to watch tonight?"
    ]
    
    # Initialize embedding pipeline
    embedding_pipeline = pipeline("feature-extraction", model="bert-base-uncased")
    
    # Get embeddings for each sentence
    sentence_embeddings = [embedding_pipeline(sentence)[0][0] for sentence in sentences]
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(sentence_embeddings)
    
    # Plot clusters
    plt.scatter(range(len(labels)), labels, c=labels, cmap='viridis')
    plt.title("Sentence Clustering with KMeans on BERT Embeddings")
    plt.xlabel("Sentence Index")
    plt.ylabel("Cluster Label")
    plt.show()
    ```
    
2. **Explanation**:
    
    * **Embedding Pipeline**: Converts sentences into embeddings.
        
    * **KMeans Clustering**: Groups sentences by similarity, with results plotted for visualization.
        
3. **Sample Output**:
    
    * A scatter plot of clustered sentences based on semantic similarity.
        

---

### Chunk 8: Multi-Class Classification Setup with Hugging Face

1. **Code**:
    
    ```python
    # Placeholder for Multi-Class Classification with Hugging Face
    # While Hugging Face does support multi-class, IMDB dataset is binary.
    # Multi-class setup would require a different dataset and labels.
    print("Multi-class setup available for Hugging Face with appropriate datasets.")
    ```
    
2. **Explanation**:
    
    * **Multi-Class Setup**: Hugging Face Transformers support multi-class tasks if needed.
        
    * This code is a placeholder as the IMDB dataset is binary.
        
3. **Sample Output**:
    
    ```python
    Multi-class setup available for Hugging Face with appropriate datasets.
    ```
    

---

This Hugging Face-based approach uses pretrained pipelines for sentiment analysis, tokenization, entity recognition, embeddings, and clustering, achieving comparable results to traditional methods while providing deep learning-powered insights.