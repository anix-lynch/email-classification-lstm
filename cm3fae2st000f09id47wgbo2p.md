---
title: "NLTK, SpaCy, Gensim VS Hugging Face 4#: POS Tagging, Parsing, TF-IDF, One-Hot Encoding, and Word2Vec"
seoTitle: "NLTK, SpaCy, Gensim VS Hugging Face"
seoDescription: "NLTK, SpaCy, Gensim VS Hugging Face 4#: POS Tagging, Parsing, TF-IDF, One-Hot Encoding, and Word2Vec"
datePublished: Wed Nov 13 2024 02:53:22 GMT+0000 (Coordinated Universal Time)
cuid: cm3fae2st000f09id47wgbo2p
slug: nltk-spacy-gensim-vs-hugging-face-4-pos-tagging-parsing-tf-idf-one-hot-encoding-and-word2vec
tags: nlp, nltk, spacy, huggingface, gensim

---

NLTK Code  
[https://gist.github.com/c969899618e37ba00be355eb676c8c39.git](https://gist.github.com/c969899618e37ba00be355eb676c8c39.git)

HuggingFace Code

[https://gist.github.com/4f9b5a23dcde12a45f87ea5254013aed.git](https://gist.github.com/4f9b5a23dcde12a45f87ea5254013aed.git)

---

# Comparison Table

Here’s a comparison table summarizing which tool (Hugging Face Transformers, SpaCy, or traditional methods like NLTK) is best suited for each NLP task based on functionality, ease of use, and task-specific strengths:

| **Task** | **Best Tool** | **Reason** |
| --- | --- | --- |
| **POS Tagging** | Hugging Face Transformers | Hugging Face’s pretrained pipelines are easy to set up and provide accurate POS tagging with deep learning models. |
| **Dependency Parsing** | SpaCy | SpaCy’s dependency parsing is robust and fast, making it ideal for syntactic analysis; Hugging Face doesn’t directly support dependency parsing. |
| **Bag of Words (BoW)** | Traditional (NLTK/Sklearn) | Simple word counting is straightforward in NLTK or Sklearn’s CountVectorizer. Hugging Face is better suited for embeddings rather than basic BoW. |
| **TF-IDF** | Sklearn | Sklearn’s `TfidfVectorizer` is specialized for TF-IDF and more flexible for text corpus analysis, while Hugging Face is better for embeddings. |
| **Embedding-Based Features** | Hugging Face Transformers | Hugging Face’s `feature-extraction` pipeline provides deep, contextualized embeddings, making it more advanced than TF-IDF for feature extraction. |
| **Integer and One-Hot Encoding** | Traditional (Sklearn) | Sklearn’s encoders (`LabelEncoder`, `OneHotEncoder`) are ideal for simple integer and one-hot encoding; Hugging Face’s transformers are too advanced for this. |
| **Word Similarity (Cosine Similarity)** | Hugging Face Transformers | Hugging Face embeddings capture word similarity effectively with contextual embeddings, providing richer comparisons than simple word vectors. |
| **Document Clustering** | Hugging Face + Sklearn (KMeans) | Hugging Face’s embeddings combined with Sklearn’s KMeans clustering capture topic clusters effectively due to contextual word meanings. |
| **Named Entity Recognition (NER)** | Hugging Face Transformers | Hugging Face’s pretrained NER models are state-of-the-art, easy to use, and outperform traditional rule-based approaches in accuracy. |
| **Tokenization with Special Characters** | Hugging Face Transformers | Hugging Face’s tokenizers handle complex tokens, special characters, and subword splits, making it ideal for modern NLP tasks. |
| **Dependency Tree Visualization** | SpaCy | SpaCy provides built-in dependency visualization (`displacy`), making it easier to visualize syntactic structures than Hugging Face, which lacks direct support. |
| **Basic Frequency Analysis** | Traditional (NLTK) | Simple word frequency analysis is easier and faster in NLTK than in Hugging Face, which is optimized for more complex, contextual embeddings. |

---

### Summary:

* **Hugging Face Transformers**: Best for advanced NLP tasks that benefit from contextual embeddings, such as POS tagging, word similarity, clustering, and NER.
    
* **SpaCy**: Preferred for syntactic analysis tasks like dependency parsing and visualization due to its optimized dependency trees and visual tools.
    
* **Traditional (NLTK/Sklearn)**: Ideal for simple frequency-based tasks, TF-IDF, and integer or one-hot encoding, where deep learning models are unnecessary.
    

# NLTK Code

### Chunk 1: POS Tagging with NLTK

1. **Code**:
    
    ```python
    # Import necessary libraries
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import movie_reviews
    
    # Load sentences from movie reviews and select one for demonstration
    example_sentences = movie_reviews.sents()
    example_sentence = example_sentences[0]
    
    # Perform POS tagging
    pos_tags = nltk.pos_tag(example_sentence)
    print("POS Tags:", pos_tags)
    ```
    
2. **Explanation**:
    
    * **Module**: `nltk.pos_tag` is used for part-of-speech (POS) tagging, identifying grammatical roles (like nouns and verbs) for each word.
        
    * **Parameter**:
        
        * `example_sentence`: A list of words representing a single sentence from the movie reviews dataset.
            
3. **Sample Output**:
    
    ```python
    POS Tags: [('plot', 'NN'), (':', ':'), ('two', 'CD'), ('teen', 'JJ'), ('couples', 'NNS'), ('go', 'VBP'), ('to', 'TO'), ('a', 'DT'), ('church', 'NN'), ('party', 'NN'), (',', ','), ('drink', 'NN'), ('and', 'CC'), ('then', 'RB'), ('drive', 'VB'), ('.', '.')]
    ```
    

---

### Chunk 2: Dependency Parsing with SpaCy

1. **Code**:
    
    ```python
    import spacy
    
    # Sample text for dependency parsing
    text = "plot: two teen couples go to a church party, drink and then drive."
    
    # Load the small English model in SpaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    # Print token information including POS tags and dependency relations
    for token in doc:
        print(f"Token: {token.text}, POS Tag: {token.tag_}, Head: {token.head.text}, Dependency: {token.dep_}")
    ```
    
2. **Explanation**:
    
    * **Module**: `spacy.load("en_core_web_sm")` loads a pretrained small English model that includes POS tagging, parsing, and NER.
        
    * **Class**: `doc` is a processed object where each `token` has attributes like `text`, `tag_` (POS tag), `head` (head word), and `dep_` (dependency type).
        
3. **Sample Output**:
    
    ```python
    Token: plot, POS Tag: NN, Head: plot, Dependency: ROOT
    Token: :, POS Tag: :, Head: plot, Dependency: punct
    Token: two, POS Tag: CD, Head: couples, Dependency: nummod
    ...
    ```
    

---

### Chunk 3: Count Bag of Words (BoW)

1. **Code**:
    
    ```python
    # Function to calculate Bag of Words
    def document_features(document):
        features = {}
        for word in word_features:
            features[word] = 0
            for doc_word in document:
                if word == doc_word:
                    features[word] += 1
        return features
    ```
    
2. **Explanation**:
    
    * **Function**: `document_features` counts occurrences of each word in `word_features` for a given document.
        
    * **Parameter**:
        
        * `document`: List of words representing a document.
            
        * `word_features`: List of important words to track in each document.
            
    * **Usage**: Creates a simple word-count-based feature representation (BoW).
        
3. **Sample Output**:
    
    * A dictionary with words as keys and their counts in the document as values (e.g., `{ 'plot': 1, 'party': 2 }`).
        

---

### Chunk 4: TF-IDF Vectorization

1. **Code**:
    
    ```python
    import string
    import os
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Maximum number of tokens (words) to consider
    max_tokens = 200
    
    # Tokenizer function to split text into words
    def tokenize(text):
        return nltk.word_tokenize(text)
    
    # Path to movie reviews (adjust path as needed)
    path = './movie_reviews/'
    token_dict = {}
    
    # Read files and remove punctuation
    for dirpath, dirs, files in os.walk(path):
        for f in files:
            fname = os.path.join(dirpath, f)
            with open(fname) as review:
                text = review.read()
                token_dict[f] = text.lower().translate(str.maketrans('', '', string.punctuation))
    
    # TF-IDF Vectorizer with a maximum of max_tokens words
    tfIdfVectorizer = TfidfVectorizer(input="content", use_idf=True, tokenizer=tokenize, max_features=max_tokens, stop_words='english')
    tfIdf = tfIdfVectorizer.fit_transform(token_dict.values())
    
    # Convert to DataFrame
    tfidf_tokens = tfIdfVectorizer.get_feature_names_out()
    final_vectors = pd.DataFrame(data=tfIdf.toarray(), columns=tfidf_tokens)
    print(final_vectors.head())
    ```
    
2. **Explanation**:
    
    * **Class**: `TfidfVectorizer` transforms text into TF-IDF vectors, representing text by word importance.
        
    * **Parameters**:
        
        * `max_features=max_tokens`: Limits to the top `max_tokens` most important words.
            
        * `stop_words='english'`: Removes common English stop words.
            
3. **Sample Output**:
    
    * A DataFrame showing TF-IDF values for each word in each document.
        

---

### Chunk 5: Integer and One-Hot Encoding

1. **Code**:
    
    ```python
    from numpy import array
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    
    # List of words from the first movie review
    data = movie_reviews.words(movie_reviews.fileids()[0])[:50]  # Only first 50 words for example
    
    # Integer encode the words
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data)
    print("Integer Encoded:", integer_encoded)
    
    # One-hot encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print("One-Hot Encoded:", onehot_encoded[0])
    
    # Decode the first one-hot vector back to the original word
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    print("Decoded Word:", inverted[0])
    ```
    
2. **Explanation**:
    
    * **Classes**: `LabelEncoder` encodes words as integers; `OneHotEncoder` converts integer encoding to one-hot encoding.
        
    * **Parameters**:
        
        * `sparse=False` in `OneHotEncoder` ensures the output is dense, not sparse.
            
3. **Sample Output**:
    
    ```python
    Integer Encoded: [integer values]
    One-Hot Encoded: [binary vector]
    Decoded Word: ['plot']
    ```
    

---

### Chunk 6: Finding Similar Words with Word2Vec

1. **Code**:
    
    ```python
    import gensim
    from gensim.models import Word2Vec
    from nltk.corpus import movie_reviews
    
    # Prepare documents for Word2Vec
    documents = [list(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids()]
    
    # Train Word2Vec model
    model = Word2Vec(documents, min_count=5)
    
    # Find words similar to 'movie'
    similar_words = model.wv.most_similar(positive=['movie'], topn=5)
    print("Words similar to 'movie':", similar_words)
    ```
    
2. **Explanation**:
    
    * **Class**: `Word2Vec` learns word embeddings, which are dense representations of words capturing semantic meaning.
        
    * **Parameters**:
        
        * `min_count=5`: Only includes words that appear at least 5 times.
            
        * `topn=5`: Returns the top 5 most similar words.
            
3. **Sample Output**:
    
    ```python
    Words similar to 'movie': [('film', 0.85), ('story', 0.76), ('plot', 0.75), ('character', 0.74), ('director', 0.73)]
    ```
    

---

# Hugging Face Code

### Chunk 1: POS Tagging with Hugging Face

1. **Code**:
    
    ```python
    # Import Hugging Face pipeline for POS tagging
    from transformers import pipeline
    
    # Initialize the POS tagging pipeline
    pos_pipeline = pipeline("token-classification", model="vblagoje/bert-english-uncased-finetuned-pos")
    
    # Define example sentence
    example_sentence = "plot: two teen couples go to a church party, drink and then drive."
    
    # Perform POS tagging
    pos_tags = pos_pipeline(example_sentence)
    print("POS Tags:", pos_tags)
    ```
    
2. **Explanation**:
    
    * **Pipeline**: Hugging Face's `pipeline` function sets up a pretrained POS tagging model, which identifies parts of speech for each word.
        
    * **Parameter**:
        
        * `model="vblagoje/bert-english-uncased-finetuned-pos"` specifies the model fine-tuned for POS tagging.
            
3. **Sample Output**:
    
    ```python
    POS Tags: [{'word': 'plot', 'entity': 'NOUN'}, {'word': ':', 'entity': 'PUNCT'}, ...]
    ```
    

---

### Chunk 2: Dependency Parsing (Using SpaCy, as Hugging Face doesn’t directly support parsing)

1. **Code**:
    
    ```python
    import spacy
    
    # Load SpaCy small English model
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("plot: two teen couples go to a church party, drink and then drive.")
    
    # Print token, POS tag, and dependency info
    print("\nDependency Parsing:")
    for token in doc:
        print(f"Token: {token.text}, POS Tag: {token.tag_}, Head: {token.head.text}, Dependency: {token.dep_}")
    ```
    
2. **Explanation**:
    
    * **SpaCy Dependency Parsing**: SpaCy provides dependency parsing as part of its NLP pipeline, analyzing grammatical roles and relationships between words.
        
    * **Parameters**:
        
        * `en_core_web_sm`: SpaCy's small English model includes POS tagging and dependency parsing.
            
3. **Sample Output**:
    
    ```python
    Token: plot, POS Tag: NN, Head: plot, Dependency: ROOT
    Token: :, POS Tag: :, Head: plot, Dependency: punct
    ...
    ```
    

---

### Chunk 3: Bag of Words Representation with Hugging Face Tokenizer

1. **Code**:
    
    ```python
    from transformers import AutoTokenizer
    from collections import Counter
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Define example text
    example_text = "plot: two teen couples go to a church party, drink and then drive."
    
    # Tokenize and count each token for BoW representation
    tokens = tokenizer.tokenize(example_text)
    token_counts = Counter(tokens)
    print("\nBag of Words:", token_counts)
    ```
    
2. **Explanation**:
    
    * **Tokenizer**: Converts text into tokens using a BERT tokenizer, generating tokens that represent words or subwords.
        
    * **Counter**: Counts occurrences of each token, mimicking a basic Bag of Words (BoW) model.
        
3. **Sample Output**:
    
    ```python
    Bag of Words: Counter({'plot': 1, ':': 1, 'two': 1, 'teen': 1, 'couples': 1, 'go': 1, ...})
    ```
    

---

### Chunk 4: TF-IDF Alternative with Hugging Face Embeddings

1. **Code**:
    
    ```python
    from transformers import pipeline
    import pandas as pd
    
    # Load Hugging Face pipeline for feature extraction
    embedding_pipeline = pipeline("feature-extraction", model="bert-base-uncased")
    
    # Sample text data
    documents = ["plot: two teen couples go to a church party, drink and then drive.",
                 "this movie was about a young couple who fell in love unexpectedly."]
    
    # Extract embeddings for each document and convert to a DataFrame
    embeddings = [embedding_pipeline(doc)[0] for doc in documents]  # [0] selects only the first layer of embeddings
    embedding_df = pd.DataFrame([embedding[0] for embedding in embeddings])  # Use first token for simplicity
    print("\nEmbeddings (first document):")
    print(embedding_df.head())
    ```
    
2. **Explanation**:
    
    * **Feature Extraction**: Converts documents into embedding vectors, which capture the contextual meaning of each word/token.
        
    * **Parameter**:
        
        * `model="bert-base-uncased"`: Uses BERT to generate embeddings based on context.
            
3. **Sample Output**:
    
    ```python
    Embeddings (first document):
          0         1         2  ...       765       766       767
    0  0.6532   0.2481   0.9732  ...   -0.8761   0.2432   0.7651
    ```
    

---

### Chunk 5: One-Hot Encoding with Hugging Face Tokenizer

1. **Code**:
    
    ```python
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    
    # Tokenize example text
    tokens = tokenizer.tokenize(example_text)
    
    # Convert tokens to unique integer IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # One-hot encode the token IDs
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = np.array(token_ids).reshape(-1, 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print("\nOne-Hot Encoded Vector for First Token:", onehot_encoded[0])
    ```
    
2. **Explanation**:
    
    * **OneHotEncoder**: Converts integer IDs of tokens into binary vectors representing each token uniquely.
        
    * **Parameter**:
        
        * `sparse=False`: Ensures dense array output.
            
3. **Sample Output**:
    
    ```python
    One-Hot Encoded Vector for First Token: [0. 0. 1. 0. ...]
    ```
    

---

### Chunk 6: Word Embeddings and Similarity with Hugging Face

1. **Code**:
    
    ```python
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Define two example words for comparison
    word1, word2 = "movie", "film"
    
    # Get embeddings for each word
    embedding1 = embedding_pipeline(word1)[0][0]  # Embedding for first token of word1
    embedding2 = embedding_pipeline(word2)[0][0]  # Embedding for first token of word2
    
    # Compute cosine similarity
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    print(f"\nCosine Similarity between '{word1}' and '{word2}':", similarity)
    ```
    
2. **Explanation**:
    
    * **Cosine Similarity**: Measures how similar two embedding vectors are in terms of direction, with 1 being identical and -1 being opposite.
        
    * **Parameters**:
        
        * `[0][0]` selects the embedding vector for the first token in the word.
            
3. **Sample Output**:
    
    ```python
    Cosine Similarity between 'movie' and 'film': 0.89
    ```
    

---

### Chunk 7: Document Clustering Using Embeddings with KMeans

1. **Code**:
    
    ```python
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    
    # Generate embeddings for clustering
    document_embeddings = [embedding_pipeline(doc)[0][0] for doc in documents]  # Embedding for each document
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(document_embeddings)
    
    # Plot clusters
    plt.scatter(range(len(labels)), labels, c=labels, cmap='viridis')
    plt.title("Document Clustering with KMeans on BERT Embeddings")
    plt.xlabel("Document Index")
    plt.ylabel("Cluster Label")
    plt.show()
    ```
    
2. **Explanation**:
    
    * **KMeans Clustering**: Groups documents into clusters based on their embeddings, finding similarities in topics or themes.
        
    * **Parameters**:
        
        * `n_clusters=2`: Specifies the number of clusters for grouping.
            
3. **Sample Output**:
    
    * A scatter plot showing clusters, where each point represents a document’s cluster assignment.
        

---