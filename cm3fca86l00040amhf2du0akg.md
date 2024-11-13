---
title: "Hugging Face #6: Topic Modeling, Sentence Embeddings, and Dynamic Clustering"
seoTitle: "Hugging Face #6: Topic Modeling, Sentence Embeddings"
seoDescription: "Hugging Face #6: Topic Modeling, Sentence Embeddings, and Dynamic Clustering"
datePublished: Wed Nov 13 2024 03:46:22 GMT+0000 (Coordinated Universal Time)
cuid: cm3fca86l00040amhf2du0akg
slug: hugging-face-6-topic-modeling-sentence-embeddings-and-dynamic-clustering
tags: ai, nlp, huggingface, topic-modeling, dynamic-clustering

---

# **Source Code Here:**

  
HuggingFace Code

[https://gist.github.com/0f88f53525d0fd992b627d62a0da13b2.git](https://gist.github.com/0f88f53525d0fd992b627d62a0da13b2.git)

---

### Chunk 1: Import Libraries and Load Data

1. [**Code**:](https://gist.github.com/a93560d8434cf4c147ed0a19e027c913.git)
    
    ```python
    !pip install bertopic
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    
    # Load text data from the 20 Newsgroups dataset
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    ```
    
2. **Explanation**:
    
    * **Libraries**:
        
        * **BERTopic**: For topic modeling.
            
        * **UMAP** (Uniform Manifold Approximation and Projection): Reduces embedding dimensionality.
            
        * **HDBSCAN**: Clustering algorithm used by BERTopic to form topic clusters.
            
    * **Dataset**: 20 Newsgroups text data.
        
3. **Sample Output**:
    
    ```python
    Loaded 20 Newsgroups dataset, containing documents on different topics.
    ```
    

---

### Chunk 2: Generating Embeddings

1. **Code**:
    
    ```python
    # Use SentenceTransformer for creating embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = embedding_model.encode(docs, show_progress_bar=True)
    ```
    
2. **Explanation**:
    
    * `SentenceTransformer`: Generates embeddings (dense vectors) from sentences using a pre-trained model.
        
    * **Parameters**:
        
        * `show_progress_bar=True`: Displays a progress bar for embedding generation.
            
3. **Sample Output**:
    
    ```python
    Corpus embeddings generated for all documents.
    ```
    

---

### Chunk 3: (Optional) View Embeddings

1. **Code**:
    
    ```python
    # View the shape and structure of the generated embeddings
    corpus_embeddings.view()
    ```
    
2. **Explanation**:
    
    * Allows inspection of the embeddings to understand dimensions and structure.
        
3. **Sample Output**:
    
    ```python
    (1000, 384) # Sample output showing the embedding dimensions (example)
    ```
    

---

### Chunk 4: Configure CountVectorizer

1. **Code**:
    
    ```python
    from sklearn.feature_extraction.text import CountVectorizer
    # Vectorizer for term frequency counts, useful for topic representation
    vectorizer_model = CountVectorizer(stop_words="english", max_df=0.95, min_df=0.01)
    ```
    
2. **Explanation**:
    
    * `CountVectorizer`: Prepares a word frequency matrix for use in BERTopic.
        
    * **Parameters**:
        
        * `max_df=0.95`: Excludes terms appearing in more than 95% of documents.
            
        * `min_df=0.01`: Includes terms present in at least 1% of documents.
            
3. **Sample Output**:
    
    ```python
    CountVectorizer configured for text frequency analysis.
    ```
    

---

### Chunk 5: Set Parameters for HDBSCAN and UMAP

1. **Code**:
    
    ```python
    # Configure clustering and dimensionality reduction models
    hdbscan_model = HDBSCAN(min_cluster_size=30, metric='euclidean', prediction_data=True)
    umap_model = UMAP(n_neighbors=15, n_components=10, metric='cosine', low_memory=False)
    ```
    
2. **Explanation**:
    
    * **HDBSCAN**:
        
        * **min\_cluster\_size=30**: Minimum size for clusters, influencing topic sizes.
            
        * **metric='euclidean'**: Distance metric used for clustering.
            
    * **UMAP**:
        
        * **n\_neighbors=15**: Balances local and global structure in data.
            
        * **n\_components=10**: Reduces to 10 dimensions for visual clarity.
            
3. **Sample Output**:
    
    ```python
    HDBSCAN and UMAP models configured for clustering and dimensionality reduction.
    ```
    

---

### Chunk 6: Train BERTopic Model

1. **Code**:
    
    ```python
    # Initialize and train BERTopic
    model = BERTopic(
        n_gram_range=(1, 3),
        vectorizer_model=vectorizer_model,
        nr_topics=40,
        top_n_words=10,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=30,
        calculate_probabilities=True
    ).fit(docs, corpus_embeddings)
    ```
    
2. **Explanation**:
    
    * **BERTopic Parameters**:
        
        * `n_gram_range=(1, 3)`: Considers 1 to 3-word combinations as terms.
            
        * `nr_topics=40`: Sets the number of topics to extract.
            
        * `top_n_words=10`: Limits the top 10 words shown for each topic.
            
        * `calculate_probabilities=True`: Enables probability distribution over topics for documents.
            
3. **Sample Output**:
    
    ```python
    BERTopic model trained with specified clustering and vectorization parameters.
    ```
    

---

### Chunk 7: Get and Visualize Topic Frequencies

1. **Code**:
    
    ```python
    # Retrieve topics and probabilities for each document
    topics, probabilities = model.transform(docs, corpus_embeddings)
    df_topic_freq = model.get_topic_freq()
    print(df_topic_freq)
    
    # Visualize topic frequencies in a bar chart
    topics_count = len(df_topic_freq) - 1
    model.visualize_barchart(top_n_topics=topics_count)
    ```
    
2. **Explanation**:
    
    * `get_topic_freq`: Shows frequency of each topic across all documents.
        
    * `visualize_barchart`: Displays a bar chart of topic frequencies.
        
3. **Sample Output**:
    
    * DataFrame with topics and frequencies, bar chart showing distribution of topics.
        

---

### Chunk 8: View Topic Information

1. **Code**:
    
    ```python
    # Get topic information
    topic_info = model.get_topic_info()
    print(topic_info)
    ```
    
2. **Explanation**:
    
    * `get_topic_info`: Retrieves detailed information about each topic, including representative words.
        
3. **Sample Output**:
    
    ```python
    Topic information with top words for each topic.
    ```
    

---

### Chunk 9: Document-Level Visualization

1. **Code**:
    
    ```python
    # Visualize document clusters based on topics
    fig = model.visualize_documents(
        docs, embeddings=corpus_embeddings, sample=0.6,
        topics=[0, 1, 2, 3, 4, 5, 6], hide_annotations=False,
        hide_document_hover=True
    )
    fig.write_image("./clusters.svg")
    ```
    
2. **Explanation**:
    
    * `visualize_documents`: Creates a scatter plot with documents grouped by topics.
        
    * **Parameters**:
        
        * `sample=0.6`: Uses a 60% sample of documents for visualization.
            
        * `topics=[0, 1, 2, ...]`: Visualizes specified topics.
            
3. **Sample Output**:
    
    * Cluster plot saved as `clusters.svg`.
        

---

### Chunk 10: Topic Bar Chart Visualization

1. **Code**:
    
    ```python
    # Create a bar chart for topic visualization
    fig2 = model.visualize_barchart()
    fig2.write_image("./barchart.svg")
    ```
    
2. **Explanation**:
    
    * `visualize_barchart`: Provides a bar chart showing each topic’s frequency.
        
3. **Sample Output**:
    
    * Bar chart saved as `barchart.svg`.
        

---

### Chunk 11: Predict Topics for New Documents

1. **Code**:
    
    ```python
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    new_docs = ["I'm looking for a new graphics card", "When is the next NASA launch?"]
    embeddings = sentence_model.encode(new_docs)
    topics, probs = model.transform(new_docs, embeddings)
    print("Topics:", topics)
    print("Probabilities:", probs)
    ```
    
2. **Explanation**:
    
    * **Predict New Topics**: Encodes new documents and predicts topic associations.
        
    * **Parameters**:
        
        * `new_docs`: List of new document texts to analyze.
            
3. **Sample Output**:
    
    ```python
    Topics: [topic_number, ...]
    Probabilities: [probability_distribution]
    ```
    

---

### Summary of Code Workflow

* **BERTopic** is trained to find clusters (topics) in text documents, which are then visualized and examined.
    
* **Embeddings** are generated by `SentenceTransformer`, which helps in topic assignments.
    
* **Visualization** tools give insight into document-topic relationships and the most frequent topics.
    

This code provides an entire topic modeling pipeline using BERTopic, from loading text data to predicting topics for new data.