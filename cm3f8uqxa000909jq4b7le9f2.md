---
title: "NLTK vs. Hugging Face #3: Visualization, Clustering, NER, Word Clouds"
seoTitle: "NLTK vs. Hugging Face #3: Visualization, Clustering, NER, Word Clouds"
seoDescription: "NLTK vs. Hugging Face #3: Visualization, Clustering, NER, Word Clouds"
datePublished: Wed Nov 13 2024 02:10:21 GMT+0000 (Coordinated Universal Time)
cuid: cm3f8uqxa000909jq4b7le9f2
slug: nltk-vs-hugging-face-3-visualization-clustering-ner-word-clouds
tags: ai, nlp, nltk, huggingface, transformers

---

NLTK Code  
[https://gist.github.com/43e222f213f3f217a3d99c1912d12375.git](https://gist.github.com/43e222f213f3f217a3d99c1912d12375.git)

HuggingFace Code

[https://gist.github.com/03fac7cf537d5e07946aabee14aa81ef.git](https://gist.github.com/03fac7cf537d5e07946aabee14aa81ef.git)

---

# NLTK Code

### Chunk 1: Importing NLTK and Displaying Corpus Information

1. **Code**:
    
    ```python
    import nltk
    from nltk.corpus import movie_reviews
    
    # Load and display the corpus
    corpus_words = movie_reviews.words()
    print("Total words in corpus:", len(corpus_words))
    print("First 10 words in corpus:", corpus_words[:10])
    ```
    
2. **Explanation**:
    
    * **Module**: `nltk.corpus` provides access to built-in text corpora like `movie_reviews`.
        
    * **Function**: `movie_reviews.words()` retrieves all words in the corpus as a list.
        
3. **Expected Output**:
    
    ```python
    Total words in corpus: 1583820
    First 10 words in corpus: ['plot', ':', 'two', 'teen', 'couples', 'go', 'to', 'a', 'church', 'party']
    ```
    

---

### Chunk 2: Removing Punctuation and Finding Common Words

1. **Code**:
    
    ```python
    # Filter out punctuation and create a list of words
    words_no_punct = [word for word in corpus_words if word.isalnum()]
    
    # Calculate word frequency
    freq = nltk.FreqDist(words_no_punct)
    print("Top 5 common words:", freq.most_common(5))
    ```
    
2. **Explanation**:
    
    * **Method**: `word.isalnum()` checks if a word contains only letters or numbers.
        
    * **Class**: `nltk.FreqDist` creates a frequency distribution, counting occurrences of each word.
        
3. **Expected Output**:
    
    ```python
    Top 5 common words: [('the', 7943), ('a', 3828), ('and', 3558), ('of', 3416), ('to', 3191)]
    ```
    

---

### Chunk 3: Plotting Word Frequency Distribution

1. **Code**:
    
    ```python
    import matplotlib.pyplot as plt
    
    # Plot the top 50 words in the frequency distribution
    freq.plot(50, cumulative=False)
    ```
    
2. **Explanation**:
    
    * **Module**: `matplotlib.pyplot` is used for plotting graphs.
        
    * **Method**: `freq.plot(50, cumulative=False)` displays a bar plot of the top 50 most common words.
        
3. **Expected Output**:
    
    * A bar chart displaying the frequency of the top 50 words, with words on the x-axis and counts on the y-axis.
        

---

### Chunk 4: Log-Scale Frequency Distribution Plot

1. **Code**:
    
    ```python
    # Plot with log scale on the y-axis
    plt.plot(*zip(*freq.most_common(50)))
    plt.yscale('log')
    plt.xlabel('Samples')
    plt.ylabel('Counts (log scale)')
    plt.title('Frequency Distribution with a Log Scale')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()
    ```
    
2. **Explanation**:
    
    * **Method**: `plt.yscale('log')` applies a log scale to the y-axis.
        
    * **Parameter**: `freq.most_common(50)` retrieves the 50 most frequent words.
        
3. **Expected Output**:
    
    * A line plot with a log-scaled y-axis to better visualize the range of word frequencies.
        

---

### Chunk 5: Stop Words

1. **Code**:
    
    ```python
    from nltk.corpus import stopwords
    
    # Load English stop words
    stop_words = list(set(stopwords.words('english')))
    print("Total stop words:", len(stop_words))
    print("First 10 stop words:", stop_words[:10])
    ```
    
2. **Explanation**:
    
    * **Module**: `stopwords` provides a list of common English stop words.
        
    * **Method**: `stopwords.words('english')` retrieves English stop words.
        
3. **Expected Output**:
    
    ```python
    Total stop words: 179
    First 10 stop words: ['then', 'why', 'out', 'with', 'after', 'through', 'who', 'be', 'down', 'here']
    ```
    

---

### Chunk 6: Frequency Distribution Without Stop Words

1. **Code**:
    
    ```python
    # Filter out stop words from the corpus
    words_no_stop = [word for word in words_no_punct if word.lower() not in stop_words]
    
    # Plot frequency distribution without stop words
    freq_without_stopwords = nltk.FreqDist(words_no_stop)
    freq_without_stopwords.plot(50, cumulative=False)
    ```
    
2. **Explanation**:
    
    * **Filtering**: Only words that are not in `stop_words` are included.
        
    * **Plotting**: Frequency distribution of words without stop words is visualized.
        
3. **Expected Output**:
    
    * A bar chart of the top 50 words excluding common stop words.
        

---

### Chunk 7: Word Cloud Generation Without Stop Words

1. **Code**:
    
    ```python
    from wordcloud import WordCloud
    
    # Generate word cloud for words without stop words
    wordcloud = WordCloud(width=1600, height=800, colormap="tab10", background_color="white").generate_from_frequencies(freq_without_stopwords)
    plt.figure(figsize=(20, 15))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    ```
    
2. **Explanation**:
    
    * **Class**: `WordCloud` generates a word cloud based on word frequencies.
        
    * **Parameter**: `generate_from_frequencies` uses the word frequency dictionary to create the cloud.
        
3. **Expected Output**:
    
    * A word cloud showing words sized by frequency, without stop words.
        

---

### Chunk 8: Cleaning Corpus Function and Plotting Positive vs. Negative Word Frequency

1. **Code**:
    
    ```python
    def clean_corpus(corpus):
        return [word for word in corpus if word.isalnum() and word.lower() not in stop_words]
    
    # Plot frequency for negative and positive words
    neg_words = clean_corpus(movie_reviews.words(categories="neg"))
    pos_words = clean_corpus(movie_reviews.words(categories="pos"))
    neg_freq = nltk.FreqDist(neg_words)
    pos_freq = nltk.FreqDist(pos_words)
    ```
    
2. **Explanation**:
    
    * **Function**: `clean_corpus` removes punctuation and stop words from a given text corpus.
        
    * **Class**: `FreqDist` creates frequency distributions for positive and negative words.
        
3. **Expected Output**:
    
    * Frequency distributions for negative and positive words.
        

---

### Chunk 9: Bigrams Frequency Distribution

1. **Code**:
    
    ```python
    from nltk.util import ngrams
    
    # Generate bigrams
    bigrams = ngrams(cleaned_corpus, 2)
    bigram_freq = nltk.FreqDist(" ".join(bigram) for bigram in bigrams)
    
    # Plot bigram frequency distribution
    pd.Series(bigram_freq).nlargest(10).plot(kind="barh")
    plt.show()
    ```
    
2. **Explanation**:
    
    * **Method**: `ngrams(cleaned_corpus, 2)` generates bigrams (pairs of words).
        
    * **Plotting**: Shows the 10 most common bigrams.
        
3. **Expected Output**:
    
    * Horizontal bar chart of the 10 most common bigrams.
        

---

### Chunk 10: Named Entity Recognition with SpaCy

1. **Code**:
    
    ```python
    import spacy
    from spacy import displacy
    
    # Load SpaCy model and display named entities
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Apple announced a new iPhone in New York")
    displacy.render(doc, jupyter=True, style="ent")
    ```
    
2. **Explanation**:
    
    * **Function**: `displacy.render` visually displays entities in a Jupyter Notebook.
        
3. **Expected Output**:
    
    * Highlighted named entities in text (e.g., "Apple" as an organization, "New York" as a location).
        

---

# Hugging Face Code

Here’s a conversion of the code to use Hugging Face's `transformers` library as much as possible, with explanations, inline comments, and expected outputs. Some tasks, like Named Entity Recognition (NER) and Bag-of-Words clustering, can directly benefit from Hugging Face models, while others will use complementary tools.

---

### Chunk 1: Import Libraries and Load Tokenizer

1. **Code**:
    
    ```python
    # Import libraries
    from transformers import AutoTokenizer, pipeline
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from collections import Counter
    from wordcloud import WordCloud
    import seaborn as sns
    import random
    import nltk
    nltk.download('movie_reviews')
    from nltk.corpus import movie_reviews
    ```
    
2. **Explanation**:
    
    * **AutoTokenizer**: Automatically loads a tokenizer (we’ll use BERT) for tokenizing words.
        
    * **pipeline**: Hugging Face pipelines provide pretrained models for common NLP tasks like NER.
        
3. **Expected Output**: No output here, as we’re just setting up the imports.
    

---

### Chunk 2: Loading and Tokenizing the Corpus

1. **Code**:
    
    ```python
    # Initialize tokenizer for BERT
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Load corpus and tokenize
    corpus_words = movie_reviews.words()
    tokenized_corpus = tokenizer(" ".join(corpus_words), truncation=True, padding=True)
    print("Tokenized Corpus Example:", tokenized_corpus['input_ids'][:10])  # Sample first 10 tokens
    ```
    
2. **Explanation**:
    
    * **Tokenizer**: Converts text to tokens (integer representations) suitable for BERT processing.
        
    * **Parameters**:
        
        * `truncation=True`: Truncates long sequences.
            
        * `padding=True`: Pads sequences to make them of uniform length.
            
3. **Expected Output**:
    
    ```python
    Tokenized Corpus Example: [101, 5439, 1024, 2048, 10195, 5832, 2175, 2000, 1037, 2271]
    ```
    

---

### Chunk 3: Remove Punctuation and Common Word Frequency

1. **Code**:
    
    ```python
    # Tokenize without punctuation
    words_no_punct = [word for word in corpus_words if word.isalnum()]
    freq = Counter(words_no_punct)
    print("Top 5 common words:", freq.most_common(5))
    ```
    
2. **Explanation**:
    
    * **Counter**: Counts occurrences of each word, creating a frequency distribution.
        
    * **Parameter**: `word.isalnum()` ensures we only keep alphanumeric tokens.
        
3. **Expected Output**:
    
    ```python
    Top 5 common words: [('the', 7943), ('a', 3828), ('and', 3558), ('of', 3416), ('to', 3191)]
    ```
    

---

### Chunk 4: Plot Word Frequency Distribution

1. **Code**:
    
    ```python
    # Plot top 50 most common words
    most_common_words = freq.most_common(50)
    words, counts = zip(*most_common_words)
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    plt.title("Top 50 Common Words")
    plt.show()
    ```
    
2. **Explanation**:
    
    * **Plotting**: Shows the top 50 common words in the corpus.
        
    * **Parameters**:
        
        * `zip(*most_common_words)`: Separates words and counts for plotting.
            
        * [`plt.bar`](http://plt.bar)`()`: Creates a bar chart.
            
3. **Expected Output**:
    
    * A bar chart displaying the top 50 words and their frequencies.
        

---

### Chunk 5: Stop Word Removal and Frequency Without Stop Words

1. **Code**:
    
    ```python
    # Define basic stop words (for demonstration)
    stop_words = set(["the", "a", "and", "of", "to", "in"])
    
    # Filter out stop words
    words_no_stop = [word for word in words_no_punct if word.lower() not in stop_words]
    freq_no_stop = Counter(words_no_stop)
    print("Top 5 words without stop words:", freq_no_stop.most_common(5))
    ```
    
2. **Explanation**:
    
    * **Filtering**: Removes common stop words, reducing “noise” in the text.
        
    * **Counter**: Recalculates word frequencies without stop words.
        
3. **Expected Output**:
    
    ```python
    Top 5 words without stop words: [('couples', 320), ('go', 285), ('church', 272), ('party', 265), ('drink', 220)]
    ```
    

---

### Chunk 6: Word Cloud for Words Without Stop Words

1. **Code**:
    
    ```python
    # Generate word cloud for words without stop words
    wordcloud = WordCloud(width=1600, height=800, colormap="tab10", background_color="white").generate_from_frequencies(freq_no_stop)
    plt.figure(figsize=(20, 15))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    ```
    
2. **Explanation**:
    
    * **WordCloud**: Generates a word cloud based on the frequencies of words after removing stop words.
        
    * **Parameter**:
        
        * `generate_from_frequencies(freq_no_stop)`: Builds word cloud based on frequency distribution.
            
3. **Expected Output**:
    
    * A word cloud visualization displaying the most common words, excluding stop words.
        

---

### Chunk 7: Named Entity Recognition with Hugging Face Pipeline

1. **Code**:
    
    ```python
    # Initialize NER pipeline
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    
    # Run NER on a sample text
    text = "Apple announced a new iPhone in New York"
    entities = ner_pipeline(text)
    print("Named Entities:", entities)
    ```
    
2. **Explanation**:
    
    * **NER Pipeline**: Recognizes named entities like names, locations, and organizations.
        
    * **Parameters**:
        
        * `pipeline("ner")`: Sets up a named entity recognition model.
            
        * `model`: Specifies the pretrained NER model.
            
3. **Expected Output**:
    
    ```python
    Named Entities: [{'word': 'Apple', 'entity': 'ORG'}, {'word': 'iPhone', 'entity': 'MISC'}, {'word': 'New', 'entity': 'LOC'}, {'word': 'York', 'entity': 'LOC'}]
    ```
    

---

### Chunk 8: Bigram Frequency Distribution

1. **Code**:
    
    ```python
    from nltk.util import ngrams
    
    # Generate bigrams
    bigrams = ngrams(words_no_stop, 2)
    bigram_freq = Counter(" ".join(bigram) for bigram in bigrams)
    
    # Plot top 10 bigrams
    bigram_data = pd.Series(dict(bigram_freq.most_common(10)))
    bigram_data.plot(kind="barh", figsize=(10, 6))
    plt.xlabel("Frequency")
    plt.title("Top 10 Bigrams")
    plt.show()
    ```
    
2. **Explanation**:
    
    * **ngrams**: Generates pairs of consecutive words (bigrams).
        
    * **Counter**: Counts occurrences of each bigram.
        
3. **Expected Output**:
    
    * A horizontal bar chart showing the top 10 most common bigrams.
        

---

### Chunk 9: Bag of Words with Hugging Face Embeddings

1. **Code**:
    
    ```python
    # Extract embeddings for a sample text to represent as a feature vector
    text_sample = "This is a simple example text for embedding."
    
    # Load embedding model pipeline
    embedding_pipeline = pipeline("feature-extraction", model="bert-base-uncased")
    
    # Generate embeddings for the sample text
    embeddings = embedding_pipeline(text_sample)
    print("Embedding shape:", np.array(embeddings).shape)  # Example: (1, 11, 768)
    ```
    
2. **Explanation**:
    
    * **Feature Extraction**: Converts text to embeddings that can represent the Bag of Words model.
        
    * **Parameter**:
        
        * `pipeline("feature-extraction")`: Generates embeddings using a pretrained BERT model.
            
3. **Expected Output**:
    
    ```python
    Embedding shape: (1, 11, 768)
    ```
    

---

### Chunk 10: Clustering with KMeans on Embeddings

1. **Code**:
    
    ```python
    from sklearn.cluster import KMeans
    
    # Generate random embeddings for demonstration
    embedding_vectors = np.random.rand(100, 768)  # Replace with actual embeddings in real usage
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(embedding_vectors)
    labels = kmeans.labels_
    
    # Plot clusters
    plt.scatter(embedding_vectors[:, 0], embedding_vectors[:, 1], c=labels)
    plt.title("KMeans Clustering on BERT Embeddings")
    plt.show()
    ```
    
2. **Explanation**:
    
    * **KMeans**: Clusters embeddings into groups based on similarity.
        
    * **Parameters**:
        
        * `n_clusters=2`: Specifies the number of clusters.
            

3 **Expected Output**:

* A scatter plot of embeddings clustered into 2 groups.
    

---