---
title: "20 Gensim concepts with Before-and-After Examples"
seoTitle: "20 Gensim concepts with Before-and-After Examples"
seoDescription: "20 Gensim concepts with Before-and-After Examples"
datePublished: Thu Oct 03 2024 13:20:13 GMT+0000 (Coordinated Universal Time)
cuid: cm1tbqa54000v09jmhcbl5j3x
slug: 20-gensim-concepts-with-before-and-after-examples
tags: ai, python, data-science, nlp, gensim

---

### 1\. **Dictionary (gensim.corpora.Dictionary)** 📖

**Boilerplate Code**:

```python
from gensim.corpora import Dictionary
```

**Use Case**: Create a **dictionary** that maps words to unique IDs, which is essential for processing text in Gensim. 📖

**Goal**: <mark>Convert raw text into a bag-of-words forma</mark>t. 🎯

**Sample Code**:

```python
# Example text
texts = [["hello", "world"], ["world", "of", "gensim"]]

# Create a dictionary
dictionary = Dictionary(texts)
print(dictionary.token2id)
```

**Before Example**: has text but doesn’t know how to represent each word with a unique ID. 🤔

```python
Texts: [["hello", "world"], ["world", "of", "gensim"]]
```

**After Example**: With **Dictionary**, each word is mapped to a unique ID! 📖

```python
Dictionary: {'hello': 0, 'world': 1, 'of': 2, 'gensim': 3}
```

**Challenge**: 🌟 Try creating a dictionary from a larger text corpus and see how many unique words are there.

---

### 2\. **Corpus (gensim.corpora.MmCorpus)** 📚

**Boilerplate Code**:

```python
from gensim import corpora
```

**Use Case**: Build a **corpus**, which represents documents as a collection of vectors (bag-of-words). 📚

**Goal**: Convert documents into a numerical format that Gensim can process. 🎯

**Sample Code**:

```python
# Example text
texts = [["hello", "world"], ["world", "of", "gensim"]]

# Create a dictionary and corpus
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)
```

**Before Example**: has raw text but doesn’t know how to represent it as numerical vectors. 🤔

```python
Texts: [["hello", "world"], ["world", "of", "gensim"]]
```

**After Example**: With **corpus**, the documents are represented as vectors! 📚

```python
Corpus: [[(0, 1), (1, 1)], [(1, 1), (2, 1), (3, 1)]]
```

**Challenge**: 🌟 Try saving the corpus to disk and reloading it using `MmCorpus`.  
You're right! The title mentions `MmCorpus`, but the example code doesn't actually use it. Let me clarify and explain how **MmCorpus** fits in, and give you a relevant example.

In the sample code, you're converting a collection of texts into a **corpus** (a collection of documents represented as vectors using the **bag-of-words** model). This means each word in a document is transformed into a unique identifier (an integer), and then the frequency of each word is stored in a vector.

The `MmCorpus` function is used when you want to **save** your corpus to disk and **reload** it later in a more efficient format. The name "Mm" stands for **Matrix Market** format, which is an efficient way to store sparse matrices (like a corpus) that you can save to a file and reuse without having to recompute the bag-of-words vector.

```python
from gensim import corpora

# Example text
texts = [["hello", "world"], ["world", "of", "gensim"]]

# Create a dictionary and corpus (bag-of-words representation)
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Save the corpus to disk using MmCorpus
corpora.MmCorpus.serialize('corpus.mm', corpus)

# Load the corpus from disk
loaded_corpus = corpora.MmCorpus('corpus.mm')

print(list(loaded_corpus))  # Corpus reloaded from disk
```

* **Creating the corpus**: You're still creating the bag-of-words corpus like before.
    
* **<mark>Saving with MmCorpus</mark>**<mark>: </mark> `MmCorpus.serialize()` <mark>saves your corpus in Matrix Market format to a file (</mark>[`corpus.mm`](http://corpus.mm)<mark>).</mark>
    
* **Loading with MmCorpus**: You can <mark>reload it later using </mark> `MmCorpus('`[`corpus.mm`](http://corpus.mm)`')` without having to recompute the entire bag-of-words model
    

---

### 3\. **TF-IDF Model (gensim.models.TfidfModel)** 📊

**Boilerplate Code**:

```python
from gensim.models import TfidfModel
```

**Use Case**: Create a **TF-IDF model** to weigh words based on their frequency and importance in a corpus. 📊

**Goal**: Adjust word importance based on how often they appear in the corpus. 🎯

**Sample Code**:

```python
# Example text
texts = [["hello", "world"], ["world", "of", "gensim"]]

# Create dictionary and corpus
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Initialize TF-IDF model
tfidf = TfidfModel(corpus)
tfidf_corpus = tfidf[corpus]
print(list(tfidf_corpus))
```

**Before Example**: we have word counts but doesn’t know which words are important in the context of the corpus. 🤔

```python
Corpus: [[(0, 1), (1, 1)], [(1, 1), (2, 1), (3, 1)]]
```

**After Example**: With **TF-IDF**, the importance of words is adjusted based on their frequency in the corpus! 📊

```python
TF-IDF Scores: Weights for each word in the corpus.
```

**Challenge**: 🌟 Try applying TF-IDF to a large document set and identify the most important words.

---

### 4\. **LDA Model (gensim.models.LdaModel)** 🔥

**Boilerplate Code**:

```python
from gensim.models import LdaModel
```

**Use Case**: Perform **<mark>Latent Dirichlet Allocation (LDA)</mark>** <mark>for </mark> **<mark>topic modeling</mark>**, which discovers topics in a set of documents. 🔥

**Goal**: Identify hidden topics in the text data. 🎯

**Sample Code**:

```python
# Example text
texts = [["hello", "world"], ["world", "of", "gensim"]]

# Create dictionary and corpus
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Train LDA model
lda = LdaModel(corpus, num_topics=2, id2word=dictionary)
print(lda.print_topics())
```

**Before Example**: doesn’t know the underlying topics. 🤔

```python
Texts: [["hello", "world"], ["world", "of", "gensim"]]
```

**After Example**: With **LDA**, we discover topics hidden in the documents! 🔥

```python
Topics: [(0, "0.5*'world' + 0.5*'hello'"), (1, "0.33*'world' + 0.33*'gensim' ...")]
```

**Challenge**: 🌟 Try experimenting with more topics and see how they cluster together.

---

### 5\. **Word2Vec Model (gensim.models.Word2Vec)** 🔠

**Boilerplate Code**:

```python
from gensim.models import Word2Vec
```

**Use Case**: Create a **Word2Vec model** to generate **word embeddings**, which represent words as vectors. 🔠

**Goal**: Represent words as vectors for machine learning tasks. 🎯

**Sample Code**:

```python
# Example text
sentences = [["hello", "world"], ["gensim", "is", "cool"]]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1)

# Get word vector
print(model.wv['hello'])
```

**Before Example**: has words but doesn’t know how to represent them as vectors. 🤔

```python
Words: ["hello", "world"]
```

**After Example**: With **Word2Vec**, we represent words as vectors! 🔠

```python
Word Vector: [0.123, -0.456, 0.789, ...]
```

**Challenge**: 🌟 Try finding similar words using `model.wv.most_similar()`.

---

### 6\. **Document Similarity (gensim.similarities.Similarity)** 🔗

**Boilerplate Code**:

```python
from gensim.similarities import MatrixSimilarity
```

**Use Case**: Compare **document similarity** by calculating how close documents are based on their vector representations. 🔗

**Goal**: Measure the similarity between documents. 🎯

**Sample Code**:

```python
# Example text
texts = [["hello", "world"], ["world", "of", "gensim"]]

# Create dictionary, corpus, and TF-IDF model
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = TfidfModel(corpus)
tfidf_corpus = tfidf[corpus]

# Compute similarity matrix
similarity_index = MatrixSimilarity(tfidf_corpus)
print(list(similarity_index))
```

**Before Example**: Has multiple documents but doesn’t know how similar they are to each other. 🤔

```python
Documents: [["hello", "world"], ["world", "of", "gensim"]]
```

**After Example**: With **similarity indexing**, we can measure the similarity between documents! 🔗

```python
Similarity: Scores between 0 and 1 for each document pair.
```

**Challenge**: 🌟 Try using the similarity matrix for document clustering or retrieval.

---

### 7\. **Coherence Model (gensim.models.CoherenceModel)** 📈

**Boilerplate Code**:

```python
from gensim.models import CoherenceModel
```

**Use Case**: <mark>Evaluate the quality of a topic model using </mark> **<mark>coherence scores</mark>**<mark>. 📈</mark>  
**High-Quality Topics**: (ie politics) The words are closely related and clearly form a distinct, interpretable topic. A human can easily understand what the topic is about.  
Words in the topic: `['election', 'government', 'candidate', 'voting', 'policy', 'debate']`

**Low-Quality Topics**(Incoherent Mix) The words seem random or unrelated, making it difficult or impossible to assign meaning to the topic. Words in the topic: `['apple', 'election', 'sky', 'music', 'policy', 'river']`

**Goal**: Assess the interpretability of the topics discovered by the LDA model. 🎯

**Sample Code**:

```python
# Example text
texts = [["hello", "world"], ["world", "of", "gensim"]]

# Create dictionary, corpus, and LDA model
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda = LdaModel(corpus, num_topics=2, id2word=dictionary)

# Compute coherence score
coherence_model_lda = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_score = coherence_model_lda.get_coherence()
print(coherence_score)
```

**Before Example**: has a topic model but doesn’t know how good the topics are. 🤔

```python
LDA Model: 2 topics
```

**After Example**: With **coherence score**, we can measures the quality of the topics!

```python
Coherence Score: A value between 0 and 1 (higher is better).
```

**Challenge**: 🌟 Try tuning the LDA model’s parameters to improve the coherence score.

---

### 8\. **Phrases (gensim.models.Phrases)** 🗣️

**Boilerplate Code**:

```python
from gensim.models import Phrases
```

**Use Case**: <mark>Detect common </mark> **<mark>phrases</mark>** or collocations in the text (e.g., "New York"). 🗣️

**Goal**: Identify and <mark>represent common multi-word expressions as single tokens</mark>. 🎯

**Sample Code**:

```python
# Example text
sentences = [["new", "york", "city"], ["new", "york", "times"]]

# Train Phrases model
phrases = Phrases(sentences, min_count=1, threshold=1)

# Apply model to detect phrases
bigram = phrases[sentences]
print(list(bigram))
```

**Before Example**: we have word sequences but doesn’t know how to detect phrases. 🤔

```python
Sentences: [["new", "york", "city"], ["new", "york", "times"]]
```

**After Example**: <mark>With </mark> **<mark>Phrases</mark>**<mark>, we identify common phrases like "New York"!</mark>

```python
Phrases: [('new_york', 'city'), ('new_york', 'times')]
```

**Challenge**: 🌟 Try detecting trigrams (three-word phrases) in a larger dataset.

---

### 9\. **KeyedVectors (gensim.models.KeyedVectors)** 💡

**Boilerplate Code**:

```python
from gensim.models import KeyedVectors
```

**Use Case**: Use <mark>pre-trained </mark> **<mark>word vectors</mark>** <mark>from sources like Word2Vec or GloVe.</mark> 💡

**Goal**: Load pre-trained vectors and apply them fo<mark>r similarity comparisons.</mark> 🎯

**Sample Code**:

```python
# Load pre-trained vectors
model = KeyedVectors.load_word2vec_format('path/to/vectors.bin', binary=True)

# Get vector for a word
vector = model['word']
print(vector)
```

**Before Example**: We need word vectors but doesn’t want to train a model from scratch. 🤔

```python
Need: Pre-trained word vectors.
```

**After Example**: <mark>With </mark> **<mark>KeyedVectors</mark>**<mark>, </mark> we load pre-trained vectors and applies them! 💡

```python
Word Vector: [0.345, -0.234, ...]
```

**Challenge**: 🌟 Try using pre-trained vectors to find similar words in a new dataset.

---

### 10\. **FastText Model (gensim.models.FastText)** 🚀

**Boilerplate Code**:

```python
from gensim.models import FastText
```

**Use Case**: Train a **FastText model** for word embeddings, <mark>which captures subword information and works well with rare words</mark>. 🚀  
**Subword** information refers to breaking words down into smaller parts, like prefixes, suffixes, or even individual characters, which can be used to build better word embeddings—especially for rare or misspelled words.

Example:

Let’s take the word **“unhappiness”**.

In a traditional word embedding model, the entire word is treated as a single unit. If the model has never seen "unhappiness" before, it won't know how to generate a good embedding for it.

In **FastText**, however, the word can be broken down into subwords, like:

* **Prefixes**: "un-", "hap-"
    
* **Root**: "happy"
    
* **Suffixes**: "-ness"
    

So, instead of treating "unhappiness" as a completely unknown word, FastText looks at the subword parts (like "happy") that are more common and uses them to create the embedding.

Why is this useful?

1. **Rare Words**: For words that don’t appear frequently, FastText can still build meaningful embeddings by analyzing the subwords it has seen before.
    
    * Example: <mark>For a rare word like "bioluminescence",</mark> <mark>FastText can break it into "bio-", "lumin-", and "-escence" </mark> to generate an embedding based on known subwords.
        
2. **Misspelled Words**: FastText can handle misspellings by looking at parts of the word that it recognizes.
    
    * Example: Even if "happiness" is misspelled as "happpiness", FastText can still generate a reasonable embedding by identifying the subword "happy".
        

Example:

* Traditional embedding: The word **"happiness"** would be treated as one unit, and if it’s not in the vocabulary, it would return an unknown vector.
    
* FastText: It breaks **"happiness"** into smaller parts like "hap-", "piness", etc., which helps it still generate an embedding based on familiar subwords.
    

FastText helps generate better word embeddings for words it hasn’t seen before or for misspelled words by breaking them into **subword components**. This allows it to capture more nuanced information about words 🚀!

**Goal**: Generate word vectors that include information about character-level features. 🎯

**Sample Code**:

```python
# Example text
sentences = [["hello", "world"], ["gensim", "is", "awesome"]]

# Train FastText model
model = FastText(sentences, vector_size=10, window=2, min_count=1)

# Get vector for a word
print(model.wv['hello'])
```

**Before Example**: has text but needs better embeddings for rare or misspelled words.

```python
Words: ["hello", "world", "gensim"]
```

**After Example**: With **FastText**, we can generate embeddings that capture subword information! 🚀

```python
Word Vector: [0.123, -0.456, ...]
```

**Challenge**: 🌟 Try training FastText on a dataset with rare or noisy text and see how it performs.

---

### 11\. **Doc2Vec Model (gensim.models.Doc2Vec)** 📝

**Boilerplate Code**:

```python
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
```

**Use Case**: <mark>Train a </mark> **<mark>Doc2Vec model</mark>** <mark>to represent entire documents as vectors,</mark> which is useful for document similarity and classification tasks. 📝

**Goal**: Convert documents into vectors while keeping context intact. 🎯

**Sample Code**:

```python
# Example text
documents = [TaggedDocument(words=["gensim", "is", "awesome"], tags=[0]), 
             TaggedDocument(words=["machine", "learning", "with", "gensim"], tags=[1])]

# Train Doc2Vec model
model = Doc2Vec(documents, vector_size=10, window=2, min_count=1)

# Get document vector
vector = model.dv[0]
print(vector)
```

**Before Example**: we doesn’t know how to represent them as vectors. 🤔

```python
Documents: ["gensim is awesome", "machine learning with gensim"]
```

**After Example**: With **Doc2Vec**, we represent entire documents as vectors! 📝

```python
Document Vector: [0.345, -0.234, ...]
```

**Challenge**: 🌟 Try using the document vectors for classification or clustering tasks.

---

### 12\. **Latent Semantic Indexing (LSI) Model (gensim.models.LsiModel)** 🔍

we want to use **LSI (Latent Semantic Indexing)** to identify **hidden relationships** between terms and reduce the dimensionality of the data.

Example:

Imagine we have these **documents**:

1. "Machine learning is great"
    
2. "Artificial intelligence is the future"
    
3. "Machine learning is part of artificial intelligence"
    
4. "The future is bright with AI"
    

Step-by-step breakdown:

1. **Texts**: These are the documents we will process using LSI.
    
    ```python
    pythonCopy codetexts = [
        ["machine", "learning", "is", "great"],
        ["artificial", "intelligence", "is", "the", "future"],
        ["machine", "learning", "is", "part", "of", "artificial", "intelligence"],
        ["the", "future", "is", "bright", "with", "ai"]
    ]
    ```
    
2. **Create Dictionary and Corpus**: We create a dictionary and a bag-of-words representation (corpus) of the documents.
    
    ```python
    from gensim.corpora import Dictionary
    from gensim.models import LsiModel
    
    # Create dictionary
    dictionary = Dictionary(texts)
    
    # Create bag-of-words corpus
    corpus = [dictionary.doc2bow(text) for text in texts]
    ```
    
3. **Train the LSI Model**: We’ll train an LSI model to find **latent topics** in the documents by reducing the dimensionality.
    
    ```python
    # Train the LSI model with 2 topics
    lsi = LsiModel(corpus, id2word=dictionary, num_topics=2)
    ```
    
4. **View the Topics**: After training, we can print the topics identified by LSI.
    
    ```python
    # Print the topics
    print(lsi.print_topics())
    ```
    

Example Output:

After running the LSI model, you might see something like this:

```python
[(0, "0.707*'learning' + 0.707*'machine'"), 
 (1, "0.577*'intelligence' + 0.577*'artificial' + 0.577*'future'")]
```

What the output means:

* **<mark>Topic 0</mark>**<mark>: This topic is mostly about </mark> **<mark>machine learning</mark>** <mark>because it heavily weights the words </mark> **<mark>"learning"</mark>** <mark>and </mark> **<mark>"machine"</mark>**<mark>. So, the documents mentioning these terms are grouped together.</mark>
    
* **Topic 1**: This topic is about **artificial intelligence** and **the future**, as it includes terms like **"intelligence"**, **"artificial"**, and **"future"**.
    
* The documents are simple and related, so you can see how LSI groups similar terms like **"machine learning"** and **"artificial intelligence"** into latent topics.
    
* LSI reduces the **dimensionality** (turning the many different words into a few meaningful topics), making it easier to understand the core themes within the documents.
    

Practical Use Case:

* **Document Similarity**: LSI is used to find hidden structures in documents and group related documents. For example, <mark>in search engines, it helps match documents with similar topics even if they don’t use the exact same keywords</mark>.
    

Challenge:

You can experiment by reducing the number of **topics** (e.g., set `num_topics=1`) and observe how LSI captures a more general pattern across the documents.

---

### 13\. **HDP Model (gensim.models.HdpModel)** 📈

Suppose we have a set of **news articles** and we want to discover the topics within them, but we don't know how many topics to expect beforehand. HDP automatically figures this out!

Step-by-Step Example:

1. **Documents (Texts)**: These are our simple documents (pretending they are news articles).
    
    ```python
    texts = [
        ["artificial", "intelligence", "future", "technology"],
        ["sports", "soccer", "goal", "team"],
        ["politics", "election", "government", "policy"],
        ["artificial", "intelligence", "machine", "learning"],
        ["soccer", "team", "win", "championship"]
    ]
    ```
    
    Here, we have a mix of topics:
    
    * Documents 1 and 4 are about **artificial intelligence**.
        
    * Documents 2 and 5 are about **soccer**.
        
    * Document 3 is about **politics**.
        
2. **Create Dictionary and Corpus**: Convert the documents into a bag-of-words representation (corpus) using a dictionary.
    
    ```python
    from gensim.corpora import Dictionary
    from gensim.models import HdpModel
    
    # Create dictionary
    dictionary = Dictionary(texts)
    
    # Convert to bag-of-words corpus
    corpus = [dictionary.doc2bow(text) for text in texts]
    ```
    
3. **Train the HDP Model**: Use HDP to discover topics dynamically, without specifying the number of topics in advance.
    
    ```python
    # Train the HDP model
    hdp = HdpModel(corpus, id2word=dictionary)
    ```
    
4. **View the Discovered Topics**: Print the topics that HDP discovered from the documents.
    
    ```python
    # Print the discovered topics
    print(hdp.print_topics())
    ```
    

Sample Output:

HDP might automatically discover several topics from the documents, such as:

```python
[(0, "0.3*'artificial' + 0.3*'intelligence' + 0.2*'machine' + 0.1*'learning'"),
 (1, "0.4*'soccer' + 0.3*'team' + 0.2*'goal' + 0.1*'win'"),
 (2, "0.5*'politics' + 0.3*'election' + 0.2*'government' + 0.1*'policy'")]
```

Explanation:

* **Topic 0**: The first topic contains words related to **artificial intelligence** ("artificial," "intelligence," "machine," "learning").
    
* **Topic 1**: The second topic relates to **soccer** ("soccer," "team," "goal," "win").
    
* **Topic 2**: The third topic captures **politics** ("politics," "election," "government," "policy").
    

Why This is Better:

* **<mark>Dynamic Topic Discovery</mark>**<mark>:</mark> <mark>Unlike LDA (Latent Dirichlet Allocation), where you need to specify the number of topics beforehand, </mark> **<mark>HDP</mark>** <mark>automatically figures out how many topics exist in the dataset. This is useful when you're not sure how many topics are present.</mark>
    
* **No Predefined Number of Topics**: HDP is great for exploring new datasets when you don’t know the structure of the topics in advance, like in news articles, customer reviews, or research papers.
    

Real-World Use Case:

* **News Categorization**<mark>: Imagine you’re analyzing thousands of news articles, but you don’t know how many distinct categories (topics) exist</mark>. HDP will automatically identify them, making it easier to group and label articles.
    
* **Customer Feedback Analysis**: For businesses analyzing customer feedback, HDP helps uncover the main themes (topics) without needing to guess how many topics (e.g., satisfaction, complaints, pricing issues) are present.
    

Challenge:

* Try applying HDP to a larger dataset (like a real news dataset or product reviews) to discover the hidden topics and see how it handles more complex data.
    

---

### 14\. **LSI Similarity Matrix (gensim.similarities.MatrixSimilarity)** 🔗

Suppose you have a collection of **news articles** about various topics like **technology**, **sports**, and **politics**. After transforming the documents using LSI, you want to measure **how similar** these articles are based on their topics.

**Example with Step-by-Step Breakdown:**

1. **Documents (Texts)**: Let’s use some news-related topics to represent articles.
    
    ```python
    texts = [
        ["technology", "ai", "innovation", "future", "tech"],
        ["soccer", "sports", "goal", "team", "win"],
        ["election", "government", "policy", "politics", "vote"],
        ["technology", "ai", "machine", "learning", "data"],
        ["sports", "soccer", "championship", "win", "tournament"]
    ]
    ```
    
    Here, we have:
    
    * Documents 1 and 4 are about **technology**.
        
    * Documents 2 and 5 are about **sports**.
        
    * Document 3 is about **politics**.
        
2. **Create Dictionary and Corpus**: We create a dictionary and a bag-of-words representation of the documents.
    
    ```python
    from gensim.corpora import Dictionary
    from gensim.models import LsiModel
    from gensim.similarities import MatrixSimilarity
    
    # Create dictionary
    dictionary = Dictionary(texts)
    
    # Convert to bag-of-words corpus
    corpus = [dictionary.doc2bow(text) for text in texts]
    ```
    
3. **Train the LSI Model**: Apply LSI to reduce the dimensionality of the data into 2 latent topics.
    
    ```python
    # Train LSI model with 2 topics
    lsi = LsiModel(corpus, id2word=dictionary, num_topics=2)
    
    # Transform corpus into LSI space
    lsi_corpus = lsi[corpus]
    ```
    
4. **Compute the Similarity Matrix**: Use **MatrixSimilarity** to compare the LSI-transformed documents and compute similarity scores between them.
    
    ```python
    # Compute similarity matrix
    similarity_matrix = MatrixSimilarity(lsi_corpus)
    
    # Print the similarity matrix
    for i, similarities in enumerate(similarity_matrix):
        print(f"Document {i} similarities: {list(similarities)}")
    ```
    

Sample Output:

The similarity matrix might show something like this:

```python
Document 0 similarities: [1.0, 0.12, 0.05, 0.95, 0.08]
Document 1 similarities: [0.12, 1.0, 0.07, 0.15, 0.92]
Document 2 similarities: [0.05, 0.07, 1.0, 0.04, 0.09]
Document 3 similarities: [0.95, 0.15, 0.04, 1.0, 0.12]
Document 4 similarities: [0.08, 0.92, 0.09, 0.12, 1.0]
```

* **Document 0** (technology) is most similar to **Document 3** (another technology-related document) with a similarity score of **0.95**.
    
* **Document 1** (sports) is highly similar to **Document 4** (another sports document) with a score of **0.92**.
    
* **Document 2** (politics) has very low similarity with the technology and sports documents, as expected.
    

Real-World Use Case:

* **Document Retrieval**: Suppose you’re building a document retrieval system where users search for documents similar to an existing one. The LSI model reduces the data complexity, and the **similarity matrix** helps efficiently find documents that are related in terms of latent topics (like **news articles**, **research papers**, etc.).
    
* **Recommendation Systems**: You can use this method to <mark>recommend articles or reports based on their similarity to the user’s reading history.</mark>
    

In summary, **MatrixSimilarity** lets you measure how similar documents are <mark>after </mark> reducing their dimensions with LSI, which is useful in cases like document retrieval or recommendation systems..

---

### 15\. **Phrases Model (gensim.models.Phrases)** 🗣️

**Boilerplate Code**:

```python
from gensim.models import Phrases
```

**Use Case**: Detect **common phrases** or bigrams (e.g., "New York") in a corpus and convert them into single tokens. 🗣️

**Goal**: Identify frequently co-occurring word pairs in text and treat them as one unit. 🎯

**Sample Code**:

```python
# Example text
sentences = [["new", "york", "city"], ["new", "york", "times"]]

# Train Phrases model
phrases = Phrases(sentences, min_count=1, threshold=1)

# Detect phrases
bigram = phrases[sentences]
print(list(bigram))
```

**Before Example**: doesn’t know how to detect frequently occurring word pairs. 🤔

```python
Sentences: [["new", "york", "city"], ["new", "york", "times"]]
```

**After Example**: With **Phrases**, we detect phrases like "New York"! 🗣️

```python
Phrases: [('new_york', 'city'), ('new_york', 'times')]
```

**Challenge**: 🌟 Try detecting trigrams (three-word phrases) and apply this to a larger corpus.

---

### 16\. **Term Frequency (gensim.models.TfidfModel)** 📊

**Boilerplate Code**:

```python
from gensim.models import TfidfModel
```

**Use Case**: Calculate **term frequency-inverse document frequency (TF-IDF)** to find important terms in a corpus. 📊

**Goal**: Identify the most significant words based on how often they appear across documents. 🎯

**Sample Code**:

```python
# Example text
texts = [["hello", "world"], ["world", "of", "gensim"]]

# Create dictionary and corpus
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Train TF-IDF model
tfidf = TfidfModel(corpus)
tfidf_corpus = tfidf[corpus]
print(list(tfidf_corpus))
```

**Before Example**: has word counts but doesn’t know which words are more important in the context of the corpus. 🤔

```python
Texts: [["hello", "world"], ["world", "of", "gensim"]]
```

**After Example**: With **TF-IDF**, we identified the most important words in the corpus! 📊

```python
TF-IDF: Scores for each word in the documents.
```

**Challenge**: 🌟 Apply TF-IDF to a news article and find the most significant words.

---

### 17\. **Random Projections (gensim.models.RpModel)** 🎯

When to Use <mark>Random Projections</mark> (RP):

1. **High Dimensionality**:
    
    * If your dataset has many features (e.g., <mark>thousands of unique words in text data), it’s high-dimensional.</mark>
        
    * Recognize this <mark>if the number of features (columns) is much larger than the number of samples (rows)</mark>.
        
2. **Slow Processing**:
    
    * High-dimensional data takes longer to process. If your computations are slow, RP can speed it up by reducing dimensions.
        
3. **<mark>Overfitting</mark>**<mark>:</mark>
    
    * Models trained on high-dimensional data often overfit because they learn from noise rather than real patterns.
        
    * <mark>Signs of overfitting include good performance on training data but poor performance on new data</mark>.
        
4. **Sparse Data**:
    
    * High-dimensional data tends to be sparse, meaning many features rarely occur (e.g., <mark>rare words in text</mark>).
        
    * RP reduces this sparsity, focusing on important features.
        
5. **<mark>Curse of Dimensionality</mark>**<mark>:</mark>
    
    * In high dimensions, <mark>distances between points can become less meaningful, affecting algorithms like clustering.</mark>
        
    * If distance-based methods are ineffective, RP can help.
        

Why Use RP:

* **Efficiency**: RP reduces dimensions quickly without much computation.
    
* **Preserves Relationships**: It retains the relative distances between data points, maintaining structure while simplifying the data.
    

Example Scenarios:

* **Text Data**: With large text corpora, RP can reduce thousands of unique words into fewer dimensions while keeping document relationships intact.
    
* **Image Data**: RP reduces the number of features (pixels) in images, making classification easier while retaining important structure.
    

How to Decide:

* Check if the number of features is much larger than the samples.
    
* If your model overfits or processing is slow, reducing dimensions with RP may help.
    

In summary, use **Random Projections** when dealing with high-dimensional, sparse, or slow-to-process data, and when you want to reduce dimensionality without losing important relationships between points!

**Boilerplate Code**:

```python
from gensim.models import RpModel
```

**Use Case**: <mark>Use </mark> **<mark>Random Projections (RP)</mark>** <mark>to reduce the dimensionality of a corpus. 🎯</mark>

**Goal**: Reduce the dimensionality of high-dimensional datasets while preserving distances. 🎯

**Sample Code**:

```python
# Example text
texts = [["hello", "world"], ["world", "of", "gensim"]]

# Create dictionary and corpus
dictionary = Dictionary(texts)
corpus = [

dictionary.doc2bow(text) for text in texts]

# Train Random Projections model
rp = RpModel(corpus, num_topics=2)
print(rp.print_topics())
```

**Before Example**: high-dimensional data, doesn’t know how to reduce it while preserving distances. 🤔

```python
Texts: [["hello", "world"], ["world", "of", "gensim"]]
```

**After Example**: With **RP**, we reduce the dimensionality while maintaining important relationships! 🎯

```python
Random Projections Topics: A reduced-dimensional representation of the data.
```

**Challenge**: 🌟 Apply random projections to a large corpus and evaluate how much dimensionality is reduced.

---

### 18\. **Online Learning (gensim.models.LdaMulticore)** 🔄

**Boilerplate Code**:

```python
from gensim.models import LdaMulticore
```

**Use Case**: Use **LDA with multicore processing** for faster **online topic modeling**. 🔄

**Goal**: Train an LDA model on large datasets efficiently by processing documents in batches. 🎯

**Sample Code**:

```python
# Example text
texts = [["hello", "world"], ["world", "of", "gensim"]]

# Create dictionary and corpus
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Train LDA model with multiple cores
lda = LdaMulticore(corpus, id2word=dictionary, num_topics=2, workers=2)
print(lda.print_topics())
```

**Before Example**: need to train a topic model on a large dataset but finds it too slow with a single core. 🤔

```python
Documents: A large set of text documents
```

**After Example**: With **LdaMulticore**,we train a topic model faster using multiple CPU cores! 🔄

```python
Topics: Thematic clusters discovered in the documents.
```

**Challenge**: 🌟 Try using online learning for large-scale text datasets to speed up processing.

---

### 19\. **Word Mover’s Distance (gensim.models.WmdSimilarity)** 🧠

**Boilerplate Code**:

```python
from gensim.similarities import WmdSimilarity
```

**Use Case**: Calculate **<mark>Word Mover’s Distance (WMD)</mark>** to measure the similarity between two documents based on word embeddings. 🧠

**Goal**: Compare documents by calculating the minimum distance to <mark>"move</mark>" words from one document to another. 🎯

**Sample Code**:

```python
# Example text
sentences = [["gensim", "is", "cool"], ["machine", "learning", "is", "great"]]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=10, min_count=1)

# Compute WMD
wmd_similarity = WmdSimilarity(sentences, model, num_best=1)
similarity = wmd_similarity[sentences[0]]
print(similarity)
```

**Before Example**: doesn’t know how to compare their similarity using word embeddings. 🤔

```python
Documents: ["gensim is cool", "machine learning is great"]
```

**After Example**: With **WMD**, we can measures how similar documents are based on word embeddings! 🧠

```python
WMD Similarity: A numerical score representing document similarity.
```

**Challenge**: 🌟 Try applying WMD to compare news articles from different categories.

---

### 20\. **Sentence Embeddings (gensim.models.FastText)** 📝

**Boilerplate Code**:

```python
from gensim.models import FastText
```

**Use Case**: Use **FastText** to generate **sentence embeddings**, capturing the meaning of entire sentences. 📝

**Goal**: Represent sentences as vectors for classification or similarity tasks. 🎯

**Sample Code**:

```python
# Example text
sentences = [["gensim", "is", "awesome"], ["machine", "learning", "with", "gensim"]]

# Train FastText model
model = FastText(sentences, vector_size=10, window=2, min_count=1)

# Get sentence embedding by averaging word vectors
sentence_vector = sum([model.wv[word] for word in sentences[0]]) / len(sentences[0])
print(sentence_vector)
```

**Before Example**: doesn’t know how to represent them as vectors. 🤔

```python
Sentences: ["gensim is awesome", "machine learning with gensim"]
```

**After Example**: With **FastText**, we represent entire sentences as vectors! 📝

```python
Sentence Embedding: [0.123, -0.456, ...]
```

**Challenge**: 🌟 Try using sentence embeddings for text classification or clustering tasks.

---