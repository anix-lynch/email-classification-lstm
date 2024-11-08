---
title: "21 NLP core concepts with sample output"
seoTitle: "21 NLP core concepts with sample output"
seoDescription: "21 NLP core concepts with sample output"
datePublished: Fri Oct 11 2024 06:34:58 GMT+0000 (Coordinated Universal Time)
cuid: cm24cry0m000g09mme4jkeupw
slug: 21-nlp-core-concepts-with-sample-output
tags: ai, nlp, deep-learning, huggingface, transformers

---

### 1\. **Tokenization**

```python
from nltk.tokenize import word_tokenize
text = "Natural Language Processing with Python is fun!"
tokens = word_tokenize(text)
print(tokens)
```

**Output**: `['Natural', 'Language', 'Processing', 'with', 'Python', 'is', 'fun', '!']`  
  
`word_tokenize`: This function splits the input text into **individual words and punctuation marks**. <mark> It is more granular than </mark> `sent_tokenize` <mark> and treats each word or punctuation mark as a separate token</mark>. It does not rely on sentence boundaries but instead splits the text based on spaces and punctuation.

### 2\. **High-level Tokenization**

```python
from nltk.tokenize import sent_tokenize
text = "I love NLP. It is very interesting."
sentences = sent_tokenize(text)
print(sentences)
```

**Output**: `['I love NLP.', 'It is very interesting.']`

  
`sent_tokenize`: This function from the **nltk.tokenize** module is used to split a given text into individual **sentences**. <mark>It relies on punctuation marks (like periods, exclamation marks, and question marks</mark>) to determine sentence boundaries.

### 3\. **Low-level Tokenization**

```python
from nltk.tokenize import regexp_tokenize
text = "Natural Language Processing"
tokens = regexp_tokenize(text, r'\w+')
print(tokens)
```

**Output**: `['Natural', 'Language', 'Processing']`  
  
`regexp_tokenize`: This function allows for more **customized tokenization** by using **regular expressions**. You define a pattern <mark>using regular expressions</mark> to specify how to break the text into tokens. This is useful when you need fine-grained control over tokenization or want to handle specific cases, like extracting only numbers, words, or specific patterns.

### 4\. **NLTK Tokenizer (**TweetTokenizer**)**

`TweetTokenizer` is designed to handle social media text (e.g., tweets) and special cases like emoticons, hashtags, and contractions.

```python
pythonCopy codefrom nltk.tokenize import TweetTokenizer

# Example text from a tweet
text = "I'm learning NLP! 😃 #excited @nltk"

# Use the TweetTokenizer
tweet_tokenizer = TweetTokenizer()
tokens = tweet_tokenizer.tokenize(text)
print(tokens)
```

**Output**:

```python
cssCopy code["I'm", 'learning', 'NLP', '!', '😃', '#excited', '@nltk']
```

**Explanation**: The `TweetTokenizer` is specialized for social media text, preserving things like emojis, <mark>hashtags (</mark>`#excited`<mark>), and mentions (</mark>`@nltk`<mark>).</mark> Unlike `word_tokenize`, it doesn’t split contractions like "I'm" into multiple tokens, which is crucial when processing informal texts like tweets.

### 5\. **Spacy Tokenizer**

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Tokenization in Spacy is powerful.")
tokens = [token.text for token in doc]
print(tokens)
```

**Output**: `['Tokenization', 'in', 'Spacy', 'is', 'powerful', '.']`  
**Spacy's tokenizer** is more **advanced and flexible**, with built-in rules and exceptions for real-world applications.

* **NLTK’s tokenizer** is good for educational purposes and simpler tasks, but Spacy excels in handling **messy text**, **abbreviations**, and **contractions** with higher accuracy.
    
* **Exception Handling**:
    
    * Spacy has built-in rules to deal with edge cases like **abbreviations, contractions, and proper nouns**. For example, Spacy handles "U.S." correctly as one token, whereas some tokenizers might split it into two tokens ("U." and "S.").
        
    * Example: If the input were `"U.S. is a country."`, Spacy would tokenize `"U.S."` as a single token, keeping the abbreviation intact.
        
* **Whitespace & Custom Rules**:
    
    * Spacy handles spaces intelligently. For example, Spacy knows when spaces are significant (between words) and when they aren't (at the beginning or end of sentences).
        
    * You can also **customize** Spacy’s tokenizer if you need to adapt it for special rules or languages.
        

### 6\. **Named Entity Recognition (NER)**

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Barack Obama was the 44th president of the United States.")
for ent in doc.ents:
    print(ent.text, ent.label_)
```

**Output**: `Barack Obama PERSON, 44th ORDINAL, United States GPE`

### 7\. **Character-Level Features**

```python
# No explicit code, but character-level tokenization could look like this:
text = "word"
characters = list(text)
print(characters)
```

**Output**: `['w', 'o', 'r', 'd']`

### 8\. **Word-Level Features**

```python
from sklearn.feature_extraction.text import CountVectorizer
text = ["This is a sample document.", "This document is the second document."]
vectorizer = CountVectorizer()
word_features = vectorizer.fit_transform(text)
print(vectorizer.get_feature_names_out())
```

**Output**: `['document', 'is', 'sample', 'second', 'the', 'this']`  
Explanation of Word-Level Features with `CountVectorizer`:

#### Step-by-Step Breakdown:

1. **Input Text**: We have two text documents:
    
    * `"This is a sample document."`
        
    * `"This document is the second document."`
        
    
    These are the two sentences we want to analyze, and we want to extract features (unique words) from them.
    
2. **CountVectorizer**: `CountVectorizer` is a tool that:
    
    * **Tokenizes** the text (splits it into individual words).
        
    * **Removes punctuation** (such as periods).
        
    * **Lowercases** everything by default.
        
    * Creates a **vocabulary** of unique words (also called features) from the documents.
        
    
    This vocabulary represents all unique words in the input data.
    
3. **Vocabulary (Word Features)**: After running `fit_transform`, the `CountVectorizer` looks at both sentences and <mark>extracts the </mark> **<mark>unique words</mark>** that appear in them. These unique words are our **word-level features**:
    
    ```python
    print(vectorizer.get_feature_names_out())
    ```
    
    The output will be:
    
    ```python
    ['document', 'is', 'sample', 'second', 'the', 'this']
    ```
    
    **Explanation of the Output**:
    
    * The output is a **list of unique words** (features) from both documents.
        
    * **Order of Words**: The words are sorted alphabetically by default. This list of words represents the **features** (or vocabulary) extracted from the documents. The words in the output are:
        
        * `'document'`: Appears in both sentences.
            
        * `'is'`: Appears in both sentences.
            
        * `'sample'`: Appears in the first sentence.
            
        * `'second'`: Appears in the second sentence.
            
        * `'the'`: Appears in the second sentence.
            
        * `'this'`: Appears in both sentences.
            
4. **Transformation**: After extracting the features, `CountVectorizer` creates a **sparse matrix** that counts how many times each word appears in each document. Although you didn’t print it, the transformation result will look like this:
    
    ```python
    Document 1: [1, 1, 1, 0, 1, 1]   -> ("This is a sample document.")
    Document 2: [2, 1, 0, 1, 1, 1]   -> ("This document is the second document.")
    ```
    
    Each number corresponds to the **frequency** of each word (feature) from the vocabulary in the respective document:
    
    * **Document 1**:
        
        * `'document'` appears **1** time.
            
        * `'is'` appears **1** time.
            
        * `'sample'` appears **1** time.
            
        * `'second'` doesn't appear (**0** times).
            
        * `'the'` appears **1** time.
            
        * `'this'` appears **1** time.
            
    * **Document 2**:
        
        * `'document'` appears **2** times.
            
        * `'is'` appears **1** time.
            
        * `'sample'` doesn't appear (**0** times).
            
        * `'second'` appears **1** time.
            
        * `'the'` appears **1** time.
            
        * `'this'` appears **1** time.
            
              
            

### 9\. **Part-of-Speech Tagging**

```python
import nltk
nltk.download('averaged_perceptron_tagger')
text = word_tokenize("NLP is exciting.")
pos_tags = nltk.pos_tag(text)
print(pos_tags)
```

**Output**: `[('NLP', 'NNP'), ('is', 'VBZ'), ('exciting', 'VBG'), ('.', '.')]`

### 10\. **Sentiment Analysis**

```python
from textblob import TextBlob
text = "I love this beautiful weather!"
analysis = TextBlob(text)
print(analysis.sentiment)
```

**Output**: `Sentiment(polarity=0.85, subjectivity=1.0)`

### 11\. **Stemming Algorithm**

  
Stemming involves chopping off the end (suffix) of a word to reduce it to its root form.

```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
print(stemmer.stem("running"))
```

**Output**: `'run'`

### 12\. **Lemma**

  
Lemmatization is a more sophisticated process that reduces a word to its **dictionary form** (lemma), considering its meaning and part of speech. It requires knowing the part of speech (POS) to find the correct form.

```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running", pos="v"))
```

**Output**: `'run'`  
Unlike stemming, lemmatization reduces **'running'** to the actual word **'run'** by considering that "running" is a verb (thanks to the `pos="v"` flag). Lemmatization is more context-aware and produces valid words.

### 13\. **N-grams (Replacing Suffix Stripping)**

**Definition**: **N-grams** are continuous sequences of **N** items (usually words) from a given text. It is commonly used for tasks like text prediction, machine translation, or understanding word dependencies in text.

* **Unigram (1-gram)**: Single words.
    
* **Bigram (2-gram)**: Pairs of consecutive words.
    
* **Trigram (3-gram)**: Triplets of consecutive words.
    

Let’s generate **bigrams** as an example:

```python
from nltk import ngrams

text = "Natural Language Processing is fun"
tokens = text.split()

# Generate bigrams (sequence of 2 words)
bigrams = list(ngrams(tokens, 2))
print(bigrams)
```

**Output**:

```python
[('Natural', 'Language'), ('Language', 'Processing'), ('Processing', 'is'), ('is', 'fun')]
```

* **N-grams** allow you to look at consecutive sequences of words in a text. In the above example, **<mark>bigrams</mark>** <mark> are formed by creating pairs of consecutive words: </mark> `('Natural', 'Language')`, `('Language', 'Processing')`, and so on.
    
* **Bigram** analysis can help in tasks like <mark>text prediction (e.g., "Natural Language" often precedes "Processing") or capturing dependencies between words.</mark>
    
* **N-grams**: N-grams are **syntax-based** and focus on the **exact sequence of words**. They don't capture deep semantic relationships between words.
    

You can extend this to **trigrams** (3-word sequences) or **n-grams** with any value of **N**:

```python
trigrams = list(ngrams(tokens, 3))
print(trigrams)
```

**Output**:

```python
[('Natural', 'Language', 'Processing'), ('Language', 'Processing', 'is'), ('Processing', 'is', 'fun')]
```

---

### 14\. **Topic Modeling (LDA)**

Latent Dirichlet Allocation (LDA) is used for finding abstract topics in a collection of documents. It identifies patterns of words that appear together across documents and groups them into topics.

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

texts = ["I love reading about machine learning.",
         "Deep learning is a subfield of machine learning.",
         "Data science includes machine learning."]

# Convert the documents into a matrix of token counts
vectorizer = CountVectorizer()
transformed_texts = vectorizer.fit_transform(texts)

# Apply LDA to extract topics
lda = LatentDirichletAllocation(n_components=2, random_state=0)
lda.fit(transformed_texts)

# Display the topics
for idx, topic in enumerate(lda.components_):
    print(f"Topic {idx + 1}:")
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]])
```

**Explanation**: The LDA model extracts **two topics** from a set of documents. It analyzes word co-occurrences and assigns them to the most probable topic. The printed words represent the most significant terms within each topic.

**Sample Output**:

```python
Topic 1: ['science', 'data', 'includes', 'learning', 'machine']
Topic 2: ['reading', 'about', 'love', 'learning', 'machine']
```

Here, **Topic 1** is about "machine learning in data science," while **Topic 2** may represent "machine learning as something the user enjoys."

---

### 15\. **Word2Vec**

Word2Vec is used to generate vector representations (embeddings) of words. It captures semantic relationships between words based on their contexts in sentences.

```python
from gensim.models import Word2Vec

# Define example sentences
sentences = [["machine", "learning", "is", "great"],
             ["NLP", "is", "fun"]]

# Train Word2Vec model
model = Word2Vec(sentences, min_count=1)

# Get the vector representation for the word 'machine'
print(model.wv['machine'])
```

**Explanation**: Word2Vec learns word representations by predicting words that appear close to each other in sentences. The model generates a **vector** (a list of numbers) for each word, capturing its meaning relative to other words in the training data.

**Sample Output**: (Vector representation of 'machine')

```python
[-0.00258912  0.00314215 -0.00196503  0.00107836 ...]
```

This vector represents the word 'machine' in multi-dimensional space, encoding its relationships with other words.

**Word2Vec**: Word2Vec captures **semantic meaning**. For instance, in Word2Vec, **<mark>“king”</mark>** <mark> and </mark> **<mark>“queen”</mark>** will have similar vectors because of their similar meanings, even if they don't appear next to each other often. N-grams can’t capture this level of meaning.

**<mark>Word2Vec</mark>** <mark> is much more sophisticated than N-grams because it captures </mark> **<mark>semantic similarity</mark>** between words, even if those words are **not directly adjacent** or **exact matches** in the text.

**Word2Vec** looks at **relationships** between words based on their context in a **larger corpus**. The vector for **"machine"** might be close to words like "learning," "AI," or "automation" in a multi-dimensional space, even though these words might not always appear directly next to "machine."

---

### 16\. **SkipGram**

<mark>SkipGram is a variant of Word2Vec</mark>. It focuses on predicting surrounding words (context) based on a target word.

```python
model = Word2Vec(sentences, min_count=1, sg=1)  # sg=1 means SkipGram model
print(model.wv['learning'])
```

**Explanation**: In SkipGram, the model learns to predict surrounding words given a target word. The vector for 'learning' reflects its role as a central word that helps predict nearby words like 'machine' and 'great.'

**Sample Output**: (Vector for 'learning')

```python
[-0.0034415   0.00489131 -0.0021938   0.00168509 ...]
```

The output is a vector representing the word 'learning,' with its meaning captured through the context provided in the training sentences.

---

### 17\. **Continuous Bag of Words (CBOW)**

To clarify, **Skip-gram** and **CBOW** (Continuous Bag of Words) are **two variants of the Word2Vec** model. Let’s explain the difference:

**Skip-gram Model** (One of the Word2Vec Models):

* **Goal**: The Skip-gram model predicts the **context words** (surrounding words) given a **target word**.
    
* **How it works**: It takes a **word in the center** of a context window and tries to predict the words around it (before and after).
    

#### Example:

Let’s say you have the sentence:

```python
"The dog is playing in the yard."
```

With <mark>Skip-gram, if the target word is </mark> **<mark>"playing"</mark>**, the model will try to predict the surrounding words (context) like:

* **"dog"**, **"is"**, **"in"**, **"the"**, **"yard"**.
    

In this case, "playing" is the **input** (center word), and the **output** is the surrounding words ("dog", "yard", etc.).

#### Key Points of Skip-gram:

* It is more effective for **small datasets** and can capture **rare words** better.
    
* The model maximizes the probability of predicting context words for a given target word.
    

**Diagram**:

```python
Skip-gram:
     Context words
     (dog, yard)
       ↑  ↑  ↑
     [ playing ]
     (Target word)
```

---

**CBOW (Continuous Bag of Words)** Model (The Other Word2Vec Model):

* **Goal**: The CBOW model predicts the **target word** based on the surrounding **context words**.
    
* **How it works**: I<mark>t takes the </mark> **<mark>context words</mark>** <mark> (before and after a word in a sentence) </mark> and tries to predict the **target word** (the word in the middle).
    

#### Example:

Using the same sentence:

```python
"The dog is playing in the yard."
```

<mark>With CBOW, if the context words are </mark> **<mark>"dog"</mark>**<mark>, </mark> **<mark>"is"</mark>**<mark>, </mark> **<mark>"in"</mark>**<mark>, </mark> **<mark>"the"</mark>**<mark>, </mark> **<mark>"yard"</mark>**<mark>, the model will try to predict the center (target) word </mark> **<mark>"playing"</mark>**<mark>.</mark>

In this case, "dog", "yard", etc., are the **input** (context), and the model tries to predict **"playing"** as the **output** (target word).

#### Key Points of CBOW:

* CBOW works better for **larger datasets** and is generally faster than Skip-gram.
    
* It tries to maximize the probability of predicting a word given the surrounding context words.
    

**Diagram**:

```python
CBOW:
     Context words
     (dog, yard)
        ↓  ↓  ↓
     [ playing ]
     (Target word)
```

---

### Summary of Differences Between Skip-gram and CBOW:

| **Feature** | **Skip-gram** | **CBOW** |
| --- | --- | --- |
| **Goal** | Predict context words given a target word | Predict target word given context words |
| **Input** | The **target word** | The **context words** |
| **Output** | The **surrounding words** (context) | The **target word** |
| **Effective For** | **Small datasets** and **rare words** | **Larger datasets**, faster training |
| **Training Speed** | **Slower** (more complex) | **Faster** |
| **Example** | Predict "dog", "yard" from "playing" | Predict "playing" from "dog", "yard" |

---

### When to Use:

* **Skip-gram** is better for **smaller datasets** and works well when you want to capture information about **rare words**.
    
* **CBOW** is better for **larger datasets** and is more efficient, especially when you care about faster training.
    

Both models belong to **Word2Vec** and are used for creating **word embeddings** based on context, but the direction of prediction is what sets them apart:

* **Skip-gram**: Word → Context.
    
* **CBOW**: Context → Word.
    

---

### 18\. **Abstractive Summarization**

The abstractive summary **rephrases** the text by condensing the idea into fewer words.

```python
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Complex text for summarization
text = """
The transformer model, introduced by researchers in 2017, has drastically changed how natural language processing tasks are approached. 
Before transformers, recurrent neural networks and convolutional networks were the go-to models for tasks such as machine translation and text summarization. 
However, transformers have outperformed these models in many ways, providing more accurate results and faster training times. 
As a result, they have been widely adopted in both academia and industry, leading to numerous breakthroughs in language-related AI applications.
"""

# Summarize the text
summary = summarizer(text, max_length=50, min_length=20, do_sample=False)
print("Original Text:\n", text)
print("\nAbstractive Summary:\n", summary[0]['summary_text'])
```

* The **input text** provides a more detailed and longer explanation of the **transformer model** and its impact on NLP.
    
* The **abstractive summary** will rephrase this text into a concise version, potentially changing the sentence structure and wording.
    

**Original Text**:

```python
The transformer model, introduced by researchers in 2017, has drastically changed how natural language processing tasks are approached. 
Before transformers, recurrent neural networks and convolutional networks were the go-to models for tasks such as machine translation and text summarization. 
However, transformers have outperformed these models in many ways, providing more accurate results and faster training times. 
As a result, they have been widely adopted in both academia and industry, leading to numerous breakthroughs in language-related AI applications.
```

**Abstractive Summary**:

```python
Introduced in 2017, the transformer model revolutionized natural language processing, outperforming previous models like recurrent and convolutional networks in tasks such as translation and summarization. It has led to breakthroughs in AI applications across academia and industry.
```

* The summary **rephrases** and **condenses** the original text, focusing on the key points without directly copying any specific sentences.
    

---

### 19\. **Extractive Summarization**

Extractive summarization selects key sentences from the original text and combines them into a summary.

```python
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

nlp = spacy.load("en_core_web_sm")
doc = nlp("The transformer model has revolutionized NLP tasks. It is widely used for translation, summarization, and question answering.")

# Extract word frequencies
word_frequencies = {}
for word in doc:
    if word.text.lower() not in STOP_WORDS and word.text.lower() not in punctuation:
        word_frequencies[word.text] = word_frequencies.get(word.text, 0) + 1

# Normalize the word frequencies
max_freq = max(word_frequencies.values())
for word in word_frequencies.keys():
    word_frequencies[word.text] = word_frequencies[word.text] / max_freq

# Score sentences based on word frequencies
sentence_scores = {}
for sent in doc.sents:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word.text]

# Select the top sentence
summarized_sentences = nlargest(1, sentence_scores, key=sentence_scores.get)
summary = ' '.join([sent.text for sent in summarized_sentences])
print(summary)
```

**Explanation**: This code identifies the most important sentence in the text based on word frequencies. The sentence with the most important words is selected for the summary.

**Sample Output**:

```python
"The transformer model has revolutionized NLP tasks."
```

This is an **extractive** summary, where the output is a direct selection of an important sentence from the original text.

**Abstractive summarization** is indeed more **intelligent** and sophisticated compared to **extractive summarization**. Here's why:

Why Abstractive Summarization is More Intelligent? Here is a quick comparison Abstractive VS Extractive:

| **Feature** | **Abstractive Summarization** | **Extractive Summarization** |
| --- | --- | --- |
| **Method** | Generates new sentences by understanding the text's meaning | Selects key sentences directly from the text |
| **Intelligence** | More intelligent and context-aware | Simpler, rule-based selection of sentences |
| **Naturalness** | Can produce summaries that sound <mark>more human-like a</mark>nd concise | Often sounds disjointed because it copies existing sentences |
| **Flexibility** | Can rephrase, paraphrase, and condense information | Rigid and dependent on sentence structure in the original text |
| **Best for** | Long, complex texts where concise and clear summaries are needed | Shorter, simpler texts where sentence selection is sufficient |
| **Model Complexity** | <mark>Uses deep learning models like transformers</mark> | Uses simple rule-based algorithms |

**Why Does Transformer-Based Summarization Code Looks Simpler?**

* **Pre-trained Models Do the Work**: In the case of **abstractive summarization**, the transformers (like GPT, BERT, T5) have already been trained. The model can summarize text with just a simple API call.
    
* **Deep Learning Power**: Transformers are designed to handle complex tasks like summarization, and once trained, they are **easy to use**. They simplify the summarization task because the model has already "learned" the task through millions of examples.
    

**Is Transformer Better than SpaCy?**

The question of whether **transformers** are better than **SpaCy** depends on the **task** and the **resources** available. Let’s compare them:

| **Feature** | **Transformers (e.g., GPT, BERT)** | **SpaCy (Traditional NLP)** |
| --- | --- | --- |
| **Capabilities** | Handles complex tasks like summarization, text generation, translation, and question answering | Great for tasks like tokenization, part-of-speech tagging, named entity recognition, dependency parsing |
| **Complexity** | Internally complex but **easy to use** through pre-trained models like Hugging Face pipelines | <mark>Requires more </mark> **<mark>manual setup</mark>** and implementation for tasks like extractive summarization |
| **Training** | Requires **pre-trained models**, often computationally expensive to train | Fast and efficient, doesn't require large models or expensive GPUs |
| **Power** | Captures **deep context** and semantics in language | Based on **rule-based** and **statistical** methods |
| **Speed & Efficiency** | **Slower**, requires more computational resources (GPU, large memory) | **Faster** and highly optimized for many NLP tasks |
| **Use Case** | Best for complex tasks like **abstractive summarization**, **translation**, and **text generation** | Best for **lighter tasks** like **tokenization**, **NER**, and **simple** text analysis |
| **Training Effort** | Uses **large pre-trained models** that take weeks or months to train on specialized hardware | Pre-trained models for tasks like NER, POS tagging, or text classification, but much simpler and faster to run |

### 20\. **Text to Speech (TTS)**

Text to Speech converts written text into spoken audio.

```python
import pyttsx3

engine = pyttsx3.init()
text = "Natural Language Processing is fun to learn."
engine.say(text)
engine.runAndWait()
```

**Explanation**: This example uses a Python TTS engine to read out the text. You won't see printed output here, but the text is spoken aloud.

**Sample Output**: Spoken version of the text: "Natural Language Processing is fun to learn."

---

### 21\. WhitespaceTokenizer

`WhitespaceTokenizer`, which only tokenizes based on spaces:

```python
from nltk.tokenize import WhitespaceTokenizer

tokenizer = WhitespaceTokenizer()
text = "Learning NLP's basics. It's quite interesting!"
tokens = tokenizer.tokenize(text)
print(tokens)
```

**Output**:

```python
['Learning', "NLP's", 'basics.', "It's", 'quite', 'interesting!']
```

### Explanation:

* `WhitespaceTokenizer` doesn't split punctuation or contractions. It treats everything between spaces as a single token, which leads to tokens like `"NLP's"` and `"basics."`.