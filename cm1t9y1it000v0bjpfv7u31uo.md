---
title: "20 NLTK concepts with Before-and-After Examples"
seoTitle: "20 NLTK concepts with Before-and-After Examples"
seoDescription: "20 NLTK concepts with Before-and-After Examples"
datePublished: Thu Oct 03 2024 12:30:16 GMT+0000 (Coordinated Universal Time)
cuid: cm1t9y1it000v0bjpfv7u31uo
slug: 20-nltk-concepts-with-before-and-after-examples
tags: ai, python, data-science, nlp, nltk

---

### 1\. **Tokenization (nltk.word\_tokenize)** ✂️

**Boilerplate Code**:

```python
from nltk.tokenize import word_tokenize
```

**Use Case**: **Break text into words or sentences** (Tokenization). ✂️

**Goal**: Split text into individual words or sentences for further analysis. 🎯

**Sample Code**:

```python
# Example text
text = "Hello world! How are you?"

# Tokenize the text into words
tokens = word_tokenize(text)
print(tokens)
```

**Before Example**: The intern has a text but doesn't know how to split it into words for analysis. 🤔

```python
Text: "Hello world! How are you?"
```

**After Example**: With **word\_tokenize()**, the text is split into tokens (words)! ✂️

```python
Tokens: ['Hello', 'world', '!', 'How', 'are', 'you', '?']
```

**Challenge**: 🌟 Try tokenizing a paragraph into sentences using `sent_tokenize()`.

---

### 2\. **Stopwords Removal (nltk.corpus.stopwords)** 🛑

**Boilerplate Code**:

```python
from nltk.corpus import stopwords
```

**Use Case**: **<mark>Remove common words</mark>** <mark> (like "the", "is", "in") that don’t add much meaning to the analysis. </mark> 🛑

**Goal**: Filter out stopwords to focus on important words. 🎯

**Sample Code**:

```python
# Get English stopwords
stop_words = set(stopwords.words('english'))

# Example sentence
words = ["This", "is", "an", "example", "sentence"]

# Filter out stopwords
filtered_words = [word for word in words if word.lower() not in stop_words]
print(filtered_words)
```

**Before Example**: The intern has a sentence but doesn’t want to analyze common words like "is" or "the". 🤔

```python
Sentence: ["This", "is", "an", "example", "sentence"]
```

**After Example**: <mark>With </mark> **<mark>stopwords removal</mark>**<mark>, only important words remain</mark>! 🛑

```python
Filtered Words: ['example', 'sentence']
```

**Challenge**: 🌟 Try applying stopwords removal to a large text and see how much it reduces the word count.

---

### 3\. **Stemming (nltk.stem.PorterStemmer)** 🌱

**Boilerplate Code**:

```python
from nltk.stem import PorterStemmer
```

**Use Case**: <mark> Reduce words to their </mark> **<mark>root form</mark>** (e.g., "running" → "run"). 🌱

**Goal**: <mark>Simplify words by removing prefixes or suffixes to group similar words together</mark>. 🎯

**Sample Code**:

```python
# Initialize the stemmer
stemmer = PorterStemmer()

# Example words
words = ["running", "runs", "runner"]

# Apply stemming
stemmed_words = [stemmer.stem(word) for word in words]
print(stemmed_words)
```

**Before Example**:

```python
Words: ["running", "runs", "runner"]
```

**After Example**: With **stemming**, all the words are reduced to their root form! 🌱

```python
Stemmed Words: ['run', 'run', 'runner']
```

**Challenge**: 🌟 Try experimenting with different stemmers like `LancasterStemmer()`.

---

### 4\. **Lemmatization (nltk.stem.WordNetLemmatizer)** 🍂

**Boilerplate Code**:

```python
from nltk.stem import WordNetLemmatizer
```

**Use Case**: Perform **lemmatization**, which reduces words to their **base form** but with context (e.g., "better" → "good"). 🍂

**Goal**: Group similar words by reducing them to their dictionary form. 🎯

**Sample Code**:

```python
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Example words
words = ["running", "better", "feet"]

# Apply lemmatization
lemmatized_words = [lemmatizer.lemmatize(word, pos="v") for word in words]
print(lemmatized_words)
```

**Before Example**:

```python
Words: ["running", "better", "feet"]
```

**After Example**: <mark> With </mark> **<mark>lemmatization</mark>**<mark>, the words are reduced to their base form! 🍂</mark>

```python
Lemmatized Words: ['run', 'good', 'foot']
```

**Challenge**: 🌟 Try lemmatizing words with different parts of speech (nouns, verbs, etc.).

---

### 5\. **Part-of-Speech Tagging (nltk.pos\_tag)** 🏷️

**Boilerplate Code**:

```python
from nltk import pos_tag
```

**Use Case**: **<mark>Tag each word</mark>** <mark> in a sentence with its part of speech (e.g., noun, verb, adjective). </mark> 🏷️

**Goal**: Understand the role each word plays in a sentence. 🎯

**Sample Code**:

```python
# Example sentence
sentence = ["This", "is", "a", "sample", "sentence"]

# Perform POS tagging
pos_tags = pos_tag(sentence)
print(pos_tags)
```

**Before Example**:

```python
Sentence: ["This", "is", "a", "sample", "sentence"]
```

**After Example**: With **pos\_tag()**, each word is tagged with its part of speech! 🏷️

```python
POS Tags: [('This', 'DT'), ('is', 'VBZ'), ('sample', 'NN'), ...]
```

**Challenge**: 🌟 Try analyzing a more complex sentence and see how the POS tags change.

---

### 6\. **Named Entity Recognition (**[**nltk.ne**](http://nltk.ne)**\_chunk)** 🏢

**Boilerplate Code**:

```python
from nltk import ne_chunk
```

**Use Case**: <mark>Identify </mark> **<mark>named entities</mark>** <mark> (like people, organizations, or locations) in a sentence. 🏢</mark>

**Goal**: Extract meaningful entities like names or places from text. 🎯

**Sample Code**:

```python
# Example sentence
sentence = "Barack Obama was the president of the United States."

# Tokenize and POS tag the sentence
words = word_tokenize(sentence)
pos_tags = pos_tag(words)

# Perform Named Entity Recognition
named_entities = ne_chunk(pos_tags)
print(named_entities)
```

**Before Example**:

```python
Sentence: "Barack Obama was the president of the United States."
```

**After Example**: <mark>With </mark> **<mark>ne_chunk()</mark>**<mark>, named entities are identified!</mark> 🏢

```python
Named Entities: Barack Obama (PERSON), United States (GPE)
```

**Challenge**: 🌟 Try applying NER to a news article and extract the key people and locations.

---

### 7\. **Frequency Distribution (nltk.FreqDist)** 📊

**Boilerplate Code**:

```python
from nltk import FreqDist
```

**Use Case**: Calculate the **frequency distribution** of words in a text. 📊

**Goal**: Find out which words appear the most in a given text. 🎯

**Sample Code**:

```python
# Example sentence
words = ["this", "is", "a", "sample", "sentence", "this", "is", "a", "test"]

# Calculate frequency distribution
fdist = FreqDist(words)
print(fdist.most_common(3))
```

**Before Example**:

```python
Words: ['this', 'is', 'a', 'sample', 'sentence', 'this', ...]
```

**After Example**: With **FreqDist()**, the intern knows which words appear the most! 📊

```python
Most Frequent: [('this', 2), ('is', 2), ('a', 2)]
```

**Challenge**: 🌟 Try plotting the frequency distribution using `fdist.plot()`.

---

### 8\. **Synonyms and Antonyms (nltk.corpus.wordnet)** 🔄

**Boilerplate Code**:

```python
from nltk.corpus import wordnet as wn
```

**Use Case**: Find **synonyms and antonyms** for a word using WordNet. 🔄

**Goal**: Get a list of similar or opposite words to expand vocabulary. 🎯

**Sample Code**:

```python
# Find synonyms and antonyms for "good"
synonyms = wn.synsets("good")
antonyms = [lemma.antonyms()[0].name() for syn in synonyms for lemma in syn.lemmas() if lemma.antonyms()]

print(s

ynonyms[:3], antonyms)
```

**Before Example**: The intern has a word but doesn’t know similar or opposite words. 🤔

```python
Word: "good"
```

**After Example**: <mark>With </mark> **<mark>WordNet</mark>**<mark>, we find synonyms and antonyms</mark>! 🔄

```python
Synonyms: 'good', 'well', 'right' | Antonyms: 'bad', 'evil'
```

**Challenge**: 🌟 Try finding synonyms and antonyms for other common words like "happy" or "sad."

---

### 9\. **Word Similarity (nltk.corpus.wordnet)** 🔗

**Boilerplate Code**:

```python
from nltk.corpus import wordnet as wn
```

**Use Case**: <mark>Calculate the </mark> **<mark>similarity between words</mark>** <mark> using semantic meaning from WordNet. 🔗</mark>

**Goal**: Measure how closely related two words are. 🎯

**Sample Code**:

```python
# Get WordNet synsets for two words
word1 = wn.synset("dog.n.01")
word2 = wn.synset("cat.n.01")

# Calculate similarity score
similarity = word1.wup_similarity(word2)
print(similarity)
```

**Before Example**: <mark>we have two words but doesn’t know how similar they are. 🤔</mark>

```python
Words: "dog", "cat"
```

**After Example**: <mark>With </mark> **<mark>wup_similarity()</mark>**<mark>, we find how closely related the words are! 🔗</mark>

```python
Similarity Score: 0.857
```

**Challenge**: 🌟 Try calculating similarities for other word pairs like "car" and "bicycle."

---

### 10\. **Text Classification (nltk.NaiveBayesClassifier)** 🏆

**Boilerplate Code**:

```python
from nltk.classify import NaiveBayesClassifier
```

**Use Case**: Build a simple **text classification model** using Naive Bayes. 🏆

**Goal**: <mark>Classify text into categories like positive/negative or spam/not spam. 🎯</mark>

**Sample Code**:

```python
# Sample training data
train_data = [({'word': 'happy'}, 'positive'), ({'word': 'sad'}, 'negative')]

# Train Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_data)

# Test the classifier
test_data = {'word': 'happy'}
print(classifier.classify(test_data))
```

**Before Example**: has labeled data but doesn’t know how to classify new text. 🤔

```python
Train Data: "happy" → positive, "sad" → negative
```

**After Example**: With **NaiveBayesClassifier()**, the text is classified into positive or negative! 🏆

```python
Test Data: "happy" → Classified as positive.
```

**Challenge**: 🌟 Try adding more training data and improve the classifier’s performance.

---

### 11\. **Text Corpora (nltk.corpus.gutenberg)** 📚

**Boilerplate Code**:

```python
from nltk.corpus import gutenberg
```

**Use Case**: Access large, public-domain texts from the **Gutenberg Project** for analysis. 📚

**Goal**: Use famous texts like Shakespeare and Austen for NLP tasks and analysis. 🎯

**Sample Code**:

```python
# Access Jane Austen's "Emma"
text = gutenberg.words('austen-emma.txt')
print(text[:20])
```

**Before Example**: need text data but doesn't know where to find famous public-domain books. 🤔

```python
Need: A large, famous text for analysis.
```

**After Example**: With **nltk.corpus.gutenberg**, the intern can access classic literature instantly! 📚

```python
Text: ['[', 'Emma', 'by', 'Jane', 'Austen', '1816', ']', 'VOLUME', 'I', ...]
```

**Challenge**: 🌟 Try analyzing the frequency distribution of words in a book like "Moby Dick."

---

### 12\. **Text Classification (nltk.DecisionTreeClassifier)** 🌳

**Boilerplate Code**:

```python
from nltk.classify import DecisionTreeClassifier
```

**Use Case**: Build a **Decision Tree Classifier** for text classification tasks. 🌳

**Goal**: <mark>Classify text into different categories using a decision tree. 🎯</mark>

**Sample Code**:

```python
# Sample training data
train_data = [({'word': 'happy'}, 'positive'), ({'word': 'sad'}, 'negative')]

# Train Decision Tree classifier
classifier = DecisionTreeClassifier.train(train_data)

# Test the classifier
test_data = {'word': 'sad'}
print(classifier.classify(test_data))
```

**Before Example**: has labeled data but doesn’t know how to classify new text. 🤔

```python
Train Data: "happy" → positive, "sad" → negative
```

**After Example**: With **DecisionTreeClassifier()**, the intern can classify text based on the tree structure! 🌳

```python
Test Data: "sad" → Classified as negative.
```

**Challenge**: 🌟 Try adding more categories and more training data to build a more complex decision tree.

---

### 13\. **N-Grams (nltk.ngrams)** 🔢

**Boilerplate Code**:

```python
from nltk import ngrams
```

**Use Case**: <mark>Generate </mark> **<mark>n-grams</mark>**<mark>, which are sequences of n words in a row (e.g., bigrams, trigrams</mark>). 🔢

**Goal**: <mark>Analyze text patterns by considering word combinations rather than single words. </mark> 🎯

**Sample Code**:

```python
# Example sentence
sentence = ['this', 'is', 'a', 'test', 'sentence']

# Generate bigrams (2-grams)
bigrams = list(ngrams(sentence, 2))
print(bigrams)
```

**Before Example**: has text data but doesn’t know how to analyze word pairs or sequences. 🤔

```python
Sentence: ['this', 'is', 'a', 'test', 'sentence']
```

**After Example**: With **ngrams()**, the intern can generate and analyze word pairs or triplets! 🔢

```python
Bigrams: [('this', 'is'), ('is', 'a'), ('a', 'test'), ...]
```

**Challenge**: 🌟 <mark>Try generating trigrams (3-grams) and find frequent word combinations in a text</mark>.

---

### 14\. **TF-IDF (nltk.text.TfidfTransformer)** 📊

**Boilerplate Code**:

```python
from sklearn.feature_extraction.text import TfidfTransformer
```

**Use Case**: Compute **<mark>TF-IDF</mark>** <mark> (Term Frequency-Inverse Document Frequency)</mark> to determine <mark>how important a word is</mark> in a document. 📊

**Goal**: <mark>Identify keywords</mark> that distinguish one document from another. 🎯

**Sample Code**:

```python
# Example word counts
word_counts = [[3, 0, 1], [2, 1, 0], [3, 0, 2]]

# Initialize TF-IDF Transformer
tfidf = TfidfTransformer()

# Fit and transform word counts
tfidf_matrix = tfidf.fit_transform(word_counts)
print(tfidf_matrix.toarray())
```

**Before Example**: has word counts but doesn’t know how to evaluate word importance across documents. 🤔

```python
Word Counts: [[3, 0, 1], [2, 1, 0], [3, 0, 2]]
```

**After Example**: With **TF-IDF**, the intern knows which words are the most important! 📊

```python
TF-IDF Scores: Calculated for each word in the corpus.
```

**Challenge**: 🌟 Try applying TF-IDF to a real-world document corpus and analyze the top keywords.

---

### 15\. **Text Generation (nltk.Text.generate)** 📝

**Boilerplate Code**:

```python
from nltk import Text
```

**Use Case**: **Generate text** based on patterns from a corpus. 📝

**Goal**: Create a model that can generate sentences based on learned word patterns. 🎯

**Sample Code**:

```python
# Example text
text = "This is a test sentence. This is another test sentence."

# Convert to NLTK Text object
text_model = Text(word_tokenize(text))

# Generate new text
text_model.generate()
```

**Before Example**: has a text corpus but doesn’t know how to generate new text based on it. 🤔

```python
Text Corpus: "This is a test sentence. This is another test sentence."
```

**After Example**: <mark>With </mark> **<mark>Text.generate()</mark>**<mark>, the intern can generate new sentences based on patterns! 📝</mark>

```python
Generated Text: "This is another test sentence."
```

**Challenge**: 🌟 Try training the model on a larger corpus and generate more complex sentences.

---

### 16\. **Sentiment Analysis (nltk.sentiment.vader)** 😃😞

**Boilerplate Code**:

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
```

**Use Case**: Perform **sentiment analysis** to determine if text is positive, negative, or neutral. 😃😞

**Goal**: Analyze the sentiment of a piece of text using V<mark>ADER (Valence Aware Dictionary and sEntiment Reasoner). 🎯</mark>

**Sample Code**:

```python
# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Example sentence
sentence = "I love this movie!"

# Perform sentiment analysis
score = analyzer.polarity_scores(sentence)
print(score)
```

**Before Example**: has a sentence but doesn’t know if the sentiment is positive or negative. 🤔

```python
Sentence: "I love this movie!"
```

**After Example**: With **VADER**, we get a sentiment score for the text! 😃

```python
Sentiment Score: {'neg': 0.0, 'neu': 0.2, 'pos': 0.8, 'compound': 0.6}
```

**Challenge**: 🌟 Try performing sentiment analysis on a set of movie reviews or tweets.

---

### 17\. **Text Collocations (nltk.Text.collocations)** 📚

**Boilerplate Code**:

```python
from nltk import Text
```

**Use Case**: <mark>Identify </mark> **<mark>collocations</mark>**<mark>,</mark> <mark>which are pairs of words that frequently appear together (e.g., "strong coffee"). 📚</mark>

**Goal**: Find commonly used word combinations in a text. 🎯

**Sample Code**:

```python
# Example text
text = "This is a test sentence. This test is very important."

# Convert to NLTK Text object
text_obj = Text(word_tokenize(text))

# Find collocations
text_obj.collocations()
```

**Before Example**: has text but doesn’t know which word pairs frequently appear together. 🤔

```python
Text: "This is a test sentence. This test is very important."
```

**After Example**: <mark>With </mark> **<mark>Text.collocations()</mark>**<mark>, the intern discovers frequent word pairs!</mark> 📚

```python
Collocations: [('test', 'sentence')]
```

**Challenge**: 🌟 Try finding collocations in a larger dataset like a book or news article.

---

### 18\. **Synset Relationships (nltk.corpus.wordnet.synsets)** 🔄

**Boilerplate Code**:

```python
from nltk.corpus import wordnet as wn
```

**Use Case**: Explore **synsets** (sets of synonyms) and their relationships (e.g., hypernyms, hyponyms). 🔄

**<mark>Goal</mark>**<mark>: Understand how words are related to each other in a hierarchy</mark>. 🎯

**Sample Code**:

```python
# Get synsets for the word 'dog'
synsets = wn.synsets('dog')

# Get hypernyms (more general terms) for 'dog'
hypernyms = synsets[0].hypernyms()
print(hypernyms)
```

**Before Example**: has a word but <mark>doesn’t know its broader or more specific meanings. 🤔</mark>

```python
Word: 'dog'
```

**After Example**: With **WordNet synsets**, the intern explores relationships like hypernyms and hyponyms! 🔄

```python
Hypernym: 'canine'
```

**Challenge**: 🌟 Try exploring other relationships like `meronyms` (part-whole relationships) or `antonyms`.

---

### 19\. **Text Parsing (nltk.RecursiveDescentParser)** 🌲

**Boilerplate Code**:

```python
from nltk import CFG
from nltk.parse import RecursiveDescentParser
```

**Use Case**: **Parse a sentence** using a grammar to understand its structure. 🌲

**Goal**: <mark>Break down sentences into their grammatical components using a recursive descent parser. 🎯</mark>

**Sample Code**:

```python
# Define a grammar
grammar = CFG.fromstring("""
    S -> NP VP
    NP -> 'I'
    VP -> 'love' NP
    NP -> 'Python'
""")

# Initialize the parser
parser = RecursiveDescentParser(grammar)

# Parse a sentence
sentence = 'I love Python'.split()
for tree in parser.parse(sentence):
    print(tree)
```

**Before Example**: has a sentence but doesn't know how to break it into its grammatical structure. 🤔

```python
Sentence: "I love Python"
```

**After Example**: With **RecursiveDescentParser()**, the sentence is parsed into its structure! 🌲

```python
Parsed Tree: (S (NP I) (VP love (NP Python)))
```

**Challenge**: 🌟 Try defining more complex grammars and parsing longer sentences.

---

### 20\. **Language Models (nltk.lm.models.MLE)** 🧠

**Boilerplate Code**:

```python
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
```

**Use Case**: Build a **<mark>Maximum Likelihood Estimation (MLE) Language Model</mark>**<mark>.</mark> 🧠

**Goal**: Create a language model that <mark> predicts the likelihood of word sequences.</mark> 🎯

**Sample Code**:

```python
# Sample text
text = [['this', 'is', 'a', 'test'], ['this', 'is', 'another', 'test']]

# Prepare data for MLE model
train_data, padded_vocab = padded_everygram_pipeline(2, text)

# Train the model
model = MLE(2)  # bigram model
model.fit(train_data, padded_vocab)

# Test the model
print(model.score('test', ['this', 'is']))
```

**Before Example**: has text data but doesn’t know how to model word sequences. 🤔

```python
Text: [['this', 'is', 'a', 'test'], ['this', 'is', 'another', 'test']]
```

**After Example**: With **MLE()**, we build a language model to predict word sequences! 🧠

```python
Score: Probability of 'test' following ['this', 'is']
```

**Challenge**: 🌟 Try training the language model on a large text corpus and generate new sentences.

---