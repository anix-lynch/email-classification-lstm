---
title: "20 Spacy concepts with Before-and-After Examples"
seoTitle: "20 Spacy concepts with Before-and-After Examples"
seoDescription: "20 Spacy concepts with Before-and-After Examples"
datePublished: Thu Oct 03 2024 13:04:41 GMT+0000 (Coordinated Universal Time)
cuid: cm1tb6alm00070aigewx6ez47
slug: 20-spacy-concepts-with-before-and-after-examples
tags: ai, python, data-science, nlp, space

---

### 1\. **Loading Language Model (spacy.load)** 🧠

**Boilerplate Code**:

```python
import spacy
nlp = spacy.load("en_core_web_sm")
```

**Use Case**: Load a **pre-trained language model** to analyze text. 🧠

**Goal**: Initialize spaCy’s language model for various NLP tasks. 🎯

**Sample Code**:

```python
# Load English language model
nlp = spacy.load("en_core_web_sm")

# Example text
doc = nlp("This is a test sentence.")
print(doc)
```

**Before Example**: needs a pre-trained model but doesn’t know how to load it. 🤔

```python
Need: Pre-trained NLP model.
```

**After Example**: With **spacy.load()**, can now analyze text using the loaded model! 🧠

```python
Loaded Model: "This is a test sentence."
```

**Challenge**: 🌟 Try loading a larger model (e.g., `en_core_web_md` or `en_core_web_lg`) for more advanced tasks.

---

### 2\. **Tokenization (spacy.tokens.Token)** ✂️

**Boilerplate Code**:

```python
from spacy.tokens import Token
```

**Use Case**: Split text into **tokens** (words or punctuation) using spaCy’s tokenizer. ✂️

**Goal**: Tokenize text into individual words or punctuation marks. 🎯

**Sample Code**:

```python
# Example text
doc = nlp("This is a test sentence.")

# Tokenize the text
tokens = [token.text for token in doc]
print(tokens)
```

**Before Example**:  
has text but doesn’t know how to split it into words. 🤔

```python
Text: "This is a test sentence."
```

**After Example**: With **spaCy tokenization**, the text is split into tokens! ✂️

```python
Tokens: ['This', 'is', 'a', 'test', 'sentence', '.']
```

**Challenge**: 🌟 Try tokenizing a more complex sentence with punctuation and special characters.

---

### 3\. **Named Entity Recognition (NER with spacy.ents)** 🏢

**Boilerplate Code**:

```python
from spacy.tokens import Doc
```

**Use Case**: Extract **named entities** (people, organizations, locations) from text. 🏢

**Goal**: Identify and classify entities like names, dates, or places. 🎯

**Sample Code**:

```python
# Example text
doc = nlp("Barack Obama was the president of the United States.")

# Extract named entities
for ent in doc.ents:
    print(ent.text, ent.label_)
```

**Before Example**: The intern has text but doesn’t know which words refer to names or places. 🤔

```python
Text: "Barack Obama was the president of the United States."
```

**After Example**: With **spaCy NER**, the intern identifies named entities! 🏢

```python
Named Entities: "Barack Obama" (PERSON), "United States" (GPE)
```

**Challenge**: 🌟 Try analyzing a news article and extract all named entities like people, locations, and organizations.

---

### 4\. **Part-of-Speech Tagging (spacy.pos\_ and spacy.tag\_)** 🏷️

**Boilerplate Code**:

```python
from spacy.tokens import Doc
```

**Use Case**: Assign **part-of-speech (POS)** tags to each word in a sentence. 🏷️

**Goal**: Understand the grammatical role of each word in a sentence. 🎯

**Sample Code**:

```python
# Example text
doc = nlp("This is a test sentence.")

# Get POS tags for each token
for token in doc:
    print(token.text, token.pos_, token.tag_)
```

**Before Example**: The intern has a sentence but doesn’t know the grammatical role of each word. 🤔

```python
Sentence: "This is a test sentence."
```

**After Example**: With **POS tagging**, each word is tagged with its grammatical role! 🏷️

```python
POS Tags: ('This', 'DET'), ('is', 'AUX'), ('a', 'DET'), ...
```

**Challenge**: 🌟 Try analyzing more complex sentences and observe how POS tags change with different sentence structures.

---

### 5\. **Dependency Parsing (spacy.dep\_)** 🌳

**Boilerplate Code**:

```python
from spacy.tokens import Doc
```

**Use Case**: Extract **dependency relationships** between words (e.g., subject-verb-object). 🌳

**Goal**: Understand the syntactic structure of a sentence. 🎯

**Sample Code**:

```python
# Example text
doc = nlp("I love programming in Python.")

# Display dependencies
for token in doc:
    print(token.text, token.dep_, token.head.text)
```

**Before Example**: The intern has a sentence but doesn’t understand the grammatical relationships between words. 🤔

```python
Sentence: "I love programming in Python."
```

**After Example**: With **dependency parsing**, the intern understands how words relate to each other! 🌳

```python
Dependencies: ('I', 'nsubj', 'love'), ('love', 'ROOT', 'love'), ...
```

**Challenge**: 🌟 Try visualizing dependencies using `spacy.displacy.render()` for a better understanding of sentence structure.

---

### 6\. **Similarity Comparison (spacy.similarity)** 🔗

**Boilerplate Code**:

```python
from spacy.tokens import Doc
```

**Use Case**: Compare the **similarity** between words, sentences, or documents. 🔗

**Goal**: Measure how similar two pieces of text are. 🎯

**Sample Code**:

```python
# Example sentences
doc1 = nlp("I love pizza.")
doc2 = nlp("I like pasta.")

# Compare similarity
similarity_score = doc1.similarity(doc2)
print(similarity_score)
```

**Before Example**: The intern has two sentences but doesn’t know how to compare their similarity. 🤔

```python
Sentences: "I love pizza." vs. "I like pasta."
```

**After Example**: With **similarity comparison**, the intern can measure how similar they are! 🔗

```python
Similarity Score: 0.8
```

**Challenge**: 🌟 Try comparing the similarity between longer documents or paragraphs.

---

### 7\. **Text Lemmatization (spacy.lemma\_)** 🍂

**Boilerplate Code**:

```python
from spacy.tokens import Token
```

**Use Case**: Perform **lemmatization**, which reduces words to their base form (e.g., "running" → "run"). 🍂

**Goal**: Normalize words to their dictionary form for easier analysis. 🎯

**Sample Code**:

```python
# Example text
doc = nlp("The cats are running in the garden.")

# Get lemmas for each token
lemmas = [token.lemma_ for token in doc]
print(lemmas)
```

**Before Example**: The intern has words in different forms but wants to normalize them. 🤔

```python
Words: "cats", "running", "garden"
```

**After Example**: With **lemmatization**, the intern reduces words to their base forms! 🍂

```python
Lemmas: ['the', 'cat', 'be', 'run', 'in', 'the', 'garden']
```

**Challenge**: 🌟 Try lemmatizing text in different tenses or forms and observe the results.

---

### 8\. **Custom Named Entity Recognition (NER)** 🔧

**Boilerplate Code**:

```python
from spacy.tokens import Span
```

**Use Case**: Create **custom named entities** by labeling specific text patterns. 🔧

**Goal**: Extend spaCy’s NER capabilities by adding custom entities. 🎯

**Sample Code**:

```python
# Define custom entity
doc = nlp("Apple is looking at buying U.K. startup for $1 billion.")
org = Span(doc, 0, 1, label="ORG")

# Add custom entity
doc.ents = list(doc.ents) + [org]
print([(ent.text, ent.label_) for ent in doc.ents])
```

**Before Example**: The intern has a company name in the text but it’s not labeled as an entity. 🤔

```python
Text: "Apple is looking at buying U.K. startup."
```

**After Example**: With **custom NER**, the intern labels "Apple" as an organization! 🔧

```python
Entities: [('Apple', 'ORG'), ('U.K.', 'GPE')]
```

**Challenge**: 🌟 Try adding custom entities for different types of data like product names or company names.

---

### 9\. **Word Vector Representation (spacy.vocab.vectors)** 🔠

\*\*Boilerplate Code

\*\*:

```python
from spacy.vocab import Vectors
```

**Use Case**: Use **word vectors** to represent words as numerical vectors for machine learning tasks. 🔠

**Goal**: Convert words into vectors to perform mathematical operations on text. 🎯

**Sample Code**:

```python
# Example word
word = nlp("apple")

# Get word vector
vector = word.vector
print(vector[:5])  # Print first 5 elements of the vector
```

**Before Example**: The intern has words but doesn’t know how to represent them as numerical vectors. 🤔

```python
Word: "apple"
```

**After Example**: With **word vectors**, the word is represented as a numerical vector! 🔠

```python
Word Vector: [0.231, 0.127, 0.654, ...]
```

**Challenge**: 🌟 Try using vectors for similarity comparison between words or performing arithmetic operations on words (e.g., king - man + woman = queen).

---

### 10\. **Visualizing Dependencies (spacy.displacy.render)** 🖼️

**Boilerplate Code**:

```python
from spacy import displacy
```

**Use Case**: **Visualize sentence structure** using dependency parsing. 🖼️

**Goal**: Generate a visual representation of how words in a sentence are related. 🎯

**Sample Code**:

```python
# Example text
doc = nlp("I love programming in Python.")

# Render dependency graph
displacy.render(doc, style="dep", jupyter=True)
```

**Before Example**: The intern has a sentence but finds it hard to understand how the words relate. 🤔

```python
Sentence: "I love programming in Python."
```

**After Example**: With **displacy**, the intern can see a visual diagram of the sentence structure! 🖼️

```python
Visual: Arrows showing grammatical relationships between words.
```

**Challenge**: 🌟 Try visualizing more complex sentences or paragraphs and see how the dependency structure changes.

---

### 11\. **Pooling Layers (spacy.tokens.Pool)** 🏊‍♂️

**Boilerplate Code**:

```python
from spacy.tokens import Doc
```

**Use Case**: Perform **pooling operations** like sum or max-pooling over tokens to reduce dimensionality. 🏊‍♂️

**Goal**: Apply pooling operations over vectors of tokens. 🎯

**Sample Code**:

```python
# Example text
doc = nlp("I love programming in Python.")

# Perform sum pooling
sum_pooling = sum(token.vector for token in doc)
print(sum_pooling[:5])  # Print the first 5 elements of the pooled vector
```

**Before Example**: The intern has word vectors but needs to reduce their size for further analysis. 🤔

```python
Word Vectors: [Vector of each word in the sentence]
```

**After Example**: With **pooling**, the intern reduces multiple vectors into a smaller vector! 🏊‍♂️

```python
Pooled Vector: [Sum of word vectors]
```

**Challenge**: 🌟 Try experimenting with different pooling methods like max-pooling and average-pooling.

---

### 12\. **Text Classification (spacy.pipeline.TextCategorizer)** 🏆

**Boilerplate Code**:

```python
from spacy.pipeline import TextCategorizer
```

**Use Case**: **Classify text** into categories like positive/negative or news/sports using spaCy's text categorizer. 🏆

**Goal**: Build a text classifier for sentiment analysis or document categorization. 🎯

**Sample Code**:

```python
# Initialize text categorizer
textcat = nlp.add_pipe("textcat")

# Add labels to text categorizer
textcat.add_label("POSITIVE")
textcat.add_label("NEGATIVE")

# Example sentence
doc = nlp("This is an awesome product!")

# Predict category
print(doc.cats)
```

**Before Example**: The intern has text but doesn’t know how to categorize it (positive or negative). 🤔

```python
Sentence: "This is an awesome product!"
```

**After Example**: With **TextCategorizer**, the text is categorized into positive or negative! 🏆

```python
Categories: {'POSITIVE': 0.85, 'NEGATIVE': 0.15}
```

**Challenge**: 🌟 Try training the text classifier on a larger dataset for better performance.

---

### 13\. **Document Similarity (doc.similarity)** 🔗

**Boilerplate Code**:

```python
from spacy.tokens import Doc
```

**Use Case**: Compare the **similarity between documents** using spaCy’s built-in similarity function. 🔗

**Goal**: Measure how similar two pieces of text are. 🎯

**Sample Code**:

```python
# Example sentences
doc1 = nlp("I love playing football.")
doc2 = nlp("I enjoy soccer.")

# Compare document similarity
similarity = doc1.similarity(doc2)
print(similarity)
```

**Before Example**: The intern has two texts but doesn’t know how similar they are. 🤔

```python
Text1: "I love playing football." 
Text2: "I enjoy soccer."
```

**After Example**: With **similarity comparison**, the intern can measure how similar the texts are! 🔗

```python
Similarity Score: 0.92
```

**Challenge**: 🌟 Try comparing the similarity between different types of documents like news articles or research papers.

---

### 14\. **Custom Token Attributes (spacy.tokens.Token.set\_extension)** 🔧

**Boilerplate Code**:

```python
from spacy.tokens import Token
```

**Use Case**: Add **custom attributes** to tokens to store additional information like polarity or frequency. 🔧

**Goal**: Extend tokens with custom attributes to suit your NLP needs. 🎯

**Sample Code**:

```python
# Define custom token attribute
Token.set_extension('is_positive', default=False)

# Example text
doc = nlp("This is a great product!")

# Set custom attribute for specific tokens
for token in doc:
    if token.text == "great":
        token._.is_positive = True
    print(token.text, token._.is_positive)
```

**Before Example**: The intern wants to tag words like "great" with a custom attribute (e.g., positivity). 🤔

```python
Sentence: "This is a great product!"
```

**After Example**: With **custom token attributes**, the intern tags specific words with custom attributes! 🔧

```python
Token Attributes: "great" → is_positive = True
```

**Challenge**: 🌟 Try adding custom attributes to other tokens like "excellent" or "awesome."

---

### 15\. **Document Vectors (doc.vector)** 🧮

**Boilerplate Code**:

```python
from spacy.tokens import Doc
```

**Use Case**: Extract the **document vector**, which is a numerical representation of the entire document. 🧮

**Goal**: Represent an entire document as a vector for similarity comparisons or machine learning tasks. 🎯

**Sample Code**:

```python
# Example text
doc = nlp("I love programming in Python.")

# Get document vector
doc_vector = doc.vector
print(doc_vector[:5])  # Print the first 5 elements of the vector
```

**Before Example**: has text but doesn’t know how to convert the entire document into a vector. 🤔

```python
Text: "I love programming in Python."
```

**After Example**: With **doc.vector**, convert the text into a numerical vector! 🧮

```python
Document Vector: [0.23, 0.56, 0.12, ...]
```

**Challenge**: 🌟 Try comparing the document vectors of two similar texts.

---

### 16\. **Matcher (spacy.matcher.Matcher)** 🔍

**Boilerplate Code**:

```python
from spacy.matcher import Matcher
```

**Use Case**: Use the **Matcher** to find specific patterns in the text (e.g., word sequences or phrases). 🔍

**<mark>Goal</mark>**<mark>: Identify specific sequences of words based on patterns</mark>. 🎯

**Sample Code**:

```python
# Initialize the matcher
matcher = Matcher(nlp.vocab)

# Define pattern (e.g., "New York City")
pattern = [{"TEXT": "New"}, {"TEXT": "York"}, {"TEXT": "City"}]

# Add pattern to matcher
matcher.add("NYC_PATTERN", [pattern])

# Example text
doc = nlp("I visited New York City last year.")

# Find matches
matches = matcher(doc)
for match_id, start, end in matches:
    print(doc[start:end].text)
```

**Before Example**: we want to find a specific phrase (e.g., "New York City") but doesn’t know how to identify it. 🤔

```python
Text: "I visited New York City last year."
```

**After Example**: With **Matcher**, we find the phrase in the text! 🔍

```python
Match Found: "New York City"
```

**Challenge**: 🌟 Try creating more complex patterns, like searching for specific parts of speech or combinations of words.

---

### 17\. **Text Entity Linking (spacy.pipeline.EntityLinker)** 🔗

**Boilerplate Code**:

```python
from spacy.pipeline import EntityLinker
```

**Use Case**: Link named entities to **external knowledge bases** like Wikipedia. 🔗

**Goal**: <mark>Provide more context for named entities by linking them to real-world information. 🎯</mark>

**Sample Code**:

```python
# Initialize entity linker
linker = nlp.add_pipe("entity_linker")

# Example text
doc = nlp("Google was founded by Larry Page and Sergey Brin.")

# Get linked entities
for ent in doc.ents:
    print(ent.text, ent.kb_id_)
```

**Before Example**: we want to identify entities but doesn’t have additional information about them. 🤔

```python
Entities: "Google", "Larry Page", "Sergey Brin"
```

**After Example**: With **EntityLinker**, we link entities to real-world knowledge! 🔗

```python
Linked Entities: "Google" → Wikipedia ID, "Larry Page" → Wikipedia ID
```

**Challenge**: 🌟 Try linking entities to other knowledge bases like Wikidata or custom datasets.

---

### 18\. **Sentence Segmentation (doc.sents)** ✂️

**Boilerplate Code**:

```python
from spacy.tokens import Doc
```

**Use Case**: Split text into **sentences** using spaCy’s sentence boundary detection. ✂️

**Goal**: Segment text into individual sentences for further analysis. 🎯

**Sample Code**:

```python
# Example text
doc = nlp("This is the first sentence. Here's another one!")

# Extract sentences
for sent in doc.sents:
    print(sent.text)
```

**Before Example**: we has a paragraph but doesn’t know how to split it into individual sentences. 🤔

```python
Text: "This is the first sentence. Here's another one!"
```

**After Example**: With <mark> **sentence segmentation**</mark>, the text is split into separate sentences! ✂️

```python
Sentences: "This is the first sentence." "Here's another one!"
```

**Challenge**: 🌟 Try segmenting a longer article or text document into individual sentences.

---

### 19\. **Pipeline Customization (spacy.pipe)** 🔄

* **Default pipeline**: When you run `nlp("Google was founded in 1998.")`, spaCy will typically process the text through all components: tokenization, POS tagging, NER, etc.
    
* **Disable NER**: In this example, you're disabling NER (which identifies entities like "Google" or "1998"). You may want to do this if you're not interested in identifying entities to speed up the process. Instead, you're only interested in **POS tagging** (figuring out if each word is a noun, verb, etc.)  
    **Boilerplate Code**:
    

```python
from spacy.language import Language
```

**Use Case**: Customize the **NLP pipeline** by adding or removing components (e.g., NER, TextCategorizer). 🔄

**Goal**: Tailor the NLP pipeline to your specific needs by adding or removing components. 🎯

**Sample Code**:

```python
# Disable Named Entity Recognition (NER)
with nlp.disable_pipes("ner"):
    doc = nlp("Google was founded in 1998.")

# Process text without NER
print([(token.text, token.pos_) for token in doc])
```

**Before Example**: We run a full NLP pipeline but doesn’t need some components like NER. 🤔

```python
Text: "Google was founded in 1998."
```

**After Example**: With **pipeline customization**, we disables unnecessary components! 🔄

```python
Pipeline: Disabled "ner", only POS tagging applied.
```

**Challenge**: 🌟 Try creating a custom pipeline with only the components you need for a specific task.

---

### 20\. **Document Extension (spacy.tokens.Doc.set\_extension)** 🛠️

**Boilerplate Code**:

```python
from spacy.tokens import Doc
```

**Use Case**: Add **custom attributes** to the entire document (not just tokens) for additional processing. 🛠️

**Goal**: Extend spaCy's `Doc` object to store custom attributes for the entire text. 🎯

**Sample Code**:

```python
# Define custom document attribute
Doc.set_extension('is_technical', default=False)

# Example text
doc = nlp("Python is a popular programming language.")

# Set custom attribute for the document
doc._.is_technical = True
print(doc._.is_technical)
```

**Before Example**: <mark>we want to tag entire documents with custom attributes (e.g., technical or non-technical). 🤔</mark>

```python
Document: "Python is a popular programming language."
```

**After Example**: With **document extension**, we add a custom attribute to the entire document! 🛠️

```python
Custom Attribute: is_technical = True
```

**Challenge**: 🌟 Try adding more custom attributes at the document level for specific types of analysis.  
  
**Bonus Point:**  
Both **NLTK** and **spaCy** are popular libraries for Natural Language Processing (NLP), but they serve slightly different purposes, and your choice depends on your needs.

When to choose **spaCy**:

* **Speed**: <mark>spaCy is faster</mark> and more efficient, making it ideal for real-time applications and larger datasets.
    
* **Modern NLP**: It's designed with modern NLP tasks in mind, like Named Entity Recognition (NER), Dependency Parsing, and Word Vectors.
    
* **Ease of use**: spaCy comes with pre-trained models that are ready to use, making it simpler to get started on common tasks without much setup.
    
* **Deep Learning**: <mark>If you plan to integrate with deep learning frameworks like TensorFlow or PyTorch, spaCy is easier to work with</mark>.
    

When to choose **NLTK**:

* **Flexibility and Variety**: NLTK offers a wider variety of tools and datasets for NLP research, covering tasks like tokenization, parsing, and corpora access.
    
* **Learning and Research**: It's a great library for teaching and learning NLP, with more academic features. It also includes a variety of text processing techniques and algorithms.
    
* **Customization**: NLTK gives you more control and customization, but it's slower and more manually intensive compared to spaCy.
    

Default Choice:

If you're looking for **speed, simplicity, and modern NLP features**, **<mark>spaCy</mark>** <mark> is the better default choice.</mark> If you need **flexibility** and **want to dive deeper into NLP theory** or work with a variety of text processing tools, then NLTK might be more suitable.

---