---
title: "NLTK VS HuggingFace #2 - Emoji replacement, smart quotes handling, stop word removal, stemming, lemmatization, and spell checking"
seoTitle: "NLTK VS HuggingFace #2"
seoDescription: "NLTK VS HuggingFace #2 - Emoji replacement, smart quotes handling, stop word removal, stemming, lemmatization, and spell checking
"
datePublished: Wed Nov 13 2024 01:54:17 GMT+0000 (Coordinated Universal Time)
cuid: cm3f8a2xw000409meeywubeb9
slug: nltk-vs-huggingface-2-emoji-replacement-smart-quotes-handling-stop-word-removal-stemming-lemmatization-and-spell-checking
tags: ai, nlp, nltk, huggingface, transformers

---

# **Source code here:**

NLTK Code

[https://gist.github.com/a60a29d0aede72d3c9f5854bdd5d7916.git](https://gist.github.com/a60a29d0aede72d3c9f5854bdd5d7916.git)

[HuggingFace Code](https://gist.github.com/2f8b8167ae5c557dc027dc19f9a84c2b.git)

[https://gist.github.com/ed6e307157cb2790285247057c25e7f0.git](https://gist.github.com/ed6e307157cb2790285247057c25e7f0.git)

# **Table summarizing scenarios where NLTK is still preferable**

Compared to Hugging Face Transformers, based on specific NLP tasks and requirements:

| **NLP Task** | **Best Tool** | **Reason to Prefer NLTK** |
| --- | --- | --- |
| **Emoji Replacement** | NLTK + `demoji` library | Hugging Face doesn’t handle emoji descriptions directly. NLTK, with `demoji`, allows emoji conversion or removal. |
| **Tokenization with Special Cases** | Hugging Face Transformers | Hugging Face is generally preferred for tokenization, as it can handle complex text in various languages effectively. |
| **Removing Smart Quotes** | NLTK + Python string functions | Text cleaning steps like removing smart quotes are outside Hugging Face’s focus, making Python string methods simpler. |
| **Stemming (e.g., Porter Stemmer)** | NLTK | Hugging Face subword tokenization doesn’t replace stemming, while NLTK’s Porter Stemmer directly produces stems. |
| **Lemmatization** | NLTK (with WordNet) | NLTK’s lemmatization with WordNet offers true lemmatization, while Hugging Face uses POS tagging as an approximation. |
| **Stop Word Removal** | NLTK | NLTK has predefined lists of stop words, making it easier to remove common words for traditional NLP preprocessing. |
| **Punctuation Removal** | Hugging Face Transformers | Tokenizers break punctuation into separate tokens, which can be filtered directly after tokenization. |
| **Spell Checking** | NLTK + `spellchecker` library | Hugging Face doesn’t have spell-checking capabilities; NLTK works well with `spellchecker` for error detection. |
| **Access to Large Text Corpora** | NLTK | NLTK includes corpora like `movie_reviews` and `brown`, which are useful for training or testing models. |
| **Syntax Tree Visualization** | NLTK | Hugging Face doesn’t support syntax tree visualizations, while NLTK has `Tree` class for diagram generation. |
| **Batch POS Tagging on Large Datasets** | NLTK or SpaCy | NLTK’s `pos_tag_sents` allows batch POS tagging efficiently, especially for large datasets. |

---

### Summary:

* **Hugging Face Transformers**: Preferred for high-level NLP tasks (tokenization, POS tagging, NER, text generation) where deep learning is beneficial.
    
* **NLTK**: Still useful for foundational NLP tasks like stemming, lemmatization, stop word removal, syntax tree visualization, and accessing built-in corpora.
    

In most cases, Hugging Face works best for sentence- or document-level tasks, while NLTK handles specific preprocessing and text normalization tasks that Transformers don’t cover directly.

# **NLTK CODE**

### [Chunk 1: R](https://gist.github.com/2f8b8167ae5c557dc027dc19f9a84c2b.git)[ep](https://gist.github.com/e293e7c3f26dd7f4104a62a9d447ec95.git)[lacing Emojis with Text Descriptions](https://gist.github.com/2f8b8167ae5c557dc027dc19f9a84c2b.git)

1. [**Code**:](https://gist.github.com/2f8b8167ae5c557dc027dc19f9a84c2b.git)
    
    ```python
    # Import demoji for emoji processing
    import demoji
    
    # Sample text with an emoji
    happy_birthday = "Happy birthday!🎂"
    
    # Replace emojis with descriptions
    text_with_emojis_replaced = demoji.replace_with_desc(happy_birthday)
    print(text_with_emojis_replaced)  # Expected output: "Happy birthday! :birthday:"
    
    # Remove emojis entirely from the text
    text_with_emojis_removed = demoji.replace(happy_birthday, "")
    print(text_with_emojis_removed)  # Expected output: "Happy birthday!"
    ```
    
2. **Sample Output**:
    
    ```python
    Happy birthday! :birthday:
    Happy birthday!
    ```
    

---

### Chunk 2: Removing Smart Quotes

1. **Code**:
    
    ```python
    # Sample text with smart quotes
    text = "here is a string with “smart” quotes"
    
    # Replace smart quotes with standard quotes
    text = text.replace("“", "\"").replace("”", "\"")
    print(text)  # Expected output: here is a string with "smart" quotes
    ```
    
2. **Sample Output**:
    
    ```python
    here is a string with "smart" quotes
    ```
    

---

### Chunk 3: Tokenization Examples

1. **Code**:
    
    ```python
    import nltk
    from nltk import word_tokenize
    
    # Sample sentence for tokenization
    text = ["Walk--don't run"]
    
    # White-space based split
    print("Split on white space")
    for sentence in text:
        tokenized = sentence.split(" ")
        print(tokenized)  # Expected output: ['Walk--don't', 'run']
    
    # NLTK tokenization, handling punctuation
    print("Using NLTK tokenization")
    for sentence in text:
        tokenized = word_tokenize(sentence)
        print(tokenized)  # Expected output: ['Walk', '--', 'do', "n't", 'run']
    ```
    
2. **Sample Output**:
    
    ```python
    Split on white space
    ['Walk--don't', 'run']
    Using NLTK tokenization
    ['Walk', '--', 'do', "n't", 'run']
    ```
    

---

### Chunk 4: Lowercasing Words

1. **Code**:
    
    ```python
    # Sample text
    mixed_text = "WALK! Going for a walk is great exercise."
    mixed_words = nltk.word_tokenize(mixed_text)
    print(mixed_words)  # Tokenized words: ['WALK', '!', 'Going', 'for', 'a', 'walk', 'is', 'great', 'exercise', '.']
    
    # Convert tokens to lowercase
    lower_words = [word.lower() for word in mixed_words]
    print(lower_words)  # Expected output: ['walk', '!', 'going', 'for', 'a', 'walk', 'is', 'great', 'exercise', '.']
    ```
    
2. **Sample Output**:
    
    ```python
    ['WALK', '!', 'Going', 'for', 'a', 'walk', 'is', 'great', 'exercise', '.']
    ['walk', '!', 'going', 'for', 'a', 'walk', 'is', 'great', 'exercise', '.']
    ```
    

---

### Chunk 5: Stemming with Porter Stemmer

1. **Code**:
    
    ```python
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    
    # Sample text for stemming
    text_to_stem = "Going for a walk is the best exercise. I've walked every evening this week."
    tokenized_to_stem = nltk.word_tokenize(text_to_stem)
    
    # Apply stemming
    stemmed = [stemmer.stem(word) for word in tokenized_to_stem]
    print(stemmed)  # Expected output: ['go', 'for', 'a', 'walk', 'is', 'the', 'best', 'exercis', 'i', "'ve", 'walk', 'everi', 'even', 'thi', 'week', '.']
    ```
    
2. **Sample Output**:
    
    ```python
    ['go', 'for', 'a', 'walk', 'is', 'the', 'best', 'exercis', 'i', "'ve", 'walk', 'everi', 'even', 'thi', 'week', '.']
    ```
    

---

### Chunk 6: Lemmatizing with WordNet

1. **Code**:
    
    ```python
    import nltk
    nltk.download("wordnet")
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.corpus import wordnet
    from collections import defaultdict
    from nltk import pos_tag
    
    # Mapping POS tags for lemmatizing
    tag_map = defaultdict(lambda: wordnet.NOUN)
    tag_map["J"] = wordnet.ADJ
    tag_map["V"] = wordnet.VERB
    tag_map["R"] = wordnet.ADV
    
    lemmatizer = WordNetLemmatizer()
    text_to_lemmatize = "going for a walk is the best exercise. i've walked every evening this week"
    print("Text to lemmatize:", text_to_lemmatize)
    
    tokens_to_lemmatize = nltk.word_tokenize(text_to_lemmatize)
    lemmatized_result = " ".join([lemmatizer.lemmatize(token, tag_map[tag[0]]) for token, tag in pos_tag(tokens_to_lemmatize)])
    print("Lemmatized result:", lemmatized_result)
    ```
    
2. **Sample Output**:
    
    ```python
    Text to lemmatize: going for a walk is the best exercise. i've walked every evening this week
    Lemmatized result: go for a walk be the best exercise . i 've walk every evening this week
    ```
    

---

### Chunk 7: NLTK Stop Words

1. **Code**:
    
    ```python
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    
    # Retrieve English stop words
    nltk_stopwords = stopwords.words('english')
    print("NLTK Stopwords:", nltk_stopwords[:10])  # Display first 10 stop words
    ```
    
2. **Sample Output**:
    
    ```python
    NLTK Stopwords: ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]
    ```
    

---

### Chunk 8: Removing Punctuation

1. **Code**:
    
    ```python
    # Sample text with punctuation
    text_to_remove_punct = "going for a walk is the best exercise!! I've walked, I believe, every evening this week."
    tokens_to_remove_punct = nltk.word_tokenize(text_to_remove_punct)
    
    # Remove punctuation by keeping only alphanumeric tokens
    tokens_no_punct = [word for word in tokens_to_remove_punct if word.isalnum()]
    print("Tokens without punctuation:", tokens_no_punct)
    ```
    
2. **Sample Output**:
    
    ```python
    Tokens without punctuation: ['going', 'for', 'a', 'walk', 'is', 'the', 'best', 'exercise', 'I', 've', 'walked', 'I', 'believe', 'every', 'evening', 'this', 'week']
    ```
    

---

### Chunk 9: Spell Checking

1. **Code**:
    
    ```python
    from spellchecker import SpellChecker
    
    # Initialize the spell checker
    spell_checker = SpellChecker()
    
    # Sample text with a spelling error
    text_to_spell_check = "Ms. Ramalingam voted agains the bill"  
    tokens_to_spell_check = nltk.word_tokenize(text_to_spell_check)
    
    # Find and correct misspelled words
    spelling_errors = spell_checker.unknown(tokens_to_spell_check)
    for misspelled in spelling_errors:
        print(misspelled, "should be", spell_checker.correction(misspelled))
    ```
    
2. **Sample Output**:
    
    ```python
    agains should be against
    ```
    

---

# Hugging Face Code

### Chunk 1: Replacing Emojis with Text Descriptions

1. **Code**:
    
    ```python
    # Import Hugging Face's tokenizer
    from transformers import AutoTokenizer
    
    # Load a pre-trained tokenizer (e.g., BERT-base)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Sample text with emoji
    text = "Happy birthday!🎂"
    
    # Tokenize text containing emojis
    tokens = tokenizer.tokenize(text)
    print("Tokens with emojis:", tokens)
    ```
    
2. **Explanation and Expected Output**:
    
    * **Explanation**: Hugging Face’s tokenizer treats emojis as unique tokens or breaks them into subword tokens based on its vocabulary.
        
    * **Expected Output**:
        
        ```python
        Tokens with emojis: ['happy', 'birthday', '!', '[UNK]']
        ```
        
    * Note: `[UNK]` indicates the emoji is unknown to the BERT tokenizer vocabulary.
        

---

### Chunk 2: Removing Smart Quotes

1. **Code**:
    
    ```python
    # Sample text with smart quotes
    text = "here is a string with “smart” quotes"
    
    # Replace smart quotes with standard quotes
    cleaned_text = text.replace("“", "\"").replace("”", "\"")
    print("Text after replacing smart quotes:", cleaned_text)
    
    # Tokenize cleaned text
    tokens = tokenizer.tokenize(cleaned_text)
    print("Tokens after cleaning smart quotes:", tokens)
    ```
    
2. **Explanation and Expected Output**:
    
    * **Explanation**: We use Python’s `replace()` function to standardize quotes, ensuring compatibility with tokenizers.
        
    * **Expected Output**:
        
        ```python
        Text after replacing smart quotes: here is a string with "smart" quotes
        Tokens after cleaning smart quotes: ['here', 'is', 'a', 'string', 'with', '"', 'smart', '"', 'quotes']
        ```
        

---

### Chunk 3: Tokenization Examples

1. **Code**:
    
    ```python
    # Sample sentence with punctuation
    text = "Walk--don't run"
    
    # Tokenize using Hugging Face
    tokens = tokenizer.tokenize(text)
    print("Tokens with Hugging Face:", tokens)
    ```
    
2. **Explanation and Expected Output**:
    
    * **Explanation**: Hugging Face tokenizers efficiently split contractions and punctuation, handling them as distinct tokens or subwords.
        
    * **Expected Output**:
        
        ```python
        Tokens with Hugging Face: ['walk', '--', 'don', "'", 't', 'run']
        ```
        

---

### Chunk 4: Lowercasing Words

1. **Code**:
    
    ```python
    # Sample text
    text = "WALK! Going for a walk is great exercise."
    
    # Tokenize and convert tokens to lowercase
    tokens = tokenizer.tokenize(text.lower())  # Transform text to lowercase before tokenization
    print("Lowercased tokens:", tokens)
    ```
    
2. **Explanation and Expected Output**:
    
    * **Explanation**: By passing `text.lower()` to the tokenizer, we handle case sensitivity efficiently.
        
    * **Expected Output**:
        
        ```python
        Lowercased tokens: ['walk', '!', 'going', 'for', 'a', 'walk', 'is', 'great', 'exercise', '.']
        ```
        

---

### Chunk 5: Stemming (Alternative: Hugging Face’s Tokenization)

While Hugging Face doesn’t support traditional stemming directly, its subword tokenization approach achieves a similar effect by breaking down words into base components.

1. **Code**:
    
    ```python
    # Sample text
    text_to_stem = "Going for a walk is the best exercise. I've walked every evening this week."
    
    # Tokenize text; subwords provide similar effects to stemming
    tokens = tokenizer.tokenize(text_to_stem)
    print("Tokens mimicking stemming:", tokens)
    ```
    
2. **Explanation and Expected Output**:
    
    * **Explanation**: Tokens are broken down into meaningful subword pieces that often resemble root words.
        
    * **Expected Output**:
        
        ```python
        Tokens mimicking stemming: ['going', 'for', 'a', 'walk', 'is', 'the', 'best', 'exercise', '.', 'i', "'", 've', 'walk', '##ed', 'every', 'evening', 'this', 'week', '.']
        ```
        

---

### Chunk 6: Lemmatizing (Alternative Using Transformers)

For lemmatization, Hugging Face models do not provide direct lemmatization, but certain models trained for POS tagging can approximate it by identifying base forms of words.

1. **Code**:
    
    ```python
    from transformers import pipeline
    
    # Load a POS tagging model and pipeline
    pos_pipeline = pipeline("token-classification", model="vblagoje/bert-english-uncased-finetuned-pos")
    
    # Text sample
    text_to_lemmatize = "going for a walk is the best exercise. I've walked every evening this week"
    
    # Run POS tagging as an alternative to lemmatization
    pos_tags = pos_pipeline(text_to_lemmatize)
    print("POS tags (approximating lemmatization):", pos_tags)
    ```
    
2. **Explanation and Expected Output**:
    
    * **Explanation**: POS tagging allows us to understand each word's role, which can inform a lemmatization-like process.
        
    * **Expected Output**:
        
        ```python
        POS tags (approximating lemmatization): [{'word': 'going', 'entity': 'VERB'}, ... ]
        ```
        

---

### Chunk 7: Stop Words

Transformers do not directly handle stop words, but tokenizers allow us to exclude specific tokens.

1. **Code**:
    
    ```python
    # Sample text
    text = "This is a simple example with common stop words."
    
    # Tokenize and remove stop words
    tokens = tokenizer.tokenize(text)
    stopwords = set(['this', 'is', 'a', 'with'])  # Define basic stop words
    tokens_no_stopwords = [word for word in tokens if word not in stopwords]
    print("Tokens without stop words:", tokens_no_stopwords)
    ```
    
2. **Expected Output**:
    
    ```python
    Tokens without stop words: ['simple', 'example', 'common', 'stop', 'words', '.']
    ```
    

---

### Chunk 8: Removing Punctuation

Hugging Face tokenizers break punctuation into separate tokens, which can then be removed.

1. **Code**:
    
    ```python
    # Sample text
    text = "going for a walk is the best exercise!! I've walked, I believe, every evening this week."
    
    # Tokenize and remove punctuation
    tokens = tokenizer.tokenize(text)
    tokens_no_punct = [token for token in tokens if token.isalnum()]
    print("Tokens without punctuation:", tokens_no_punct)
    ```
    
2. **Expected Output**:
    
    ```python
    Tokens without punctuation: ['going', 'for', 'a', 'walk', 'is', 'the', 'best', 'exercise', 'I', 've', 'walked', 'I', 'believe', 'every', 'evening', 'this', 'week']
    ```
    

---

### Chunk 9: Spell Checking (Requires External Library)

For spell-checking, you’ll still need an external library like `SpellChecker`, as Transformers don’t directly handle this task.

1. **Code**:
    
    ```python
    from spellchecker import SpellChecker
    
    # Initialize the spell checker
    spell_checker = SpellChecker()
    
    # Sample text with a spelling error
    text_to_spell_check = "Ms. Ramalingam voted agains the bill"
    tokens = tokenizer.tokenize(text_to_spell_check)
    
    # Identify and correct misspelled words
    spelling_errors = spell_checker.unknown(tokens)
    for word in spelling_errors:
        print(word, "should be", spell_checker.correction(word))
    ```
    
2. **Expected Output**:
    
    ```python
    agains should be against
    ```
    

---

This Hugging Face adaptation consolidates tokenization and removes the need for traditional NLP-specific libraries where possible, leveraging Transformers' capabilities for an efficient alternative.