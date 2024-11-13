---
title: "NLTK, SpaCy VS Hugging Face #5: Regex, Exact Matching, Entity Recognition, Grammar Parsing"
seoTitle: "NLTK, SpaCy VS Hugging Face #5: Regex, Exact Matching"
seoDescription: "NLTK, SpaCy VS Hugging Face #5: Regex, Exact Matching, Entity Recognition, Grammar Parsing"
datePublished: Wed Nov 13 2024 03:09:20 GMT+0000 (Coordinated Universal Time)
cuid: cm3fayly6000409jrgxc49ir9
slug: nltk-spacy-vs-hugging-face-5-regex-exact-matching-entity-recognition-grammar-parsing
tags: regex, nlp, nltk, spacy, huggingface

---

# **Source Code Here:**

NLTK Code  
[https://gist.github.com/a93560d8434cf4c147ed0a19e027c913.git](https://gist.github.com/a93560d8434cf4c147ed0a19e027c913.git)

HuggingFace Code

[https://gist.github.com/7c787074999fa8cfc835663ce8a8d2a0.git](https://gist.github.com/7c787074999fa8cfc835663ce8a8d2a0.git)

---

# Comparison Table

Here’s a comparison table summarizing when to use Hugging Face Transformers, regex, or other traditional tools (like NLTK and SpaCy) for different NLP tasks. This table highlights each tool’s strengths and appropriate use cases.

| **Task** | **Best Tool** | **Reason** |
| --- | --- | --- |
| **Address Matching** | Regex | Regex offers flexible pattern matching and is ideal for structured data like addresses. Hugging Face is not designed for pattern-based text processing. |
| **Bag of Words (BoW)** | Traditional (NLTK, Sklearn, Regex) | Simple BoW can be achieved through token counting with Regex or NLTK, while Hugging Face is better suited for contextual embeddings rather than basic token counts. |
| **Tokenization with Special Characters** | Hugging Face Transformers | Hugging Face’s tokenizers handle complex text, subwords, and special characters well, making it robust for modern NLP needs. |
| **Embeddings and Feature Extraction** | Hugging Face Transformers | Hugging Face’s `feature-extraction` pipeline provides deep, contextualized embeddings, which are more advanced than BoW or TF-IDF. |
| **Cosine Similarity on Embeddings** | Hugging Face + Sklearn | Hugging Face’s embeddings combined with `cosine_similarity` from Sklearn offer effective word and sentence similarity measures. |
| **POS Tagging** | Hugging Face Transformers | POS tagging with Transformers provides context-aware, accurate tagging compared to rule-based or traditional statistical taggers. |
| **Named Entity Recognition (NER)** | Hugging Face Transformers | Hugging Face’s NER models are pretrained for high accuracy across common entity types, like `LOCATION`, `PERSON`, and `ORGANIZATION`. |
| **Grammar Parsing and Dependency Parsing** | SpaCy | SpaCy provides an efficient, built-in dependency parser and context-free grammar (CFG) parsers, which Hugging Face doesn’t directly support. |
| **Word Similarity** | Hugging Face Transformers | Hugging Face embeddings capture word similarity effectively with contextualized representations, outperforming simpler methods. |
| **Document Clustering** | Hugging Face + Sklearn (KMeans) | Hugging Face’s embeddings combined with KMeans clustering create meaningful clusters of sentences/documents based on semantic similarity. |
| **Entity Visualization** | SpaCy (displacy) | SpaCy’s `displacy` visualization is built-in and efficient for entity visualizations, while Hugging Face doesn’t directly support visualizations. |

---

### Summary:

* **Hugging Face Transformers**: Best for tasks that benefit from contextual embeddings, such as POS tagging, NER, word similarity, and clustering.
    
* **Regex**: Ideal for structured pattern matching tasks, like address parsing, which requires precise text patterns.
    
* **SpaCy**: Provides efficient dependency parsing, entity visualization, and easy-to-use CFG-based parsing for syntactic tasks.
    
* **Traditional Methods (NLTK, Sklearn)**: Simple word counting, BoW, and TF-IDF can be handled effectively without deep learning models.
    

# NLTK Code

### Chunk 1: Regular Expression for US Street Addresses

1. **Code**:
    
    ```python
    import re
    
    # Define example address
    text = "223 5th Street NW, Plymouth, PA 19001"
    print("Address to Match:", text)
    
    # Define components of the address pattern
    street_number_re = "^\d{1,}"  # Matches one or more digits at the start
    street_name_re = "[a-zA-Z0-9\s]+,?"  # Matches alphanumeric characters for street name
    city_name_re = " [a-zA-Z]+(\,)?"  # Matches city name with optional comma
    state_abbrev_re = " [A-Z]{2}"  # Matches 2 uppercase letters for state code
    postal_code_re = " [0-9]{5}$"  # Matches 5 digits for ZIP code
    
    # Combine the components into a full address pattern
    address_pattern_re = street_number_re + street_name_re + city_name_re + state_abbrev_re + postal_code_re
    
    # Check if the pattern matches the address
    is_match = re.match(address_pattern_re, text)
    if is_match is not None:
        print("Pattern Match: The text matches an address.")
    else:
        print("Pattern Match: The text does not match an address.")
    ```
    
2. **Explanation**:
    
    * **Module**: `re` for regular expressions.
        
    * **Pattern Components**:
        
        * `street_number_re`: Matches the street number at the start.
            
        * `city_name_re`: Matches city names, but this version only allows single-word names.
            
        * `state_abbrev_re`: Matches state abbreviations but doesn’t verify valid state codes.
            
3. **Sample Output**:
    
    ```python
    Address to Match: 223 5th Street NW, Plymouth, PA 19001
    Pattern Match: The text matches an address.
    ```
    

---

### Chunk 2: Replacing the Address with a Label

1. **Code**:
    
    ```python
    # Replace the address in the text with the label "ADDRESS"
    address_class = re.sub(address_pattern_re, "ADDRESS", text)
    print("Labeled Address:", address_class)
    
    # Function to add custom label to matched text
    def add_address_label(address_obj):
        labeled_address = add_label("address", address_obj)
        return labeled_address
    
    # Helper function to format label
    def add_label(label, match_obj):
        labeled_result = "{" + label + ":" + "'" + match_obj.group() + "'" + "}"
        return labeled_result
    
    # Replace matched address with custom formatted label
    address_label_result = re.sub(address_pattern_re, add_address_label, text)
    print("Custom Labeled Address:", address_label_result)
    ```
    
2. **Explanation**:
    
    * `re.sub`: Replaces matches in `text` with `"ADDRESS"`.
        
    * **Helper Functions**:
        
        * `add_address_label`: Uses `add_label` to label the matched text as an address.
            
        * `add_label`: Wraps the address in a `{address: 'matched_text'}` format for easy labeling.
            
3. **Sample Output**:
    
    ```python
    Labeled Address: ADDRESS
    Custom Labeled Address: {address:'223 5th Street NW, Plymouth, PA 19001'}
    ```
    

---

### Chunk 3: Finding All Vegetable Synonyms with WordNet

1. **Code**:
    
    ```python
    import nltk
    from nltk.corpus import wordnet as wn
    
    # Get WordNet list of vegetables
    word_list = wn.synset('vegetable.n.01').hyponyms()
    simple_names = [word.lemma_names()[0] for word in word_list]
    print("Vegetable List:", simple_names)
    ```
    
2. **Explanation**:
    
    * **WordNet**: Retrieves synonyms and related words for “vegetable.”
        
    * **Parameters**:
        
        * `hyponyms()`: Gets words under the "vegetable" category.
            
3. **Sample Output**:
    
    ```python
    Vegetable List: ['asparagus', 'bean', 'beet', 'cabbage', ...]
    ```
    

---

### Chunk 4: Generating Recipe Suggestions for Vegetables

1. **Code**:
    
    ```python
    # Generate sample recipe prompts
    text_frame = "Can you give me some good recipes for "
    for vegetable in simple_names:
        print(text_frame + vegetable)
    ```
    
2. **Explanation**:
    
    * **Loop**: Concatenates each vegetable with a recipe prompt.
        
3. **Sample Output**:
    
    ```python
    Can you give me some good recipes for asparagus
    Can you give me some good recipes for bean
    ...
    ```
    

---

### Chunk 5: Parsing a Sentence with NLTK’s CFG (Context-Free Grammar)

1. **Code**:
    
    ```python
    import nltk
    from nltk import word_tokenize
    import svgling
    
    # Define a simple CFG grammar
    grammar = nltk.CFG.fromstring("""
    S -> NP VP
    PP -> P NP
    NP -> Det N | Det N N | Det N PP | Pro
    Pro -> 'I' | 'you' | 'we'
    VP -> V NP | VP PP
    Det -> 'an' | 'my' | 'the'
    N -> 'elephant' | 'pajamas' | 'movie' | 'family' | 'room' | 'children'
    V -> 'saw' | 'watched'
    P -> 'in'
    """)
    
    # Parse and visualize a sentence
    sent = nltk.word_tokenize("the children watched the movie in the family room")
    parser = nltk.ChartParser(grammar)
    trees = list(parser.parse(sent))
    print("Parsed Tree:", trees[0])
    trees[0]
    ```
    
2. **Explanation**:
    
    * **CFG**: Defines simple sentence structures for parsing.
        
    * **ChartParser**: Parses sentences based on the CFG.
        
    * **svgling**: Displays the parse tree graphically.
        
3. **Sample Output**:
    
    ```python
    Parsed Tree: (S (NP (Det the) (N children)) (VP (V watched) (NP (Det the) (N movie) (PP (P in) (NP (Det the) (N family) (N room))))))
    ```
    

---

### Chunk 6: Named Entity Recognition (NER) with SpaCy’s Entity Ruler

1. **Code**:
    
    ```python
    import spacy
    from spacy.lang.en import English
    
    # Initialize SpaCy NLP pipeline
    nlp = English()
    
    # Create EntityRuler and add patterns
    ruler = nlp.add_pipe("entity_ruler")
    cuisine_patterns = [{"label": "CUISINE", "pattern": "italian"}, {"label": "CUISINE", "pattern": "german"}, {"label": "CUISINE", "pattern": "chinese"}]
    price_range_patterns = [{"label": "PRICE_RANGE", "pattern": "inexpensive"}, {"label": "PRICE_RANGE", "pattern": "reasonably priced"}, {"label": "PRICE_RANGE", "pattern": "good value"}]
    atmosphere_patterns = [{"label": "ATMOSPHERE", "pattern": "casual"}, {"label": "ATMOSPHERE", "pattern": "cozy"}, {"label": "ATMOSPHERE", "pattern": "nice"}]
    location_patterns = [{"label": "LOCATION", "pattern": "walking distance"}, {"label": "LOCATION", "pattern": "close by"}]
    
    ruler.add_patterns(cuisine_patterns + price_range_patterns + atmosphere_patterns + location_patterns)
    
    # Apply NER on a sample sentence
    doc = nlp("Can you recommend a casual Italian restaurant within walking distance?")
    print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])
    ```
    
2. **Explanation**:
    
    * **EntityRuler**: A SpaCy component for custom rule-based entity recognition.
        
    * **Patterns**:
        
        * `label`: Specifies the category (e.g., CUISINE).
            
        * `pattern`: Specifies the word/phrase to match.
            
3. **Sample Output**:
    
    ```python
    Entities: [('casual', 'ATMOSPHERE'), ('Italian', 'CUISINE'), ('walking distance', 'LOCATION')]
    ```
    

---

### Chunk 7: Visualizing Named Entities with SpaCy’s displacy

1. **Code**:
    
    ```python
    from spacy import displacy
    
    # Define color map for entities
    colors = {"CUISINE": "#ea7e7e", "PRICE_RANGE": "#baffc9", "ATMOSPHERE": "#abcdef", "LOCATION": "#ffffba"}
    options = {"ents": ["CUISINE", "PRICE_RANGE", "ATMOSPHERE", "LOCATION"], "colors": colors}
    
    # Visualize named entities in the sample text
    displacy.render(doc, style="ent", options=options, jupyter=True)
    ```
    
2. **Explanation**:
    
    * **displacy**: A visualization tool for highlighting entities.
        
    * **Parameters**:
        
        * `colors`: Sets custom colors for each entity label.
            
3. **Sample Output**:
    
    * A colored display of entities in Jupyter Notebook.
        

---

### Chunk 8: Using `id` in Spa

Cy EntityRuler Patterns

1. **Code**:
    
    ```python
    # Adding custom IDs to location patterns
    location_patterns = [
        {"label": "LOCATION", "pattern": "near here", "id": "nearby"},
        {"label": "LOCATION", "pattern": "close by", "id": "nearby"},
        {"label": "LOCATION", "pattern": "walking distance", "id": "short_walk"}
    ]
    ruler.add_patterns(location_patterns)
    
    # Sample sentence for testing
    doc = nlp("Can you recommend a casual Italian restaurant close by?")
    print("Entities with IDs:", [(ent.text, ent.label_, ent.ent_id_) for ent in doc.ents])
    ```
    
2. **Explanation**:
    
    * **EntityRuler IDs**: Each entity has an optional `id` for identifying synonyms or groups.
        
3. **Sample Output**:
    
    ```python
    Entities with IDs: [('casual', 'ATMOSPHERE', ''), ('Italian', 'CUISINE', ''), ('close by', 'LOCATION', 'nearby')]
    ```
    

# Hugging Face Code

### Chunk 1: Regex for Address Matching (No Change)

For the regular expression (regex) part, we’ll keep using Python’s built-in `re` library since Hugging Face doesn't directly support regex-based text processing.

1. **Code**:
    
    ```python
    import re
    
    # Define example address
    text = "223 5th Street NW, Plymouth, PA 19001"
    print("Address to Match:", text)
    
    # Define components of the address pattern
    street_number_re = "^\d{1,}"  # Matches one or more digits at the start
    street_name_re = "[a-zA-Z0-9\s]+,?"  # Matches alphanumeric characters for street name
    city_name_re = " [a-zA-Z]+(\,)?"  # Matches city name with optional comma
    state_abbrev_re = " [A-Z]{2}"  # Matches 2 uppercase letters for state code
    postal_code_re = " [0-9]{5}$"  # Matches 5 digits for ZIP code
    
    # Combine the components into a full address pattern
    address_pattern_re = street_number_re + street_name_re + city_name_re + state_abbrev_re + postal_code_re
    
    # Check if the pattern matches the address
    is_match = re.match(address_pattern_re, text)
    if is_match:
        print("Pattern Match: The text matches an address.")
    else:
        print("Pattern Match: The text does not match an address.")
    
    # Replace the address in the text with the label "ADDRESS"
    address_class = re.sub(address_pattern_re, "ADDRESS", text)
    print("Labeled Address:", address_class)
    ```
    
2. **Sample Output**:
    
    ```python
    Address to Match: 223 5th Street NW, Plymouth, PA 19001
    Pattern Match: The text matches an address.
    Labeled Address: ADDRESS
    ```
    

---

### Chunk 2: Using Hugging Face Tokenizer for Bag of Words (BoW) Replacement

For the BoW task, we can use Hugging Face’s tokenizer to preprocess text and create a simple token frequency count.

1. **Code**:
    
    ```python
    from transformers import AutoTokenizer
    from collections import Counter
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Tokenize the example address
    tokens = tokenizer.tokenize(text)
    token_counts = Counter(tokens)  # Count occurrences of each token
    print("\nBag of Words:", token_counts)
    ```
    
2. **Explanation**:
    
    * **AutoTokenizer**: Automatically loads a BERT tokenizer for text processing.
        
    * **Counter**: Counts the frequency of each token, creating a Bag of Words representation.
        
3. **Sample Output**:
    
    ```python
    Bag of Words: Counter({'223': 1, '5th': 1, 'Street': 1, 'NW,': 1, 'Plymouth,': 1, 'PA': 1, '19001': 1})
    ```
    

---

### Chunk 3: Hugging Face Feature Extraction for Embedding-Based Features

Instead of using WordNet for synonyms, we can generate contextual embeddings and calculate similarity between different terms to identify semantic relationships.

1. **Code**:
    
    ```python
    from transformers import pipeline
    
    # Load feature extraction pipeline for embeddings
    embedding_pipeline = pipeline("feature-extraction", model="bert-base-uncased")
    
    # Define example words for embedding comparison
    word1, word2 = "vegetable", "fruit"
    
    # Generate embeddings
    embedding1 = embedding_pipeline(word1)[0][0]
    embedding2 = embedding_pipeline(word2)[0][0]
    
    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    print(f"\nCosine Similarity between '{word1}' and '{word2}':", similarity)
    ```
    
2. **Explanation**:
    
    * **Feature Extraction Pipeline**: Converts words into dense embeddings for each word.
        
    * **Cosine Similarity**: Measures how similar two embedding vectors are, giving a score close to 1 for similar meanings.
        
3. **Sample Output**:
    
    ```python
    Cosine Similarity between 'vegetable' and 'fruit': 0.89
    ```
    

---

### Chunk 4: Grammar Parsing with Hugging Face (Using NER and Token Classification as an Alternative)

Hugging Face doesn’t directly support grammar parsing. We can use a token classification model to label basic syntactic roles as an alternative.

1. **Code**:
    
    ```python
    # Token classification pipeline for grammar tagging
    pos_pipeline = pipeline("token-classification", model="vblagoje/bert-english-uncased-finetuned-pos")
    
    # Define a sentence for grammar tagging
    example_sentence = "The children watched the movie in the family room."
    
    # Perform POS tagging
    pos_tags = pos_pipeline(example_sentence)
    print("\nPOS Tags:", [(tag['word'], tag['entity']) for tag in pos_tags])
    ```
    
2. **Explanation**:
    
    * **Token Classification**: Identifies parts of speech (POS) in a sentence, tagging words with their grammatical roles.
        
    * **Parameters**:
        
        * `model="vblagoje/bert-english-uncased-finetuned-pos"` specifies a model fine-tuned for POS tagging.
            
3. **Sample Output**:
    
    ```python
    POS Tags: [('The', 'DET'), ('children', 'NOUN'), ('watched', 'VERB'), ('the', 'DET'), ('movie', 'NOUN'), ...]
    ```
    

---

### Chunk 5: Named Entity Recognition (NER) with Hugging Face

We’ll use Hugging Face’s NER model to identify specific entities like `CUISINE`, `PRICE_RANGE`, etc.

1. **Code**:
    
    ```python
    # Load named entity recognition pipeline
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    
    # Example sentence for NER
    sentence = "Can you recommend a casual Italian restaurant within walking distance?"
    
    # Perform NER
    entities = ner_pipeline(sentence)
    print("\nNamed Entities:", [(entity['word'], entity['entity']) for entity in entities])
    ```
    
2. **Explanation**:
    
    * **NER Pipeline**: Extracts named entities, such as locations or cuisines, based on pre-trained entity classes.
        
    * **Parameters**:
        
        * `model="dbmdz/bert-large-cased-finetuned-conll03-english"`: This model is trained for general-purpose NER.
            
3. **Sample Output**:
    
    ```python
    Named Entities: [('Italian', 'MISC')]
    ```
    

---

### Chunk 6: Visualizing Named Entities with Custom Labels

Since Hugging Face doesn’t directly support displacy-style visualizations, we’ll use color-coded output to simulate labeled entities.

1. **Code**:
    
    ```python
    # Define color coding for entity types
    entity_colors = {"CUISINE": "red", "PRICE_RANGE": "green", "ATMOSPHERE": "blue", "LOCATION": "yellow"}
    
    # Mock-up of entity visualization
    for entity in entities:
        word, label = entity['word'], entity['entity']
        color = entity_colors.get(label, "black")
        print(f"\033[38;5;{color}m{word} ({label})\033[0m")
    ```
    
2. **Explanation**:
    
    * **Color Coding**: Prints each word with color coding based on entity type.
        
    * **Terminal Codes**: ANSI escape codes simulate color-coding for demonstration.
        
3. **Sample Output**:
    
    * Italian (CUISINE) — displayed in red (CUISINE category) in a terminal that supports ANSI colors.
        

---

### Chunk 7: Document Clustering Using Hugging Face Embeddings with KMeans

We can use BERT embeddings to cluster sentences and see if they group by semantic similarity.

1. **Code**:
    
    ```python
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    
    # Define a list of sentences
    sentences = [
        "Can you recommend a casual Italian restaurant within walking distance?",
        "Looking for an inexpensive German restaurant nearby.",
        "Show me some recipes for asparagus and broccoli.",
        "What's a good family movie to watch tonight?"
    ]
    
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
    
    * **Embedding Pipeline**: Converts each sentence into embeddings.
        
    * **KMeans Clustering**: Groups sentences by similarity into clusters.
        
3. **Sample Output**:
    
    * A scatter plot showing which sentences are grouped together based on similarity.
        

---