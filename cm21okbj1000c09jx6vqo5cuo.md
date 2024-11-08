---
title: "6 NLP tasks by HuggingFace Transformers"
seoTitle: "6 NLP tasks by HuggingFace Transformers"
seoDescription: "6 NLP tasks by HuggingFace Transformers"
datePublished: Wed Oct 09 2024 09:41:39 GMT+0000 (Coordinated Universal Time)
cuid: cm21okbj1000c09jx6vqo5cuo
slug: 6-nlp-tasks-by-huggingface-transformers
tags: ai, nlp, deep-learning, huggingface, transformers

---

Natural Language Processing (NLP) encompasses various tasks that enable machines to understand, interpret, and generate human language. Below are some common NLP tasks, along with sample code snippets using Python and the Hugging Face Transformers library.

### 1\. Token Classification

**What it does**: Instead of classifying the entire sequence as a single label, **token classification** assigns a label to **each individual token** (or word) in a sequence.

* **Use cases**:
    
    * **Named Entity Recognition (NER)**: Labeling tokens as specific entities (e.g., names, locations).
        
    * **Part-of-Speech Tagging**: Assigning grammatical roles to each word (e.g., noun, verb, etc.).
        
    * **Chunking**: Grouping words into phrases or syntactic units.
        

**Sample Code**:

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Input text
text = "Hugging Face is based in New York."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs).logits

# Get predictions
predictions = torch.argmax(outputs, dim=2)
labels = [model.config.id2label[p.item()] for p in predictions[0]]

print("Tokens:", tokenizer.tokenize(text))
print("Labels:", labels)
```

**Sample Output**:

```python
Tokens: ['Hugging', 'Face', 'is', 'based', 'in', 'New', 'York', '.']
Labels: ['ORG', 'ORG', 'O', 'O', 'O', 'LOC', 'LOC', 'O']
```

* **Input**: `"Hugging Face is based in New York."`
    
* **Output**: `['ORG', 'O', 'O', 'O', 'LOC', 'LOC']`
    
    * Hugging Face = Organization (ORG)
        
    * New York = Location (LOC)
        
* **Model for this task**: `AutoModelForTokenClassification`
    

### 2\. Sequence-to-Sequence Tasks

* **What it does**: This involves converting one sequence into another sequence. It’s often used when you need the model to **generate text**, not just classify it.
    
* **Use cases**:
    
    * **Machine Translation**: Translating text from one language to another.
        
    * **Text Summarization**: Creating a summary of a longer text.
        
    * **Text Generation**: Generating new text from an input prompt.
        

**Sample Code**:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Input text
text = "Translate this sentence to French."
inputs = tokenizer(text, return_tensors="pt")

# Generate translation
outputs = model.generate(**inputs)
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Translated Text:", translated_text)
```

**Sample Output**:

```python
Translated Text: Traduire cette phrase en français.
```

**Model for this task**: `AutoModelForSeq2SeqLM` (Sequence-to-Sequence Language Modeling)

### 3\. Text Generation

* **What it does**: The model generates new text based on a given prompt or sequence.
    
* **Use cases**:
    
    * **Autocompletion**: Predicting the next word or sentence in a sequence.
        
    * **Story/Dialogue Generation**: Creating coherent, contextually relevant text.
        

**Sample Code**:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Input prompt
prompt = "The weather today is"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(**inputs, max_length=30)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Text:", generated_text)
```

**Sample Output**:

```python
Generated Text: The weather today is sunny and warm with a slight breeze.
```

* **Input**: `"The weather today is"`
    
* **Output**: `"The weather today is sunny and warm with a slight breeze."`
    
* **Model for this task**: `AutoModelForCausalLM` (Causal Language Modeling), often used with models like **GPT** (Generative Pretrained Transformer)
    

### 4\. Question Answering

* **What it does**: The model finds and returns an **answer** to a given question, usually by extracting relevant information from a piece of text.
    
* **Use cases**:
    
    * **Extractive QA**: Pulling the answer directly from a text passage.
        
    * **Generative QA**: Generating an answer based on a text passage.
        

**Sample Code**:

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Load model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Input context and question
context = "Hugging Face is based in New York."
question = "Where is Hugging Face based?"
inputs = tokenizer(question, context, return_tensors="pt")

# Get answer
outputs = model(**inputs)
answer_start_index = torch.argmax(outputs.start_logits)
answer_end_index = torch.argmax(outputs.end_logits) + 1

answer_tokens = inputs['input_ids'][0][answer_start_index:answer_end_index]
answer = tokenizer.decode(answer_tokens)

print("Answer:", answer)
```

**Sample Output**:

```python
Answer: New York
```

**Model for this task**: `AutoModelForQuestionAnswering`

### 5\. Multiple Choice

* **What it does**: The model chooses the most likely answer from a set of predefined options.
    
* **Use cases**:
    
    * **Standardized Test Questions**: Automatically solving multiple-choice questions.
        
    * **Trivia/Quiz Applications**: Answering a question by selecting from multiple possible choices.
        

**Sample Code**:

```python
from transformers import AutoModelForMultipleChoice, AutoTokenizer

# Load model and tokenizer
model_name = "valhalla/distilbart-mnli-12-9"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMultipleChoice.from_pretrained(model_name)

# Input question and options
question = "What is the capital of France?"
options = ["Berlin", "Paris", "Madrid"]
inputs = [question + " " + option for option in options]
encoding = tokenizer(inputs, return_tensors="pt", padding=True)

# Get predictions
outputs = model(**encoding)
predicted_index = outputs.logits.argmax()

print("Predicted Answer:", options[predicted_index])
```

**Sample Output**:

```python
Predicted Answer: Paris
```

**Model for this task**: `AutoModelForMultipleChoice`

### **6\. Sentence Pair Classification**

* **What it does**: Classifies the relationship between two sentences (rather than a single sentence).
    
* **Use cases**:
    
    * **Natural Language Inference (NLI)**: Determining if one sentence logically follows from another (e.g., entailment, contradiction, or neutral).
        
    * **Paraphrase Detection**: Determining if two sentences have the same meaning.
        
* In this example, we'll use a pre-trained transformer model fine-tuned for **Natural Language Inference (NLI)** from Hugging Face, specifically **BERT**.
    
    #### **Code**:
    
    ```python
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    
    # Load a pre-trained model and tokenizer for NLI (BERT fine-tuned on the MultiNLI dataset)
    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Define two sentences to classify their relationship
    sentence_1 = "A man is playing guitar."
    sentence_2 = "A person is making music."
    
    # Tokenize the sentences and prepare them as input for the model
    inputs = tokenizer(sentence_1, sentence_2, return_tensors='pt', padding=True, truncation=True)
    
    # Forward pass to get logits (raw model output)
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Convert logits to probabilities using softmax
    probs = torch.softmax(logits, dim=1)
    
    # The model predicts three classes: 0 -> contradiction, 1 -> neutral, 2 -> entailment
    predicted_class = torch.argmax(probs).item()
    
    # Output results
    labels = ['Contradiction', 'Neutral', 'Entailment']
    print(f"Prediction: {labels[predicted_class]}")
    print(f"Probabilities: {probs}")
    ```
    
    ---
    
    ### **Explanation**:
    
    * **Model**: We’re using a model fine-tuned for **Natural Language Inference** (NLI). In this case, **"facebook/bart-large-mnli"**.
        
    * **Input**: We pass two sentences (`sentence_1` and `sentence_2`) to the model.
        
    * **Output**: The model returns logits, which represent the predicted relationship between the sentences: **Contradiction**, **Neutral**, or **Entailment**.
        
    * **Softmax**: We apply a softmax function to convert the logits into probabilities and select the most likely class (predicted relationship).
        
    
    **Sample Output**:
    
    ```bash
    Prediction: Entailment
    Probabilities: tensor([[0.0321, 0.1493, 0.8186]])
    ```
    
    * **Prediction**: The model predicts that **Sentence 2** is an **entailment** of **Sentence 1** (i.e., "A person is making music" logically follows from "A man is playing guitar").
        
    * **Probabilities**:
        
        * Contradiction: 3.2%
            
        * Neutral: 14.9%
            
        * Entailment: 81.9%
            
    
    The **probabilities** indicate how confident the model is in each of the three possible relationships between the two sentences:
    
    * **Contradiction (3.2%)**: The model thinks there's only a **3.2% chance** that the two sentences contradict each other (i.e., **"A man is playing guitar"** and **"A person is making music"** are not in direct conflict).
        
    * **Neutral (14.9%)**: The model assigns a **14.9% chance** that the two sentences are unrelated or neutral (i.e., the two sentences don't directly support or contradict each other, but describe different things that are not logically connected).
        
    * **Entailment (81.9%)**: The model is **81.9% confident** that **"A person is making music"** is an **entailment** of **"A man is playing guitar"**, meaning the second sentence logically follows from the first. In other words, if someone is playing a guitar, it makes sense to say they are making music, which is why the model gives this a high probability.
        
    * The model is saying that there's a very high chance (81.9%) that **"A person is making music"** logically follows from **"A man is playing guitar"**. Therefore, it predicts **entailment** as the most likely relationship between the two sentences.
        

You're right! In the refined version, I kept the focus on organizing and simplifying what you already had, rather than adding the additional tasks (7-10) like **Text Summarization**, **Machine Translation**, **Speech Recognition**, etc.

So technically, there's nothing new compared to what you already had.

If you'd like to add **more tasks (7-10)** to the existing list (like **Text Summarization**, **Machine Translation**, etc.), I can now extend it with **sample code** and **output** for those tasks. Here's a quick list of what those additional tasks might look like:

---

### **7\. Text Summarization**

**What it does**: Summarizes longer texts into shorter, meaningful summaries.

**Use cases**:

* Summarizing articles, reports, or documents.
    

**Sample Code**:

```python
from transformers import pipeline

# Load a summarization pipeline
summarizer = pipeline("summarization")

# Input long text
text = """Hugging Face provides state-of-the-art models and easy-to-use tools for building
          machine learning models that can perform various tasks like text classification,
          summarization, and question answering."""

# Summarize text
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(summary)
```

**Sample Output**:

```python
Hugging Face provides state-of-the-art models for building machine learning models that perform tasks like
text classification, summarization, and question answering.
```

---

### **8\. Text Classification (Sentiment Analysis)**

**What it does**: Classifies text into predefined categories, often used for tasks like sentiment analysis (positive/negative/neutral) or topic classification.

**Use cases**:

* **Sentiment Analysis**: Classifying the sentiment of a product review or social media post.
    
* **Topic Classification**: Categorizing documents based on topics like sports, politics, technology, etc.
    

**Sample Code**:

```python
pythonCopy codefrom transformers import pipeline

# Load sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis")

# Input text to classify
text = "I love using Hugging Face Transformers!"
result = classifier(text)

print("Sentiment:", result)
```

**Sample Output**:

```python
cssCopy codeSentiment: [{'label': 'POSITIVE', 'score': 0.9998}]
```

---

### **9\. Speech Recognition (Automatic Speech Recognition, ASR)**

**What it does**: Converts spoken language (audio) into written text.

**Use cases**:

* Transcribing podcasts, interviews, or conversations into text.
    

**Sample Code**:

```python
from transformers import pipeline

# Load ASR pipeline
asr = pipeline("automatic-speech-recognition")

# Example audio file path (local or URL)
audio_file = "path_to_audio_file.wav"

# Transcribe the speech to text
transcription = asr(audio_file)
print("Transcription:", transcription["text"])
```

**Sample Output**:

```python
Transcription: Hugging Face is an amazing platform for natural language processing tasks.
```

---

### **10\. Text-to-Speech (TTS)**

**What it does**: Converts written text into spoken audio (speech synthesis).

**Use cases**:

* Reading out loud text documents or chat messages.
    

**Sample Code**:

```python
# For TTS, external libraries like pyttsx3 are often used, as Hugging Face mainly focuses on text processing.

import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()

# Input text to convert to speech
text = "Hugging Face Transformers makes NLP easy and fun!"

# Convert text to speech
engine.say(text)
engine.runAndWait()
```

**Sample Output**:

```python
(Spoken audio of the input text)
```

---