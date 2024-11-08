---
title: "20 Huggingface Tokenizers concepts with Examples"
seoTitle: "20 Huggingface Tokenizers concepts with Before-and-After Examples"
seoDescription: "20 Huggingface Tokenizers concepts with Before-and-After Examples"
datePublished: Fri Oct 04 2024 12:43:22 GMT+0000 (Coordinated Universal Time)
cuid: cm1upuqt3001j09l60pr198zv
slug: 20-huggingface-tokenizers-concepts-with-examples
tags: ai, nlp, text-processing, huggingface, tokenizer

---

### 1\. **Installing Hugging Face Tokenizers 📦**

**Boilerplate Code**:

```bash
pip install tokenizers
```

**Use Case**: Install the Hugging Face Tokenizers library to tokenize text data efficiently.

**Goal**: Set up the `tokenizers` library to quickly tokenize and process large text datasets. 🎯

**Sample Code**:

```bash
pip install tokenizers
```

**Before Example**: You manually tokenize text using regular expressions or basic string manipulations, which can be slow.

```bash
# Manually splitting text into words:
tokens = text.split(" ")
```

**After Example**: With Hugging Face `tokenizers`, tokenization is highly optimized and much faster.

```bash
Successfully installed tokenizers
# Hugging Face Tokenizers library installed and ready for use.
```

---

### 2\. **Loading a Pre-trained Tokenizer 📖**

**Boilerplate Code**:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

**Use Case**: <mark>Load a pre-trained tokenizer to tokenize text data according to a specific model (e.g., BERT).</mark>

**Goal**: Tokenize text using the same tokenizer that was used to train a pre-trained model. 🎯

**Sample Code**:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer("Hugging Face is great!")
print(tokens)
```

**Before Example**: You use simple tokenization methods that don’t match the format needed for pre-trained models.

```bash
# Manually tokenizing text:
tokens = text.split(" ")
```

**After Example**: With a pre-trained tokenizer, the text is tokenized in a way that matches the model’s expected input format.

```bash
{'input_ids': [101, 17662, 2227, 2003, 2307, 999, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}
# Text tokenized using the BERT tokenizer.
```

---

### 3\. **Tokenizing and Decoding Text ✍️**

**Boilerplate Code**:

```python
tokens = tokenizer.encode("I love NLP", return_tensors="pt")
decoded = tokenizer.decode(tokens[0])
```

**Use Case**: <mark>Tokenize text into input IDs and later decode them back to the original text.</mark>

**Goal**: Convert text to model-readable token IDs and then decode them back into human-readable text. 🎯

**Sample Code**:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.encode("I love NLP", return_tensors="pt")
decoded = tokenizer.decode(tokens[0])
print(decoded)
```

**Before Example**: You manually map tokens to numbers and struggle with converting tokens back into text.

```bash
# Manually mapping words to IDs:
tokens = [word_to_id[word] for word in sentence]
```

**After Example**: With `tokenizers`, encoding and decoding text is handled automatically.

```bash
I love NLP
# Text tokenized and then decoded back into its original form.
```

---

### 4\. **Handling Tokenizer Special Tokens 🔖**

**What are Special Tokens?**

* **Special tokens** are extra tokens that certain models like **BERT** require to perform tasks like text classification, sequence-to-sequence tasks, or sentence pair classification.
    
* These tokens help models understand the **structure of input sequences**. The most common special tokens are:
    
    * **<mark>[CLS]</mark>**<mark>: Marks the beginning of a sequence. BERT uses this token for classification tasks.</mark>
        
    * **<mark>[SEP]</mark>**<mark>: Marks the end of a sequence or separates two sequences (for sentence pair tasks).</mark>
        
    * **<mark>[PAD]</mark>**<mark>: Used for padding shorter sequences to the same length during batching.</mark>
        

**Why are they needed?**

* <mark>These tokens ensure that models like </mark> **<mark>BERT</mark>** <mark> can handle tasks like </mark> **<mark>classification</mark>**<mark>, </mark> **<mark>question answering</mark>**<mark>, or </mark> **<mark>sentence pair tasks</mark>** <mark> by marking the </mark> **<mark>beginning</mark>** <mark> and </mark> **<mark>end</mark>** <mark> of sequences.</mark>
    
* <mark>Without these tokens, the model might not understand where the input starts or ends, which could lead to poor performance or even errors.</mark>  
    
    **Analogy**:
    
    * Imagine you’re writing a **formal letter**. You need a **greeting** (like “Dear \[Name\]”) to introduce the letter, and a **closing** (like “Sincerely, \[Your Name\]”) to end it. Without these formal markers, your letter would seem incomplete or confusing.
        
        * **\[CLS\]** is like the **greeting** of your letter, marking the start.
            
        * **\[SEP\]** is like the **closing**, signaling the end.
            
        * **<mark>[PAD]</mark>** <mark> is like adding </mark> **<mark>blank space</mark>** at the bottom of the page to make sure all letters are the same length (for models that process multiple letters at once).
            
    
    **Special Tokens Example with Human-Friendly Output:**  
    
    Imagine we are working with the sentence: `"I love NLP"`. The **special tokens** **\[CLS\]** and **\[SEP\]** are added automatically when tokenizing this input for a model like **BERT**.
    
    #### **Code**:
    
    ```python
    from transformers import AutoTokenizer
    
    # Load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Tokenize a sentence with special tokens
    tokens = tokenizer("I love NLP", add_special_tokens=True)
    
    # Print special tokens and the resulting tokenized sequence
    print("Special tokens:", tokenizer.cls_token, tokenizer.sep_token)
    print("Tokenized output:", tokens)
    ```
    
    #### **Expected Output (with Analogy)**:
    
    ```bash
    Special tokens: [CLS] [SEP]
    Tokenized output: 
    {
      'input_ids': [101, 1045, 2293, 17953, 102], 
      'attention_mask': [1, 1, 1, 1, 1]
    }
    ```
    
    **Explanation**:
    
    * **<mark>[CLS]</mark>** <mark> (</mark>`101`<mark>): Like the </mark> **<mark>greeting</mark>** <mark> of a letter, it marks the </mark> **<mark>beginning</mark>** <mark> of the sentence. This is important for the model to know where the sentence starts.</mark>
        
    * **<mark>[SEP]</mark>** <mark> (</mark>`102`<mark>): Like the </mark> **<mark>closing</mark>** <mark> of a letter, it marks the </mark> **<mark>end</mark>** <mark> of the sentence. This tells the model where the sentence finishes.</mark>
        
    * <mark>The numbers </mark> `1045, 2293, 17953` <mark> are the token IDs for the words </mark> `"I love NLP"`<mark>.</mark>
        
    * **<mark>attention_mask</mark>**<mark>: Each </mark> `1` <mark> here means the token is important for the model to process, helping the model focus on real words rather than padding.</mark>  
        
    
    Here’s how the **input** looks:
    
    * **Original sentence**: `"I love NLP"`
        
    * **Formatted sentence with special tokens**: `"[CLS] I love NLP [SEP]"`  
        
    
    Using **AutoTokenizer**:
    
    ```python
    tokens = tokenizer("I love NLP", add_special_tokens=True)
    ```
    
    * **Output**:
        
        ```bash
        {'input_ids': [101, 1045, 2293, 17953, 102], 'attention_mask': [1, 1, 1, 1, 1]}
        ```
        
        * **\[CLS\]** and **\[SEP\]** are automatically added,
            
        * Now, you won’t accidentally forget the special tokens, and the model will correctly interpret where the sentence starts and ends.
            

---

### 5\. **Batch Tokenization for Multiple Sentences 🗂️**

#### **What is Batch Tokenization?**

* **<mark>Batch tokenization</mark>** <mark> is the process of converting multiple sentences or paragraphs into token IDs simultaneously,</mark> <mark>while ensuring that all sequences are the </mark> **<mark>same length</mark>**. <mark>This is achieved through </mark> **<mark>padding</mark>** <mark> (adding extra tokens to shorter sequences) and </mark> **<mark>truncation</mark>** <mark> (shortening longer sequences)</mark>.
    

#### **Why is it needed?**

* Models like **BERT** require all input sequences in a batch to be of **equal length**. <mark>Padding ensures that shorter sentences match the length of the longest one,</mark> <mark>while truncation shortens longer sequences to a specified limit</mark>. <mark>This way, the model can process all inputs efficiently in one go.</mark>
    

**Boilerplate Code**:

```python
tokenized_batch = tokenizer(
    ["I love NLP", "Transformers are amazing"], 
    padding=True, truncation=True, return_tensors="pt"
)
```

* **return\_tensors="pt"** means the output will be in **PyTorch tensor** format, which is useful for direct input into models.
    

**Sample Code**:

```python
from transformers import AutoTokenizer

# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Sentences to be tokenized
sentences = ["I love NLP", "Transformers are amazing"]

# Tokenize the batch of sentences with automatic padding and truncation
tokenized_batch = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Print the tokenized output
print(tokenized_batch)
```

#### **Expected Output**:

```bash
{
  'input_ids': tensor([[ 101,  1045,  2293, 17953,   102,     0], 
                       [ 101,  19081,  2024,  6429,   102,     0]]), 
  'attention_mask': tensor([[1, 1, 1, 1, 1, 0], 
                            [1, 1, 1, 1, 1, 0]])
}
```

* **input\_ids**: The token IDs for each sentence, including **padding tokens** (`0`) at the end to ensure that all sequences have the same length.
    
    * `101` is the **\[CLS\]** token.
        
    * `102` is the **\[SEP\]** token.
        
    * The sentence `"I love NLP"` has been padded with `0` (representing the **\[PAD\]** token).
        
* **<mark>attention_mask</mark>**<mark>: This tells the model which tokens are actual words (</mark>`1`<mark>) and which are padding (</mark>`0`<mark>).</mark> In this case, only the actual words and special tokens (`[CLS]`, `[SEP]`) are marked as important.
    

---

### 6\. **Adding Custom Tokens to a Pre-trained Tokenizer 🆕**

**What is Adding Custom Tokens?**

Adding custom tokens allows you to extend a **pre-trained tokenizer** to handle domain-specific words, <mark> jargon</mark>, or abbreviations that were not part of the original tokenizer's vocabulary. <mark>This is useful when working with specialized datasets (e.g., medical or legal documents)</mark>.

**Why is it needed?**

In many cases, the pre-trained tokenizer may not recognize new or specific terms that are critical to your task. By adding these custom tokens, you ensure that your model can accurately process and understand these unique terms.

**Boilerplate Code**:

```python
tokenizer.add_tokens(["newtoken1", "newtoken2"])
```

**Sample Code**:

```python
from transformers import AutoTokenizer

# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Add custom tokens
tokenizer.add_tokens(["nlp_token", "deep_learning"])

# Check that the tokens were added
print(f"Added tokens: {tokenizer.additional_special_tokens}")

# Tokenize a sentence containing the new tokens
tokens = tokenizer("I love nlp_token and deep_learning", add_special_tokens=True)

# Print the tokenized output
print(tokens)
```

**Expected Output**:

```bash
Added tokens: ['nlp_token', 'deep_learning']
{
  'input_ids': [101, 1045, 2293, 32000, 1998, 32001, 102], 
  'attention_mask': [1, 1, 1, 1, 1, 1, 1]
}
```

* **input\_ids**: The token IDs for each word in the sentence. The new custom tokens (`"nlp_token"` and `"deep_learning"`) are represented by newly assigned token IDs (`32000`, `32001`).
    
* **attention\_mask**: All tokens are marked as important (`1`), meaning no padding was added.
    

---

### 7\. **Tokenizing Text with Attention Masks 🎭**

**What is Tokenizing with Attention Masks?**

<mark>When tokenizing text, </mark> **<mark>attention masks</mark>** <mark> help the model understand which tokens should be attended to (actual words) and which ones should be ignored</mark> (like padding tokens). This is especially important when dealing with batches of sentences of different lengths, as some sentences will need padding.

**Why is it needed?**

In **transformer-based models** like BERT, attention masks allow the model to focus on the actual content and ignore any padded tokens that were added to match the sequence length. This prevents the model from "learning" patterns from the padding.

**Boilerplate Code**:

```python
tokenized_output = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
attention_mask = tokenized_output['attention_mask']
```

**Sample Code**:

```python
from transformers import AutoTokenizer

# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Sentences to be tokenized
sentences = ["I love NLP", "Transformers are amazing"]

# Tokenize the batch of sentences with automatic padding and truncation, return attention masks
tokenized_output = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Print the tokenized output with attention masks
print(tokenized_output['input_ids'])
print(tokenized_output['attention_mask'])
```

**Expected Output**:

```bash
tensor([[ 101, 1045,  2293, 17953,  102,     0],
        [ 101, 19081,  2024,  6429,  102,     0]])

tensor([[1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0]])
```

* **input\_ids**: The token IDs for the sentences, including padding tokens (`0`) to ensure all sequences have the same length.
    
* **attention\_mask**:
    
* * **1**: Pay attention to this token.
        
    * **0**: Ignore this token..
        

---

### 8\. **Padding and Truncating Sequences to a Fixed Length ⏳**

**Boilerplate Code**:

```python
tokenized_batch = tokenizer(sentences, padding='max_length', truncation=True, max_length=8, return_tensors="pt")
```

* **padding='max\_length'**: Ensures that every sentence is padded to a specified maximum length.
    
* **truncation=True**: Cuts off longer sequences to fit within the max length.
    
* **max\_length=8**: The specified maximum length for sequences.
    
      
    **Sample Code**:
    
    ```python
    from transformers import AutoTokenizer
    
    # Load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Sentences to be tokenized
    sentences = ["I love NLP", "Transformers are amazing but complex"]
    
    # Tokenize the batch of sentences with padding and truncation to max length of 8 tokens
    tokenized_batch = tokenizer(sentences, padding='max_length', truncation=True, max_length=8, return_tensors="pt")
    
    # Print the tokenized output
    print(tokenized_batch['input_ids'])
    print(tokenized_batch['attention_mask'])
    ```
    
    ---
    
    **Expected Output**:
    
    ```python
    tensor([[ 101, 1045,  2293, 17953,  102,     0,     0,     0], 
            [ 101, 19081,  2024,  6429,  2021,  3372,  102,     0]])
    
    tensor([[1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0]])
    ```
    
    * **input\_ids**:
        
        * The first sequence `"I love NLP"` was padded with three **\[PAD\]** tokens (`0`) to reach the max length of 8.
            
        * The second sequence `"Transformers are amazing but complex"` was truncated to the first 8 tokens.
            
    * **attention\_mask**:
        
        * The **1s** indicate tokens that the model should focus on (actual words and special tokens like **\[CLS\]** and **\[SEP\]**).
            
        * The **0s** represent the padding tokens, which should be ignored by the model.
            

Both **batch tokenization** and **padding/truncating sequences** are closely related concepts, but they have subtle differences in emphasis. Let’s compare them:

**Key Differences**:

* **Batch Tokenization**: It's primarily about processing **multiple inputs** (a batch) and ensuring all inputs are padded/truncated <mark>based on the longest sequence.</mark>
    
    * Example: `"I love NLP"` vs. `"Transformers are amazing"`.
        
    * The shorter sentence will be padded to match the length of the longer one.
        
* **Padding/Truncation to a Fixed Length**: The emphasis here is on ensuring **every input (or batch)** h<mark>as a </mark> **<mark>specific length</mark>** <mark> (e.g., 8 tokens), regardless of the original sentence length</mark>. This might involve **truncating** longer sequences or **padding** shorter ones.
    

**Similarities**:

* Both <mark>involve </mark> **<mark>padding</mark>** <mark> and </mark> **<mark>truncation</mark>** <mark> to ensure input sequences are of </mark> **<mark>equal length</mark>**<mark>.</mark>
    
* Both use **tokenizers** to automatically manage these tasks.
    

**Summary**:

* **Batch Tokenization** is about tokenizing **multiple sequences** and ensuring they are of equal length for batching.
    
* **Padding/Truncating to Fixed Length** ensures that **all sequences** (even if single) are padded/truncated to a **predefined maximum length**.
    

---

### 9\. **Saving and Loading Custom Tokenizers 💾**

**Boilerplate Code**:

```python
# Save tokenizer
tokenizer.save_pretrained('./my_custom_tokenizer')

# Load tokenizer
custom_tokenizer = AutoTokenizer.from_pretrained('./my_custom_tokenizer')
```

---

**Sample Code**:

```python
from transformers import AutoTokenizer

# Load a pre-trained BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Add custom tokens to the tokenizer
tokenizer.add_tokens(["custom_token1", "custom_token2"])

# Save the modified tokenizer to a directory
tokenizer.save_pretrained('./my_custom_tokenizer')

# Load the custom tokenizer from the saved directory
custom_tokenizer = AutoTokenizer.from_pretrained('./my_custom_tokenizer')

# Check if the custom tokens were preserved
print(f"Custom tokens: {custom_tokenizer.additional_special_tokens}")

# Tokenize a sentence with the custom tokenizer
tokens = custom_tokenizer("I love custom_token1 and custom_token2", add_special_tokens=True)
print(tokens)
```

---

**Expected Output**:

```bash
Custom tokens: ['custom_token1', 'custom_token2']
{
  'input_ids': [101, 1045, 2293, 32000, 1998, 32001, 102],
  'attention_mask': [1, 1, 1, 1, 1, 1, 1]
}
```

* **Custom tokens**: This shows that the tokenizer preserved your custom tokens (`custom_token1`, `custom_token2`) after saving and loading.
    
* **input\_ids**: The custom tokens were correctly assigned new token IDs (`32000` and `32001`), showing that the tokenizer recognizes them after being reloaded.
    
* **attention\_mask**: All tokens are marked as important (`1`), meaning there’s no padding.
    

---

### 10\. **Using Byte-Pair Encoding (BPE) Tokenization 🧩**

**What is <mark>Byte-Pair Encoding (BPE) </mark> Tokenization?**

**BPE Tokenization** is a subword tokenization technique that <mark>splits words into subword units based on their frequency. </mark> T<mark>his helps models handle </mark> **<mark>rare words</mark>** <mark> and </mark> **<mark>out-of-vocabulary words</mark>** <mark> by breaking them into smaller, more common subwords.</mark>

**Example**:

Let’s take the word **"unhappiness"**.

* A regular tokenizer might treat this as a single token.
    
* **BPE tokenization** splits it into:
    
    * **"un"** (prefix)
        
    * **"happiness"** (root word)
        
    * But if "happiness" is too rare, BPE might even split it further:
        
        * **"hap"**
            
        * **"pi"**
            
        * **"ness"**
            

This way, the model can process each smaller piece, even if it’s never seen the whole word before.

**Boilerplate Code**:

```python
from transformers import GPT2Tokenizer

# Load GPT-2 tokenizer (which uses BPE)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
```

**Sample Code**:

```python
from transformers import GPT2Tokenizer

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Example sentence
sentence = "Tokenization is awesome, right?"

# Tokenize the sentence using BPE
tokenized_output = tokenizer(sentence, add_special_tokens=True)

# Print the tokens and their corresponding IDs
print("Tokens:", tokenizer.convert_ids_to_tokens(tokenized_output['input_ids']))
print("Token IDs:", tokenized_output['input_ids'])
```

**Expected Output**:

```bash
Tokens: ['<|endoftext|>', 'Token', 'ization', 'Ġis', 'Ġawesome', ',', 'Ġright', '?', '<|endoftext|>']
Token IDs: [50256, 19204, 3034, 318, 10433, 11, 4283, 30, 50256]
```

* **Tokens**:
    
    * The sentence `"Tokenization"` is broken into `['Token', 'ization']` due to BPE.
        
    * The spaces before some tokens like `Ġis` and `Ġawesome` represent the start of a new word or space.
        
    * **<mark>&lt;|endoftext|&gt;</mark>** <mark> is a special token used to mark the end of the input.</mark>
        
* **Token IDs**: The token IDs are the unique numbers assigned to each token, which will be fed into the model for processing.
    

---

### 11\. **Loading Tokenizer from Hub 🔄**

**Boilerplate Code**:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

**Use Case**: Load a tokenizer from Hugging Face's Model Hub to ensure compatibility with a specific model.

**Goal**: Download and initialize a tokenizer from a pre-trained model available on the Hugging Face Hub. 🎯

**Sample Code**:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer("Hugging Face is awesome!", return_tensors="pt")
print(tokens)
```

**Output sample**: The Hugging Face library simplifies loading a pre-trained tokenizer directly from the Model Hub.

```bash
{'input_ids': tensor([[15496,  2159,   133,  8377,     0]])}
# GPT-2 tokenizer loaded from the Hub and tokenized the input text.
```

---

### 12\. **Detokenizing: Converting IDs Back to Text 🔄**

**Boilerplate Code**:

```python
decoded_text = tokenizer.decode(token_ids)
```

**Use Case**: <mark>Convert token IDs back into human-readable text</mark> after tokenization and model inference.

**Goal**: Detokenize a sequence of token IDs back into a string of readable text. 🎯

**Sample Code**:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer("Hugging Face is awesome!", return_tensors="pt")
decoded_text = tokenizer.decode(tokens["input_ids"][0])
print(decoded_text)
```

**Output:** With Hugging Face's `tokenizers`, detokenization is handled automatically, producing clean text.

```bash
hugging face is awesome!
# Token IDs decoded back into readable text.
```

---

### 13\. **Fast Tokenizers: Speeding Up Tokenization 🚀**

**Boilerplate Code**:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
```

**Use Case**: Use a fast version of the tokenizer that dramatically speeds up the tokenization process.

**Goal**: Speed up tokenization using the `Fast` tokenizer implementation, which leverages Rust for faster performance. 🎯

**Sample Code**:

```python
from transformers import AutoTokenizer

# Use the fast tokenizer implementation
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
tokens = tokenizer("Hugging Face is awesome!", return_tensors="pt")
print(tokens)
```

**Output**: With the `Fast` tokenizers, tokenization is significantly faster without losing accuracy.

```bash
{'input_ids': tensor([[101, 17662, 2227, 2003, 2307, 999, 102]])}
# Fast tokenization of the input text using the optimized tokenizer.
```

---

### 14\. **Padding Tokenized Inputs for Batch Processing 🛠️**

**Boilerplate Code**:

```python
tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

**Use Case**: Automatically pad tokenized inputs to the maximum sequence length in a batch for easier batching and processing.

**Goal**: Ensure all tokenized sequences in a batch have the same length by adding padding tokens where necessary. 🎯

**Sample Code**:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
texts = ["I love NLP", "Transformers are amazing"]
tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

print(tokens["input_ids"], tokens["attention_mask"])
```

**Output**: Padding and attention masks are automatically handled, ensuring all sequences in a batch are of equal length.

```bash
tensor([[ 101, 1045, 2293, 17953,   102,     0], [ 101, 19081, 2024, 6429,   102,     0]])
tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]])
# Sequences padded and attention masks generated for batch processing.
```

---

### 15\. **Custom Pre-tokenization Rules with Whitespace or Splitters ✂️**

**Boilerplate Code**:

```python
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()
```

**Use Case**: Apply custom pre-tokenization rules (e.g., splitting based on whitespace or punctuation) before applying subword tokenization.

**Goal**: Customize how the input text is split into tokens before further processing, enabling more control over tokenization. 🎯

**Sample Code**:

```python
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace

# Create a custom tokenizer
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

encoded = tokenizer.encode("Hugging Face is awesome!")
print(encoded.tokens)
```

With Hugging Face `tokenizers`, you can define pre-tokenization rules for greater flexibility.

```bash
['Hugging', 'Face', 'is', 'awesome', '!']
# Text pre-tokenized using whitespace splitting.
```

### 16\. **Training a New Tokenizer from Scratch 🆕**

**Boilerplate Code**:

```python
from tokenizers import Tokenizer, models, trainers

# Initialize tokenizer with BPE model
tokenizer = Tokenizer(models.BPE())
trainer = trainers.BpeTrainer(vocab_size=30000, min_frequency=2)

# Train tokenizer on your dataset
tokenizer.train(files=["your_data.txt"], trainer=trainer)
```

**Use Case**: Train a brand new tokenizer from scratch using your own dataset.

**Goal**: Build a tokenizer that is specifically tailored to your dataset by training it from raw text. 🎯

**Sample Code**:

```python
from tokenizers import Tokenizer, models, trainers

tokenizer = Tokenizer(models.BPE())
trainer = trainers.BpeTrainer(vocab_size=30000, min_frequency=2)

# Assuming you have a file with your dataset
tokenizer.train(files=["your_data.txt"], trainer=trainer)
print(tokenizer.get_vocab_size())
```

With Hugging Face `tokenizers`, you can train a tokenizer on your own dataset, tailored to its vocabulary and structure.

```bash
30000
# Tokenizer trained on your data with a vocabulary size of 30,000 tokens.
```

---

### 17\. **Subword Tokenization with WordPiece 🧩**

**Boilerplate Code**:

```python
from tokenizers import Tokenizer, models

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
```

**Use Case**: <mark>Tokenize text into subword units using the WordPiece algorithm, which is widely used in models like BERT.</mark>

**Goal**: Break words into smaller subwords or characters when they are not part of the tokenizer’s vocabulary. 🎯

**Sample Code**:

```python
from tokenizers import Tokenizer, models

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.add_tokens(["hugging", "face", "is", "awesome", "!"])

encoded = tokenizer.encode("Hugging Face is awesome!")
print(encoded.tokens)
```

With WordPiece, unknown or rare words are broken into smaller, more frequent subword units.

```bash
['[UNK]', '[UNK]', 'is', 'awesome', '!']
# Text tokenized using the WordPiece model, unknown words represented by [UNK].
```

---

### 18\. **Using Byte-Level BPE Tokenization 🧬**

**Boilerplate Code**:

```python
from tokenizers import Tokenizer, models

tokenizer = Tokenizer(models.ByteLevelBPETokenizer())
```

**Use Case**: Tokenize text at the byte level, enabling the tokenizer to handle any input text, even <mark> rare characters or non-standard inputs.</mark>

**Goal**: Use byte-level tokenization to create robust tokenizers that can process any text, <mark> including emojis, symbols, or different languages. 🎯</mark>

**Sample Code**:

```python
from tokenizers import Tokenizer, models

# Initialize the Byte-Level BPE tokenizer
tokenizer = Tokenizer(models.ByteLevelBPETokenizer())
tokenizer.add_tokens(["Hugging", "Face", "is", "awesome", "!"])

encoded = tokenizer.encode("Hugging Face is awesome! 😊")
print(encoded.tokens)
```

Byte-level BPE tokenization automatically handles any text, including special characters and emojis.

```bash
['Hugging', 'Face', 'is', 'awesome', '!', '😊']
# Text tokenized at the byte level, handling all characters correctly.
```

---

### 19\. **Post-processing: Adding Special Tokens Automatically 📜**

**Boilerplate Code**:

```python
from tokenizers import Tokenizer, processors

tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 101), ("[SEP]", 102)]
)
```

**Use Case**: Automatically add special tokens (like `[CLS]`, `[SEP]`, etc.) after tokenization for sequence classification tasks.

**Goal**: Ensure that special tokens like `[CLS]` and `[SEP]` are included in tokenized outputs for models that require them. 🎯

**Sample Code**:

```python
from tokenizers import Tokenizer, models, processors

# Initialize tokenizer
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# Add post-processing for special tokens
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 101), ("[SEP]", 102)]
)

encoded = tokenizer.encode("Hugging Face is awesome!")
print(encoded.tokens)
```

With post-processing, special tokens are automatically added to every tokenized sequence.

```bash
['[CLS]', 'Hugging', 'Face', 'is', 'awesome', '!', '[SEP]']
# Special tokens `[CLS]` and `[SEP]` are automatically added.
```

---

### 20\. **Managing Padding and Truncation Dynamically ✂️**

**Boilerplate Code**:

```python
tokens = tokenizer.encode("Hugging Face is awesome!", padding=True, truncation=True, max_length=10)
```

**Use Case**: Automatically pad and truncate sequences dynamically based on the specified maximum length.

**Goal**: Ensure all tokenized sequences are of uniform length by padding shorter sequences and truncating longer ones. 🎯

**Sample Code**:

```python
from transformers import AutoTokenizer

# Load pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Dynamically pad and truncate sequences
tokens = tokenizer.encode("Hugging Face is awesome!", padding=True, truncation=True, max_length=10)
print(tokens)
```

With dynamic padding and truncation, sequences are automatically adjusted to the desired length.

```bash
[101, 17662, 2227, 2003, 2307, 999, 102, 0, 0, 0]
# Sequence padded and truncated to a length of 10 tokens.
```