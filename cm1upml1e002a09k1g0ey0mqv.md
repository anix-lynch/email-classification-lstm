---
title: "20 Huggingface Datasets concepts with  Examples"
seoTitle: "20 Huggingface Datasets concepts with Before-and-After Examples"
seoDescription: "20 Huggingface Datasets concepts with Before-and-After Examples"
datePublished: Fri Oct 04 2024 12:37:01 GMT+0000 (Coordinated Universal Time)
cuid: cm1upml1e002a09k1g0ey0mqv
slug: 20-huggingface-datasets-concepts-with-examples
tags: ai, machine-learning, huggingface, datapreparation, huggingface-dataset

---

### 1\. **Installing the Datasets Library 📦**

**Boilerplate Code**:

```bash
pip install datasets
```

**Use Case**: Install the Hugging Face Datasets library to load, process, and analyze large datasets.

**Goal**: Set up the `datasets` library to access a variety of NLP and machine learning datasets. 🎯

### 2\. **Loading a Dataset from the Hub 📂**

**Use Case**: Load a dataset directly from the Hugging Face Hub.

**Goal**: Access popular datasets like IMDb, MNIST, or SQuAD with one line of code. 🎯

**Sample Code**:

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset["train"][0])
```

**Before Example**: You manually search for, download, and format datasets for machine learning tasks.

```bash
# Manually fetching dataset files:
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
```

**After Example**: The `datasets` library automatically loads datasets, including metadata and splits.

```bash
{'text': 'I absolutely loved this movie...', 'label': 1}
# IMDb dataset loaded and ready for use.
```

---

**Where you can run this code**:

1. **Google Colab**: It works perfectly in Google Colab, which has the Hugging Face `datasets` library pre-installed (or you can install it easily with `!pip install datasets`).
    
2. **JupyterLab**: You can run this in your local JupyterLab setup after installing the Hugging Face datasets library:
    
    ```bash
    !pip install datasets
    ```
    
3. **Locally**: This will also work on your local environment or any notebook interface, as long as you have the `datasets` library installed.
    

**Can this be done on the Hugging Face website?**

Yes, Hugging Face has a platform called **"Hugging Face Datasets Viewer"** where you can view and explore datasets, but the code execution (e.g., loading datasets, manipulating them) happens outside the website in environments like **Colab, JupyterLab, or your local environment**.

### **3\. Inspecting Dataset Features and Splits 🔍**

**Boilerplate Code**:

```python
print(dataset.features)
print(dataset["train"].split)
```

**Use Case**: Explore the structure of a dataset, including its features and available splits (e.g., train, test, validation).

**Goal**: Understand the dataset's structure before processing or training. 🎯

**Sample Code**:

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset.features)
print(dataset["train"].split)
```

**<mark>Before Example</mark>**<mark>: You manually inspect the dataset file formats, often needing custom code to explore features.</mark>

```bash
# Manually loading and inspecting data:
import pandas as pd
df = pd.read_csv("dataset.csv")
print(df.columns)
```

**After Example**: With the `datasets` library, the dataset's features and splits are automatically presented.

```bash
Features: {'text': Value(dtype='string'), 'label': ClassLabel(num_classes=2)}
# Features and splits of the dataset displayed.
```

---

### 4\. **Processing Datasets with Map Function 🗺️**

**Preprocessing** is the step that converts **human-readable text** into **machine-readable numbers** (token IDs) that the model can use for training or inference.

**Boilerplate Code**:

```python
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

**Use Case**: Apply transformations or preprocessing to the entire dataset using the `map()` function.

**Goal**: Tokenize text or perform any custom processing on a dataset. 🎯

**Sample Code**:

```python
from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
print(tokenized_dataset["train"][0])
```

**Before Example**: You write custom loops to preprocess datasets, which can be inefficient and harder to maintain.

```bash
# Manually tokenizing dataset:
tokenized_texts = [tokenizer(text) for text in texts]
```

**After Example**: The `map()` function applies the preprocessing efficiently across the dataset.

```bash
{'input_ids': [...], 'attention_mask': [...], 'label': 1}
# Dataset tokenized using the map function.
```

---

### 5\. **Filtering Data with Datasets Library 🧹**

**Boilerplate Code**:

```python
filtered_dataset = dataset.filter(lambda example: example["label"] == 1)
```

**Use Case**: Filter datasets based on specific conditions (e.g., selecting positive sentiment examples).

**Goal**: Reduce dataset size by selecting only relevant examples. 🎯

**Sample Code**:

```python
from datasets import load_dataset

dataset = load_dataset("imdb")

filtered_dataset = dataset.filter(lambda example: example["label"] == 1)
print(filtered_dataset["train"][0])
```

**After Example**: With the `datasets` library, filtering is simple and efficient, even for large datasets.

```bash
{'text': 'I absolutely loved this movie...', 'label': 1}
# Dataset filtered to only include examples with positive sentiment.
```

### 6\. **Dataset Shuffling 🔀**

**Boilerplate Code**:

```python
shuffled_dataset = dataset.shuffle(seed=42)
```

**Use Case**: Randomly shuffle the order of examples in a dataset.

**Goal**: Shuffle the dataset to ensure that the training examples are not in any particular order. 🎯

**Sample Code**:

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
shuffled_dataset = dataset.shuffle(seed=42)
print(shuffled_dataset["train"][0])
```

**After Example**: With the `datasets` library, you can easily shuffle large datasets with a single function.

```bash
{'text': 'A must-watch for movie lovers...', 'label': 1}
# Dataset shuffled and ready for training.
```

---

### 7\. **Dataset Batching for Training ⚙️**

**Boilerplate Code**:

```python
batched_dataset = dataset.with_format("torch").train_test_split(test_size=0.1)
```

**Use Case**: <mark>Split the dataset into smaller batches, which can then be used during training.</mark>

**Goal**: Split the dataset into train and test sets, and prepare it for model training. 🎯

**Sample Code**:

```python
from datasets import load_dataset

dataset = load_dataset("imdb").with_format("torch")
batched_dataset = dataset["train"].train_test_split(test_size=0.1)
print(batched_dataset["train"][0])
```

With `datasets`, batching and splitting are automatic and efficient.

```bash
{'text': 'A great film that is a timeless classic...', 'label': 0}
# Data split into training and testing batches.
```

---

### 8\. **Saving Processed Datasets 💾**

**Boilerplate Code**:

```python
dataset.save_to_disk("path/to/dataset")
```

**Use Case**: Save a preprocessed dataset to disk for later use.

**Goal**: Store datasets locally after processing them, so you can reload them without reprocessing. 🎯

**Sample Code**:

```python
from datasets import load_dataset

dataset = load_dataset("imdb").shuffle(seed=42)
dataset.save_to_disk("path/to/dataset")

# Later load the dataset
from datasets import load_from_disk
loaded_dataset = load_from_disk("path/to/dataset")
print(loaded_dataset["train"][0])
```

**After Example**: With `datasets`, saving and reloading datasets is streamlined.

```bash
# Dataset saved and reloaded from disk.
{'text': 'Amazing movie with great performances...', 'label': 1}
```

---

### 9\. **Loading Datasets from Disk 📂**

**Boilerplate Code**:

```python
from datasets import load_from_disk

dataset = load_from_disk("path/to/dataset")
```

**Use Case**: Load a previously saved dataset from your local disk.

**Goal**: Reload a dataset from disk without having to reprocess or reload it from the Hugging Face Hub. 🎯

**Sample Code**:

```python
from datasets import load_from_disk

dataset = load_from_disk("path/to/dataset")
print(dataset["train"][0])
```

**After Example**: With the `datasets` library, loading datasets from disk is simple and efficient.

```bash
{'text': 'Fantastic storyline with deep characters...', 'label': 0}
# Dataset loaded from disk, ready for use.
```

---

### 10\. **Streaming Large Datasets 🌊**

**Boilerplate Code**:

```python
dataset = load_dataset("imdb", split="train", streaming=True)
```

**Use Case**: Stream large datasets that do not fit into memory.

**Goal**: Load large datasets efficiently by streaming them instead of loading everything into memory at once. 🎯

**Sample Code**:

```python
from datasets import load_dataset

dataset = load_dataset("imdb", split="train", streaming=True)
for example in dataset.take(5):
    print(example)
```

**Before Example**: You struggle with large datasets that don't fit into memory, requiring custom code for data management.

```bash
# Manually streaming large datasets:
for chunk in pd.read_csv("large_dataset.csv", chunksize=1000):
    process(chunk)
```

**After Example**: With `datasets`, streaming large datasets is handled automatically, making it memory-efficient.

```bash
{'text': 'A beautiful movie with deep meaning...', 'label': 1}
# Streaming dataset processed efficiently.
```

### 11\. **Dataset Concatenation and Merging ➕**

**Boilerplate Code**:

```python
concatenated_dataset = dataset1.concatenate(dataset2)
```

**Use Case**: Combine two datasets into one by concatenating or merging them.

**Goal**: Merge multiple datasets to create a larger dataset for training or evaluation. 🎯

**Sample Code**:

```python
from datasets import load_dataset

dataset1 = load_dataset("imdb", split="train[:50%]")
dataset2 = load_dataset("imdb", split="train[50%:]")
concatenated_dataset = dataset1.concatenate(dataset2)

print(len(concatenated_dataset))
```

**Before Example**: You manually join datasets by reading them into memory and concatenating them using custom code.

```bash
# Manually concatenating datasets:
combined_dataset = dataset1 + dataset2
```

**After Example**: With the `datasets` library, you can seamlessly concatenate datasets.

```bash
25000
# Two IMDb dataset splits merged into a single dataset.
```

---

### 12\. **Dataset Sorting 🔄**

**Boilerplate Code**:

```python
sorted_dataset = dataset.sort("label")
```

**Use Case**: Sort a dataset by a specific feature (e.g., sorting by label for classification tasks).

**Goal**: Sort dataset rows based on a feature to arrange them in a specific order. 🎯

**Sample Code**:

```python
from datasets import load_dataset

dataset = load_dataset("imdb", split="train")
sorted_dataset = dataset.sort("label")
print(sorted_dataset["train"][0])
```

**Before Example**: You manually sort data using pandas or other tools, which can be inefficient.

```bash
# Manually sorting dataset:
sorted_dataset = dataset.sort_values(by="label")
```

**After Example**: With the `datasets` library, sorting by any feature is quick and straightforward.

```bash
{'text': 'Worst movie ever...', 'label': 0}
# Dataset sorted by the "label" feature.
```

---

### 13\. **Dataset Casting for Feature Types 🎭**

**Boilerplate Code**:

```python
dataset = dataset.cast_column("label", ClassLabel(num_classes=3))
```

**Use Case**: <mark>Change the type of a dataset’s feature, such as converting labels to a categorical type.</mark>

**Goal**: Modify the data types of specific columns, e.g., converting integer labels to class labels. 🎯

**Sample Code**:

```python
from datasets import load_dataset
from datasets import ClassLabel

dataset = load_dataset("imdb", split="train")
dataset = dataset.cast_column("label", ClassLabel(num_classes=2))
print(dataset.features["label"])
```

**Before Example**: You manually modify data types using pandas or custom functions.

```bash
# Manually changing data types:
df['label'] = df['label'].astype('category')
```

**After Example**: With the `datasets` library, you can cast features to specific types with minimal code.

```bash
ClassLabel(num_classes=2, names=['negative', 'positive'])
# Labels cast to class label type for easier manipulation.
```

---

### 14\. **Dataset Stratified Splitting ⚖️**

**Boilerplate Code**:

```python
train_test_split = dataset.train_test_split(test_size=0.2, stratify_by_column="label")
```

**Use Case**: <mark>Split the dataset into training and test sets while maintaining class balance (stratified split).</mark>

**Goal**: Create train/test splits while ensuring that the distribution of labels is preserved. 🎯

**Sample Code**:

```python
from datasets import load_dataset

dataset = load_dataset("imdb", split="train")
train_test_split = dataset.train_test_split(test_size=0.2, stratify_by_column="label")
print(train_test_split["train"][0])
```

**Before Example**: You write custom code to perform stratified splits, ensuring balanced label distribution.

```bash
# Manually creating a stratified split:
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(dataset, stratify=dataset['label'])
```

**After Example**: With the `datasets` library, stratified splitting is simple and automatic.

```bash
{'text': 'A fascinating movie...', 'label': 1}
# Dataset split with balanced label distribution.
```

---

### 15\. **Applying Preprocessing Pipelines 🛠️**

**Boilerplate Code**:

```python
def preprocess_function(examples):
    return {"length": len(examples["text"].split())}

processed_dataset = dataset.map(preprocess_function)
```

**Use Case**: Apply <mark>custom</mark> preprocessing functions (e<mark>.g., tokenization, feature extraction) to the dataset.</mark>

**Goal**: Add new features or preprocess the dataset <mark>before training or evaluation</mark>. 🎯

**Sample Code**:

```python
from datasets import load_dataset

dataset = load_dataset("imdb", split="train")

def preprocess_function(examples):
    return {"length": len(examples["text"].split())}

processed_dataset = dataset.map(preprocess_function)
print(processed_dataset["train"][0])
```

**Before Example**: You manually preprocess datasets, adding features and applying transformations with loops.

```bash
# Manually adding a new feature:
df['length'] = df['text'].apply(lambda x: len(x.split()))
```

**After Example**: With `datasets`, preprocessing functions are applied efficiently to every row.

```bash
{'text': 'A great story with amazing actors...', 'label': 1, 'length': 6}
# Dataset processed with a new "length" feature added.
```

### 16\. **Dataset Column Renaming 🔄**

**Boilerplate Code**:

```python
renamed_dataset = dataset.rename_column("label", "sentiment")
```

**Use Case**: Rename dataset columns to make them more descriptive or easier to work with.

**Goal**: Change the name of a specific column in the dataset. 🎯

**Sample Code**:

```python
from datasets import load_dataset

dataset = load_dataset("imdb", split="train")
renamed_dataset = dataset.rename_column("label", "sentiment")
print(renamed_dataset["train"][0])
```

**Before Example**: You manually rename columns using pandas or other tools.

```bash
# Manually renaming columns in pandas:
df.rename(columns={"label": "sentiment"}, inplace=True)
```

**After Example**: With the `datasets` library, renaming columns is straightforward and can be done with a single command.

```bash
{'text': 'Amazing movie!', 'sentiment': 1}
# "label" column renamed to "sentiment".
```

---

### 17\. **Dataset Column Removal 🗑️**

**Boilerplate Code**:

```python
dataset = dataset.remove_columns(["text"])
```

**Use Case**: Remove unnecessary columns from a dataset to focus on relevant features.

**Goal**: Drop specific columns from the dataset to reduce its size or complexity. 🎯

**Sample Code**:

```python
from datasets import load_dataset

dataset = load_dataset("imdb", split="train")
reduced_dataset = dataset.remove_columns(["text"])
print(reduced_dataset["train"][0])
```

**Before Example**: You manually remove columns using pandas or write custom code for column management.

```bash
# Manually dropping columns:
df.drop(columns=["text"], inplace=True)
```

**After Example**: With the `datasets` library, removing columns is quick and efficient.

```bash
{'label': 1}
# "text" column removed from the dataset.
```

---

### 18\. **Dataset Bucketing and Binning 📊**

**Boilerplate Code**:

```python
def bucket_function(examples):
    return {"length_bucket": int(len(examples["text"].split()) / 10)}

binned_dataset = dataset.map(bucket_function)
```

**Use Case**: Group continuous values into buckets or bins (e.g., text length categories).

**Goal**: Create buckets to categorize data based on a feature (e.g., sentence length, age group). 🎯

**Sample Code**:

```python
from datasets import load_dataset

dataset = load_dataset("imdb", split="train")

def bucket_function(examples):
    return {"length_bucket": int(len(examples["text"].split()) / 10)}

binned_dataset = dataset.map(bucket_function)
print(binned_dataset["train"][0])
```

**Before Example**: You manually create bins or buckets for data based on continuous features.

```bash
# Manually binning data:
df['length_bucket'] = df['length'] // 10
```

**After Example**: With `datasets`, bucketing or binning data is easy and done with a simple function.

```bash
{'text': 'Great acting!', 'label': 1, 'length_bucket': 0}
# Dataset bucketed based on text length.
```

---

### 19\. **Dataset Imputation for Missing Values 🩹**

**Boilerplate Code**:

```python
def impute_function(examples):
    if examples["text"] is None:
        examples["text"] = "N/A"
    return examples

imputed_dataset = dataset.map(impute_function)
```

**Use Case**: <mark>Handle missing or null values in the dataset by filling them with default values.</mark>

**Goal**: Impute missing values to ensure clean and complete data. 🎯

**Sample Code**:

```python
from datasets import load_dataset

dataset = load_dataset("imdb", split="train")

def impute_function(examples):
    if examples["text"] is None:
        examples["text"] = "N/A"
    return examples

imputed_dataset = dataset.map(impute_function)
print(imputed_dataset["train"][0])
```

**Before Example**: You manually handle missing values, which can be tedious, especially for large datasets.

```bash
# Manually filling missing values:
df['text'].fillna("N/A", inplace=True)
```

**After Example**: The `datasets` library allows easy imputation of missing values in large datasets.

```bash
{'text': 'Fantastic!', 'label': 1}
# Missing values filled with default values.
```

---

### 20\. **Dataset Sampling 🧪**

**Boilerplate Code**:

```python
sampled_dataset = dataset.shuffle(seed=42).select(range(100))
```

**Use Case**: Sample a subset of data from a large dataset for quick testing or analysis.

**Goal**: Randomly sample a fixed number of examples from the dataset. 🎯

**Sample Code**:

```python
from datasets import load_dataset

dataset = load_dataset("imdb", split="train")
sampled_dataset = dataset.shuffle(seed=42).select(range(100))
print(len(sampled_dataset))
```

**Before Example**: You manually sample data by writing custom sampling functions, which can be slow.

```bash
# Manually sampling data:
sampled_df = df.sample(n=100, random_state=42)
```

**After Example**: The `datasets` library provides efficient sampling functionality with built-in methods.

```bash
100
# Random sample of 100 examples selected from the dataset.
```