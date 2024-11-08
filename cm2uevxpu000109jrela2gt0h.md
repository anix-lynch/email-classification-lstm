---
title: "Hugging Face & Google Colab"
seoTitle: "Hugging Face & Google Colab"
seoDescription: "Hugging Face & Google Colab"
datePublished: Tue Oct 29 2024 12:16:04 GMT+0000 (Coordinated Universal Time)
cuid: cm2uevxpu000109jrela2gt0h
slug: hugging-face-google-colab
tags: google-colab, huggingface

---

# **How do I integrate Hugging Face with Google Colab?**

Integrating Hugging Face with Google Colab is simple and powerful, especially with Hugging Face’s `transformers` library. Here’s a step-by-step guide to get started:

---

### Step 1: Install Hugging Face Libraries

First, install the necessary Hugging Face libraries in your Colab environment. Run this in a Colab cell:

```python
!pip install transformers datasets huggingface_hub
```

---

### Step 2: Log In to Hugging Face (Optional)

If you want to access private models or upload models to the Hugging Face Hub, log in using your Hugging Face account token:

1. **Get your token**: Go to [your Hugging Face account](https://huggingface.co/settings/tokens) and create an API token.
    
2. **Log in on Colab**:
    
    ```python
    from huggingface_hub import login
    login(token="YOUR_HUGGING_FACE_API_TOKEN")
    ```
    

*If you don’t need access to private models or Hub uploads, you can skip this step.*

---

### Step 3: Load Models and Tokenizers

Load any model directly from the Hugging Face Hub. Here’s an example with `distilbert-base-uncased` for sentiment analysis:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")
result = classifier("I love using Hugging Face with Colab!")
print(result)
```

---

### Step 4: Load Datasets

Use `datasets` to load a dataset directly from the Hugging Face Hub, like IMDb for sentiment analysis:

```python
from datasets import load_dataset

dataset = load_dataset("imdb")
print(dataset["train"][0])
```

---

### Step 5: Training on Google Colab (Optional)

If you want to fine-tune a model in Colab, Hugging Face’s `Trainer` API can automate the process, including data handling, model training, and evaluation.

**Example: Fine-tuning on IMDb Dataset**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize dataset
encoded_dataset = dataset.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True), batched=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=8
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"]
)

# Start training
trainer.train()
```

---

### Step 6: Save or Upload Models (Optional)

To save the model to your Google Drive:

```python
model.save_pretrained("/content/drive/MyDrive/my-model")
```

Or, upload the model directly to Hugging Face Hub (if logged in):

```python
model.push_to_hub("your-huggingface-username/my-model")
```

---

# What are the steps to open a Hugging Face notebook in Google Colab?

If you want to open a Hugging Face notebook directly in Google Colab, here are the steps:

### Step 1: Find the Hugging Face Notebook

1. Go to the [Hugging Face GitHub repository](https://github.com/huggingface/transformers) or the [Hugging Face Hub](https://huggingface.co/).
    
2. Look for the "Examples" or "Notebooks" section, where Hugging Face provides various notebooks to demonstrate tasks like model training, fine-tuning, and deployment.
    

### Step 2: Open the Notebook in Colab

Once you find a notebook you want to open:

1. **From GitHub**:
    
    * Open the notebook (`.ipynb` file) on GitHub.
        
    * Click on the **"Open in Colab"** button if available, or
        
    * Replace [`github.com`](http://github.com) in the URL with [`colab.research.google.com/github`](http://colab.research.google.com/github).
        
    * Example: Change [`https://github.com/huggingface/transformers/blob/main/notebooks/01_how_to_train.ipynb`](https://github.com/huggingface/transformers/blob/main/notebooks/01_how_to_train.ipynb) to [`https://colab.research.google.com/github/huggingface/transformers/blob/main/notebooks/01_how_to_train.ipynb`](https://colab.research.google.com/github/huggingface/transformers/blob/main/notebooks/01_how_to_train.ipynb).
        
2. **From the Hugging Face Hub**:
    
    * Some models or datasets may have linked Colab notebooks. Look for an "Open in Colab" button or notebook link.
        
    * Click on it to directly open the notebook in Colab.
        

### Step 3: Run the Notebook in Colab

Once in Colab:

1. Run the first cell to ensure all dependencies are installed.
    
2. Follow the instructions in each cell to execute the code.
    

### Optional: Save the Notebook to Google Drive

* If you want to keep a copy in your Google Drive, go to `File > Save a copy in Drive`.
    

By following these steps, you can quickly open and work with Hugging Face notebooks in Google Colab.

# **How can I fine-tune a model using Hugging Face in Google Colab?**

To fine-tune a model using Hugging Face in Google Colab, follow these steps:

---

### Step 1: Set Up the Environment

1. **Install Hugging Face Libraries**: Run the following cell in Colab to install the required libraries:
    
    ```python
    !pip install transformers datasets
    ```
    

### Step 2: Load a Dataset

Hugging Face’s `datasets` library provides many datasets you can use directly. Here’s an example with the IMDb sentiment dataset:

```python
from datasets import load_dataset

# Load IMDb dataset for sentiment analysis
dataset = load_dataset("imdb")
```

### Step 3: Preprocess the Data

Tokenize the dataset to prepare it for training. This step uses the model's tokenizer to convert text into input IDs.

```python
from transformers import AutoTokenizer

# Choose a model; we'll use 'distilbert-base-uncased' for this example
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the text
def tokenize_data(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

# Apply tokenization to the dataset
encoded_dataset = dataset.map(tokenize_data, batched=True)
```

### Step 4: Load the Model

Load a pre-trained model with a classification head. For binary sentiment analysis, Hugging Face's `AutoModelForSequenceClassification` is useful.

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
```

### Step 5: Set Up Training Arguments

Define training parameters like batch size, learning rate, and the number of epochs.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",          # Where to save the model
    evaluation_strategy="epoch",     # Evaluate at the end of each epoch
    per_device_train_batch_size=8,   # Batch size for training
    per_device_eval_batch_size=8,    # Batch size for evaluation
    num_train_epochs=3,              # Number of training epochs
    weight_decay=0.01,               # Strength of weight decay
)
```

### Step 6: Initialize the Trainer

The `Trainer` class handles training, evaluation, and logging automatically.

```python
from transformers import Trainer

trainer = Trainer(
    model=model,                       # Model to be fine-tuned
    args=training_args,                # Training arguments
    train_dataset=encoded_dataset["train"],  # Training dataset
    eval_dataset=encoded_dataset["test"]     # Evaluation dataset
)
```

### Step 7: Train the Model

Start the fine-tuning process.

```python
trainer.train()
```

### Step 8: Evaluate the Model (Optional)

After training, you can evaluate the model on the test set.

```python
trainer.evaluate()
```

### Step 9: Save the Model (Optional)

If you want to save the fine-tuned model to Google Drive:

```python
model.save_pretrained("/content/drive/MyDrive/my-fine-tuned-model")
tokenizer.save_pretrained("/content/drive/MyDrive/my-fine-tuned-model")
```

---

# **How do I install the huggingface\_hub library in Google Colab?**

To install the `huggingface_hub` library in Google Colab, follow these steps:

### Step 1: Install `huggingface_hub`

In a Colab cell, run:

```python
!pip install huggingface_hub
```

### Step 2: Import and Log In (Optional)

If you plan to interact with the Hugging Face Hub (e.g., uploading models or datasets), you’ll need to log in with your Hugging Face account token.

```python
from huggingface_hub import login

# Replace 'YOUR_TOKEN' with your actual token from Hugging Face
login(token="YOUR_TOKEN")
```

You can obtain your token from [Hugging Face Account Settings](https://huggingface.co/settings/tokens).

---

# What hugging face CLI can i use in colab?

In Google Colab, you can use Hugging Face's Command Line Interface (CLI) to interact with the Hugging Face Hub directly from the terminal. Here’s a quick guide to commonly used Hugging Face CLI commands in Colab.

---

### Step 1: Install the CLI Tools

Install the `huggingface_hub` package, which includes the CLI:

```python
!pip install huggingface_hub
```

### Step 2: Login Using CLI

The `huggingface-cli login` command lets you authenticate with the Hugging Face Hub.

1. Run the login command:
    
    ```python
    !huggingface-cli login
    ```
    
2. You’ll be prompted to enter your Hugging Face token (found in [Hugging Face Settings](https://huggingface.co/settings/tokens)). Paste the token and press enter.
    

---

### Common Hugging Face CLI Commands in Colab

Here are some useful commands to use in Colab after logging in.

#### 1\. **Upload a Model**

If you’ve fine-tuned a model and want to upload it to the Hugging Face Hub:

```python
!huggingface-cli upload ./path_to_your_model -y
```

Replace `./path_to_your_model` with the path to your model directory.

#### 2\. **Create a New Repository**

Create a new model repository on the Hugging Face Hub:

```python
!huggingface-cli repo create your-username/your-model-name
```

This command creates a new repository under your Hugging Face username.

#### 3\. **List Repositories**

List all repositories associated with your Hugging Face account:

```python
!huggingface-cli repo list
```

#### 4\. **Delete a Repository**

Delete a specific repository (use with caution):

```python
!huggingface-cli repo delete your-username/your-model-name
```

This permanently deletes the specified repository.

#### 5\. **Download Files from the Hub**

To download files directly from the Hugging Face Hub:

```python
!huggingface-cli download your-username/your-model-name --include="*.bin"
```

Replace `"*.bin"` with the specific file pattern you want to download.

---

### Example: Upload a Fine-Tuned Model to the Hub

If you’ve saved a model locally in Colab, you can upload it to your Hugging Face Hub repository.

1. **Save your model**:
    
    ```python
    model.save_pretrained("./my-fine-tuned-model")
    tokenizer.save_pretrained("./my-fine-tuned-model")
    ```
    
2. **Upload using CLI**:
    
    ```python
    !huggingface-cli upload ./my-fine-tuned-model -y
    ```
    

This command uploads all files in the `my-fine-tuned-model` directory to the Hugging Face Hub.

---

### Summary of Useful CLI Commands in Colab

| Command | Purpose |
| --- | --- |
| `!huggingface-cli login` | Log in to Hugging Face Hub |
| `!huggingface-cli upload <path>` | Upload a model or dataset to the Hub |
| `!huggingface-cli repo create <repo-name>` | Create a new repository |
| `!huggingface-cli repo list` | List your repositories |
| `!huggingface-cli repo delete <repo-name>` | Delete a repository (use with caution) |
| `!huggingface-cli download <repo-name>` | Download files from a Hub repository |

These CLI commands make it easy to manage your Hugging Face models, datasets, and repositories directly from Colab.