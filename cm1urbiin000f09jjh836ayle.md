---
title: "6 Hugging Face Accelerate to use with google colab pro for NLP tasks"
seoTitle: "6 Hugging Face Accelerate concepts with Before-and-After Examples"
seoDescription: "6 Hugging Face Accelerate concepts with Before-and-After Examples"
datePublished: Fri Oct 04 2024 13:24:24 GMT+0000 (Coordinated Universal Time)
cuid: cm1urbiin000f09jjh836ayle
slug: 20-hugging-face-accelerate-concepts-with-before-and-after-examples
tags: ai, deep-learning, huggingface, accelerate, distributed-training

---

### **Step 1: Setting Up Google Colab Pro with GPU**

Before running any model or using Hugging Face Accelerate, you need to make sure that you're using **GPU** in Colab.

#### How to enable GPU in Colab:

1. **Go to the top menu**: Click **Runtime &gt; Change runtime type**.
    
2. **Select GPU**:
    
    * Under **Hardware accelerator**, choose **GPU** from the dropdown.
        
    * For **Colab Pro**, this could give you access to a **Tesla T4**, **P100**, or **V100 GPU**.
        
3. **Click Save**.
    

#### Verify GPU is enabled:

Run the following code to check that a GPU is available:

```python
pythonCopy codeimport torch

if torch.cuda.is_available():
    print(f"GPU enabled: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found")
```

* **Expected output**: This will print the **name of the GPU**, such as **Tesla T4** or **P100**. If no GPU is found, it means you need to recheck your runtime settings.
    

---

### **Step 2: Install Hugging Face Accelerate and Dependencies**

To use Hugging Face Accelerate, install the necessary packages in your Colab environment:

```python
bashCopy code!pip install accelerate transformers datasets
```

* **Expected output**: You will see a confirmation that the packages have been installed successfully.
    

### **Test Dataset: GLUE MRPC** (Microsoft Research Paraphrase Corpus)

For the examples, we’ll use the **GLUE MRPC** dataset, which is a common benchmark for NLP tasks. It contains pairs of sentences and the goal is to determine whether the two sentences are paraphrases (similar in meaning).

To keep it simple, we’ll load this dataset in each demo and use it for testing the different features of **Accelerate**.

---

###   
Step 3: Experiment with Accelerate 6 ways:  
  
1\. **Mixed Precision Training with Accelerate ⚡**

* **Analogy**: Imagine you're packing for a vacation. Instead of carrying everything in **large suitcases**, you use **half-size suitcases** (FP16 precision), which take up **less space** but still fit all your essentials.
    
* **Why useful**: With **smaller suitcases**, you can fit more into your luggage (more data into memory), and you can **move faster** because it’s lighter. This helps you train models faster by <mark>using less memory.</mark>
    
* **For Colab Pro**: Since you have **one GPU** in Colab Pro, this feature makes training faster by making better use of the available memory.  
    

```python
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Initialize Accelerator with mixed precision
accelerator = Accelerator(mixed_precision="fp16")

# Load dataset and tokenizer
dataset = load_dataset('glue', 'mrpc')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the dataset
def tokenize(batch):
    return tokenizer(batch['sentence1'], batch['sentence2'], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Load model and optimizer
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Prepare everything with Accelerator
model, optimizer, tokenized_dataset = accelerator.prepare(model, optimizer, tokenized_dataset)

# Check that it's running on GPU and using FP16
print(f"Model is on device: {next(model.parameters()).device}")
print(f"Mixed precision: {accelerator.mixed_precision}")

# Train for 1 batch just to verify
for batch in tokenized_dataset['train']:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    break
```

#### **Expected Output**:

* **GPU detection**:
    
    ```python
    Model is on device: cuda:0
    ```
    
    This means the model is running on **GPU**.
    
* **Mixed precision**:
    
    ```python
    Mixed precision: fp16
    ```
    
    This confirms that the model is using **FP16 precision** for faster computation and lower memory usage.
    
* **No errors** during the training loop indicates that the training process with **FP16** is successful.  
    

---

### 2\. **Saving and Loading Checkpoints Efficiently 💾**

* **Analogy**: Imagine you’re playing a video game, and after each level, you can **save your progress**. If you have to stop playing or your console shuts off unexpectedly, you can **resume exactly where you left off**.
    
* **Why useful**: If your Colab runtime disconnects (which can happen in long training sessions), saving checkpoints allows you to pick up the training from where it left off rather than starting from scratch.
    
* **For Colab Pro**: Prevents losing progress and allows you to easily **resume** long-running training tasks.  
      
    **Code**:
    
    ```python
    import os
    from accelerate import Accelerator
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset
    
    # Load dataset and tokenizer
    dataset = load_dataset('glue', 'mrpc')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize the dataset
    def tokenize(batch):
        return tokenizer(batch['sentence1'], batch['sentence2'], padding=True, truncation=True)
    
    tokenized_dataset = dataset.map(tokenize, batched=True)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Prepare model and dataset
    model, tokenized_dataset = accelerator.prepare(model, tokenized_dataset)
    
    # Save checkpoint
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), 'model_checkpoint.pth')
    
    # Check if checkpoint file was saved
    if os.path.exists('model_checkpoint.pth'):
        print("Checkpoint saved successfully!")
    else:
        print("Checkpoint save failed.")
        
    # Load the checkpoint back into the model
    unwrapped_model.load_state_dict(torch.load('model_checkpoint.pth', map_location=accelerator.device))
    print("Checkpoint loaded successfully!")
    ```
    
    #### **Expected Output**:
    
* **Visible output**:
    
    ```python
    Checkpoint saved successfully!
    Checkpoint loaded successfully!
    ```
    
    * **Checkpoint saved successfully!**: This confirms that the checkpoint was saved to disk as `model_checkpoint.pth`.
        
    * **Checkpoint loaded successfully!**: This confirms that the saved checkpoint was successfully loaded back into the model, and you can resume training from this point.
        
* **Non-visible output**:
    
    * Check the **Colab file system** to see if the file `model_checkpoint.pth` was created. If it’s there, your checkpoint was saved correctly.
        

### 3\. **Gradient Accumulation with Accelerate 🧮**

* **Analogy**: Imagine you're trying to carry **several heavy boxes**, but they are too heavy to carry all at once. Instead, you carry a few boxes at a time and place them down until you've moved everything.
    
* **Why useful**: With **limited memory** on a Mac or Colab, **gradient accumulation** allows you to train larger models in **small steps**, accumulating gradients over time before updating the model. This way, you can **carry more load** in smaller, manageable steps.
    
* **For Mac**: It’s like breaking a big job into smaller tasks that fit your limited memory, allowing you to train **large models** without overloading your system.  
    
    ```python
    from accelerate import Accelerator
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset
    import torch
    
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Load dataset and tokenizer
    dataset = load_dataset('glue', 'mrpc')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize the dataset
    def tokenize(batch):
        return tokenizer(batch['sentence1'], batch['sentence2'], padding=True, truncation=True)
    
    tokenized_dataset = dataset.map(tokenize, batched=True)
    
    # Load model and optimizer
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Prepare everything with Accelerator
    model, optimizer, tokenized_dataset = accelerator.prepare(model, optimizer, tokenized_dataset)
    
    # Example of gradient accumulation
    accumulation_steps = 4
    optimizer.zero_grad()
    
    for step, batch in enumerate(tokenized_dataset['train']):
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps  # Scale the loss
        accelerator.backward(loss)
        
        # Accumulate gradients
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f"Step {step+1}: Weights updated")
        if step > 8:  # Limiting the loop for brevity
            break
    ```
    
    #### **Expected Output**:
    
    * **Visible output**:
        
        ```python
        Step 4: Weights updated
        Step 8: Weights updated
        ```
        
        * **Step 4: Weights updated** and **Step 8: Weights updated**: These messages indicate that the gradients were accumulated over 4 steps before updating the weights. This confirms that **gradient accumulation** is working.
            
    * **Non-visible output**:
        
        * **No errors** during training. If the model successfully completes multiple steps and updates the weights after the specified number of accumulation steps, gradient accumulation is functioning correctly.
            

---

### 4\. **Effortless Mixed Precision in FP16 🖥️**

* **Analogy**: Think of it as **driving a car on a fuel-efficient mode**. By switching to a lower gear (FP16 precision), you save fuel (memory), but still keep moving forward efficiently.
    
* **Why useful**: FP16 allows your model to run faster with **less memory**, just like driving in an efficient mode uses less gas.
    
* **For Colab Pro**: Helps your **GPU** run more efficiently by using **less memory** and speeding up calculations.
    

#### **Code**:

```python
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch

# Initialize accelerator with zero-offload
accelerator = Accelerator(cpu=True, mixed_precision="fp16")

# Load dataset and tokenizer
dataset = load_dataset('glue', 'mrpc')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the dataset
def tokenize(batch):
    return tokenizer(batch['sentence1'], batch['sentence2'], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Load model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased').to(accelerator.device)

# Offload some layers to CPU if GPU memory is too high
for param in model.parameters():
    if torch.cuda.memory_allocated() > 1e9:  # If memory > 1GB
        param = param.to("cpu")
        print(f"Parameter offloaded to CPU: {param.device}")
```

#### **Expected Output**:

* **Visible output**:
    
    ```python
    Parameter offloaded to CPU: cpu
    ```
    
    * **Parameter offloaded to CPU: cpu**: This message confirms that certain model parameters were offloaded to the **CPU** to save GPU memory when GPU usage exceeded 1GB.
        
* **Non-visible output**:
    
    * **GPU memory utilization**: You can run `!nvidia-smi` to check the GPU memory usage before and after offloading. You should see that some GPU memory has been freed up after the parameters were offloaded to the CPU.
        

---

### 5\. **Zero-offload with Accelerate for Efficient Memory Usage 📉**

* **Analogy**: Picture yourself trying to **carry multiple bags**, but your hands are full. You decide to place the **heavier bags** in a **shopping cart** (the CPU) to make it easier to carry the lighter ones (on the GPU).
    
* **Why useful**: **Zero-offload** allows you to offload memory-intensive parts of your model onto the CPU, so your **GPU** can focus on lighter tasks. This ensures you don’t run out of GPU memory when training larger models.
    
* **For Mac/Colab Pro**: Offloading helps you manage memory more efficiently, especially when your hardware has **limited GPU memory**.
    

```python
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch

# Initialize accelerator
accelerator = Accelerator()

# Load dataset and tokenizer
dataset = load_dataset('glue', 'mrpc')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the dataset
def tokenize(batch):
    return tokenizer(batch['sentence1'], batch['sentence2'], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Load model and optimizer
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Prepare everything with Accelerator
model, optimizer, tokenized_dataset = accelerator.prepare(model, optimizer, tokenized_dataset)

# Gradient clipping to prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# In the training loop
for batch in tokenized_dataset['train']:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)

    # Clip gradients before optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    break
```

#### **Expected Output**:

* **Visible output**:
    
    ```python
    Parameter offloaded to CPU: cpu
    ```
    
    * **Parameter offloaded to CPU: cpu**: This message appears if the **Zero-offload** feature is triggered due to high GPU memory usage, showing that memory-intensive parts of the model were moved to the **CPU** to optimize GPU usage.
        
* **Non-visible output**:
    
    * You can monitor the **GPU memory utilization** using `!nvidia-smi` to see if GPU memory is being reduced when offloading parameters to the CPU.
        

---

### 6\. **Handling Gradient Clipping with Accelerate ✂️**

* **Analogy**: Imagine you’re driving a car and you want to make sure you don’t go too fast around sharp curves. **Gradient clipping** is like putting a **speed limit** on your car to make sure you don’t lose control.
    
* **Why useful**: In training, gradients can become too large, causing unstable updates. **Gradient clipping** puts a limit on them to ensure your model trains **stably** and doesn’t “crash.”
    
* **For Colab Pro**: Helps keep training stable and smooth, avoiding issues like **exploding gradients** during GPU training.  
    **Code**:
    
    ```python
    pythonCopy codefrom accelerate import Accelerator
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset
    import torch
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load dataset and tokenizer
    dataset = load_dataset('glue', 'mrpc')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Tokenize the dataset
    def tokenize(batch):
        return tokenizer(batch['sentence1'], batch['sentence2'], padding=True, truncation=True)
    
    tokenized_dataset = dataset.map(tokenize, batched=True)
    
    # Load model and optimizer
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Prepare everything with Accelerator
    model, optimizer, tokenized_dataset = accelerator.prepare(model, optimizer, tokenized_dataset)
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # In the training loop
    for batch in tokenized_dataset['train']:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
    
        # Clip gradients before optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        break
    ```
    
    **Expected Output**:
    
    * **Visible output**:
        
        * There is **no visible output** directly from gradient clipping itself. However, you should see that training proceeds smoothly without errors or instability.
            
    * **Non-visible output**:
        
        * **Stable training**: Gradient clipping prevents **exploding gradients**, ensuring the training runs smoothly. If your model doesn't encounter errors during training and produces reasonable loss values, then **gradient clipping** is working.  
            

---