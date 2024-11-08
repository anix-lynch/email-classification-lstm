---
title: "From Setup to Inference: Running Language Models on Google Colab GPU with LangChain"
seoTitle: "Running Language Models on Google Colab GPU with LangChain"
seoDescription: "Running Language Models on Google Colab GPU with LangChain"
datePublished: Tue Oct 29 2024 14:26:34 GMT+0000 (Coordinated Universal Time)
cuid: cm2ujjrd6000308mi7jtmd3d5
slug: from-setup-to-inference-running-language-models-on-google-colab-gpu-with-langchain
tags: gpu, google-colab, huggingface, llm, langchain

---

### **1\. Initial Setup: Checking for GPU Availability**

```python
# Import torch library, which is essential for deep learning tasks in PyTorch
import torch

# Determine if a GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

* **Explanation**: We use PyTorch (`torch`) to check if Colab has an available GPU.
    
* **Purpose**: Assigning `device` as `cuda` (GPU) or `cpu` ensures that our models run on the GPU if available, making them faster. If not, it defaults to CPU (slower but works if GPU isn’t enabled).
    
* **Why it’s useful**: Running on a GPU can accelerate model inference significantly, especially useful in Colab Pro where we have access to better GPUs.
    

---

### **2\. Installing Necessary Libraries**

```python
# Install the necessary libraries to handle large language models and LangChain functionality
!pip install accelerate transformers langchain
```

* **Explanation**:
    
    * `accelerate`: A library to optimize distributed and multi-GPU setups, so it’s especially helpful if we’re running on a powerful GPU.
        
    * `transformers`: A core library from Hugging Face that makes it easy to load and use pre-trained models.
        
    * `langchain`: This library allows us to create chains and integrate LLMs into workflows smoothly.
        
* **Why it’s useful**: These libraries give us all the tools we need for text generation with high-performance model chaining in LangChain on Colab’s GPU.
    

---

### **3\. Setting Up the Text Generation Pipeline with GPU**

```python
from transformers import pipeline

# Set up a Hugging Face pipeline for text generation with a specific model, set to use GPU if available
generate_text = pipeline(
    task="text-generation",                # Specifies we're doing text generation
    model="liminerity/Phigments12",         # Model choice; we can switch to "gpt2" or other models as needed
    trust_remote_code=True,                 # Allows use of any model-specific code if defined in the model’s repo
    torch_dtype="auto",                     # Automatically selects the best tensor type (float32/float16) for hardware
    device=0 if device == "cuda" else -1,   # Use GPU (device 0) if available, otherwise CPU (-1)
    max_new_tokens=100                      # Limits generated text length to 100 tokens for efficiency
)

# Test the pipeline with a simple prompt
print(generate_text("In this chapter, we'll discuss first steps with generative AI in Python."))
```

* **Explanation**:
    
    * **Pipeline**: The Hugging Face `pipeline` function simplifies model loading and makes it easy to run models directly on tasks.
        
    * **Model Choice**: `liminerity/Phigments12` is the model specified here, but you can experiment with other models (e.g., `gpt2`).
        
    * **trust\_remote\_code**: Some models have custom code that this flag enables; it’s essential for certain advanced or modified models.
        
    * **torch\_dtype**: Automatically selects the best tensor type, like `float32` or `float16`, based on GPU availability, making the code more efficient.
        
    * **device**: This tells the model whether to use GPU (`0`) or CPU (`-1`) for processing.
        
    * **max\_new\_tokens**: Keeps generation within 100 tokens, preventing overly lengthy responses which can use too much memory.
        
* **Sample Output**:
    
    ```plaintext
    "In this chapter, we'll discuss first steps with generative AI in Python, including setup and usage..."
    ```
    

---

### **4\. Wrapping the Pipeline with LangChain for Compatibility**

```python
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

# Wrap our Hugging Face pipeline to make it compatible with LangChain workflows
hf = HuggingFacePipeline(pipeline=generate_text)
```

* **Explanation**:
    
    * **HuggingFacePipeline Wrapper**: This wraps our `generate_text` pipeline so that it can integrate smoothly with LangChain.
        
    * **Why LangChain**: LangChain allows us to chain prompts and model responses efficiently, making complex workflows simpler and reusable.
        
    * **Compatibility**: With this setup, we can create chains and templates, controlling how models interact with user prompts.
        

---

### **5\. Setting Up Prompt Templates and Creating an LLM Chain**

```python
from langchain import PromptTemplate, LLMChain

# Define a simple prompt template with placeholders
template = """{question} Be concise!"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create an LLM Chain that feeds the prompt template into the Hugging Face pipeline model
llm_chain = LLMChain(prompt=prompt, llm=hf)

# Test the chain with a specific question
question = "What is electroencephalography?"
print(llm_chain.invoke(question))
```

* **Explanation**:
    
    * **PromptTemplate**: This template sets up a simple prompt structure with `{question}` as a variable placeholder, allowing us to reuse this format with different questions.
        
    * **LLMChain**: Combines the prompt template with our Hugging Face pipeline model, creating a workflow where the prompt is automatically fed into the model.
        
    * **Invoke Method**: `invoke()` sends the `question` directly through the pipeline, getting an answer formatted according to the prompt.
        
* **Sample Output**:
    
    ```plaintext
    "Electroencephalography is a method to record brain electrical activity."
    ```
    

---

### **6\. Switching to a Different Model (e.g., GPT-2) on GPU**

```python
# Try using GPT-2 model with the same setup for variety
hf = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100, "device": 0 if device == "cuda" else -1}
)

# Create an LLM Chain that combines the prompt with this new model
llm_chain = prompt | hf

# Test with the same question using GPT-2
question = "What is electroencephalography?"
print(llm_chain.invoke(question))
```

* **Explanation**:
    
    * **from\_model\_id**: This method allows us to load a different model (GPT-2) into our Hugging Face pipeline while keeping the same settings.
        
    * **pipeline\_kwargs**: We’re setting max tokens and device options directly in this method to customize GPT-2’s behavior.
        
    * **Chain Setup**: Here, `prompt | hf` is shorthand for feeding `prompt` into `hf` to create the LLM chain.
        
    * **Versatility**: Switching models like this lets you experiment with various LLMs to find the best fit for different tasks.
        
* **Sample Output**:
    
    ```plaintext
    "Electroencephalography, or EEG, records electrical signals in the brain using sensors on the scalp."
    ```
    

---

### **Notes for Cloud Use**

* **Avoid Local Models**: Since Colab doesn’t support local storage of massive model files long-term, we stick to cloud-hosted models (Hugging Face).
    
* **GPU Compatibility**: Setting this up on GPU makes the setup faster and more memory-efficient for inference. Colab’s GPU is well-suited to handle these Hugging Face model pipelines.