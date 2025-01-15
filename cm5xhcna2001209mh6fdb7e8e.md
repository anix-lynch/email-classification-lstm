---
title: "Generative AI for Research Summarization w/Langchain & HuggingFace Hub"
seoTitle: "Generative AI for Research Summarization w/Langchain & HuggingFace Hub"
seoDescription: "Generative AI for Research Summarization w/Langchain & HuggingFace Hub"
datePublished: Wed Jan 15 2025 05:47:29 GMT+0000 (Coordinated Universal Time)
cuid: cm5xhcna2001209mh6fdb7e8e
slug: generative-ai-for-research-summarization-wlangchain-huggingface-hub
tags: llm, langchain, genai, agentic-ai, huggingfacehub

---

To implement a tool that summarizes recent publications using **Generative AI**, here’s a Python example leveraging **LangChain** and **Hugging Face Transformers** for summarizing an article from arXiv.

---

### **Code Example**

```python
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests

# Hugging Face API Key (replace with your key)
huggingface_api_key = "YOUR_HUGGINGFACE_API_KEY"

# Initialize Hugging Face model (e.g., GPT-4 equivalent model like "google/flan-t5-large")
llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0}, huggingfacehub_api_token=huggingface_api_key)

# Define a custom prompt template
summary_prompt = PromptTemplate(
    template="Summarize the following research paper abstract in a concise and clear way:\n\nAbstract: {abstract}\n\nSummary:",
    input_variables=["abstract"]
)

# Create an LLMChain
summarizer_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Fetch the arXiv abstract
def fetch_arxiv_abstract(arxiv_id):
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url)
    if response.status_code == 200:
        # Extract abstract
        start = response.text.find("<summary>") + len("<summary>")
        end = response.text.find("</summary>")
        abstract = response.text[start:end].strip()
        return abstract
    else:
        raise Exception(f"Failed to fetch arXiv abstract: {response.status_code}")

# Test with an arXiv ID
arxiv_id = "2401.06795"  # Replace with a valid arXiv ID
abstract = fetch_arxiv_abstract(arxiv_id)

# Run the summarizer
summary = summarizer_chain.run({"abstract": abstract})
print("🔬 Abstract Summary:", summary)
```

---

### **Sample Output**

Assuming the input abstract is:

> "This paper introduces a novel approach for real-time multimodal interaction using generative AI models. We explore the integration of text, image, and audio inputs to provide richer user experiences. Experiments demonstrate significant improvements in understanding user intent, paving the way for more intuitive AI systems."

#### **Output Summary**:

```python
🔬 Abstract Summary: This paper presents a novel generative AI method for multimodal interactions, integrating text, image, and audio for intuitive user experiences and improved intent understanding.
```

---

### **How It Works**

1. **Fetch Abstract**: The script queries the arXiv API for the paper's abstract.
    
2. **Custom Prompt**: A LangChain pipeline feeds the abstract into a summarization model.
    
3. **Hugging Face Model**: Uses the **FLAN-T5** model (you can replace it with any summarization model).
    
4. **Output**: Generates a concise summary for easy understanding.
    

---

### **Impact**

* **Time-Saving**: Summarizes lengthy abstracts into actionable insights.
    
* **Scalable**: Process hundreds of papers in a batch for literature reviews.
    
* **Customizable**: Modify the prompt for specific research domains (e.g., medicine, physics).
    

Would you like me to adjust this further or add batch processing for multiple papers? 🚀

# **Here’s how you can run the arXiv abstract summarization tool step-by-step in Google Colab:**

### **Steps to Implement in Colab**

#### **Step 1: Open Google Colab**

* Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.
    

---

#### **Step 2: Install Required Libraries**

Install the necessary packages for Hugging Face and LangChain.

```python
!pip install langchain huggingface-hub requests
```

---

#### **Step 3: Set Up Hugging Face API Key**

* Create a [Hugging Face account](https://huggingface.co/join) if you don’t already have one.
    
* Generate an **API key** from the [Hugging Face settings](https://huggingface.co/settings/tokens).
    

In Colab, set your API key as an environment variable:

```python
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_HUGGINGFACE_API_KEY"  # Replace with your key
```

---

#### **Step 4: Write the Summarization Code**

Copy the following code into a Colab cell:

```python
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests

# Initialize Hugging Face model
llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0})

# Define a custom prompt template
summary_prompt = PromptTemplate(
    template="Summarize the following research paper abstract in a concise and clear way:\n\nAbstract: {abstract}\n\nSummary:",
    input_variables=["abstract"]
)

# Create an LLMChain
summarizer_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Fetch the arXiv abstract
def fetch_arxiv_abstract(arxiv_id):
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url)
    if response.status_code == 200:
        # Extract abstract
        start = response.text.find("<summary>") + len("<summary>")
        end = response.text.find("</summary>")
        abstract = response.text[start:end].strip()
        return abstract
    else:
        raise Exception(f"Failed to fetch arXiv abstract: {response.status_code}")

# Test with an arXiv ID
arxiv_id = "2401.06795"  # Replace with a valid arXiv ID
abstract = fetch_arxiv_abstract(arxiv_id)

# Run the summarizer
summary = summarizer_chain.run({"abstract": abstract})
print("🔬 Abstract Summary:", summary)
```

---

#### **Step 5: Execute the Code**

* Run the cells one by one in Colab.
    
* Ensure the abstract from the specified arXiv ID is fetched and summarized by the model.
    

---

### **Output Example in Colab**

When you run the script, you’ll see an output like this in your Colab notebook:

```python
🔬 Abstract Summary: This paper presents a novel generative AI method for multimodal interactions, integrating text, image, and audio for intuitive user experiences and improved intent understanding.
```

---

### **Optional: Batch Process Multiple Papers**

To summarize multiple abstracts at once:

1. Create a list of arXiv IDs:
    
    ```python
    arxiv_ids = ["2401.06795", "2301.12345", "2205.67890"]
    ```
    
2. Fetch and summarize abstracts in a loop:
    
    ```python
    for arxiv_id in arxiv_ids:
        abstract = fetch_arxiv_abstract(arxiv_id)
        summary = summarizer_chain.run({"abstract": abstract})
        print(f"🔬 Summary for {arxiv_id}:\n{summary}\n")
    ```
    

---

### **What’s Next?**

* **Customization**: Edit the `PromptTemplate` for specific domains (e.g., "medical research," "physics").
    
* **Visualization**: Export summaries to CSV or Google Sheets for easy sharing.