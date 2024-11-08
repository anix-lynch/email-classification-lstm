---
title: "Customer Service AI Assistant with LangChain, Hugging Face, and Vertex AI"
seoTitle: "Customer Service AI Assistant with LangChain, Hugging Face, and Vertex"
seoDescription: "Customer Service AI Assistant with LangChain, Hugging Face, and Vertex AI"
datePublished: Tue Oct 29 2024 14:32:52 GMT+0000 (Coordinated Universal Time)
cuid: cm2ujrv8v000b09mo720t4tar
slug: customer-service-ai-assistant-with-langchain-hugging-face-and-vertex-ai
tags: ai, huggingface, llm, vertex-ai, langchain

---

### **1\. Listing the Most Popular Hugging Face Models for Specific Tasks**

```python
from huggingface_hub import list_models

# Define a function to list the top 5 most downloaded models for a specified task
def list_most_popular(task: str):
    for rank, model in enumerate(
        list_models(filter=task, sort="downloads", direction=-1)
):
        if rank == 5:
            break
        print(f"{model.id}, {model.downloads}\n")

# List popular models for text classification
list_most_popular("text-classification")

# List popular models for summarization
list_most_popular("summarization")
```

* **Explanation**:
    
    * **Purpose**: This code lists the top 5 models for each task (`text-classification` and `summarization`), sorted by the highest number of downloads on Hugging Face.
        
    * **Why it’s useful**: Knowing the popular models can help you quickly choose reliable models for customer service applications without trial and error.
        
* **Sample Output**:
    
    ```plaintext
    facebook/bart-large-mnli, 500000
    distilbert-base-uncased, 300000
    ...
    ```
    

---

### **2\. Setting Up a Hugging Face Summarization Endpoint**

```python
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

# Sample customer email text
customer_email = """..."""  # Customer’s detailed complaint (see full text above).

# Set up the summarization model, requiring an API token (set in environment as HUGGINGFACEHUB_API_TOKEN)
summarizer = HuggingFaceEndpoint(
    repo_id="facebook/bart-large-cnn",
    model_kwargs={"temperature":0, "max_length":180}
)

# Function to summarize a text
def summarize(llm, text) -> str:
    return llm(f"Summarize this: {text}!")

# Summarize the customer email
print(summarize(summarizer, customer_email))
```

* **Explanation**:
    
    * **HuggingFaceEndpoint**: Wraps the `facebook/bart-large-cnn` summarization model for easy use within LangChain. It requires an API token from Hugging Face.
        
    * **Summarize Function**: Sends the text to the model with the `Summarize this:` prompt to get a concise summary.
        
    * **Why it’s useful**: Summarizing customer emails helps quickly capture the essence of lengthy complaints, improving response efficiency.
        
* **Sample Output**:
    
    ```plaintext
    "Customer received a damaged coffee machine and expresses disappointment in the product and brand quality."
    ```
    

---

### **3\. Analyzing Sentiment with Hugging Face’s Sentiment Model**

```python
from transformers import pipeline

# Set up a sentiment analysis model for evaluating customer emotions
sentiment_model = pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

# Run sentiment analysis on the customer email
print(sentiment_model(customer_email))

# Test the model on different example texts
print(sentiment_model("I am so angry and sad, I want to kill myself!"))
print(sentiment_model("I am elated, I am so happy, this is the best thing that ever happened to me!"))
print(sentiment_model("I don't care. I guess it's ok, or not, I couldn't care one way or the other"))
```

* **Explanation**:
    
    * **Sentiment Analysis Pipeline**: Loads a pre-trained sentiment model to analyze the tone and emotion in customer messages.
        
    * **Purpose**: Provides insights into the customer’s emotional state, helping the support team handle emotionally charged situations effectively.
        
    * **Different Inputs**: Tests include strongly negative, positive, and neutral statements to assess the model’s accuracy.
        
* **Sample Output**:
    
    ```plaintext
    [{'label': 'NEGATIVE', 'score': 0.95}]
    [{'label': 'NEGATIVE', 'score': 0.99}]
    [{'label': 'POSITIVE', 'score': 0.98}]
    [{'label': 'NEUTRAL', 'score': 0.92}]
    ```
    

---

### **4\. Categorizing Customer Concerns with Vertex AI and LangChain**

```python
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain

# Define a template to classify the customer's issue based on the email content
template = """Given this text, decide what is the issue the customer is concerned about. Valid categories are these:
* product issues
* delivery problems
* missing or late orders
* wrong product
* cancellation request
* refund or exchange
* bad support experience
* no clear reason to be upset

Text: {email}
Category:
"""
# Create the prompt template
prompt = PromptTemplate(template=template, input_variables=["email"])

# Initialize VertexAI to handle text generation
llm = VertexAI()

# Create a chain that runs the prompt through VertexAI
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

# Run the customer email through the chain to classify the issue
print(llm_chain.run(customer_email))
```

* **Explanation**:
    
    * **PromptTemplate**: Provides a structured prompt with predefined categories, guiding the model to classify the complaint correctly.
        
    * **LLMChain**: Uses LangChain to pass the email content through the VertexAI model, leveraging its classification capabilities.
        
    * **Purpose**: Helps identify the specific category of the complaint (e.g., product issue, delivery problem), enabling better response routing and prioritization.
        
* **Sample Output**:
    
    ```plaintext
    "product issues"
    ```
    

---

### **Summary**

This notebook sets up a **Customer Service Helper** that can:

1. **Identify popular models** to quickly access reliable tools.
    
2. **Summarize customer emails** for efficient review.
    
3. **Analyze emotional sentiment** to gauge the intensity of the complaint.
    
4. **Classify the issue type** for streamlined support and response.