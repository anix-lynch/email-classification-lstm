---
title: "Open AI API"
seoTitle: "Open AI API"
seoDescription: "Open AI API"
datePublished: Tue Nov 26 2024 12:16:09 GMT+0000 (Coordinated Universal Time)
cuid: cm3yf7w8u001709la2yn8atw7
slug: open-ai-api
tags: openai

---

# CASE 1: Set tone

---

### **1\. Import Libraries and Load API Key**

```python
# Import necessary libraries
import os  # For accessing environment variables
import openai  # OpenAI API library for interacting with models
import tiktoken  # Tokenizer library for token management
from dotenv import load_dotenv, find_dotenv  # For securely loading environment variables from .env file

# Load the environment variables from a .env file (e.g., API key)
_ = load_dotenv(find_dotenv())  # Locate and load .env file
openai.api_key = os.environ['OPENAI_API_KEY']  # Set the API key for OpenAI
```

**Output:**  
No output.

---

### **2\. Define a Basic Helper Function**

```python
# Define a helper function to send a single prompt to the OpenAI model
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]  # Set up a single user message
    response = openai.ChatCompletion.create(  # Call the OpenAI API
        model=model,  # Specify the model
        messages=messages,  # Provide the messages
        temperature=0,  # Set randomness to 0 (deterministic response)
    )
    return response.choices[0].message["content"]  # Return the model's response
```

**Output:**  
No output.

---

### **3\. Basic Prompt Example**

```python
# Ask a simple question using the helper function
response = get_completion("What is the capital of France?")
print(response)
```

**Expected Output:**

```python
Paris
```

---

### **4\. Tokenization Example**

```python
# Example: Reverse the letters in a word
response = get_completion("Take the letters in lollipop and reverse them")
print(response)
```

**Expected Output:**

```python
popillol
```

```python
# Example: Same prompt but with hyphenated text
response = get_completion("""Take the letters in l-o-l-l-i-p-o-p and reverse them""")
print(response)
```

**Expected Output:**

```python
p-o-p-i-l-l-o-l
```

---

### **5\. Define an Advanced Helper Function (Message-Based)**

```python
# Helper function to send multiple messages to the model
def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,  # Specify the model
        messages=messages,  # Pass the messages list (with roles)
        temperature=temperature,  # Randomness of the output
        max_tokens=max_tokens,  # Limit on response length
    )
    return response.choices[0].message["content"]  # Extract and return the content
```

**Output:**  
No output.

---

### **6\. Example: Custom Style Responses**

```python
# Example: Respond in the style of Dr. Seuss
messages = [  
    {'role': 'system', 'content': "You are an assistant who responds in the style of Dr. Seuss."},  
    {'role': 'user', 'content': "Write me a very short poem about a happy carrot."},  
] 
response = get_completion_from_messages(messages, temperature=1)
print(response)
```

**Expected Output:**

```python
There once was a carrot, so orange and bright,  
It danced in the garden, from morning to night!  
With a laugh and a hop, it spread so much cheer,  
The happiest carrot you'd find anywhere near!
```

---

### **7\. Combining Multiple Behaviors**

```python
# Example: Combine Dr. Seuss style with a one-sentence constraint
messages = [  
    {'role': 'system', 'content': "You are an assistant who responds in the style of Dr. Seuss. All your responses must be one sentence long."},  
    {'role': 'user', 'content': "Write me a story about a happy carrot."},
]
response = get_completion_from_messages(messages, temperature=1)
print(response)
```

**Expected Output:**

```python
A carrot named Claire danced in the sun, spreading joy to everyone.
```

---

### **8\. Define a Token-Aware Helper Function**

```python
# Function to get response and token usage
def get_completion_and_token_count(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,  # Specify the model
        messages=messages,  # Provide the messages
        temperature=temperature,  # Set randomness
        max_tokens=max_tokens,  # Limit response length
    )
    # Extract response content
    content = response.choices[0].message["content"]

    # Extract token usage details
    token_dict = {
        'prompt_tokens': response['usage']['prompt_tokens'],  # Tokens used in the input
        'completion_tokens': response['usage']['completion_tokens'],  # Tokens in the response
        'total_tokens': response['usage']['total_tokens'],  # Total tokens used
    }
    return content, token_dict  # Return both the response and token usage
```

**Output:**  
No output.

---

### **9\. Token Usage Example**

```python
# Example: Get response and token count
messages = [
    {'role': 'system', 'content': "You are an assistant who responds in the style of Dr. Seuss."},  
    {'role': 'user', 'content': "Write me a very short poem about a happy carrot."},  
]
response, token_dict = get_completion_and_token_count(messages)

# Print the response
print(response)  # Response content

# Print token usage details
print(token_dict)  # Token details
```

**Expected Output:**  
**Response:**

```python
A carrot named Fred with a smile so wide,  
Laughed with the turnips and danced with pride!
```

**Token Dictionary:**

```python
{'prompt_tokens': 31, 'completion_tokens': 24, 'total_tokens': 55}
```

---

### **10\. Notes on Setup**

```python
# Install OpenAI Python library if not installed
# !pip install openai

# Set the API key in your environment (alternative method)
# import openai
# openai.api_key = "sk-..."  # Replace with your actual API key
```

**Output:**  
No output unless installation is required. During installation:

```python
Collecting openai
  Downloading openai-0.27.2-py3-none-any.whl (72 kB)
...
Successfully installed openai-0.27.2
```

---

# CASE 2: W/Chain of thoughts

Here's the code with inline comments and expected outputs where applicable:

---

### **1\. Import Libraries and Load API Key**

```python
# Import necessary libraries
import os  # For accessing environment variables
import openai  # OpenAI API library
import sys  # For handling system-level operations (not used here)
from dotenv import load_dotenv, find_dotenv  # For loading environment variables from .env file

# Load the API key securely from a .env file
_ = load_dotenv(find_dotenv())  # Locate and load .env file
openai.api_key = os.environ['OPENAI_API_KEY']  # Set the API key for OpenAI
```

**Output:**  
No output.

---

### **2\. Define a Helper Function**

```python
# Function to send messages to the OpenAI model and get responses
def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,  # Model to use
        messages=messages,  # List of messages including system and user roles
        temperature=temperature,  # Controls randomness of the response
        max_tokens=max_tokens,  # Limit the number of tokens in the response
    )
    return response.choices[0].message["content"]  # Extract and return the content
```

**Output:**  
No output.

---

### **3\. Define System Message with Chain-of-Thought Prompting**

```python
# Delimiter for separating reasoning steps in the response
delimiter = "####"

# System message providing detailed step-by-step reasoning instructions
system_message = f"""
Follow these steps to answer the customer queries.
The customer query will be delimited with four hashtags,\
i.e. {delimiter}. 

Step 1:{delimiter} First decide whether the user is \
asking a question about a specific product or products. \
Product cateogry doesn't count. 

Step 2:{delimiter} If the user is asking about \
specific products, identify whether \
the products are in the following list.
All available products: 
1. Product: TechPro Ultrabook
   ...
(Full product list continues)
...
Step 5:{delimiter}: First, politely correct the \
customer's incorrect assumptions if applicable. \
Only mention or reference products in the list of \
5 available products, as these are the only 5 \
products that the store sells. \
Answer the customer in a friendly tone.

Use the following format:
Step 1:{delimiter} <step 1 reasoning>
Step 2:{delimiter} <step 2 reasoning>
Step 3:{delimiter} <step 3 reasoning>
Step 4:{delimiter} <step 4 reasoning>
Response to user:{delimiter} <response to customer>

Make sure to include {delimiter} to separate every step.
"""
```

**Output:**  
No output.

---

### **4\. Query: Compare Product Prices**

```python
# User query about comparing prices of two products
user_message = f"""
by how much is the BlueWave Chromebook more expensive \
than the TechPro Desktop"""

# Define the conversation with system instructions and user query
messages = [  
    {'role':'system', 'content': system_message},    
    {'role':'user', 'content': f"{delimiter}{user_message}{delimiter}"},  
]

# Get and print the response
response = get_completion_from_messages(messages)
print(response)
```

**Expected Output:**

```python
Step 1:#### The user is asking about specific products.
Step 2:#### The products mentioned are "BlueWave Chromebook" and "TechPro Desktop," which are in the available list.
Step 3:#### The user assumes the BlueWave Chromebook is more expensive than the TechPro Desktop.
Step 4:#### The assumption is incorrect. The TechPro Desktop costs $999.99, while the BlueWave Chromebook costs $249.99.
Response to user:#### The BlueWave Chromebook is actually less expensive than the TechPro Desktop by $750.00.
```

---

### **5\. Query: Asking About Non-Listed Products**

```python
# User query about unrelated product (e.g., TVs)
user_message = f"""
do you sell tvs"""

# Define the conversation with system instructions and user query
messages = [  
    {'role':'system', 'content': system_message},    
    {'role':'user', 'content': f"{delimiter}{user_message}{delimiter}"},  
]

# Get and print the response
response = get_completion_from_messages(messages)
print(response)
```

**Expected Output:**

```python
Step 1:#### The user is not asking about specific products.
Step 2:#### The user is asking about a product category (TVs), which is not in the available product list.
Response to user:#### Sorry, we currently do not sell TVs. Our store specializes in the five listed products.
```

---

### **6\. Inner Monologue: Hide Chain-of-Thought Reasoning**

```python
# Split the response to show only the final message to the user
try:
    final_response = response.split(delimiter)[-1].strip()  # Extract final part after delimiter
except Exception as e:
    final_response = "Sorry, I'm having trouble right now, please try asking another question."

# Print the simplified response for the user
print(final_response)
```

**Expected Output for Query 1:**

```python
The BlueWave Chromebook is actually less expensive than the TechPro Desktop by $750.00.
```

**Expected Output for Query 2:**

```python
Sorry, we currently do not sell TVs. Our store specializes in the five listed products.
```

---