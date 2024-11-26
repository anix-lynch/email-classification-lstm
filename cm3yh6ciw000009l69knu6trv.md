---
title: "How AI can classify customer queries like a pro?"
seoTitle: "How AI can classify customer queries like a pro?"
seoDescription: "How AI can classify customer queries like a pro?"
datePublished: Tue Nov 26 2024 13:10:56 GMT+0000 (Coordinated Universal Time)
cuid: cm3yh6ciw000009l69knu6trv
slug: how-ai-can-classify-customer-queries-like-a-pro
tags: chatbot, classification, openai

---

In this blog, we’re diving into how **AI** can classify customer queries like a pro, ensuring that each request gets routed to the right department. Whether your customer needs help with billing 💳, technical support 💻, or just wants some product info 📱, AI can efficiently categorize and respond based on predefined categories. Think of it as your personal customer service assistant, but smarter! 🤖 Join me as we explore how to set up AI for the ultimate **query classification.**

### **System Message and How it Classifies Queries:**

In this blog, we’ve set up a system message to **guide the AI** in **classifying customer queries** into appropriate categories. Here’s how it works:

### **1\. System Message Overview:**

The system message provides the **AI** with the rules and logic needed to classify incoming customer queries into specific categories. It includes:

* **Primary Categories**: Broad categories like **Billing**, **Technical Support**, **Account Management**, and **General Inquiry**.
    
* **Secondary Categories**: More specific subcategories under each primary category that further classify the query, such as:
    
    * **Billing**: Unsubscribe or upgrade, Add a payment method, Explanation for charge, Dispute a charge.
        
    * **Technical Support**: General troubleshooting, Device compatibility, Software updates.
        
    * **Account Management**: Password reset, Update personal information, Close account, Account security.
        
    * **General Inquiry**: Product information, Pricing, Feedback, Speak to a human.
        

### **2\. Classification Process:**

When a customer query is received, the AI follows these steps:

1. **<mark>Step 1</mark>**<mark>: </mark> **<mark>Identify Primary Category</mark>**<mark>:</mark>
    
    * The AI first identifies which **primary category** the query belongs to. For example, if the query is about deleting a user account, the primary category would be **Account Management**.
        
    * If the query is about product information, it would be classified under **General Inquiry**.
        
2. **<mark>Step 2</mark>**<mark>: </mark> **<mark>Identify Secondary Category</mark>**<mark>:</mark>
    
    * After determining the primary category, the AI then identifies which **secondary category** the query fits. For example, if the query is about deleting a user account, it will fall under the secondary category of **Close account** under **Account Management**.
        
    * If the query is about getting more information on flat screen TVs, it will fall under **Product information** under **General Inquiry**.
        
3. **<mark>Step 3</mark>**<mark>: </mark> **<mark>Provide the Classification in JSON Format</mark>**<mark>:</mark>
    
    * Once the query is classified, the AI returns a structured **JSON response** showing the **primary and secondary categories**.
        

### **How Classification Works in Action:**

1. **Example 1:**
    
    * **User Query**: "I want you to delete my profile and all of my user data."
        
    * **AI Classification**:
        
        * **Primary Category**: **Account Management**
            
        * **Secondary Category**: **Close account**
            
    * **Output**:
        
        ```json
        {
          "primary": "Account Management",
          "secondary": "Close account"
        }
        ```
        
2. **Example 2:**
    
    * **User Query**: "Tell me more about your flat screen TVs."
        
    * **AI Classification**:
        
        * **Primary Category**: **General Inquiry**
            
        * **Secondary Category**: **Product information**
            
    * **Output**:
        
        ```json
        {
          "primary": "General Inquiry",
          "secondary": "Product information"
        }
        ```
        

---

### **Conclusion:**

The system message essentially provides the **AI** with the framework for **classifying** customer queries into clear, actionable categories. It ensures that each query is processed efficiently, helping route it to the right department or providing a quick, accurate response. The AI then returns the classification in a structured **JSON format** for easy use in further workflows.

Let's break this code down into **chunks**, explain what each part does, and provide **sample outputs**.

---

### **Chunk 1: Setup and Loading the API Key**

```python
import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # Read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']  # Set OpenAI API key
```

**Explanation:**

* This section loads the **API key** from the `.env` file (securely) and sets it to use OpenAI's API.
    
* **No output** here—this is just setting things up.
    

---

### **Chunk 2: Helper Function for Chat Completion**

```python
def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, 
                                 max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # Controls randomness (0 for deterministic responses)
        max_tokens=max_tokens,  # Limit the number of tokens in the response
    )
    return response.choices[0].message["content"]  # Return the content of the response
```

**Explanation:**

* This function sends a **list of messages** to OpenAI's **Chat API**.
    
* It returns the **model’s response** based on the messages it receives.
    
* The **temperature** setting controls the randomness of the response (a lower value like 0 gives more predictable responses).
    
* **No output** here either—it's just a helper function.
    

---

### **<mark>Chunk 3: System Message Setup (Classification Instructions)</mark>**

```python
delimiter = "####"
system_message = f"""
You will be provided with customer service queries. \
The customer service query will be delimited with \
{delimiter} characters.
Classify each query into a primary category \
and a secondary category. 
Provide your output in json format with the \
keys: primary and secondary.

Primary categories: Billing, Technical Support, \
Account Management, or General Inquiry.

Billing secondary categories:
Unsubscribe or upgrade
Add a payment method
Explanation for charge
Dispute a charge

Technical Support secondary categories:
General troubleshooting
Device compatibility
Software updates

Account Management secondary categories:
Password reset
Update personal information
Close account
Account security

General Inquiry secondary categories:
Product information
Pricing
Feedback
Speak to a human
"""
```

**Explanation:**

* This **system message** guides the model to classify customer queries into one of the **primary categories** (Billing, Technical Support, Account Management, or General Inquiry).
    
* Each primary category also has **secondary categories** that further specify the type of query.
    
* **No output** here either—this sets the context for how the model should classify the queries.
    

---

### **Chunk 4: Example Query 1 - User Wants Profile Deleted**

```python
user_message = f"""\
I want you to delete my profile and all of my user data"""
messages =  [  
    {'role':'system', 
     'content': system_message},    
    {'role':'user', 
     'content': f"{delimiter}{user_message}{delimiter}"},  
] 
response = get_completion_from_messages(messages)
print(response)
```

**Explanation:**

* The **user query** asks to delete their profile and data. Based on the system message, the model should classify this query into **Account Management** (secondary category: **Close account**).
    
* The **delimiter** separates the query from the reasoning steps that the model will perform.
    

**Sample Output:**

```json
{
  "primary": "Account Management",
  "secondary": "Close account"
}
```

* **Explanation**: The model correctly classifies this as an **Account Management** query with the secondary category of **Close account**.
    

---

### **Chunk 5: Example Query 2 - Asking About Flat Screen TVs**

```python
user_message = f"""\
Tell me more about your flat screen tvs"""
messages =  [  
    {'role':'system', 
     'content': system_message},    
    {'role':'user', 
     'content': f"{delimiter}{user_message}{delimiter}"},  
] 
response = get_completion_from_messages(messages)
print(response)
```

**Explanation:**

* The **user query** is asking about flat screen TVs. This falls under the **General Inquiry** category (secondary category: **Product information**).
    
* The system message guides the model to classify this query properly.
    

**Sample Output:**

```json
{
  "primary": "General Inquiry",
  "secondary": "Product information"
}
```

* **Explanation**: The model correctly classifies this as a **General Inquiry** with the secondary category of **Product information**.
    

---

### **Final Explanation**:

The code is designed to take a **customer query**, classify it into a **primary** and **secondary category**, and return the classification in **JSON format**. The primary categories are predefined (e.g., **Billing**, **Technical Support**, etc.), and the secondary categories are more specific to the nature of the inquiry.

---

### **What Happens in Each Chunk**:

1. **Setup**: Loads the OpenAI API key.
    
2. **Helper Function**: Defines how to get a response from OpenAI’s API.
    
3. **System Message**: Sets up classification logic for queries.
    
4. **First User Query**: Classifies the query "I want you to delete my profile and all of my user data".
    
5. **Second User Query**: Classifies the query "Tell me more about your flat screen tvs".
    

### **Summary of Expected Outputs**:

* For deletion of user data:
    
    ```json
    { "primary": "Account Management", "secondary": "Close account" }
    ```
    
* For asking about TVs:
    
    ```json
    { "primary": "General Inquiry", "secondary": "Product information" }
    ```
    

This process helps you categorize customer queries effectively, making it easier to route them to the right team or provide the correct response.