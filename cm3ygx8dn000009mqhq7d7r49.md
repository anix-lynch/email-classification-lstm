---
title: "How to setup Chain-of-Thought Prompting with OpenAI"
seoTitle: "How to setup Chain-of-Thought Prompting with OpenAI"
seoDescription: "How to setup Chain-of-Thought Prompting with OpenAI"
datePublished: Tue Nov 26 2024 13:03:51 GMT+0000 (Coordinated Universal Time)
cuid: cm3ygx8dn000009mqhq7d7r49
slug: how-to-setup-chain-of-thought-prompting-with-openai
tags: openai, chain-of-thought-prompting

---

1. **Chain-of-Thought Prompting**:
    
    * **System Message**: The system message contains detailed instructions on how the model should process the user query. This is where the **chain-of-thought reasoning** comes in. The steps break down how the model should:
        
        * First, determine if the user is asking about specific products.
            
        * Check if the products mentioned match any in the list of available products.
            
        * Identify any assumptions made in the user’s query and whether they are true or false.
            
        * Respond politely, making sure only the available products are mentioned.
            
2. **Handling User Messages**:
    
    * In each example (e.g., "by how much is the BlueWave Chromebook more expensive than the TechPro Desktop?"), the query is processed step by step to ensure that assumptions are validated and the response is well-structured.
        
    * The **delimiter (**`####`) is used to separate each reasoning step, allowing the model to logically analyze the query before responding.
        
3. **Inner Monologue**:
    
    * The final response displayed to the user excludes the internal reasoning steps (hidden by the delimiter) and only shows the **final answer** to the user.
        

### **Chunk 1: Setup and API Key**

```python
import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # Load environment variables (API key)

openai.api_key = os.environ['OPENAI_API_KEY']  # Set the API key from the environment variable
```

**Explanation:**

* This section **loads your API key** using the `dotenv` package, so you can securely interact with OpenAI's API.
    
* **No output** here; it's just setting things up for later.
    

---

### **Chunk 2: Helper Function for Getting Completion from OpenAI**

```python
def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    return response.choices[0].message["content"]
```

**Explanation:**

* This function sends a list of **messages** to the OpenAI model and gets a **response**.
    
* It takes the `temperature` parameter, which controls the randomness of the response (set to 0 here for deterministic responses).
    
* **Output**: It returns the content of the model’s response, which can be printed or processed further.
    

---

### **Chunk 3: System Message with Chain-of-Thought Prompting**

```python
delimiter = "####"
system_message = f"""
Follow these steps to answer the customer queries.
The customer query will be delimited with four hashtags, i.e. {delimiter}. 

Step 1:{delimiter} First decide whether the user is \
asking a question about a specific product or products. \
Product category doesn't count. 

Step 2:{delimiter} If the user is asking about \
specific products, identify whether the products are in the following list.
All available products: 
1. Product: TechPro Ultrabook
   Category: Computers and Laptops
   Brand: TechPro
   Model Number: TP-UB100
   Warranty: 1 year
   Rating: 4.5
   Features: 13.3-inch display, 8GB RAM, 256GB SSD, Intel Core i5 processor
   Description: A sleek and lightweight ultrabook for everyday use.
   Price: $799.99

2. Product: BlueWave Gaming Laptop
   Category: Computers and Laptops
   Brand: BlueWave
   Model Number: BW-GL200
   Warranty: 2 years
   Rating: 4.7
   Features: 15.6-inch display, 16GB RAM, 512GB SSD, NVIDIA GeForce RTX 3060
   Description: A high-performance gaming laptop for an immersive experience.
   Price: $1199.99

Step 3:{delimiter} If the message contains products \
in the list above, list any assumptions that the user is making in their message e.g. that Laptop X is bigger than Laptop Y.

Step 4:{delimiter} If the user made any assumptions, figure out whether the assumption is true based on your product information. 

Step 5:{delimiter} First, politely correct the customer's incorrect assumptions if applicable. Only mention or reference products in the list of 5 available products.
"""
```

**Explanation:**

* This is where you define the **reasoning steps** for the model.
    
* The model is guided through five **steps** to logically process the customer query, check the products mentioned, and respond with the correct information.
    
* **No output** at this stage—this is just a setup for the reasoning steps.
    

---

### **Chunk 4: User Query Example 1 - Price Comparison**

```python
user_message = f"""
by how much is the BlueWave Chromebook more expensive \
than the TechPro Desktop"""

messages =  [  
    {'role':'system', 'content': system_message},    
    {'role':'user', 'content': f"{delimiter}{user_message}{delimiter}"},  
] 

response = get_completion_from_messages(messages)
print(response)
```

**Explanation:**

* This chunk sends a **user query** asking for a **price comparison** between two products: the BlueWave Chromebook and the TechPro Desktop.
    
* The **delimiter** helps separate reasoning steps for the model.
    
* The **response** will be the model’s reasoning and final answer.
    

**Sample Output:**

```python
Step 1:#### The user is asking about specific products: BlueWave Chromebook and TechPro Desktop.
Step 2:#### Both products are in the list of available products.
Step 3:#### The user assumes the BlueWave Chromebook is more expensive than the TechPro Desktop.
Step 4:#### The TechPro Desktop costs $999.99, and the BlueWave Chromebook costs $249.99. The BlueWave Chromebook is actually less expensive by $750.
Response to user:#### The BlueWave Chromebook is $750 less expensive than the TechPro Desktop.
```

---

### **Chunk 5: User Query Example 2 - Asking About TV Sales**

```python
user_message = f"""
do you sell tvs"""
messages =  [  
    {'role':'system', 'content': system_message},    
    {'role':'user', 'content': f"{delimiter}{user_message}{delimiter}"},  
] 
response = get_completion_from_messages(messages)
print(response)
```

**Explanation:**

* The user asks whether the store sells **TVs**, but since TVs are not listed in the available products, the system needs to respond politely and clarify this.
    

**Sample Output:**

```python
Step 1:#### The user is asking about a product category, not specific products.
Step 2:#### TVs are not in the list of available products.
Response to user:#### Sorry, we do not sell TVs. We only offer laptops and desktops.
```

---

### **Chunk 6: Hide Inner Reasoning (Chain-of-Thought)**

```python
try:
    final_response = response.split(delimiter)[-1].strip()  # Hide reasoning and return final response
except Exception as e:
    final_response = "Sorry, I'm having trouble right now, please try asking another question."
    
print(final_response)
```

**Explanation:**

* This chunk **splits** the reasoning steps and only shows the **final response** to the user.
    
* The model uses the **delimiter** to separate the reasoning, and the final user-facing output is just the **answer** without the chain-of-thought breakdown.
    

**Sample Output:**

```python
The BlueWave Chromebook is $750 less expensive than the TechPro Desktop.
```

---

### **Summary of the Code Flow**:

1. **Setup**: Loads the environment variables and initializes OpenAI API.
    
2. **Helper Function**: Defines a function to call OpenAI's `ChatCompletion` API.
    
3. **System Message**: Guides the model to use **Chain-of-Thought reasoning** to break down and analyze customer queries.
    
4. **User Query 1**: Analyzes the **price comparison** query between two products, outputting reasoning and the answer.
    
5. **User Query 2**: Handles a **TV sales** query, explaining that the store doesn't sell TVs.
    
6. **Inner Reasoning**: Hides the internal reasoning steps and only returns the final output to the user.
    

By chunking the code this way, you allow the model to reason through each customer query logically and return accurate answers in a **well-structured manner**!