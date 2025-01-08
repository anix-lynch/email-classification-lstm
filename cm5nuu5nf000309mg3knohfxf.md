---
title: "Semantic Routing VS LLM-Based Routing"
datePublished: Wed Jan 08 2025 12:07:19 GMT+0000 (Coordinated Universal Time)
cuid: cm5nuu5nf000309mg3knohfxf
slug: semantic-routing-vs-llm-based-routing

---

## **1\. Semantic Routing 🧭 – Based on Similarity**

Think of **semantic routing** like a **matchmaking service** 💘 that uses **similarity scores** to route queries. It checks **how close the user query is** to pre-defined **questions or prompts** based on **text embeddings** (fancy word for vectorized meaning).

### **How Does It Work?**

1. **Create Embeddings** for the query and pre-defined domain questions.
    
2. **Measure Similarity** using cosine distance (how close meanings are).
    
3. **Route Based on Score** – The domain with the **highest similarity** handles the query.
    

### **Key Strengths**

* **No labels needed** 📦 – Works directly with **semantic similarity**, skipping manual classification!
    
* **Dynamic Matching** 🔄 – Ideal for unstructured queries when domains overlap.
    

### **Key Weaknesses**

* **Limited Flexibility** 🛑 – Can't handle **edge cases** or ambiguous queries well.
    
* **No Learning Ability** 📚 – Doesn’t **learn from examples**—relies purely on embeddings.
    

---

## **2\. LLM-Based Classifier 🚦 – Based on Context**

Now, think of **LLM-based classifiers** as **customized experts** 🧑‍🏫 who **read the query** and **label it** into categories using **examples** or **rules**. Instead of measuring similarity, they **understand intent** and **predict categories**.

### **How Does It Work?**

1. **Pre-train Examples** – Teach the model **patterns** for query classification (e.g., fitness vs finance).
    
2. **Classify Query** – Use an **LLM template** to categorize the query.
    
3. **Route Based on Label** – Forward it to the **appropriate domain template** based on the predicted label.
    

### **Key Strengths**

* **Better for Ambiguous Queries** 🤔 – Can **reason through context** rather than rely on keywords.
    
* **Learns Patterns** 📖 – Adapts with **few-shot learning** using labeled examples.
    

### **Key Weaknesses**

* **Requires Examples** ✏️ – Needs pre-defined **templates or categories**.
    
* **Slightly Slower** 🐢 – Depends on **LLM processing** each query (vs direct embeddings).
    

---

## **Quick Example: Find the Difference 🚀**

### **User Query: "What’s the best way to save for retirement?"**

### **Semantic Routing** 🧭:

* **Matches by Similarity** – Compares query embedding to **personal finance questions**.
    
* Routes query based on the **highest similarity score**—simple but limited to matching patterns.
    

**Output:** "🪙 Routed to Personal Finance template!"

### **LLM-Based Routing** 🚦:

* **Reads Context and Classifies** – Checks if the question mentions **saving** or **retirement**.
    
* Labels it as **Personal Finance** even if phrased differently.
    

**Output:** "🪙 Routed to Personal Finance template!"

---

### **Key Takeaway 🍰**

* Use **Semantic Routing** if your queries are **short and well-defined** or when **embeddings work well** for pattern-matching.
    
* Use **LLM-Based Classifiers** for **complex, ambiguous questions** that need **reasoning** or **learning from examples**.
    

---

### **Routing with LLM-Based Classifiers: The Query Matchmaker 💌**

Imagine you're at a huge info desk 🏢, asking questions about **finance**, **fitness**, **books**, or **travel**. Instead of guessing answers, the receptionist (our classifier) smartly routes your question to an **expert** in that area. 💡

**LLM-Based Classifiers** make this routing dynamic—no hardcoding rules! They classify queries based on content and **redirect them to domain-specific prompts** for precise answers. 🚀

---

### **Why Use LLM-Based Routing?**

* **Auto Classification** 🤖 – Learns patterns, handles new categories.
    
* **Scalable & Flexible** 🌱 – Adapts as queries evolve—no manual updates.
    
* **Context-Driven Answers** 🎯 – Connects queries to **domain-relevant prompts** for clarity.
    

---

### **<mark>Where Does Routing Kick In? 🚦</mark>**

* **<mark>Step 6:</mark>** `prompt_router()` <mark> – This function:</mark>
    
    1. **Classifies the Query** 🔍
        
    2. **Chooses the Best Template** 📝 (e.g., finance vs. fitness)
        
    3. **Routes for Processing** 🚀
        

---

### **Full Code for LLM-Based Routing 🧑‍💻**

### **Step 1: Import Modules 📦**

```python
import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
```

---

### **Step 2: Set Up API Keys 🔑**

```python
os.environ['OPENAI_API_KEY'] = ""  # Add OpenAI API Key here
if os.environ['OPENAI_API_KEY'] == "":
    raise ValueError("Please set the OPENAI_API_KEY environment variable")
```

---

### **Step 3: Define LLM Templates 📝**

```python
# Domain-specific templates
personal_finance_template = "You are a finance expert. Help with budgeting, savings, and investments."
book_review_template = "You are a literary critic. Recommend books and provide reviews."
health_fitness_template = "You are a fitness coach. Offer workout plans and health tips."
travel_guide_template = "You are a travel expert. Suggest places to visit and travel tips."
```

---

### **Step 4: Classification Template 🧠**

```python
classification_template = ChatPromptTemplate.from_template(
    """
    You are good at classifying user queries into the following categories:
    - Personal Finance
    - Book Review
    - Health & Fitness
    - Travel Guide

    Given the user's question, classify it into one of the categories above.

    Question: {question}

    Classification:
    """
)
```

---

### **Step 5: Build Classification Chain 🔗**

```python
# Initialize LLM with output parser
classification_chain = (
    classification_template
    | ChatOpenAI(temperature=0)  # Query classification
    | StrOutputParser()          # Parses output (classification label)
)
```

---

### **<mark>Step 6: Route Queries (Core Logic) 🚦</mark>**

```python
def prompt_router(input_query):
    """Classify query and return the matching prompt."""
    try:
        # Get classification label
        classification = classification_chain.invoke({"question": input_query["query"]})

        # Route to the matching template
        if "Personal Finance" in classification:
            print("🪙 Routed to Personal Finance!")
            return personal_finance_template
        elif "Book Review" in classification:
            print("📚 Routed to Book Review!")
            return book_review_template
        elif "Health & Fitness" in classification:
            print("💪 Routed to Health & Fitness!")
            return health_fitness_template
        elif "Travel Guide" in classification:
            print("✈️ Routed to Travel Guide!")
            return travel_guide_template
        else:
            print("❌ No matching category found!")
            return None
    except Exception as e:
        print(f"Error during routing: {e}")
        return None
```

---

### **<mark>Step 7: Use the Router 🚀</mark>**

```python
# User input queries
input_query_1 = {"query": "What are the best exercises for weight loss?"}
input_query_2 = {"query": "Can you suggest must-see places in Italy?"}

# Route the first query
prompt = prompt_router(input_query_1)

if prompt:
    # Build processing chain for selected prompt
    chain = (
        RunnablePassthrough()
        | ChatPromptTemplate.from_template(prompt)  # Use selected template
        | ChatOpenAI(temperature=0)                # Generate answer
        | StrOutputParser()                        # Parse output
    )
    # Get response
    response = chain.invoke(input_query_1)
    print("\nAI Response:", response)
else:
    print("❌ Unable to classify query!")

# Route the second query
prompt_2 = prompt_router(input_query_2)

if prompt_2:
    # Build chain and get response
    chain_2 = (
        RunnablePassthrough()
        | ChatPromptTemplate.from_template(prompt_2)
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
    )
    response_2 = chain_2.invoke(input_query_2)
    print("\nAI Response:", response_2)
else:
    print("❌ Unable to classify query!")
```

---

### **Example Output 📝**

**Query 1:**

> *"What are the best exercises for weight loss?"*

* **Routed to:** 💪 *Health & Fitness*
    
* **AI Response:**
    

> "For weight loss, combine cardio (e.g., running, swimming) with strength training like squats and lunges. Include HIIT for faster results."

**Query 2:**

> *"Can you suggest must-see places in Italy?"*

* **Routed to:** ✈️ *Travel Guide*
    
* **AI Response:**
    

> "In Italy, visit Rome's Colosseum, Venice's canals, Florence's art museums, and the Amalfi Coast for stunning views."

---

### **Quick Recap: Where Does Routing Kick In? 🚦**

1. **<mark>Step 6 – </mark>** `prompt_router()`
    
    * <mark>Classifies </mark> **<mark>query intent</mark>** <mark> and matches </mark> **<mark>domain templates</mark>**<mark>.</mark>
        
2. **<mark>Step 7 – Query Execution 🤖</mark>**
    
    * <mark>Routes to </mark> **<mark>domain-specific LLM</mark>** <mark> for generating answers.</mark>
        

---

### **Why Is This Awesome? 🤩**

* **Smart Classifier** – Dynamically adjusts without manual updates!
    
* **Scales Easily** – Add new domains by tweaking templates—no big changes needed.
    
* **Domain Expertise** – Uses specialized prompts for **sharper answers**.