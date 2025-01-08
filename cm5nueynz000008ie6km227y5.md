---
title: "Semantic Routing: Your RAG Query’s GPS for Precise Answers! 🚦"
seoTitle: "Semantic Routing: Your RAG Query’s GPS for Precise Answers! 🚦"
seoDescription: "Semantic Routing: Your RAG Query’s GPS for Precise Answers! 🚦"
datePublished: Wed Jan 08 2025 11:55:30 GMT+0000 (Coordinated Universal Time)
cuid: cm5nueynz000008ie6km227y5
slug: semantic-routing-your-rag-querys-gps-for-precise-answers
tags: ai, llm, langchain, rag, semanticrouting

---

Imagine asking an AI assistant, *“Can you recommend a book?”* or *“What exercises help with weight loss?”*

Without routing, the assistant might treat both questions the same way—causing chaos! 😱 Semantic Routing fixes this by acting like **a GPS that sends queries to the right experts** based on intent. 🧭

---

### **Why Semantic Routing? 🚀**

1. **Laser-Focused Answers 🎯** – Queries are matched to the **right domain** (e.g., finance, health).
    
2. **Context-Aware Matching 🔍** – Uses embeddings (fancy math for understanding meaning) to match query intent with pre-trained categories.
    
3. **Saves Time ⏳** – Instead of reading all data, it goes directly to the most relevant info.
    

---

### **Where Does Semantic Routing Kick In?**

Semantic Routing shines in **Step 5** when it:

1. **Calculates Similarity 🔢** between the query and pre-defined domain prompts.
    
2. **Routes to the Closest Match 🚦** (e.g., Book Reviews vs Health Tips).
    
3. **Uses the Selected Template ✍️** for generating answers.
    

---

## **Full Code for Semantic Routing 🧑‍💻**

---

### **Step 1: Import Modules** 📦

```python
import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
```

---

### **Step 2: Set Up API Keys 🔑**

```python
os.environ['OPENAI_API_KEY'] = ""  # Add OpenAI API key here
if os.environ['OPENAI_API_KEY'] == "":
    raise ValueError("Please set the OPENAI_API_KEY environment variable")
```

---

### **Step 3: Define Domain Prompts & Questions 📝**

```python
# Domain-specific prompts
book_review_template = "You are a book critic. Provide insightful book reviews."
health_fitness_template = "You are a fitness coach. Offer advice on health and exercise routines."
travel_guide_template = "You are a travel expert. Share travel tips and destination recommendations."
personal_finance_template = "You are a finance advisor. Provide tips on budgeting and investments."

# Sample questions to define domain intent
book_review_questions = ["Can you recommend a book?", "What is the best novel of 2023?"]
health_fitness_questions = ["What exercises help with weight loss?", "How can I build muscle?"]
travel_guide_questions = ["Where should I go for my honeymoon?", "Best places to visit in Europe?"]
personal_finance_questions = ["How can I save money?", "What’s the best investment strategy?"]
```

---

### **Step 4: Create Text Embeddings 🧠**

```python
# Initialize embedding model
embeddings = OpenAIEmbeddings()

# Generate embeddings for each domain’s sample questions
book_review_embeddings = embeddings.embed_documents(book_review_questions)
health_fitness_embeddings = embeddings.embed_documents(health_fitness_questions)
travel_guide_embeddings = embeddings.embed_documents(travel_guide_questions)
personal_finance_embeddings = embeddings.embed_documents(personal_finance_questions)
```

---

### **<mark>Step 5: Route Queries Based on Similarity 🚦 (Semantic Routing Kicks In Here!)</mark>**

```python
# Function to calculate similarity and route query
def prompt_router(input):
    # Embed the user query
    query_embedding = embeddings.embed_query(input["query"])

    # Calculate cosine similarities
    similarities = {
        "Book Reviews": cosine_similarity([query_embedding], book_review_embeddings).max(),
        "Health & Fitness": cosine_similarity([query_embedding], health_fitness_embeddings).max(),
        "Travel Guide": cosine_similarity([query_embedding], travel_guide_embeddings).max(),
        "Personal Finance": cosine_similarity([query_embedding], personal_finance_embeddings).max(),
    }

    # Find the domain with the highest similarity
    best_domain = max(similarities, key=similarities.get)
    print(f"Query routed to: {best_domain}")  # Debugging output

    # Return the template based on the best match
    if best_domain == "Book Reviews":
        return book_review_template
    elif best_domain == "Health & Fitness":
        return health_fitness_template
    elif best_domain == "Travel Guide":
        return travel_guide_template
    else:
        return personal_finance_template
```

---

### **Step 6: Generate Responses Using Selected Prompt 🤖**

```python
# Define a sample query
user_input = {"query": "What exercises can I do at home to lose weight?"}

# Define LLM and routing pipeline
llm = ChatOpenAI(temperature=0)
chain = (
    RunnablePassthrough()
    | RunnableLambda(prompt_router)  # Route query based on similarity
    | ChatOpenAI(temperature=0)      # Use selected domain prompt
    | StrOutputParser()              # Parse final output
)

# Invoke the chain with user input
response = chain.invoke(user_input)

print("\nAI Response:")
print(response)
```

---

### **Example Output** 📝

**Input Query:**

> *"What exercises can I do at home to lose weight?"*

**Routing Decision:**

> Query routed to: **Health & Fitness**

**AI Response:**

> "For weight loss at home, focus on bodyweight exercises like squats, lunges, push-ups, and burpees. Combine these with cardio activities such as jump rope or high-intensity interval training (HIIT) for maximum results."

---

### **<mark>Quick Recap: Where Does Semantic Routing Kick In? 🚦</mark>**

1. **<mark>Step 5 – Similarity Calculation</mark>** <mark> 🧠</mark>
    
    * <mark>Matches query embedding to domain embeddings (e.g., Health vs Finance).</mark>
        
2. **<mark>Step 6 – Prompt Selection</mark>** <mark> 📄</mark>
    
    * <mark>Chooses the most </mark> **<mark>relevant prompt template</mark>** <mark> based on similarity score.</mark>
        
3. **<mark>Final Step – Query Execution</mark>** <mark> 🤖</mark>
    
    * <mark>Generates the </mark> **<mark>answer</mark>** <mark> using the selected domain template.</mark>
        

---

### **Why Does This Matter?**

Semantic Routing ensures your query lands in the right **expert zone**, saving time and improving relevance. 🚀 Whether it’s **books, workouts, travel tips, or budgeting**, this approach keeps things **smart and scalable**!