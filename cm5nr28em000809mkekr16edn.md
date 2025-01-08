---
title: "Self-Querying Retrieval (SQR): Smarter Search with LLMs 🧠🔍"
datePublished: Wed Jan 08 2025 10:21:37 GMT+0000 (Coordinated Universal Time)
cuid: cm5nr28em000809mkekr16edn
slug: self-querying-retrieval-sqr-smarter-search-with-llms
tags: llm, langchain, rag, sqr, naturallangauge

---

Imagine asking a librarian:

> *"Can you find highly-rated historical fiction books published after 2010 with deep themes?"*

And voilà—they pull out **exactly what you need** without any keyword tricks or fancy search syntax! That’s **Self-Querying Retrieval (SQR)**—a system where your queries sound **natural** but are backed by **powerful AI magic.✨**

---

## **Why SQR Rocks 🚀**

1. **Talk Like a Human 🗣️** – No weird query formats. Just type naturally!
    
2. **Rich Filters 🎯** – Search by **content** and **metadata** (author, year, genre).
    
3. **Smart Follow-Ups 🔄** – Refine results with follow-up questions.
    
4. **Flexible for Any Dataset 📚** – Handles books, legal files, or even customer data.
    

---

## **How Does It Work? ⚙️**

1. **Documents** ➡️ Turned into smart embeddings (numeric fingerprints).
    
2. **Query Understanding 🤖** ➡️ The LLM figures out *what you mean*.
    
3. **Matching 🔍** ➡️ Finds relevant chunks based on content **and metadata**.
    
4. **Refinement 🛠️** ➡️ Lets you filter results step-by-step (e.g., rating &gt; 4.5).
    

---

## **Let’s Build It! 🛠️**

---

### **1\. Import Tools 🧰**

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
import os
```

---

### **2\. Set Up API Key 🔑**

```python
os.environ["OPENAI_API_KEY"] = "sk-your-api-key"

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Oops! Missing API key. Add yours to continue.")
```

---

### **3\. Create Example Data 📚**

```python
from langchain.schema import Document

docs = [
    Document(
        page_content="A gripping historical novel set in ancient Rome.",
        metadata={
            "title": "The Last Gladiator",
            "author": "Marcus Green",
            "year": 2020,
            "genre": "Historical Fiction",
            "rating": 4.8,
            "language": "English",
            "country": "USA"
        },
    ),
    Document(
        page_content="A dystopian thriller about survival in a broken society.",
        metadata={
            "title": "Shattered World",
            "author": "Lucy Stone",
            "year": 2015,
            "genre": "Dystopian",
            "rating": 4.6,
            "language": "English",
            "country": "UK"
        },
    ),
]
```

---

### **4\. Define Embeddings 🌐**

```python
embeddings = OpenAIEmbeddings()
```

---

### **5\. Build the Vector Store 📦**

```python
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(docs, embeddings)
```

---

### **6\. Create the LLM and Retriever 🤖**

```python
# Metadata descriptions
metadata_field_info = [
    AttributeInfo(name="title", description="The title of the book.", type="string"),
    AttributeInfo(name="author", description="The author of the book.", type="string"),
    AttributeInfo(name="year", description="Year the book was published.", type="integer"),
    AttributeInfo(name="genre", description="Genre of the book.", type="string"),
    AttributeInfo(name="rating", description="Rating of the book.", type="float"),
    AttributeInfo(name="country", description="Country of origin.", type="string"),
]

document_content_description = "Summary of the book."

# Create LLM and retriever
llm = ChatOpenAI(model="gpt-4o", temperature=0)

retriever = SelfQueryRetriever.from_llm(
    llm, 
    vectorstore, 
    document_content_description, 
    metadata_field_info, 
    verbose=True
)
```

---

### **7\. Example Queries 🔎**

**1\. Search by Genre:**

```python
print(retriever.get_relevant_documents("Show me historical fiction books."))
```

💬 **Output:**  
*A gripping historical novel set in ancient Rome.*

---

**2\. Filter by Rating:**

```python
print(retriever.get_relevant_documents("Find books rated above 4.7."))
```

💬 **Output:**  
*The Last Gladiator (4.8 rating).*

---

**3\. Combine Filters (Genre + Rating):**

```python
print(retriever.get_relevant_documents("Dystopian books rated above 4.5."))
```

💬 **Output:**  
*Shattered World (4.6 rating).*

---

**4\. Search by Year and Theme:**

```python
print(retriever.get_relevant_documents("Books published after 2018 with deep themes."))
```

💬 **Output:**  
*The Last Gladiator (2020).*

---

**5\. Limit Results:**

```python
print(retriever.get_relevant_documents("Show 1 book with the highest rating."))
```

💬 **Output:**  
*The Last Gladiator (4.8 rating).*

---

### **Why Does It Work? 🌟**

* **Embeddings = Smart Memory 🧠** – Captures meaning, not just keywords.
    
* **Metadata = Filters 🎯** – Adds precision by combining text with structured data.
    
* **LLM Power 🚀** – Understands natural language queries without needing formulas.
    

---

### **What’s Next? 🚦**

1. **Add More Metadata 🏷️** – Support fields like price, themes, or even reviews.
    
2. **Try PDFs or HTML Files 📄** – Use real-world datasets instead of plain text.
    
3. **Multi-Step Queries 🔄** – Refine answers with follow-ups like:
    
    > "Now show only books by female authors."
    

---

### **Why It’s Like Magic? ✨**

**SQR** turns a basic search engine into a **genius librarian** who:

* Reads **every book.📚**
    
* Knows **your taste.🎭**
    
* Finds **exact matches.🔍**
    

Let me know if you’d like **more upgrades** for this! 😊