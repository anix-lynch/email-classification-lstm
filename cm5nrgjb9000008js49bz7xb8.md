---
title: "Parent Document Retrieval (PDR): Smarter Context Retrieval 📚🔍"
datePublished: Wed Jan 08 2025 10:32:44 GMT+0000 (Coordinated Universal Time)
cuid: cm5nrgjb9000008js49bz7xb8
slug: parent-document-retrieval-pdr-smarter-context-retrieval
tags: llm, langchain, rag, retriever, parentdocumentretriever

---

Ever asked an AI something complex, and it gave you an answer without enough context? 😅 That’s where **Parent Document Retrieval (PDR)** saves the day! It fetches the **full parent document** tied to relevant snippets, giving richer, more **accurate answers** for tough queries. 🚀

---

## **Why PDR? 🤔**

1. **Better Context 🧠** – Retrieves entire sections instead of tiny snippets.
    
2. **Handles Complex Queries 🔄** – Perfect for multi-step or deep-dive questions.
    
3. **Scales with Data 📊** – Handles big datasets (PDFs, research papers) like a pro.
    
4. **Combats Hallucination 🌈** – Provides more context, reducing AI guesswork.
    

---

## **How Does PDR Work? 🛠️**

1. **Split Data 📂** – Break big docs into smaller **chunks** (children).
    
2. **Embed Chunks 🌐** – Turn each chunk into numbers for easy search.
    
3. **Retrieve Parent Docs 📋** – Find chunks and pull their **parent docs** for context.
    
4. **Answer with Depth 📖** – AI responds with richer insights.
    

---

## **Let’s Build It! 🛠️**

---

### **1\. Import Tools 🧰**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.document_compressors import ParentDocumentRetriever
from langchain_openai import ChatOpenAI
import os
```

---

### **2\. Set API Key 🔑**

```python
os.environ["OPENAI_API_KEY"] = "sk-your-api-key"

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("API key missing! Add yours to continue.")
```

---

### **3\. Load Data 📚**

```python
from langchain.schema import Document

docs = [
    Document(
        page_content="LangSmith is a platform for managing AI applications with tools for debugging, logging, and deployment.",
        metadata={"source": "langsmith_overview.txt"}
    ),
    Document(
        page_content="LangChain simplifies AI workflows through modular pipelines and integrations with tools like OpenAI and Chroma.",
        metadata={"source": "langchain_intro.txt"}
    ),
]
```

---

### **4\. Split Text into Chunks 🧩**

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)
```

💬 **Output Example:**  
**Chunk 1:**

```python
LangSmith is a platform for managing AI applications...
```

**Chunk 2:**

```python
LangChain simplifies AI workflows through modular pipelines...
```

---

### **5\. Create Embeddings 🌐**

```python
embeddings = OpenAIEmbeddings()
```

---

### **6\. Store Chunks in Vector DB 📦**

```python
# Vector Store for Children
vectorstore = Chroma.from_documents(chunks, embeddings)

# Store Full Parent Docs
from langchain.storage import InMemoryStore
doc_store = InMemoryStore()
```

---

### **7\. Set Up Parent Document Retriever 📋**

```python
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=doc_store,
    child_splitter=splitter
)

# Add Full Docs
retriever.add_documents(docs)
```

---

### **8\. Search & Retrieve 🕵️‍♂️**

**🔍 Find Child Chunks First:**

```python
results = vectorstore.similarity_search("What is LangSmith?", k=2)
for res in results:
    print(res.page_content)
```

💬 **Output:**

```python
LangSmith is a platform for managing AI applications...
```

**📖 Retrieve Parent Docs:**

```python
parent_docs = retriever.invoke("What is LangSmith?")
for doc in parent_docs:
    print(doc.page_content)
```

💬 **Output:**

```python
LangSmith is a platform for managing AI applications with tools for debugging, logging, and deployment.
```

---

### **9\. Combine with AI QA Model 🤖**

```python
llm = ChatOpenAI(model="gpt-4o", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, chain_type="stuff"
)

response = qa_chain.invoke("What is LangSmith?")
print(response)
```

💬 **Output:**

```python
LangSmith is a platform for managing AI applications. It offers tools for debugging, logging, and deployment, helping streamline AI workflows.
```

---

## **Why Does It Work? 🌟**

1. **Chunk + Context Combo 🔄** – Finds snippets first, then retrieves **full docs** for clarity.
    
2. **No More Missing Details 🧩** – Large chunks ensure **context isn’t lost.**
    
3. **Flexible Scaling 📊** – Handles PDFs, articles, research papers—whatever you throw at it!
    

---

## **Next Steps 🚦**

1. **Test with PDFs 📄** – Swap `TextLoader` for `PyPDFLoader` for PDFs.
    
2. **Add Metadata Filters 🏷️** – Filter by author, topic, or date.
    
3. **Optimize Retrieval 🚀** – Combine with **reranking** or **query expansion** for sharper searches.
    

---

## **Why PDR Feels Like Magic? ✨**

It’s like asking:

> *“What’s the best part of this book?”*

And instead of pulling out one line, it gives you the **whole chapter**—so you **get the context too.** 📖💡

Let me know if you’d like more tweaks or examples! 😊