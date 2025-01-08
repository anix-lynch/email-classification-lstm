---
title: "Hypothetical Document Embeddings (HyDE): Your Smart Study Buddy for RAG! 📚🤖"
seoTitle: "Hypothetical Document Embeddings (HyDE): Your Smart Study Buddy for RA"
seoDescription: "Hypothetical Document Embeddings (HyDE): Your Smart Study Buddy for RAG! 📚🤖"
datePublished: Wed Jan 08 2025 11:50:50 GMT+0000 (Coordinated Universal Time)
cuid: cm5nu8yrd000e09l47voo1d0t
slug: hypothetical-document-embeddings-hyde-your-smart-study-buddy-for-rag
tags: ai, llm, langchain, rag

---

Ever felt like your search results miss the mark when asking complex questions? That’s where **HyDE** jumps in! 🚀

<mark>Think of HyDE as your </mark> **<mark>study buddy</mark>** <mark> who doesn’t just search for existing notes but also </mark> **<mark>imagines</mark>** <mark> what a </mark> **<mark>perfect answer might look like</mark>**<mark>—even if it doesn’t exist yet! It then uses this </mark> **<mark>hypothetical document</mark>** <mark> to find </mark> **<mark>real data</mark>** <mark> that matches its vibe.</mark>

This boosts **accuracy** and **creativity** when retrieving info, especially for tough or unique questions. 🎯

---

## **Where Does HyDE Kick In? 🤔**

HyDE **kicks in** when it **creates a hypothetical document** based on your query! Instead of looking for exact matches, it imagines the kind of document that might contain the answer—and then searches for **real-world data** similar to it. 💡

In the code below, this happens in **Step 7** where it **generates the hypothetical document** and **Step 6** where it queries the vector store using embeddings. 🌟

---

## **Full Code for HyDE Implementation 🧑‍💻**

### **Step 1: Import Libraries** 📦

```python
import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.schema.runnable import RunnablePassthrough
```

---

### **Step 2: Set Up API Keys** 🔑

```python
# Set OpenAI and LangChain API Keys
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = ''  # Add LangChain API Key
os.environ['LANGCHAIN_PROJECT'] = 'HyDE'

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] = ""  # Add OpenAI API Key
if OPENAI_API_KEY == "":
    raise ValueError("Please set the OPENAI_API_KEY environment variable")
```

---

### **Step 3: Load and Split Documents** 📄

```python
# Load example documents
loaders = [
    TextLoader("example_doc1.txt"),
    TextLoader("example_doc2.txt")
]

# Load content
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split into smaller chunks for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
splits = text_splitter.split_documents(docs)
```

---

### **Step 4: Create Vector Store for Search** 🔍

```python
# Create vector store with document embeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
```

---

### **Step 5: Generate Embeddings for Search Queries** 🧠

```python
# Define a query to test HyDE
query = "What is LangSmith, and why do we need it?"

# Create embeddings for the query
embedding_model = OpenAIEmbeddings()
query_embedding = embedding_model.embed_query(query)

# Search vector store for documents similar to query embedding
retrieved_docs = vectorstore.similarity_search(query, k=3)

print("\nRetrieved Documents (Based on Query):")
for doc in retrieved_docs:
    print(doc.page_content)
```

---

### **<mark>Step 6: HyDE Creates Hypothetical Document</mark>** <mark> ✨</mark>

**<mark>(Where HyDE Kicks In 🚀)</mark>**

```python
# Generate a hypothetical document (HyDE Magic! ✨)
template = """You are a research assistant tasked with generating hypothetical documents.  
Write a brief summary or webpage content to address the query:  
"{query}"  
"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(temperature=0)

hypothetical_doc_chain = (
    prompt
    | llm
    | StrOutputParser()
)

# Generate hypothetical document based on query
hypothetical_doc = hypothetical_doc_chain.invoke({"query": query})
print("\nHypothetical Document:")
print(hypothetical_doc)
```

---

### **Step 7: Query Vector Store Using HyDE Document** 🎯

```python
# Embed the hypothetical document
hypothetical_doc_embedding = embedding_model.embed_query(hypothetical_doc)

# Search for real documents similar to the hypothetical doc
hyde_retrieved_docs = vectorstore.similarity_search(hypothetical_doc, k=3)

print("\nRetrieved Documents (Based on HyDE Document):")
for doc in hyde_retrieved_docs:
    print(doc.page_content)
```

---

### **Step 8: Return the Final Answer** 📝

```python
# Combine retrieved results and generate the final answer
final_template = """Answer the query using the following context:
{context}

Query: {query}
"""

final_prompt = ChatPromptTemplate.from_template(final_template)
final_chain = (
    final_prompt
    | llm
    | StrOutputParser()
)

context = "\n".join([doc.page_content for doc in hyde_retrieved_docs])
final_answer = final_chain.invoke({"context": context, "query": query})

print("\nFinal Answer:")
print(final_answer)
```

---

## **Example Output** 📝

**Input Query:**

> *"What is LangSmith, and why do we need it?"*

**<mark>Hypothetical Document (HyDE Generated):</mark>**

> <mark>"LangSmith is a tool designed for managing language model applications by offering features like debugging, version control, and monitoring. It helps developers streamline LLM workflows, ensuring reliability and scalability."</mark>

**Retrieved Documents (Similar to HyDE):**

> 1. "LangSmith simplifies AI workflows by providing logging and debugging capabilities for prompt engineering."
>     
> 2. "Developers can use LangSmith for performance tracking and rapid prototyping of LLM-powered tools."
>     
> 3. "LangSmith integrates with LangChain for better testing and monitoring of LLM applications."
>     

**Final Answer (Generated by RAG):**

> "LangSmith is a tool for managing LLM applications, offering debugging, performance tracking, and integration capabilities. It simplifies workflows, making it ideal for AI developers who need scalable solutions for prompt engineering and monitoring."

---

## **Quick Recap – Where Does HyDE Kick In? 🚀**

1. **Step 6 – Generate Hypothetical Document:**
    
    * HyDE imagines a document **based on the query** to simulate what the ideal answer should look like.
        
2. **Step 7 – Retrieve Similar Documents:**
    
    * <mark>Uses the </mark> **<mark>hypothetical document embedding</mark>** <mark> to search for </mark> **<mark>real data</mark>** <mark> matching its content.</mark>
        
3. **Step 8 – Final Answer Generation:**
    
    * Combines retrieved documents and HyDE’s output to **answer the query**.
        

---