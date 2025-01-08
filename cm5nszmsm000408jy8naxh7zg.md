---
title: "Multi-Query Techniques for Complex Information Retrieval 🚦"
seoTitle: "Multi-Query Techniques for Complex Information Retrieval 🚦"
seoDescription: "Multi-Query Techniques for Complex Information Retrieval 🚦"
datePublished: Wed Jan 08 2025 11:15:35 GMT+0000 (Coordinated Universal Time)
cuid: cm5nszmsm000408jy8naxh7zg
slug: multi-query-techniques-for-complex-information-retrieval
tags: ai, llm, langchain, rag, multi-query

---

Multi-query techniques **supercharge RAG systems** by **expanding a user’s question** into **multiple perspectives**—ensuring **broader, deeper searches** for **better answers** even when documents lack exact keywords.

---

### **Why Multi-Query Rocks 🌟**

1. **Improves Coverage 📚**—Captures different angles of the query.
    
2. **Finds Hidden Context 🔍**—Picks up documents with synonyms or related ideas.
    
3. **Boosts Recall 🔄**—Grabs more relevant data while avoiding missed details.
    
4. **Saves Time & Costs 💸**—Fewer retries, better hits on the first try.
    

---

### **How Multi-Query Works 🔧**

**Step 1: Generate Diverse Queries**

* Starts with **1 user question** → Expands into **3–5 rephrased questions**.
    

**Example:**  
💬 **Original Query:** "What is LangSmith and why do we need it?"  
✨ **Generated Queries:**

1. "What are the features of LangSmith?"
    
2. "How does LangSmith improve language models?"
    
3. "Why is LangSmith important for AI development?"
    

**Step 2: Parallel Search 🧭**

* Runs **all queries at once** → Combines **results**.
    

**Step 3: Deduplicate Results 📌**

* Removes **duplicates** → Keeps **unique hits** only.
    

**Step 4: Context-Driven Answers ✍️**

* Passes combined results to **LLM** → Generates **focused responses.**
    

### **Step 1: Load and Split Documents (PREP WORK)**

Purpose:

* Load documents. 📚
    
* Split them into **smaller chunks** for better **indexing** and **search** later.
    

🛠️ **Code Block**

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load docs
docs = []
loaders = [TextLoader("/path/to/doc1.txt"), TextLoader("/path/to/doc2.txt")]

for loader in loaders:
    docs.extend(loader.load())

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
chunks = splitter.split_documents(docs)
```

💡 **Multi-Query NOT here yet**—This is just **prep work** to set up the data!

---

### **Step 2: Index Documents (VECTOR STORE)**

Purpose:

* Store chunks into a **searchable vector database** (Chroma) for **fast retrieval.**
    

🛠️ **Code Block**

```python
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Create vector store
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
```

💡 **Multi-Query NOT here yet**—We’re still setting up the **storage** for documents.

---

### **Step 3: Generate Multi-Query Variations (KICK-IN 🔥)**

Purpose:

* THIS is where the **Multi-Query technique begins.**
    
* LLM **rephrases** the user’s question into **multiple variations** to **capture different angles.**
    

🛠️ **Code Block**

```python
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# Template for generating query variations
template = """
Generate 3 variations of the question to capture different perspectives:
Question: {question}
Variations:
1.
2.
3.
"""

prompt_perspectives = ChatPromptTemplate.from_template(template)

# Create variations using LLM
generate_queries = (
    prompt_perspectives
    | ChatOpenAI(temperature=0)  # Set temperature to 0 for deterministic outputs
    | (lambda x: x.split("\n"))  # Split variations into a list
)

# Example user query
question = "What is LangSmith, and why do we need it?"
query_variations = generate_queries.invoke({"question": question})

print(query_variations)  # Output: 3 rephrased queries
```

🎯 **Multi-Query STARTS HERE!**

* Generates **multiple versions** of the question (e.g., synonyms, rephrased formats).
    
* Prepares for **broader, deeper searches.**
    

Example Output:

```python
1. What features does LangSmith provide?  
2. How does LangSmith improve AI workflows?  
3. Why is LangSmith useful for AI applications?  
```

---

### **Step 4: Retrieve Documents with Multi-Query (KICK-IN CONTINUES 🚀)**

Purpose:

* Use **each variation** of the query to **retrieve results.**
    
* **Combine results** while **removing duplicates.**
    

🛠️ **Code Block**

```python
def get_unique_union(results):
    # Flatten nested lists of documents
    flat_results = [doc for result in results for doc in result]

    # Deduplicate results based on content
    seen = set()
    unique_docs = []
    for doc in flat_results:
        key = doc.page_content
        if key not in seen:
            unique_docs.append(doc)
            seen.add(key)
    return unique_docs

# Perform searches for each query variation
results = [retriever.get_relevant_documents(q) for q in query_variations]

# Merge and remove duplicates
unique_docs = get_unique_union(results)

print(f"Retrieved {len(unique_docs)} unique documents.")
```

🎯 **Multi-Query IN ACTION here!**

* **Parallel searches**: Uses **each query variation** to **fetch documents**.
    
* **Merge & Deduplicate**: Combines results into **one set** without duplicates.
    

---

### **Step 5: Generate Answers (FINAL STEP—NO MORE QUERIES)**

Purpose:

* Use the **retrieved documents** as **context** to answer the user’s question.
    

🛠️ **Code Block**

```python
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

# Template for response generation
template = """
Use the following context to answer the question accurately:
Context: {context}
Question: {question}
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# Create final RAG pipeline
final_rag_chain = (
    {
        "context": unique_docs,
        "question": question,
    }
    | prompt
    | ChatOpenAI(temperature=0)
)

# Generate answer
answer = final_rag_chain.invoke({"question": question})
print(answer)
```

💡 **Multi-Query ENDS HERE!**

* Uses **already retrieved documents** to **generate the final answer.**
    
* Doesn’t expand queries further—just builds the response.
    

---

### **Multi-Query in Action Summary 🚦**

1. **Kick-In Point #1 (Step 3):**
    
    * Generates **query variations** using LLM.
        
    * Captures **different perspectives.**
        
2. **Kick-In Point #2 (Step 4):**
    
    * Uses **all query variations** to **fetch documents**.
        
    * **Merges results** while removing duplicates.
        
3. **Stop Point (Step 5):**
    
    * Uses **retrieved documents** to **generate the answer.**
        

---

### **Final Takeaway 🌟**

* **Multi-Query** **expands searches** to **increase relevance** and **reduce misses.**
    
* It **kicks in at Steps 3 & 4**, but **stops at Step 5** when the **final response is built.**
    
* Think of it as **casting a wider net** (queries) before **filtering** and **serving the catch** (answers). 🎣