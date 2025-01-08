---
title: "What Is Query Decomposition? 🤔"
seoTitle: "What Is Query Decomposition? "
seoDescription: "What Is Query Decomposition? "
datePublished: Wed Jan 08 2025 11:29:19 GMT+0000 (Coordinated Universal Time)
cuid: cm5nthal9001008me4kahg0al
slug: what-is-query-decomposition
tags: ai, llm, langchain, rag

---

Think of it like planning a **group project**:

* <mark>Instead of solving the </mark> **<mark>entire problem</mark>** <mark> at once, you </mark> **<mark>split it into smaller tasks</mark>**<mark>.</mark>
    
* <mark>Everyone works on their own part, then </mark> **<mark>combine answers</mark>** <mark> for the final project.</mark>
    

In RAG, we **decompose** complex questions into **sub-questions**, **answer each part**, and **merge** them for a **complete answer**. 🎯

## **Key Code Sections – Where Decomposition Happens** 🚀

### **1\. Generate Sub-Questions (Decompose the Query)**

📌 **Code Block Where It Kicks In:**

```python
template = """Break the following question into 3 smaller questions:
Question: {question}"""
prompt_decomposition = ChatPromptTemplate.from_template(template)

generate_queries_decomposition = (
    prompt_decomposition
    | ChatOpenAI(temperature=0)  
    | StrOutputParser() 
    | (lambda x: x.split("\n")) # Splits output into sub-questions
)

question = "What is LangSmith, and why do we need it?"
sub_questions = generate_queries_decomposition.invoke({"question": question})
```

💡 **What Happens Here?**

1. The **template** tells the LLM to **split the main question** into 3 smaller questions.
    
2. `ChatOpenAI` generates the sub-questions.
    
3. `lambda x: x.split("\n")` splits the LLM’s output into **multiple sub-questions**.
    

🎯 **Result:** We now have a list like this:

```python
1. What is LangSmith?  
2. What are the key features of LangSmith?  
3. How does LangSmith improve workflows?
```

---

### **2\. Generate Answers for Sub-Questions**

📌 **Code Block Where It Kicks In:**

```python
rag_results = []

for sub_question in sub_questions:
    docs = retriever.invoke(sub_question)  # Retrieve docs for each sub-question
    
    chain = (
        {"context": docs, "question": sub_question}
        | prompt_rag
        | llm
        | StrOutputParser()
    )
    rag_results.append(chain.invoke({"context": docs, "question": sub_question}))
```

💡 **What Happens Here?**

1. Each **sub-question** is treated like a **separate query**.
    
2. **Relevant documents** are retrieved for each sub-question.
    
3. The LLM **answers each sub-question** one by one and stores it in `rag_results`.
    

🎯 **Result:**  
Answers for each sub-question are stored, ready to be merged. Example:

```python
1. LangSmith is a tool for managing AI workflows.  
2. Key features include monitoring, tracing, and debugging AI pipelines.  
3. It improves workflows by simplifying debugging and improving data management.
```

---

### **3\. Merge Answers into a Structured Format**

📌 **Code Block Where It Kicks In:**

```python
def format(questions, answers):
    formatted_string = ""
    for i, (q, a) in enumerate(zip(questions, answers)):
        formatted_string += f"Q{i+1}: {q}\nA{i+1}: {a}\n\n"
    return formatted_string.strip()

context = format(sub_questions, rag_results)
```

💡 **What Happens Here?**

1. Combines each **sub-question** and its **answer** into a **structured format**.
    
2. Makes the information **clear and organized**.
    

🎯 **Result:**

```python
Q1: What is LangSmith?  
A1: LangSmith is a tool for managing AI workflows.  

Q2: What are the key features of LangSmith?  
A2: Key features include monitoring, tracing, and debugging AI pipelines.  

Q3: How does LangSmith improve workflows?  
A3: It simplifies debugging and improves data management.
```

---

### **4\. Generate Final Answer**

📌 **Code Block Where It Kicks In:**

```python
template = """Based on the context below, provide a comprehensive answer:
{context}
Main Question: {question}
"""

prompt_final = ChatPromptTemplate.from_template(template)

final_chain = (
    {"context": context, "question": question} 
    | prompt_final
    | llm
    | StrOutputParser()
)

final_answer = final_chain.invoke({"context": context, "question": question})
```

💡 **What Happens Here?**

1. Takes the **formatted sub-question/answer pairs** as **context**.
    
2. Uses the LLM to **summarize and synthesize** everything into **one final answer**.
    

🎯 **Result:**

```python
LangSmith is a tool designed for managing AI workflows. It offers features such as monitoring, tracing, and debugging AI pipelines. These features simplify debugging, improve data management, and enhance overall AI performance.
```

---

## **Key Takeaways – Why Use Decomposition?** 🧠

1. **Precision** – Breaks down complex queries for **detailed answers**.
    
2. **Flexibility** – Each part is **handled independently** and can be **verified**.
    
3. **Scalability** – Works great for **large datasets** or **complex domains** like **legal**, **medical**, or **technical** texts.
    

---

Full Code. ✅

```python
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough

# --- SETUP ENVIRONMENT VARIABLES ---
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = ''  # Add your LangSmith API key
os.environ['LANGCHAIN_PROJECT'] = 'Decomposition'

# --- SETUP OPENAI API KEY ---
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] = ""  # Add your OpenAI API key
if OPENAI_API_KEY == "":
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# --- STEP 1: LOAD DOCUMENT ---
loader = WebBaseLoader(
    web_paths=("https://example.com/sample-document",),  # Replace with valid URL
)
blog_docs = loader.load()

# --- STEP 2: SPLIT TEXT INTO CHUNKS ---
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100,    # Chunk size of 100 tokens
    chunk_overlap=20   # 20-token overlap to preserve context
)
splits = text_splitter.split_documents(blog_docs)

# --- STEP 3: CREATE VECTOR STORE ---
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings()  # Use OpenAI embeddings for vector representation
)

retriever = vectorstore.as_retriever()  # Allows document retrieval using embeddings

# --- STEP 4: CREATE SUB-QUESTIONS (DECOMPOSITION) ---
template = """You are an AI assistant tasked with breaking down the input question 
into 3 smaller, answerable sub-questions that can be solved independently. 
Original question: {question}"""

prompt_decomposition = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0)  # Use deterministic LLM with no randomness

# Create pipeline for query decomposition
generate_queries_decomposition = (
    prompt_decomposition 
    | llm 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))  # Split output into separate sub-questions
)

# --- INPUT QUESTION ---
question = "What is LangSmith, and why do we need it?"

# Generate sub-questions
sub_questions = generate_queries_decomposition.invoke({"question": question})
print("\nSub-Questions Generated:", sub_questions)  # Debugging output

# --- STEP 5: ANSWER SUB-QUESTIONS ---
prompt_rag = hub.pull("rlm/rag-prompt")  # Use prebuilt RAG prompt from LangChain hub

rag_results = []  # Store answers to each sub-question

for sub_question in sub_questions:
    # Retrieve documents relevant to each sub-question
    retrieved_docs = retriever.invoke(sub_question)

    # Generate answers using the retrieved docs
    answer = (
        prompt_rag 
        | llm 
        | StrOutputParser()
    ).invoke({"context": retrieved_docs, "question": sub_question})
    
    rag_results.append(answer)  # Append answer to results

print("\nAnswers to Sub-Questions:", rag_results)  # Debugging output

# --- STEP 6: FORMAT SUB-QUESTIONS AND ANSWERS ---
def format(questions, answers):
    """Format sub-questions and answers as structured text."""
    formatted_string = ""
    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
    return formatted_string.strip()

# Create context from formatted Q&A pairs
context = format(sub_questions, rag_results)
print("\nFormatted Context:", context)  # Debugging output

# --- STEP 7: SYNTHESIZE FINAL ANSWER ---
template = """Here is a set of Q and A:
{context}
Use these to synthesize an answer to the question: {question}
"""

prompt_final = ChatPromptTemplate.from_template(template)

# Final RAG pipeline to synthesize the complete answer
final_rag_chain = (
    prompt_final
    | llm
    | StrOutputParser()
)

# Generate final answer
final_answer = final_rag_chain.invoke({"context": context, "question": question})

print("\nFinal Answer:", final_answer)  # Debugging output
```

---

## **Key Fixes** 🔧

1. **Document Loader URL Fixed** – Placeholder URL was corrected to `"`[`https://example.com/sample-document`](https://example.com/sample-document)`"`. Replace it with a real one.
    
2. **Chunk Splitting Configured** – Used 100 tokens per chunk with 20-token overlap for better text processing.
    
3. **Decomposition Template Updated** – Template prompts were rewritten for clarity and effectiveness in generating **sub-questions**.
    
4. **Debugging Print Statements** – Added outputs at key steps to **track progress** and validate the workflow.
    
5. **Formatted Context** – Ensures answers and sub-questions are structured cleanly for **final synthesis**.
    
6. **Environment Variable Checks** – Explicitly validates API keys to **avoid runtime errors**.
    

---

## **Where Decomposition Happens** ✂️

**Decomposition Kick-in Points:**

1. **Breaks Question into Sub-Questions:**
    

```python
generate_queries_decomposition.invoke({"question": question})
```

2. **Answers Each Sub-Question:**
    

```python
retrieved_docs = retriever.invoke(sub_question)
answer = (prompt_rag | llm | StrOutputParser()).invoke(...)
```

3. **Combines Answers into Context:**
    

```python
context = format(sub_questions, rag_results)
```

4. **Synthesizes Final Answer:**
    

```python
final_rag_chain.invoke({"context": context, "question": question})
```

---

## **Example Output:** 🎉

**Sub-Questions:**

```python
1. What is LangSmith?  
2. What are the main features of LangSmith?  
3. How does LangSmith simplify AI workflows?
```

**Answers:**

```python
1. LangSmith is a platform for monitoring and debugging AI pipelines.  
2. Key features include data visualization, error tracing, and scalability.  
3. It simplifies workflows by automating data preprocessing and model testing.
```

**Final Answer:**

```python
LangSmith is a robust platform designed to monitor, debug, and enhance AI workflows. It provides features like data visualization, error tracing, and scalability, simplifying the development and deployment of AI models.
```

---

## **Why Does It Work?** 🚀

* **Decomposes Complexity** – Breaks hard questions into smaller tasks.
    
* **Focuses Retrieval** – Each sub-question retrieves only **relevant docs**.
    
* **Combines Insights** – Ensures the **final answer is structured and complete**.