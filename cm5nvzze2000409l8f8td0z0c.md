---
title: "Quick Intro to RAG-Fusion 🚀"
seoTitle: "Quick Intro to RAG-Fusion 🚀"
seoDescription: "Quick Intro to RAG-Fusion 🚀"
datePublished: Wed Jan 08 2025 12:39:50 GMT+0000 (Coordinated Universal Time)
cuid: cm5nvzze2000409l8f8td0z0c
slug: quick-intro-to-rag-fusion
tags: ai, llm, langchain, rag

---

**RAG-Fusion** is like having a **team of detectives** 🕵️‍♂️ investigating every angle of a case, instead of relying on just one perspective. It boosts **retrieval quality** <mark> by generating </mark> **<mark>multiple variations</mark>** <mark> o</mark>f a query and then **reranking the results** for maximum relevance—ensuring your AI always focuses on the **most important evidence**!

---

### **Why Does RAG-Fusion Matter? 🎯**

1. **Covers More Ground 🌍** – Generates multiple query variations to catch different angles of user intent.
    
2. **Improves Accuracy 🔍** – Uses **Reciprocal Rank Fusion (RRF)** to combine scores and prioritize the best results.
    
3. **Reduces Bias 🛡️** – Avoids over-relying on exact keyword matches by reranking based on **semantic meaning**.
    
4. **Enhances Context 📚** – Pulls richer data, ensuring **contextually aware responses** from the LLM.
    

---

### **<mark>Where Does RAG-Fusion Kick In? 🔄</mark>**

**<mark>Key Steps</mark>**<mark>:</mark>

1. **<mark>Multi-Query Generation (Step 5)</mark>** <mark> – Generates different </mark> **<mark>versions of the query</mark>** <mark> to widen the search.</mark>
    
2. **<mark>Reciprocal Rank Fusion (Step 6)</mark>** <mark> – </mark> **<mark>Reranks results</mark>** <mark> based on scores from multiple queries.</mark>
    
3. **<mark>Final Answer Generation (Step 7)</mark>** <mark> – Combines </mark> **<mark>optimized results</mark>** <mark> into a </mark> **<mark>precise response</mark>**<mark>.</mark>
    

---

### **Full RAG-Fusion Code 💻**

```python
import os
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter

# 1. Set Environment Keys
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = ''  # Add your LangSmith API key
os.environ['LANGCHAIN_PROJECT'] = 'RAG-Fusion'
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] = ''  # Add your OpenAI API key
if OPENAI_API_KEY == "":
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# 2. Load and Split Documents
docs = ["example_doc1.txt", "example_doc2.txt"]  # Replace with your file paths
loaded_docs = []
for doc in docs:
    with open(doc, 'r') as file:
        loaded_docs.append(file.read())

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
chunks = text_splitter.split_documents(loaded_docs)

# 3. Index Documents
vectorstore = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# 4. Generate Multi-Queries (RAG-Fusion Step 5)
template = """You are an AI assistant tasked with generating search queries for a vector search engine. 
Generate 5 variations of the following question to capture different aspects of the query:
Original question: {question}"""
prompt_template = ChatPromptTemplate.from_template(template)

multi_query_chain = (
    prompt_template
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

# 5. RRF: Reciprocal Rank Fusion (Step 6)
def reciprocal_rank_fusion(results, k=60):
    """Re-rank results based on Reciprocal Rank Fusion."""
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = str(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc[0] for doc in reranked_results]

# Process Queries
query = "What is LangSmith, and why do we need it?"
query_variations = multi_query_chain.invoke({"question": query})

retrieved_docs = []
for variation in query_variations:
    retrieved_docs.append(retriever.invoke(variation))  # Retrieve docs for each variation

reranked_docs = reciprocal_rank_fusion(retrieved_docs)

# 6. Final RAG Model (Step 7)
response_template = """Answer the following question based on this context:
{context}
Question: {question}"""

prompt = ChatPromptTemplate.from_template(response_template)

llm = ChatOpenAI(temperature=0)

final_rag_chain = (
    {"context": RunnablePassthrough(), "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

result = final_rag_chain.invoke({"context": reranked_docs, "question": query})
print(result)
```

---

### **Key RAG-Fusion Steps Highlighted 🔥**

1. **<mark>Multi-Query Generation</mark>** <mark> (Step 5)</mark>
    
    ```python
    multi_query_chain = (
        prompt_template
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )
    ```
    
    * **Purpose**: Generates **5 rephrased queries** to capture **different intents**.
        
2. **<mark>Reciprocal Rank Fusion (RRF)</mark>** <mark> (Step 6)</mark>
    
    ```python
    reranked_docs = reciprocal_rank_fusion(retrieved_docs)
    ```
    
    * **Purpose**: Combines results across all queries and **reranks based on scores**.
        
3. **<mark>Final Answer Generation</mark>** <mark> (Step 7)</mark>
    
    ```python
    final_rag_chain = (
        {"context": RunnablePassthrough(), "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )
    ```
    
    * **Purpose**: Uses **reranked documents** to generate the **final answer**.
        

---

### **Key Takeaway 🍰**

RAG-Fusion is like hiring a **detective squad** 🕵️‍♂️ to analyze your question from **multiple angles**, filter out **irrelevant data**, and deliver the **best evidence** to the LLM for answering accurately. 💡