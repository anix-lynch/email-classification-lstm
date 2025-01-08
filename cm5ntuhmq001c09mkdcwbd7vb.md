---
title: "Step-Back Prompting with RAG 🧠"
seoTitle: "Step-Back Prompting with RAG 🧠"
seoDescription: "Step-Back Prompting with RAG 🧠"
datePublished: Wed Jan 08 2025 11:39:35 GMT+0000 (Coordinated Universal Time)
cuid: cm5ntuhmq001c09mkdcwbd7vb
slug: step-back-prompting-with-rag
tags: ai, llm, langchain, prompt-engineering, rag

---

### **Why Step-Back Prompting?** 🤔

Step-Back Prompting is like asking **"Wait, what’s the bigger picture?"** before answering a question. Instead of jumping straight to an answer, it **rewrites the question** into a **broader, more general version** to:

---

### **Key Benefits** 🌟

1. **Fixes Ambiguity** 🔄
    
    * Simplifies **tricky or vague questions** by focusing on the **core concept**.
        
    * Example:  
        **Original**: *"Did Leonardo da Vinci invent the printing press?"*  
        **Step-Back**: *"What are Leonardo da Vinci’s key contributions?"*
        
2. **Improves Context** 📚
    
    * Retrieves **more relevant information** by searching **general knowledge** first.
        
3. **Enhances Accuracy** 🎯
    
    * Avoids **overfitting** to a single query, resulting in **richer answers**.
        
4. **Handles Complex Queries** 🧩
    
    * Breaks down **multi-layered questions** and connects **missing details**.
        

---

### **When to Use It?** 🚀

* **Vague Questions** 😕 – When the query lacks clarity or direction.
    
* **Multi-Part Questions** 🧩 – Questions that need **background context** first.
    
* **Conceptual Queries** 🔍 – When the answer relies on **broader knowledge**.
    

---

**1\. Imports and Setup** 📦

```python
import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain_community.tools import DuckDuckGoSearchAPIWrapper
from langchain import hub

# Environment Setup 🌍
os.environ['OPENAI_API_KEY'] = ""  # Add your OpenAI API key
if os.environ["OPENAI_API_KEY"] == "":
    raise ValueError("Please set the OPENAI_API_KEY environment variable")
```

**What's happening here?**

* We’re bringing in all the **tools** we need.
    
* Connecting to **OpenAI API** so we can chat with GPT later! 🤖
    

---

### **2\. Few-Shot Learning for Step-Back Prompting** 📚

**💡 This is where the Step-Back Prompting concept kicks in!** It teaches the model to **generalize questions** by showing examples! ✨

```python
# Examples to Teach the Model 📖
examples = [
    {"input": "Did Leonardo da Vinci invent the printing press?",
     "output": "What are some significant inventions and contributions of Leonardo da Vinci?"},
    {"input": "Is climate change reversible?",
     "output": "What are the main causes and solutions to climate change?"}
]
```

**What’s happening here?**

* We're feeding **examples** 📝 to the model to **learn how to 'step back'** from specific to general questions!
    
* Think of it as teaching the model how to **rephrase tricky questions** into broader ones for better results! 🔄
    

---

### **3\. Building the Step-Back Prompt** 🎯

**💥 This is the heart of Step-Back Prompting!**  
We’re crafting the **prompt** that asks the model to rewrite the user’s question into a more **general form**. 🌐

```python
# Define the Prompt Template 🛠️
prompt = ChatPromptTemplate.from_messages([
    {"role": "system", "content": "You will rewrite specific questions into more general questions."},
    *[
        {"role": "human", "content": ex["input"]} for ex in examples
    ],
    *[
        {"role": "ai", "content": ex["output"]} for ex in examples
    ],
    {"role": "human", "content": "{question}"}
])

# Model Setup 🤖
llm = ChatOpenAI(temperature=0)
question_gen = prompt | llm | StrOutputParser()

# User's Original Question ❓
question = "Did Leonardo da Vinci invent the printing press?"

# 🛠️ **Step-Back Kicks In Here!** 🔥
step_back_question = question_gen.invoke({"question": question})
print("Step-Back Question:", step_back_question)
```

---

**What's happening here?**

1. 🛠️ We **instruct the model** to **rewrite** questions into broader ones based on examples.
    
2. 🔄 The **step-back question** generalizes the original question.
    
    * Example Input: *"Did Leonardo da Vinci invent the printing press?"*
        
    * Step-Back Output: *"What are some significant inventions and contributions of Leonardo da Vinci?"*
        

🎉 **Result:** The model **steps back** to ask a broader question that captures **context** and improves retrieval! 🌟

---

### **4\. Retrieve Information from the Web** 🌍

**Why this step?**  
Now that we have a **step-back question**, we use it to fetch **relevant documents** for a better answer. 📚

```python
# Web Search Setup 🔎
search = DuckDuckGoSearchAPIWrapper(max_results=4)

# Retrieve Context 📖
retrieved_docs_original = search.run(question)  # Original Question
retrieved_docs_step_back = search.run(step_back_question)  # Step-Back Question

print("Original Context:", retrieved_docs_original[:300])  # Show first 300 chars
print("Step-Back Context:", retrieved_docs_step_back[:300])
```

**What's happening here?**

1. 🔎 **Two searches**:
    
    * **Original Question** for focused results.
        
    * **Step-Back Question** for **broader context**.
        
2. 🌟 **Compare results**: Often, the step-back question retrieves **more relevant details**!
    

---

### **5\. Combine Contexts and Answer** 🧩

We merge **all the information** and let GPT generate the **final answer**! 💡

```python
# Combine Context 🧩
context = f"Original Context:\n{retrieved_docs_original}\n\nStep-Back Context:\n{retrieved_docs_step_back}"

# Prompt Template 📝
answer_template = """You are an AI assistant tasked with answering a question using the provided context. 
Context: {context}
Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(answer_template)

# Final RAG Chain 🌐
final_rag_chain = prompt | llm | StrOutputParser()

# 🔥 Generate the Answer
final_answer = final_rag_chain.invoke({"context": context, "question": question})
print("\nFinal Answer:", final_answer)
```

---

**What’s happening here?**

1. 🧩 We **merge** the results from both the **original** and **step-back** questions.
    
2. 📝 Use **GPT** to **write the final answer**, combining **all insights**.
    
3. 🎉 The **step-back technique** ensures we **don’t miss important details** that were outside the scope of the initial question!
    

---

### **Key Takeaways** 🎯

1. **Step-Back Prompting** 🛠️
    
    * Teaches GPT to **rewrite narrow questions** into **broader ones** for **better results**!
        
    * Kicks in when we invoke:
        
        ```python
        step_back_question = question_gen.invoke({"question": question})
        ```
        
2. **Better Context = Better Answers** 📚
    
    * Combines **specific** and **broad** search results for a **complete view**.
        
3. **Flexible for Complex Queries** 🔄
    
    * Handles **vague or tricky questions** without missing the **core meaning**.
        

---

### **Output Example** 🖥️

```python
Step-Back Question: What are some significant inventions and contributions of Leonardo da Vinci?

Original Context: Leonardo da Vinci, born in 1452, was known for...
Step-Back Context: Leonardo da Vinci was a polymath with inventions ranging from flying machines...

Final Answer: Leonardo da Vinci did not invent the printing press, but he made numerous contributions, including...
```

---

✨ **Now you’re a pro at Step-Back Prompting!** 🚀

Full code  
Here’s the **full code** for **Step-Back Prompting** 🧠✨ in LangChain with OpenAI, along with comments to help you understand each step:

---

### **Step 1: Import Libraries** 📦

```python
import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import DuckDuckGoSearchAPIWrapper
from langchain.schema.runnable import RunnablePassthrough
from operator import itemgetter
```

---

### **Step 2: Set Up API Keys** 🔑

```python
# Set LangChain and OpenAI environment variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = ''  # Add LangChain API key
os.environ['LANGCHAIN_PROJECT'] = 'Step-Back'

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] = ""  # Add OpenAI API key
if OPENAI_API_KEY == "":
    raise ValueError("Please set the OPENAI_API_KEY environment variable")
```

---

### **Step 3: Few-Shot Examples for Step-Back Prompting** 🌟

```python
# Step-Back Examples
examples = [
    {"input": "Did Leonardo da Vinci invent the printing press?",
     "output": "What contributions did Leonardo da Vinci make to art and science?"},

    {"input": "What was Marie Curie’s biggest discovery?",
     "output": "What discoveries did Marie Curie make in the field of chemistry?"}
]

# Format prompt using examples
few_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rephrase the question into a broader query to provide more context."),
    *[(f"Human: {ex['input']}\nAI: {ex['output']}") for ex in examples],
    ("human", "{question}")
])
```

---

### **Step 4: Generate Step-Back Questions** 🔄

```python
# Model and pipeline to generate Step-Back questions
llm = ChatOpenAI(temperature=0)

step_back_gen = (
    few_shot_prompt  # Use few-shot prompt
    | llm            # Pass through the model
    | StrOutputParser()  # Parse output as text
)

# Example Question
question = "Did Leonardo da Vinci invent the printing press?"

# Generate Step-Back Question
step_back_question = step_back_gen.invoke({"question": question})
print("Step-Back Question:", step_back_question)
```

**🛑 Where does Step-Back Kick in?**  
👉 It **kicks in here** by **rephrasing** the input question into a **broader query** using the **few-shot examples**.

---

### **Step 5: Retrieve Context** 🔍

```python
# Web Search Setup
search = DuckDuckGoSearchAPIWrapper(max_results=4)

# Retrieve Contexts for Original and Step-Back Questions
retriever = lambda q: search.run(q)

original_context = retriever(question)  # Search based on original question
step_back_context = retriever(step_back_question)  # Search using Step-Back question

print("Original Context:", original_context)
print("Step-Back Context:", step_back_context)
```

---

### **Step 6: Build the RAG Chain** 🔗

```python
# RAG Template for Combining Information
template = """Use the following context to answer the question:

Original Context: {original_context}
Step-Back Context: {step_back_context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Final Pipeline (RAG Chain)
rag_chain = (
    {
        "original_context": RunnablePassthrough(),
        "step_back_context": RunnablePassthrough(),
        "question": itemgetter("question")
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Final Answer
answer = rag_chain.invoke({
    "original_context": original_context,
    "step_back_context": step_back_context,
    "question": question
})

print("\nFinal Answer:", answer)
```

---

### **Output Example** 📝

**Input Question:**

> "Did Leonardo da Vinci invent the printing press?"

**Step-Back Question:**

> "What contributions did Leonardo da Vinci make to art and science?"

**Final Answer:**

> "Leonardo da Vinci did not invent the printing press. The printing press was invented by Johannes Gutenberg around 1440. However, Leonardo made significant contributions to art, science, and engineering, including designs for machines and anatomical studies."

---

### **Quick Recap – Where Does Step-Back Kick In?** 🚦

1. **Step 4:** 🔄
    
    * **Rephrases the question** into a **broader query** using the **few-shot prompt**.
        
2. **Step 5:** 🔍
    
    * Searches both **original** and **step-back contexts** to **widen the knowledge scope**.
        
3. **Step 6:** 🔗
    
    * Combines **both contexts** in the RAG chain to **generate the final answer**.
        

---

Let me know if you’d like me to simplify or expand any part! 😊