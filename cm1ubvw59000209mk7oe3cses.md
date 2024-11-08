---
title: "20 Langchain concepts with Before-and-After Examples"
seoTitle: "20 Langchain concepts with Before-and-After Examples"
seoDescription: "20 Langchain concepts with Before-and-After Examples"
datePublished: Fri Oct 04 2024 06:12:21 GMT+0000 (Coordinated Universal Time)
cuid: cm1ubvw59000209mk7oe3cses
slug: 20-langchain-concepts-with-before-and-after-examples
tags: ai, python, data-science, llm, langchain

---

### 1\. **Installing LangChain via pip 📦**

**Boilerplate Code**:

```bash
pip install langchain
```

**Use Case**: Install LangChain to build language model-powered applications.

**Goal**: Set up LangChain for LLM (Large Language Model) powered workflows. 🎯

**Sample Code**:

```bash
pip install langchain
```

**Before Example**: You manually interact with language models using APIs without a framework.

```bash
# Using OpenAI API directly:
import openai
response = openai.Completion.create(model="text-davinci-003", prompt="Hello world!")
```

**After Example**: LangChain simplifies the interaction with LLMs and provides tools for chaining tasks.

```bash
Successfully installed langchain
# LangChain installed and ready for building LLM-powered applications.
```

---

### 2\. **Creating a Simple Language Model Chain 🧠**

**Boilerplate Code**:

```python
from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(input_variables=["name"], template="Hello, {name}!")
chain = LLMChain(llm=llm, prompt=prompt)
```

**Use Case**: Chain together prompts and LLMs for seamless execution.

**Goal**: Create a basic LLM chain for generating responses. 🎯

**Sample Code**:

```python
from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(input_variables=["name"], template="Hello, {name}!")
chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run("Alice")
print(response)
```

**Before Example**: You interact with LLMs through isolated prompts without chaining them efficiently.

```bash
# Single prompt execution:
response = openai.Completion.create(prompt="Hello, Alice!")
```

**After Example**: LangChain allows chaining prompts for more structured LLM interaction.

```bash
Hello, Alice!
# Output generated using the LangChain pipeline.
```

---

### 3\. **Customizing Prompt Templates 📝**

**Boilerplate Code**:

```python
from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["name", "task"],
    template="{name}, your task is {task}. Please complete it."
)
```

**Use Case**: Customize prompts for specific tasks using templates.

**Goal**: Create flexible prompts that adapt to different inputs. 🎯

**Sample Code**:

```python
template = PromptTemplate(
    input_variables=["name", "task"],
    template="{name}, your task is {task}. Please complete it."
)

prompt = template.format(name="Alice", task="analyzing data")
print(prompt)
```

**Before Example**: You manually construct and modify prompts for each new task or input.

```bash
# Manually creating the prompt:
prompt = "Alice, your task is analyzing data."
```

**After Example**: With LangChain, you dynamically generate prompts based on inputs.

```bash
Alice, your task is analyzing data. Please complete it.
# Prompt generated using a flexible template.
```

---

### 4\. **Using Memory in Chains 🧠⏳**

**Boilerplate Code**:

```python
from langchain.chains import ConversationChain
from langchain.memory import SimpleMemory

memory = SimpleMemory()
chain = ConversationChain(llm=llm, memory=memory)
```

**Use Case**: Allow the LLM to retain context between interactions.

**Goal**: Implement memory to hold conversations or inputs for continuous workflows. 🎯

**Sample Code**:

```python
from langchain.chains import ConversationChain
from langchain.memory import SimpleMemory

memory = SimpleMemory()
chain = ConversationChain(llm=llm, memory=memory)

chain.run("What's the capital of France?")
chain.run("What about Italy?")
```

**Before Example**: You manually re-enter previous context when interacting with LLMs.

```bash
# Each prompt needs full context:
"What is the capital of France?"
"What is the capital of Italy?"
```

**After Example**: With LangChain memory, the LLM remembers the conversation flow.

```bash
France is Paris.
The capital of Italy is Rome.
# LLM retains context between interactions.
```

---

### 5\. **Creating a Sequential Chain ⛓️**

**Boilerplate Code**:

```python
from langchain import SequentialChain

chain = SequentialChain(chains=[chain1, chain2], input_variables=["input"], output_variables=["output"])
```

**Use Case**: Combine multiple chains to execute them sequentially.

**Goal**: Chain together several tasks or models for sequential execution. 🎯

**Sample Code**:

```python
from langchain import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name="text-davinci-003")

# Chain 1
prompt1 = PromptTemplate(input_variables=["input"], template="Summarize the following text: {input}")
chain1 = LLMChain(llm=llm, prompt=prompt1)

# Chain 2
prompt2 = PromptTemplate(input_variables=["summary"], template="What are the key takeaways from this summary: {summary}")
chain2 = LLMChain(llm=llm, prompt=prompt2)

# Sequential Chain
seq_chain = SequentialChain(chains=[chain1, chain2], input_variables=["input"], output_variables=["output"])

output = seq_chain.run("LangChain is a framework for developing applications powered by language models.")
print(output)
```

**Before Example**: You manually execute tasks in sequence without a structured chain.

```bash
# Manually run tasks one after another:
summary = openai.Completion.create(prompt="Summarize the following text...")
takeaways = openai.Completion.create(prompt="What are the key takeaways...")
```

**After Example**: LangChain automates the sequential execution of tasks.

```bash
The key takeaways are...
# LangChain executes the summarization and extraction of key takeaways sequentially.
```

---

### 6\. **Adding Custom Tools to a LangChain Agent 🛠️**

**Boilerplate Code**:

```python
from langchain.tools import Tool

def custom_tool(input: str) -> str:
    return f"Processed {input}"

tool = Tool.from_function(func=custom_tool, name="Custom Tool", description="A tool for processing inputs.")
```

**Use Case**: Extend LangChain agents by adding custom tools to process specific inputs.

**Goal**: Create and integrate a custom tool within an agent for specialized tasks. 🎯

**Sample Code**:

```python
from langchain.tools import Tool

def custom_tool(input: str) -> str:
    return f"Processed {input}"

tool = Tool.from_function(func=custom_tool, name="Custom Tool", description="A tool for processing inputs.")
output = tool.run("Data")
print(output)
```

**Before Example**: You manually run functions outside of the LangChain agent, losing the advantage of agent orchestration.

```bash
# Running a function manually:
def custom_tool(input: str) -> str:
    return f"Processed {input}"
print(custom_tool("Data"))
```

**After Example**: LangChain agents can use the custom tool for more automated and structured interactions.

```bash
Processed Data
# Custom tool integrated with LangChain and used within an agent.
```

---

### 7\. **Using the LangChain Agent for Zero-Shot Question Answering 🤖**

**Boilerplate Code**:

```python
from langchain.agents import initialize_agent, Tool
from langchain import OpenAI

llm = OpenAI(model_name="text-davinci-003")
tools = [Tool.from_function(func=custom_tool, name="Custom Tool")]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
```

**Use Case**: Automatically generate answers without prior task-specific training using the LangChain agent.

**Goal**: Implement zero-shot reasoning for answering questions using the agent. 🎯

**Sample Code**:

```python
from langchain.agents import initialize_agent, Tool
from langchain import OpenAI

llm = OpenAI(model_name="text-davinci-003")

def custom_tool(input: str) -> str:
    return f"Processed {input}"

tools = [Tool.from_function(func=custom_tool, name="Custom Tool")]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

response = agent.run("What is LangChain?")
print(response)
```

**Before Example**: You manually create prompts and responses, requiring more time for task execution.

```bash
# Manually creating prompt:
response = openai.Completion.create(prompt="What is LangChain?")
```

**After Example**: The LangChain agent automatically processes the question and generates an answer.

```bash
LangChain is a framework for building applications powered by language models.
# Answer generated using the LangChain zero-shot agent.
```

---

### 8\. **Retrieving Information from Documents with LangChain 🔍**

**Boilerplate Code**:

```python
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

loader = TextLoader("document.txt")
index = VectorstoreIndexCreator().from_loaders([loader])
```

**Use Case**: Retrieve answers or summaries from documents using a language model.

**Goal**: Create a system to search and retrieve information from documents using LangChain. 🎯

**Sample Code**:

```python
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

loader = TextLoader("document.txt")
index = VectorstoreIndexCreator().from_loaders([loader])

query = "What is the key takeaway from this document?"
result = index.query_with_sources(query)
print(result)
```

**Before Example**: You manually search through documents or write specific scripts to extract information.

```bash
# Manually scanning through documents:
open("document.txt").read()
```

**After Example**: LangChain automates information retrieval from documents based on queries.

```bash
The key takeaway is...
# The system retrieves the key information from the document automatically.
```

---

### 9\. **Building a Conversational Agent with LangChain 🗣️**

**Boilerplate Code**:

```python
from langchain.chains import ConversationChain

chain = ConversationChain(llm=llm, memory=SimpleMemory())
```

**Use Case**: Build a conversational agent that retains memory between interactions.

**Goal**: Create an agent that can handle multi-turn conversations, remembering past context. 🎯

**Sample Code**:

```python
from langchain.chains import ConversationChain
from langchain.memory import SimpleMemory
from langchain import OpenAI

llm = OpenAI(model_name="text-davinci-003")
memory = SimpleMemory()

conversation_chain = ConversationChain(llm=llm, memory=memory)

response1 = conversation_chain.run("What's your name?")
response2 = conversation_chain.run("Where do you live?")
print(response1)
print(response2)
```

**Before Example**: You interact with a language model that does not retain past interactions, losing conversation flow.

```bash
# Each prompt is isolated:
openai.Completion.create(prompt="What's your name?")
openai.Completion.create(prompt="Where do you live?")
```

**After Example**: The LangChain conversational agent remembers past interactions and responds with context.

```bash
My name is LangBot.
I live in the cloud.
# Multi-turn conversation with memory retention.
```

---

### 10\. **Using LangChain with OpenAI Functions 🧩**

**Boilerplate Code**:

```python
from langchain import OpenAI

llm = OpenAI(model_name="text-davinci-003", function_call=True)
```

**Use Case**: Incorporate OpenAI function calling to integrate external logic with LLM outputs.

**Goal**: Use OpenAI function calling with LangChain to run specific logic. 🎯

**Sample Code**:

```python
from langchain import OpenAI

llm = OpenAI(model_name="text-davinci-003", function_call=True)

functions = [
    {
        "name": "calculate",
        "description": "Perform a calculation.",
        "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
    }
]

response = llm.run("What is 2 + 3?", functions=functions)
print(response)
```

**Before Example**: You manually integrate external logic like mathematical calculations by handling outputs yourself.

```bash
# Manually handle calculations outside the LLM:
result = 2 + 3
print(result)
```

**After Example**: With OpenAI function calling, the LLM performs the calculations directly and provides results.

```bash
5
# LLM performs the calculation using the defined functions.
```

### 11\. **LangChain for Web Scraping 🌐**

**Boilerplate Code**:

```python
from langchain.tools import Tool
from langchain import OpenAI

def web_scraping_tool(url: str) -> str:
    # Add your web scraping logic here
    return f"Scraped content from {url}"

tool = Tool.from_function(func=web_scraping_tool, name="Web Scraper")
llm = OpenAI(model_name="text-davinci-003")
```

**Use Case**: Scrape content from websites and use it within LangChain.

**Goal**: Integrate a web scraping tool into LangChain for automatic content extraction. 🎯

**Sample Code**:

```python
def web_scraping_tool(url: str) -> str:
    return f"Scraped content from {url}"

tool = Tool.from_function(func=web_scraping_tool, name="Web Scraper")
output = tool.run("https://example.com")
print(output)
```

**Before Example**: You manually scrape web pages and process the data outside of the LangChain framework.

```bash
# Manually scraping a webpage using requests and BeautifulSoup:
response = requests.get("https://example.com")
```

**After Example**: LangChain integrates the scraping tool, allowing automated scraping within your workflows.

```bash
Scraped content from https://example.com
# Web scraping performed using the integrated tool.
```

---

### 12\. **LangChain Agents with Google Search 🕵️‍♂️**

**Boilerplate Code**:

```python
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain import OpenAI

search_tool = Tool(name="Google Search", func=google_search)
llm = OpenAI(model_name="text-davinci-003")
```

**Use Case**: Use LangChain agents to perform Google searches and retrieve information.

**Goal**: Automate Google search queries through LangChain for fetching information. 🎯

**Sample Code**:

```python
from langchain.tools import Tool
from langchain import OpenAI

def google_search(query: str) -> str:
    # Simulated search result
    return f"Top search result for {query}"

search_tool = Tool.from_function(func=google_search, name="Google Search")
llm = OpenAI(model_name="text-davinci-003")
agent = initialize_agent([search_tool], llm, agent="zero-shot-react-description")

response = agent.run("Search for LangChain tutorials")
print(response)
```

**Before Example**: You manually perform searches, then copy and process the results.

```bash
# Perform search manually and extract the top result:
results = google_search("LangChain tutorials")
```

**After Example**: LangChain agents automate the search and return results without manual input.

```bash
Top search result for LangChain tutorials
# Search result automatically fetched and processed by the agent.
```

---

### 13\. **LangChain with SQL Databases 💾**

**Boilerplate Code**:

```python
from langchain import SQLDatabase
from langchain import OpenAI
from langchain.chains import SQLDatabaseChain

db = SQLDatabase.from_uri("sqlite:///example.db")
llm = OpenAI(model_name="text-davinci-003")
sql_chain = SQLDatabaseChain(llm=llm, database=db)
```

**Use Case**: Interact with SQL databases and perform queries using LangChain.

**Goal**: Automate SQL queries and return data using language models. 🎯

**Sample Code**:

```python
from langchain import SQLDatabase, OpenAI
from langchain.chains import SQLDatabaseChain

db = SQLDatabase.from_uri("sqlite:///example.db")
llm = OpenAI(model_name="text-davinci-003")
sql_chain = SQLDatabaseChain(llm=llm, database=db)

query = "SELECT * FROM users"
result = sql_chain.run(query)
print(result)
```

**Before Example**: You manually write and execute SQL queries, then process the results.

```bash
# Running SQL queries manually in a database:
SELECT * FROM users;
```

**After Example**: LangChain executes the SQL query and processes the data automatically.

```bash
Returned data: [(1, 'Alice'), (2, 'Bob')]
# Data fetched automatically using LangChain with SQL integration.
```

---

### 14\. **Summarization with LangChain 📋**

**Boilerplate Code**:

```python
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(input_variables=["text"], template="Summarize the following: {text}")
summary_chain = LLMChain(llm=llm, prompt=prompt)
```

**Use Case**: Summarize large texts or documents using LangChain’s LLM chain.

**Goal**: Generate concise summaries of large texts with LangChain. 🎯

**Sample Code**:

```python
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(input_variables=["text"], template="Summarize the following: {text}")
summary_chain = LLMChain(llm=llm, prompt=prompt)

text = "LangChain is a framework for building applications powered by language models..."
summary = summary_chain.run(text)
print(summary)
```

**Before Example**: You manually summarize large texts, which is time-consuming.

```bash
# Reading through the document and summarizing:
"LangChain is a framework..."
```

**After Example**: LangChain automates the summarization of text using a language model.

```bash
LangChain simplifies language model workflows...
# Summary generated automatically using LangChain.
```

---

### 15\. **Question Answering from Documents with LangChain 📚**

**Boilerplate Code**:

```python
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

loader = TextLoader("document.txt")
index = VectorstoreIndexCreator().from_loaders([loader])
qa_chain = RetrievalQA(llm=llm, retriever=index.as_retriever())
```

**Use Case**: Answer questions based on the contents of a document.

**Goal**: Retrieve answers from a document based on user queries. 🎯

**Sample Code**:

```python
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain import OpenAI

llm = OpenAI(model_name="text-davinci-003")
loader = TextLoader("document.txt")
index = VectorstoreIndexCreator().from_loaders([loader])
qa_chain = RetrievalQA(llm=llm, retriever=index.as_retriever())

query = "What is LangChain?"
answer = qa_chain.run(query)
print(answer)
```

**Before Example**: You manually search through documents for answers, which is time-consuming.

```bash
# Searching for information manually:
open("document.txt").read()
```

**After Example**: LangChain retrieves the answer automatically from the document.

```bash
LangChain is a framework for building applications powered by language models.
# Answer retrieved automatically using LangChain's document retriever.
```

### 16\. **LangChain for Translation 🌍**

**Boilerplate Code**:

```python
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(input_variables=["text"], template="Translate this to French: {text}")
translation_chain = LLMChain(llm=llm, prompt=prompt)
```

**Use Case**: Translate text into different languages using LangChain’s LLM chain.

**Goal**: Automate language translation tasks using LangChain. 🎯

**Sample Code**:

```python
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(input_variables=["text"], template="Translate this to French: {text}")
translation_chain = LLMChain(llm=llm, prompt=prompt)

text = "LangChain is a powerful tool."
translation = translation_chain.run(text)
print(translation)
```

**Before Example**: You manually translate texts using dictionaries or online translation services.

```bash
# Manually translating:
"LangChain is a powerful tool" → "LangChain est un outil puissant."
```

**After Example**: LangChain automates translation between languages.

```bash
LangChain est un outil puissant.
# Translation done automatically using LangChain.
```

---

### 17\. **Sentiment Analysis with LangChain 😊😡**

**Boilerplate Code**:

```python
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(input_variables=["text"], template="Analyze the sentiment of this text: {text}")
sentiment_chain = LLMChain(llm=llm, prompt=prompt)
```

**Use Case**: Perform sentiment analysis on text data.

**Goal**: Automate sentiment analysis using a language model with LangChain. 🎯

**Sample Code**:

```python
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(input_variables=["text"], template="Analyze the sentiment of this text: {text}")
sentiment_chain = LLMChain(llm=llm, prompt=prompt)

text = "I love using LangChain, it's amazing!"
sentiment = sentiment_chain.run(text)
print(sentiment)
```

**Before Example**: You manually assess the tone or sentiment of text by reading it.

```bash
# Manually analyze sentiment:
"This text seems positive."
```

**After Example**: LangChain automatically classifies the sentiment as positive, negative, or neutral.

```bash
The sentiment is positive.
# Sentiment analysis performed automatically using LangChain.
```

---

### 18\. **Named Entity Recognition (NER) with LangChain 🏷️**

**Boilerplate Code**:

```python
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(input_variables=["text"], template="Extract entities from this text: {text}")
ner_chain = LLMChain(llm=llm, prompt=prompt)
```

**Use Case**: Extract named entities (like people, organizations, locations) from text.

**Goal**: Use LangChain to automatically detect and extract entities from text. 🎯

**Sample Code**:

```python
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(input_variables=["text"], template="Extract entities from this text: {text}")
ner_chain = LLMChain(llm=llm, prompt=prompt)

text = "LangChain was founded by Harrison Chase in 2021."
entities = ner_chain.run(text)
print(entities)
```

**Before Example**: You manually highlight or extract named entities from text documents.

```bash
# Manually extracting entities:
"LangChain" → Organization
"Harrison Chase" → Person
"2021" → Date
```

**After Example**: LangChain automatically identifies and extracts entities from text.

```bash
Entities: LangChain (Organization), Harrison Chase (Person), 2021 (Date)
# Named entities detected and extracted using LangChain.
```

---

### 19\. **Text Classification with LangChain 🗂️**

**Boilerplate Code**:

```python
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(input_variables=["text"], template="Classify the following text: {text}")
classification_chain = LLMChain(llm=llm, prompt=prompt)
```

**Use Case**: Automatically classify text into predefined categories.

**Goal**: Use LangChain to categorize text into relevant classes. 🎯

**Sample Code**:

```python
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(input_variables=["text"], template="Classify the following text: {text}")
classification_chain = LLMChain(llm=llm, prompt=prompt)

text = "LangChain simplifies building language model applications."
classification = classification_chain.run(text)
print(classification)
```

**Before Example**: You manually classify or label text based on its content.

```bash
# Manually classify text:
"LangChain simplifies building language model applications" → Technology
```

**After Example**: LangChain automatically classifies the text.

```bash
Category: Technology
# Text classification performed automatically using LangChain.
```

---

### 20\. **Text Generation with LangChain ✍️**

**Boilerplate Code**:

```python
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(input_variables=["text"], template="Continue writing this story: {text}")
generation_chain = LLMChain(llm=llm, prompt=prompt)
```

**Use Case**: Generate creative text, such as stories or articles.

**Goal**: Use LangChain to automatically generate extended content based on a prompt. 🎯

**Sample Code**:

```python
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(model_name="text-davinci-003")
prompt = PromptTemplate(input_variables=["text"], template="Continue writing this story: {text}")
generation_chain = LLMChain(llm=llm, prompt=prompt)

text = "Once upon a time, in a land far away..."
story = generation_chain.run(text)
print(story)
```

**Before Example**: You manually write or extend text, such as stories or articles.

```bash
# Manually continuing a story:
"Once upon a time, in a land far away..."
```

**After Example**: LangChain generates creative extensions for the text automatically.

```bash
Once upon a time, in a land far away, there was a magical forest where animals could talk...
# Story extended automatically using LangChain.
```