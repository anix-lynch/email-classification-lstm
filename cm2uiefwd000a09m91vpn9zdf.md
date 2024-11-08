---
title: "Using LangChain with OpenAI, Hugging Face, and Google Vertex AI"
seoTitle: "Using LangChain with OpenAI, Hugging Face, and Google Vertex AI"
seoDescription: "Using LangChain with OpenAI, Hugging Face, and Google Vertex AI"
datePublished: Tue Oct 29 2024 13:54:26 GMT+0000 (Coordinated Universal Time)
cuid: cm2uiefwd000a09m91vpn9zdf
slug: using-langchain-with-openai-hugging-face-and-google-vertex-ai
tags: openai, llm, langchain, claudeai, google-vertex

---

### **1\. Setting Environment Variables**

```python
# Import system and operating system libraries
import sys
import os

# Insert parent directory to system path so we can import configurations from there
sys.path.insert(0, os.path.abspath('..'))

# Import and set environment variables, such as API keys (e.g., for LLMs)
from config import set_environment
set_environment()
```

**Explanation**: This block prepares the environment by setting paths and loading variables like API keys.

**Sample Output**: No output here; it simply sets up environment variables.

---

### **2\. Initializing an OpenAI Language Model (LLM)**

```python
# Import OpenAI's LLM module from LangChain
from langchain_openai import OpenAI

# Initialize an OpenAI LLM instance
llm = OpenAI()

# Ask the model to tell a joke about light bulbs
summary = llm.invoke("Tell me a joke about light bulbs!")
print(summary)
```

**Explanation**: This initializes an OpenAI model and sends a simple joke request.

**Sample Output**:

```plaintext
"How many light bulbs does it take to change? Just one, but it takes a programmer to fix it!"
```

---

### **3\. Hugging Face Hub Integration**

```python
from langchain.llms import HuggingFaceHub

# Initialize a Hugging Face model with specific parameters
llm = HuggingFaceHub(
    model_kwargs={"temperature": 0.5, "max_length": 64},
    repo_id="google/flan-t5-xxl"  # specifies the model to use
)

# Ask the model a question about geography
prompt = "In which country is Tokyo?"
completion = llm.invoke(prompt)
print(completion)
```

**Explanation**: This uses the Hugging Face model `flan-t5-xxl` to answer a trivia question.

**Sample Output**:

```plaintext
"Tokyo is in Japan."
```

---

### **4\. FakeListLLM - Mock Model for Testing**

```python
from langchain.llms.fake import FakeListLLM

# Set up a mock model that always replies with a predefined answer
fake_llm = FakeListLLM(responses=["Hello"])
print(fake_llm.invoke("Hi and goodbye, FakeListLLM!"))
```

**Explanation**: Creates a mock LLM that replies with "Hello" to any input, helpful for testing workflows without real model responses.

**Sample Output**:

```plaintext
"Hello"
```

---

### **5\. Chat Models with OpenAI’s ChatGPT (GPT-4)**

```python
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Initialize the ChatGPT model from OpenAI
llm = ChatOpenAI(model_name='gpt-4-0613')

# Ask the model to write "Hello world" in Python
response = llm.invoke([HumanMessage(content='Say "Hello world" in Python.')])
print(response)
```

**Explanation**: Uses GPT-4 to provide a basic Python code snippet.

**Sample Output**:

```plaintext
"print('Hello world')"
```

---

### **6\. Multi-turn Conversation Setup**

```python
from langchain_core.messages import SystemMessage

# Start a chat where the assistant is given the role of a helpful guide
chat_output = llm.invoke([
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(content="What is the purpose of model regularization?")
])
print(chat_output)
```

**Explanation**: Defines a conversation where the assistant explains model regularization.

**Sample Output**:

```plaintext
"Model regularization helps prevent overfitting by adding constraints or penalties to the model parameters."
```

---

### **7\. Chat with Anthropic’s Claude Model**

```python
from langchain_anthropic import ChatAnthropic

# Initialize the Claude model from Anthropic for chat functionality
llm = ChatAnthropic(model='claude-3-opus-20240229')

# Ask the model to name the best large language model
response = llm.invoke([HumanMessage(content='What is the best large language model?')])
print(response)
```

**Explanation**: Uses Anthropic’s Claude model to discuss LLM preferences.

**Sample Output**:

```plaintext
"There isn't a single 'best' LLM; each has strengths and is suited to different tasks."
```

---

### **8\. Prompt Templates with LangChain**

```python
from langchain.prompts import PromptTemplate

# Create a flexible prompt template with placeholders for adjective and content
prompt_template = PromptTemplate.from_template("Tell me a {adjective} joke about {content}.")

# Format the template with specific words
formatted_prompt = prompt_template.format(adjective="funny", content="chickens")
print(formatted_prompt)
```

**Explanation**: Sets up a reusable prompt template for joke generation.

**Sample Output**:

```plaintext
"Tell me a funny joke about chickens."
```

---

### **9\. Using ChatPromptTemplate for Specific Conversations**

```python
from langchain.prompts import ChatPromptTemplate

# Define a chat template for translating text to French
template = ChatPromptTemplate.from_messages([
    ('system', 'You are an English to French translator.'),
    ('user', 'Translate this to French: {text}')
])

# Use the GPT model to translate a joke to French
response = llm.invoke(template.format_messages(text='How many programmers does it take to change a light bulb?'))
print(response)
```

**Explanation**: This prompts the model to act as a French translator for a given text.

**Sample Output**:

```plaintext
"Combien de programmeurs faut-il pour changer une ampoule?"
```

---

### **10\. Chat with Google Vertex AI Model**

```python
from langchain_google_vertexai import ChatVertexAI
from langchain import PromptTemplate, LLMChain

# Initialize the Google Vertex AI model
llm = ChatVertexAI(model_name="gemini-pro")

# Define a question template for structured answers
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

# Ask about a specific Super Bowl event
question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
print(llm_chain.run(question))
```

**Explanation**: Uses Google Vertex AI's Gemini model with a structured prompt template.

**Sample Output**:

```plaintext
"The Dallas Cowboys won the Super Bowl in 1994, the year Justin Bieber was born."
```

---

### **11\. Text-to-Image Generation with DALL-E**

```python
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

# Generate a DALL-E image based on a custom description
image_url = DallEAPIWrapper().run("halloween night at a haunted museum")

# Display the generated image
from IPython.display import Image, display
display(Image(url=image_url))
```

**Explanation**: Uses DALL-E to create an image prompt based on a description.

**Sample Output**: Displays an image of a "Halloween night at a haunted museum."

---

### **12\. Using Replicate’s Stable Diffusion Model**

```python
from langchain_community.llms import Replicate

# Use the Stable Diffusion model from Replicate for text-to-image generation
text2image = Replicate(
    model="stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
    model_kwargs={"image_dimensions": "512x512"}
)
image_url = text2image("a book cover for a book about creating generative ai applications in Python")

# Display the generated image
display(Image(url=image_url))
```

**Explanation**: Creates a book cover using the Stable Diffusion model.

**Sample Output**: Displays a generated book cover.

---

### **13\. Complex Image Generation Pipeline**

```python
# Define a complex architectural description
langchain_image = "The image appears to be a diagram representing the architecture of a system named LangChain."

# Convert the description into a concise image prompt
prompt = PromptTemplate(
    input_variables=["image_desc"],
    template=("Simplify this image description into a concise prompt to generate an image: {image_desc}")
)
chain = LLMChain(llm=llm, prompt=prompt)
prompt = chain.run(langchain_image)
print(prompt)

# Generate the image based on the simplified prompt
image_url = DallEAPIWrapper().run(prompt)
display(Image(url=image_url))
```

**Explanation**: Processes a complex text description to generate an architecture diagram using LangChain and DALL-E.

**Sample Output**: Displays an image of the LangChain architecture diagram.

---