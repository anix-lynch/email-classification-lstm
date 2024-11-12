---
title: "Langchain Project 2: Mini Chatbot"
seoTitle: "Langchai Project 2: Mini Chatbot"
seoDescription: "Langchai Project 2: Mini Chatbot"
datePublished: Tue Nov 12 2024 06:07:58 GMT+0000 (Coordinated Universal Time)
cuid: cm3e1wgv600070amihzuogdec
slug: langchain-project-2-mini-chatbot
tags: chatbot, streamlit, openai, llm, langchain

---

### 1\. **Imports and Setup**

**Code**:

```python
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
```

* **Explanation**: Import necessary libraries. `os` is for environment variable access, `streamlit` for the app interface, and LangChain’s `ChatOpenAI` along with message types for conversation structure.
    
* What Are `langchain_openai` and `langchain.schema`?
    
    * `langchain_openai`:
        
        * This is a **specialized module** that specifically integrates LangChain with OpenAI’s models.
            
        * It provides classes like `ChatOpenAI` that make it easier to interact with OpenAI’s GPT-based models (e.g., GPT-3.5-turbo, GPT-4) by simplifying setup and usage.
            
        * Instead of directly using `langchain.llms` (which is more general), `langchain_openai` gives you OpenAI-specific functionality and optimizations.
            
    * `langchain.schema`:
        
        * This module provides **data structures** that standardize how messages and interactions are defined in LangChain.
            
        * It includes classes like `HumanMessage`, `SystemMessage`, and `AIMessage`:
            
            * `HumanMessage`: Represents the user’s input.
                
            * `SystemMessage`: Sets the AI’s role or guidelines.
                
            * `AIMessage`: Represents the AI's response back to the user.
                
        * These standardized message types are crucial for creating organized and understandable interactions with the model.
            
    

**Demo Output**: No visible output. Just setting up the necessary imports.

---

### 2\. **Set Environment Variables for API Key**

**Code**:

```python
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
```

* **Explanation**: Set up OpenAI API key for accessing OpenAI’s models. Replace `"your_openai_api_key_here"` with your actual API key if testing locally.
    

**Demo Output**: No visible output, but the API key is now accessible for making requests to the OpenAI API.

---

### 3\. **Initialize the OpenAI Chat Model**

**Code**:

```python
chat_model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
```

* **Explanation**: Initialize `ChatOpenAI` with a temperature setting of 0.7 (adds creativity to responses) and specify the model as `"gpt-3.5-turbo"`, which is suitable for chat-based applications.
    

**Demo Output**: No visible output, but `chat_model` is now set up and ready to process conversations.

---

### 4\. **Streamlit App Setup: Title, Header, and Session State Initialization**

**Code**:

```python
st.set_page_config(page_title="My Mini Chatbot", page_icon=":robot:")
st.header("Welcome to My Mini Chatbot!")

# Session state to maintain conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a friendly and helpful assistant.")
    ]
```

* **Explanation**:
    
    * Set the page title and icon for a more personalized feel.
        
    * Add a welcome header.
        
    * Initialize `st.session_state.messages` to store conversation history if it doesn’t already exist. A system message sets the chatbot's persona as friendly and helpful.
        

**Demo Output**:

```python
Welcome to My Mini Chatbot!
```

* **Explanation**: The title and header appear at the top of the page, and the chatbot’s initial session memory is set up.
    

---

### 5\. **User Input Box**

**Code**:

```python
def get_user_input():
    user_input = st.text_input("You: ", key="input")
    return user_input

# Display input box and get user input
user_input = get_user_input()
```

* **Explanation**: Define a function to capture user input through a text box labeled "You:". This input is stored in the `user_input` variable.
    

**Demo Output**:

```python
You: [__________]
```

* **Explanation**: This creates an input box where the user can type their questions or comments.
    

---

### 6\. **Generate Button and Response Processing**

**Code**:

```python
# Generate button for the chatbot's response
if st.button("Generate"):
    if user_input:
        # Add user message to session messages
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        # Invoke the chat model with the session messages
        response = chat_model.invoke(st.session_state.messages)
        
        # Add AI's response to the conversation history
        st.session_state.messages.append(AIMessage(content=response.content))
        
        # Display the AI's response
        st.subheader("Assistant:")
        st.write(response.content)
    else:
        st.write("Please enter a message to start the conversation!")
```

* **Explanation**:
    
    * A button labeled "Generate" triggers the chatbot’s response.
        
    * If `user_input` contains text:
        
        * It’s added to `st.session_state.messages` as a `HumanMessage`.
            
        * The entire conversation history (`st.session_state.messages`) is passed to `chat_model.invoke()`, which generates a response.
            
        * The response is appended to the conversation history as an `AIMessage` and displayed on the interface.
            
    * If `user_input` is empty, a prompt asks the user to enter a message.
        

**Demo Output**:

* If **user\_input** is `"Hello!"`, the assistant might respond with:
    
    ```python
    Assistant:
    Hello! How can I assist you today?
    ```
    
* The assistant’s response updates each time you click “Generate” with new input, and conversation history is retained.
    

---

### Full Code Recap

Here’s the full code in one place for easy copy-pasting:

```python
# Import necessary libraries
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Set environment variable for OpenAI API Key (replace with actual API key or set in deployment environment)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")

# Initialize the OpenAI chat model
chat_model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

# Streamlit app setup: Page title and header
st.set_page_config(page_title="My Mini Chatbot", page_icon=":robot:")
st.header("Welcome to My Mini Chatbot!")

# Session state to maintain conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a friendly and helpful assistant.")
    ]

# Function to get user input
def get_user_input():
    user_input = st.text_input("You: ", key="input")
    return user_input

# Display input box and get user input
user_input = get_user_input()

# Generate button for the chatbot's response
if st.button("Generate"):
    if user_input:
        # Add user message to session messages
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        # Invoke the chat model with the session messages
        response = chat_model.invoke(st.session_state.messages)
        
        # Add AI's response to the conversation history
        st.session_state.messages.append(AIMessage(content=response.content))
        
        # Display the AI's response
        st.subheader("Assistant:")
        st.write(response.content)
    else:
        st.write("Please enter a message to start the conversation!")
```

### Running the App

1. **Locally**:
    
    * Save as [`app.py`](http://app.py).
        
    * Run in the terminal:
        
        ```bash
        streamlit run app.py
        ```
        
2. **Hugging Face Spaces**:
    
    * Create a new Space with Streamlit as the SDK.
        
    * Upload [`app.py`](http://app.py) and add `requirements.txt` with:
        
        ```python
        streamlit
        langchain-openai==0.1.0
        ```
        
    * Set the OpenAI API key in the Space’s environment settings.
        

This app provides an interactive chatbot experience that remembers the conversation context across exchanges.