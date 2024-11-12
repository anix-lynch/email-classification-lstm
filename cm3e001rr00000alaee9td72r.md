---
title: "Langchain Project 1:  Basic ChatGPT clone"
seoTitle: "Langchain Project 1:  Basic ChatGPT clone"
seoDescription: "Langchain Project 1:  Basic ChatGPT clone"
datePublished: Tue Nov 12 2024 05:14:45 GMT+0000 (Coordinated Universal Time)
cuid: cm3e001rr00000alaee9td72r
slug: langchain-project-1-basic-chatgpt-clone
tags: ai, llm, langchain

---

### 1\. **Install Necessary Packages**

* **Code**:
    
    ```python
    !pip install langchain==0.1.13 openai==1.14.2 langchain-openai==0.1.0 huggingface-hub==0.21.4 streamlit
    ```
    
* **Output**: No visible output, but these installations ensure all necessary packages are available for interacting with OpenAI, Hugging Face, and Streamlit.
    

---

### 2\. **Set Up OpenAI API Access**

* **Code**:
    
    ```python
    import os
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
    ```
    
* **Output**: No visible output, but behind the scenes, the `OPENAI_API_KEY` is stored securely as an environment variable. This key is required for any OpenAI API interactions.
    

---

### 3\. **Initialize OpenAI Model Using LangChain**

* **Code**:
    
    ```python
    from langchain_openai import OpenAI
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
    ```
    
* **Output**: No visible output, but here, LangChain initializes the OpenAI model (`gpt-3.5-turbo-instruct`) for subsequent use. The model can now be invoked to process prompts.
    

---

### 4\. **Generate an Answer Using OpenAI Model**

* **Code**:
    
    ```python
    our_query = "What is the currency of India?"
    completion = llm.invoke(our_query)
    print(completion)
    ```
    
* **Output**:
    
    ```python
    The currency of India is the Indian Rupee.
    ```
    
* **Explanation**: The code sends the query "What is the currency of India?" to the initialized OpenAI model. The model responds with an answer based on its trained knowledge.
    

---

### 5\. **Set Up Hugging Face API Access**

* **Code**:
    
    ```python
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_token_here"
    ```
    
* **Output**: No visible output, but the Hugging Face API token is securely set as an environment variable. This token enables access to models hosted on Hugging Face.
    

---

### 6\. **Initialize Hugging Face Model Using LangChain**

* **Code**:
    
    ```python
    from langchain.llms import HuggingFaceEndpoint
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
    ```
    
* **Output**: No visible output, but LangChain initializes the `Mistral-7B-Instruct` model hosted on Hugging Face, ready to be used for generating responses to queries.
    

---

### 7\. **Generate an Answer Using Hugging Face Model**

* **Code**:
    
    ```python
    our_query = "What is the currency of India?"
    completion = llm.invoke(our_query)
    print(completion)
    ```
    
* **Output**:
    
    ```python
    Rupee
    ```
    
* **Explanation**: The code sends the query "What is the currency of India?" to the Hugging Face model, which returns the answer. It may vary slightly from OpenAI’s response due to differences in training data or model structure.
    

---

### 8\. **Build the Streamlit App UI**

* **Code**:
    
    ```python
    import streamlit as st
    
    def load_answer(question):
        llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
        answer = llm.invoke(question)
        return answer
    
    st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
    st.header("LangChain Demo")
    
    def get_text():
        input_text = st.text_input("You: ", key="input")
        return input_text
    
    user_input = get_text()
    response = load_answer(user_input)
    
    submit = st.button('Generate')
    
    if submit:
        st.subheader("Answer:")
        st.write(response)
    ```
    
* **Output**:
    
    * When run in a Streamlit environment, the app appears as a minimal chat interface with:
        
        * A **title** ("LangChain Demo").
            
        * A **text input box** labeled "You:" for user queries.
            
        * A **Generate button** to trigger responses.
            
        * An **Answer section** displaying the model’s response when the Generate button is clicked.
            
* **Explanation**:
    
    * This sets up a single-page app for interacting with an LLM. The user’s input is sent to the `load_answer()` function, which queries OpenAI’s model. The response is then displayed within the app when the Generate button is clicked.
        

---

### **Run the Streamlit Application on Hugging Face Spaces**

* * Hugging Face Spaces supports **Streamlit apps** and allows you to deploy your app publicly.
        
    * For deployment on Hugging Face Spaces:
        
        * Create a **new Space** on Hugging Face.
            
        * Choose **Streamlit** as the SDK.
            
        * Upload your [`app.py`](http://app.py) file (the entire Streamlit section of your code).
            
        * Add a `requirements.txt` file to specify dependencies (e.g., `streamlit`, `langchain`, `openai`, `huggingface-hub`).
            
        * Add any **API keys** under "Settings" &gt; "Environment Variables" in your Hugging Face Space for secure access.
            
    * **Note**: This will allow you to interact with the app as a standalone, public-facing application.
        
    
    Here's a breakdown of each section, with expected demo outputs or descriptions of what’s being set up behind the scenes:
    
    ---
    
    ### 1\. **Install Necessary Packages**
    
    * **Code**:
        
        ```python
        !pip install langchain==0.1.13 openai==1.14.2 langchain-openai==0.1.0 huggingface-hub==0.21.4 streamlit
        ```
        
    * **Output**: No visible output, but these installations ensure all necessary packages are available for interacting with OpenAI, Hugging Face, and Streamlit.
        
    
    ---
    
    ### 2\. **Set Up OpenAI API Access**
    
    * **Code**:
        
        ```python
        import os
        os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
        ```
        
    * **Output**: No visible output, but behind the scenes, the `OPENAI_API_KEY` is stored securely as an environment variable. This key is required for any OpenAI API interactions.
        
    
    ---
    
    ### 3\. **Initialize OpenAI Model Using LangChain**
    
    * **Code**:
        
        ```python
        from langchain_openai import OpenAI
        llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
        ```
        
    * **Output**: No visible output, but here, LangChain initializes the OpenAI model (`gpt-3.5-turbo-instruct`) for subsequent use. The model can now be invoked to process prompts.
        
    
    ---
    
    ### 4\. **Generate an Answer Using OpenAI Model**
    
    * **Code**:
        
        ```python
        our_query = "What is the currency of India?"
        completion = llm.invoke(our_query)
        print(completion)
        ```
        
    * **Output**:
        
        ```python
        The currency of India is the Indian Rupee.
        ```
        
    * **Explanation**: The code sends the query "What is the currency of India?" to the initialized OpenAI model. The model responds with an answer based on its trained knowledge.
        
    
    ---
    
    ### 5\. **Set Up Hugging Face API Access**
    
    * **Code**:
        
        ```python
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_token_here"
        ```
        
    * **Output**: No visible output, but the Hugging Face API token is securely set as an environment variable. This token enables access to models hosted on Hugging Face.
        
    
    ---
    
    ### 6\. **Initialize Hugging Face Model Using LangChain**
    
    * **Code**:
        
        ```python
        from langchain.llms import HuggingFaceEndpoint
        llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
        ```
        
    * **Output**: No visible output, but LangChain initializes the `Mistral-7B-Instruct` model hosted on Hugging Face, ready to be used for generating responses to queries.
        
    
    ---
    
    ### 7\. **Generate an Answer Using Hugging Face Model**
    
    * **Code**:
        
        ```python
        our_query = "What is the currency of India?"
        completion = llm.invoke(our_query)
        print(completion)
        ```
        
    * **Output**:
        
        ```python
        Rupee
        ```
        
    * **Explanation**: The code sends the query "What is the currency of India?" to the Hugging Face model, which returns the answer. It may vary slightly from OpenAI’s response due to differences in training data or model structure.
        
    
    ---
    
    ### 8\. **Build the Streamlit App UI**
    
    * **Code**:
        
        ```python
        import streamlit as st
        
        def load_answer(question):
            llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
            answer = llm.invoke(question)
            return answer
        
        st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
        st.header("LangChain Demo")
        
        def get_text():
            input_text = st.text_input("You: ", key="input")
            return input_text
        
        user_input = get_text()
        response = load_answer(user_input)
        
        submit = st.button('Generate')
        
        if submit:
            st.subheader("Answer:")
            st.write(response)
        ```
        
    * **Output**:
        
        * When run in a Streamlit environment, the app appears as a minimal chat interface with:
            
            * A **title** ("LangChain Demo").
                
            * A **text input box** labeled "You:" for user queries.
                
            * A **Generate button** to trigger responses.
                
            * An **Answer section** displaying the model’s response when the Generate button is clicked.
                
    * **Explanation**:
        
        * This sets up a single-page app for interacting with an LLM. The user’s input is sent to the `load_answer()` function, which queries OpenAI’s model. The response is then displayed within the app when the Generate button is clicked.
            
    
    ---
    
    This breakdown provides an overview of what each section sets up or accomplishes, along with expected outputs where relevant. Let me know if you need further detail on any part!
    

Here’s a consolidated Python script that combines all components to create a simple ChatGPT-style app using LangChain, OpenAI, Hugging Face, and Streamlit. This script will run in a **Streamlit** environment, such as **Hugging Face Spaces** or your local machine.

---

### Full Python Code for Streamlit App

```python
# Import necessary libraries
import os
import streamlit as st
from langchain_openai import OpenAI
from langchain.llms import HuggingFaceEndpoint

# Set up environment variables for API keys
# Make sure to set your API keys in the environment or replace 'your_openai_api_key_here' and 'your_huggingface_token_here' with your actual API keys.
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "your_huggingface_token_here")

# Initialize the OpenAI model (proprietary LLM)
def openai_model():
    return OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# Initialize the Hugging Face model (open-source LLM)
def huggingface_model():
    return HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")

# Load the answer from the selected model
def load_answer(question, model_choice="OpenAI"):
    if model_choice == "OpenAI":
        llm = openai_model()
    else:
        llm = huggingface_model()
    # Invoke the model with the question
    answer = llm.invoke(question)
    return answer

# Streamlit app UI setup
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo - OpenAI and Hugging Face Q&A")

# Dropdown to choose the model
model_choice = st.selectbox("Choose the model:", ["OpenAI", "Hugging Face"])

# Get user input
def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text

user_input = get_text()

# Button to generate the answer
if st.button('Generate'):
    if user_input:
        response = load_answer(user_input, model_choice=model_choice)
        st.subheader("Answer:")
        st.write(response)
    else:
        st.write("Please enter a question!")
```

---

### Explanation of the Code

1. **Environment Variables**:
    
    * OpenAI and Hugging Face API keys are set using `os.environ`. Replace `"your_openai_api_key_here"` and `"your_huggingface_token_here"` with actual keys if running locally or set them as environment variables in your deployment environment (e.g., Hugging Face Spaces).
        
2. **Model Initialization Functions**:
    
    * `openai_model()`: Initializes the OpenAI model (`gpt-3.5-turbo-instruct`).
        
    * `huggingface_model()`: Initializes the Hugging Face model (`Mistral-7B-Instruct`).
        
3. **load\_answer Function**:
    
    * Loads the response from the selected model (either OpenAI or Hugging Face) based on the dropdown selection in Streamlit.
        
4. **Streamlit UI**:
    
    * **Dropdown**: Allows users to select between the OpenAI and Hugging Face models.
        
    * **Text Input**: Captures user questions.
        
    * **Generate Button**: Sends the question to the selected model and displays the answer.
        

---

### Instructions to Run the App

1. **Locally**:
    
    * Save the code as [`app.py`](http://app.py).
        
    * Run in the terminal:
        
        ```bash
        streamlit run app.py
        ```
        
2. **On Hugging Face Spaces**:
    
    * Create a new Space, select **Streamlit** as the SDK.
        
    * Upload [`app.py`](http://app.py).
        
    * Add a `requirements.txt` with the following dependencies:
        
        ```python
        streamlit
        langchain-openai==0.1.0
        huggingface-hub==0.21.4
        ```
        
    * Set API keys in the Hugging Face Space environment settings.
        

---

This single code file will create a functional Q&A app, where users can ask questions and select which model (OpenAI or Hugging Face) they want to use for responses.