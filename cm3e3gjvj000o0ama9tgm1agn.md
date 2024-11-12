---
title: "Langchain Project  3: Similarity Search Application"
seoTitle: "Langchain Project  3: Similarity Search Application"
seoDescription: "Langchain Project  3: Similarity Search Application"
datePublished: Tue Nov 12 2024 06:51:34 GMT+0000 (Coordinated Universal Time)
cuid: cm3e3gjvj000o0ama9tgm1agn
slug: langchain-project-3-similarity-search-application
tags: streamlit, langchain, faiss, tiktoken, similaritysearch

---

In this project, we're building a **Similarity Search Application** that leverages **text embeddings** to identify and display items similar to a user-provided word. Specifically, this app provides suggestions for words or objects that are related to an input word by comparing it with a predefined list of items and their embeddings.

The goal is to find the closest matches to a given word based on **cosine similarity** in the embedding space, which represents the semantic closeness between words.

### Project Overview

1. **Purpose**:
    
    * This project is designed to help users (such as children or language learners) find semantically related items or concepts. For example, if a child types "cat," the app might return "dog" and "tiger" because these animals are conceptually similar in the embeddings space.
        
2. **Use Case**:
    
    * A simple educational tool or assistant where users can type a word and receive suggestions of similar words or concepts.
        
    * For example, a child might type "apple" and the app could return "orange," "banana," and "grape," helping them understand the concept of fruits.
        
3. **Components**:
    
    * **Text Embeddings**: We use OpenAI’s embeddings through LangChain to represent each word as a vector, capturing semantic relationships.
        
    * **Cosine Similarity**: To measure similarity between the input word and words in our dataset, we use cosine similarity.
        
    * **Streamlit**: Provides a simple user interface for input and output, enabling users to enter a word and see related suggestions.
        
    * **FAISS (Facebook AI Similarity Search)**: This open-source library by Meta/Facebook allows efficient searching within the high-dimensional vector space.
        
4. **Data**:
    
    * A sample dataset of items (e.g., animals, fruits, sports) stored in a CSV file, which is converted into embeddings to be compared with the user's input.
        
5. **Key Technologies**:
    
    * **LangChain**: Manages interactions with OpenAI’s embedding model.
        
    * **OpenAI**: Provides the embedding model to convert text into a numerical form that can be compared.
        
    * **Streamlit**: Enables quick and easy web app development for a responsive user interface.
        
    * **FAISS**: Speeds up similarity searches in the embedding space, making the app responsive even with larger datasets.
        

.

---

### User Interface Overview

* **Input Field**: Accepts a word from the user, such as "cat" or "strawberry."
    
* **Generate Button**: When clicked, performs the similarity search to find related words or items.
    
* **Output Display**: Shows a list of items related to the input, helping the user discover items in the same category or conceptually similar items.
    

### Code Walkthrough and How Each Part Fits In

The code is structured to:

1. **Load Dependencies and Models**: Set up LangChain, OpenAI, FAISS, and Streamlit.
    
2. **Convert Data into Embeddings**: Preprocess the CSV data file, storing each item as an embedding.
    
3. **User Interaction**: Receive a word from the user, compute its embedding, and search for similar items in the dataset.
    
4. **Display Results**: Use Streamlit to show the top results.
    

Let me know if you would like to walk through specific code sections to see how each part is implemented!

---

### Step 1: Import Required Libraries

```python
# Allows you to use Streamlit, a framework for building interactive web applications.
import streamlit as st

# Provides a way to interact with the operating system, such as accessing environment variables, working with files.
import os

# Imports OpenAI Embeddings from LangChain
from langchain_openai import OpenAIEmbeddings

# FAISS library for similarity search
from langchain_community.vectorstores import FAISS

# Load environment variables
from dotenv import load_dotenv
```

* **Explanation**: This section imports all necessary libraries:
    
    * **Streamlit** (`st`): Creates the user interface and displays output on a web page.
        
    * **os**: Manages environment variables and interacts with the file system.
        
    * **OpenAIEmbeddings**: Generates vector representations (embeddings) for words or phrases, allowing us to search for similar terms.
        
    * **FAISS**: Efficient similarity search library, used here to store and retrieve embeddings for related terms.
        
    * **dotenv** (`load_dotenv`): Loads environment variables from a `.env` file, especially useful for sensitive keys like API keys.
        

---

### Step 2: Configure Streamlit and Load Environment Variables

```python
load_dotenv()  # Load environment variables from .env file

# Set up Streamlit page
st.set_page_config(page_title="Educate Kids", page_icon=":robot:")
st.header("Hey, Ask me something & I will give out similar things")
```

* **Explanation**:
    
    * `load_dotenv()`: This command loads environment variables (like API keys) from a `.env` file, which is essential for accessing external services.
        
    * `st.set_page_config`: Configures the page title and icon in Streamlit.
        
    * `st.header`: Sets a title at the top of the page for user guidance.
        
* **Expected Output**: This configures the page. When you load the app, you’ll see a title like:
    
    **Educate Kids** *Hey, Ask me something & I will give out similar things*
    

---

### Step 3: Initialize the Embedding and CSV Data Loader

```python
# Initialize the OpenAIEmbeddings object
embeddings = OpenAIEmbeddings()

# Import and load CSV file data for our tasks
from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='/mnt/data/myData.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['Words']
})

# Assigning the data inside the CSV to a variable
data = loader.load()
```

* **Explanation**:
    
    * **OpenAIEmbeddings**: Creates an `embeddings` object that can convert words into numerical representations.
        
    * **CSVLoader**: Loads a CSV file (here `myData.csv`), specifying its structure and delimiter type. The data is then assigned to `data` as a list of documents.
        
* **Expected Output**:
    
    * You won’t see anything visually yet on the Streamlit app, but internally, `data` now contains a list of terms from `myData.csv`.
        

---

### Step 4: Create FAISS Database

```python
# Initialize FAISS database with embeddings
db = FAISS.from_documents(data, embeddings)
```

* **Explanation**:
    
    * `FAISS.from_documents` takes `data` (words from CSV) and `embeddings`, generating vector embeddings for each word and storing them in a searchable FAISS database.
        
    * `db`: Now holds our searchable database, allowing us to find similar items by their vector representation.
        
* **Expected Output**:
    
    * No immediate output is shown, but the app now has a database (`db`) that can search for terms similar to a given input.
        

---

### Step 5: Define User Input Function and Capture Input

```python
# Function to receive input from the user and store it in a variable
def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text

# Capture user input
user_input = get_text()
```

* **Explanation**:
    
    * `get_text()`: Defines a function to capture user input in the app’s text box.
        
    * `user_input`: Stores whatever the user types into the text box, awaiting submission.
        
* **Expected Output**:
    
    * A text input box appears with the prompt “You: ”, where users can type their question or keyword.
        

---

### Step 6: Define Similarity Search and Display Results

```python
submit = st.button('Find similar Things')

if submit:
    # If the button is clicked, fetch similar text
    docs = db.similarity_search(user_input)
    st.subheader("Top Matches:")
    st.text(docs[0].page_content)
    st.text(docs[1].page_content)
```

* **Explanation**:
    
    * **submit**: A button labeled "Find similar Things". When clicked, it triggers a similarity search.
        
    * **similarity\_search**: This searches `db` (our FAISS database) for items similar to `user_input`.
        
    * **Displaying Results**:
        
        * `st.subheader("Top Matches:")`: Adds a section header.
            
        * `st.text(docs[0].page_content)`: Shows the top match for the user’s input.
            
        * `st.text(docs[1].page_content)`: Shows the second-best match.
            
* **Expected Output**:
    
    * After typing a word in the input box and pressing **Find similar Things**, the app displays the two most similar words from `myData.csv` in the **Top Matches** section.
        

---

### Example of Expected User Interaction

Suppose the CSV file contains the following words:

* `Words`: \["apple", "banana", "pear", "fruit", "dog", "cat", "animal"\]
    

If the user enters **"apple"**, the app might display:

**Top Matches:**

* `banana`
    
* `pear`
    

If the user enters **"cat"**, it could display:

**Top Matches:**

* `dog`
    
* `animal`
    

---

### Full Code

Here's the consolidated code for your app:

```python
import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Educate Kids", page_icon=":robot:")
st.header("Hey, Ask me something & I will give out similar things")

embeddings = OpenAIEmbeddings()

from langchain.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='/mnt/data/myData.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['Words']
})
data = loader.load()

db = FAISS.from_documents(data, embeddings)

def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text

user_input = get_text()
submit = st.button('Find similar Things')

if submit:
    docs = db.similarity_search(user_input)
    st.subheader("Top Matches:")
    st.text(docs[0].page_content)
    st.text(docs[1].page_content)
```

---

Although `tiktoken` appears in the `requirements.txt` file, it’s not explicitly used in the code. This can happen for a few reasons:

1. **Indirect Dependency**: Sometimes, libraries like `langchain` or `OpenAI API client` may internally rely on `tiktoken` for tokenization, especially when interacting with OpenAI’s models. Even though it's not directly used in the code, it ensures compatibility if these libraries call `tiktoken` under the hood.
    
2. **Prepared for Future Use**: It might have been included in the `requirements.txt` file as a precaution, allowing for tokenization management if necessary in future modifications of the project (e.g., handling token limits, or dynamically adjusting input lengths to avoid exceeding model constraints).
    
3. **Token Management**: In a project that involves fine-tuning responses or ensuring inputs don’t exceed certain token limits, `tiktoken` might be added to manage or check token counts directly within the code.
    

If you want to use `tiktoken` directly, you can add it to monitor or adjust token counts, especially if you're sending large text inputs. Here’s how you might integrate it:

```python
import tiktoken

# Initialize encoding
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Example usage
user_input = "Your text input here"
tokens = encoding.encode(user_input)

# Check the number of tokens
print("Token count:", len(tokens))

# Optionally, truncate if token count exceeds limit
MAX_TOKENS = 4096  # Example limit
if len(tokens) > MAX_TOKENS:
    tokens = tokens[:MAX_TOKENS]
    truncated_text = encoding.decode(tokens)
    print("Truncated text:", truncated_text)
```

Including this in the project would provide direct control over token limits and ensure smoother API interactions.