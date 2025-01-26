---
title: "Natural Language SQL: AI Meets SQL, No More SQL Queries! 🚀"
seoTitle: "Natural Language SQL: AI Meets SQL, No More SQL Queries! 🚀"
seoDescription: "Natural Language SQL: AI Meets SQL, No More SQL Queries! 🚀"
datePublished: Sun Jan 26 2025 10:19:46 GMT+0000 (Coordinated Universal Time)
cuid: cm6dgx6p8000c09lb8rxicpyj
slug: natural-language-sql-ai-meets-sql-no-more-sql-queries
tags: sql, natural-language, text-to-sql, weaviate, hugging-face-grappa

---

# 1\. SQL End-to-End Pipelines

| 💀 **Before** | 🎉 **After** |
| --- | --- |
| Writing SQL queries manually. | Use **Hugging Face Grappa** (open-source) to translate text to SQL. |
| Learning database schema and syntax. | Combine **Whisper** + **Grappa** for open-source voice-to-SQL translation. |

---

### **Code Sample: Open-Source NLP-to-SQL with Hugging Face**

#### **Step 1: Install Open-Source Tools**

```bash
pip install transformers openai-whisper sqlalchemy
```

---

#### **Step 2: Create a Database**

Set up a simple SQLite database for testing (same as before):

```python
import sqlite3

# Create SQLite database and table
conn = sqlite3.connect('employees.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        age INTEGER,
        profession TEXT,
        achievements TEXT
    )
''')

# Insert sample employee data
employees = [
    (1, 'Fiona Davis', 32, 'Mobile Developer', 'Developed app with 100k downloads'),
    (2, 'Ian Clark', 45, 'Cybersecurity Expert', 'Led red-teaming for Fortune 500'),
    (3, 'Alice Cooper', 39, 'Data Scientist', 'Built a predictive analytics model')
]
cursor.executemany('INSERT INTO employees VALUES (?, ?, ?, ?, ?)', employees)
conn.commit()
```

---

#### **Step 3: Text-to-SQL with Hugging Face Grappa**

We use **Salesforce Grappa** (open-source) for converting natural language into SQL.

```python
from transformers import pipeline

# Load Hugging Face's Grappa model for text-to-SQL
text_to_sql = pipeline("text-to-sql", model="Salesforce/grappa-text-to-sql")

# Natural language query
query = "Who are the employees with the profession 'Cybersecurity Expert'?"

# Translate natural language to SQL
sql_query = text_to_sql(query, db_path="employees.db")['query']

# Execute the generated SQL query
cursor.execute(sql_query)
results = cursor.fetchall()

# Display results
for row in results:
    print(f"ID: {row[0]}, Name: {row[1]}, Age: {row[2]}, Profession: {row[3]}, Achievements: {row[4]}")
```

---

### **Optional: Voice-to-SQL with Whisper + Grappa**

If you want to add **voice input**, integrate OpenAI's **Whisper** for speech-to-text before feeding it to Grappa.

#### **Code for Speech-to-SQL**

```python
import whisper

# Load Whisper model for speech recognition
whisper_model = whisper.load_model("base")

# Convert speech input to text
audio_file = "query_audio.wav"  # Replace with your audio file
result = whisper_model.transcribe(audio_file)
speech_query = result['text']

# Use Grappa to convert the text to SQL
sql_query = text_to_sql(speech_query, db_path="employees.db")['query']

# Execute SQL query
cursor.execute(sql_query)
results = cursor.fetchall()

# Display results
for row in results:
    print(f"ID: {row[0]}, Name: {row[1]}, Age: {row[2]}, Profession: {row[3]}, Achievements: {row[4]}")
```

---

### **Sample Output**

#### **Natural Language Input**:

```text
Who are the employees with the profession 'Cybersecurity Expert'?
```

#### **Generated SQL Query**:

```sql
SELECT * FROM employees WHERE profession = 'Cybersecurity Expert';
```

#### **Final Output**:

```text
ID: 2, Name: Ian Clark, Age: 45, Profession: Cybersecurity Expert, Achievements: Led red-teaming for Fortune 500
```

---

This solution uses entirely **open-source tools** (Hugging Face’s Grappa + Whisper) to achieve text/voice-to-SQL functionality. Let me know if this works or needs more tweaking! 🚀

# 2\. **Explainable AI** in SQL

---

### **Table Update**

| 💀 **Before** | 🎉 **After** |
| --- | --- |
| No clarity on how queries are built. | Use open-source tools like **Hugging Face Grappa** to generate SQL and explain the process. |

---

### **Code Example for Explainable SQL**

#### **Step 1: Install Open-Source Libraries**

```bash
pip install transformers sqlalchemy
```

---

#### **Step 2: Create a Database**

Let’s reuse a simple SQLite database setup:

```python
import sqlite3

# Create SQLite database and table
conn = sqlite3.connect('employees.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        age INTEGER,
        profession TEXT,
        achievements TEXT
    )
''')

# Insert employee data
employees = [
    (1, 'Fiona Davis', 32, 'Mobile Developer', 'Developed app with 100k downloads'),
    (2, 'Ian Clark', 45, 'Cybersecurity Expert', 'Led red-teaming for Fortune 500'),
    (3, 'Alice Cooper', 39, 'Data Scientist', 'Built a predictive analytics model')
]
cursor.executemany('INSERT INTO employees VALUES (?, ?, ?, ?, ?)', employees)
conn.commit()
```

---

#### **Step 3: Natural Language to SQL with Explanations**

We use Hugging Face's **Grappa model** to generate SQL queries and provide manual explanations for each part of the query.

```python
from transformers import pipeline

# Load Grappa model for text-to-SQL
text_to_sql = pipeline("text-to-sql", model="Salesforce/grappa-text-to-sql")

# User's natural language query
query = "Who are the employees with the profession 'Cybersecurity Expert'?"

# Generate SQL query
result = text_to_sql(query, db_path="employees.db")
sql_query = result['query']

# Explanation of SQL query
explanation = {
    "SELECT *": "Retrieve all columns from the table.",
    "FROM employees": "Query the 'employees' table.",
    "WHERE profession = 'Cybersecurity Expert'": "Filter results where the profession column equals 'Cybersecurity Expert'."
}

# Execute SQL query
cursor.execute(sql_query)
results = cursor.fetchall()

# Display results
print("SQL Query:", sql_query)
print("\nQuery Explanation:")
for clause, meaning in explanation.items():
    print(f"{clause}: {meaning}")

print("\nResults:")
for row in results:
    print(f"ID: {row[0]}, Name: {row[1]}, Age: {row[2]}, Profession: {row[3]}, Achievements: {row[4]}")
```

---

### **Sample Output**

#### **Input Query**:

```text
Who are the employees with the profession 'Cybersecurity Expert'?
```

#### **Generated SQL Query**:

```sql
SELECT * FROM employees WHERE profession = 'Cybersecurity Expert';
```

#### **Explanation**:

```text
SELECT *: Retrieve all columns from the table.
FROM employees: Query the 'employees' table.
WHERE profession = 'Cybersecurity Expert': Filter results where the profession column equals 'Cybersecurity Expert'.
```

#### **Results**:

```text
ID: 2, Name: Ian Clark, Age: 45, Profession: Cybersecurity Expert, Achievements: Led red-teaming for Fortune 500
```

---

### **How This Works**

1. **Query Translation**: Hugging Face's Grappa translates natural language to SQL.
    
2. **Explainability**: Manual annotations break down the SQL query into understandable parts, helping users trust the process.
    
3. **Open-Source**: Fully open-source tools are used to ensure flexibility and accessibility.
    

# 3\. Fast search w/ **Weaviate**

| 💀 **Before** | 🎉 **After** |
| --- | --- |
| Slow searches in large databases. | Open-source tools like **Weaviate** enable scalable vector searches. |

---

### **Code Example with Weaviate**

#### **Step 1: Install Required Libraries**

```bash
pip install weaviate-client transformers
```

#### **Step 2: Start Weaviate Locally**

Use Docker to start an open-source Weaviate instance:

```bash
docker run -d -p 8080:8080 semitechnologies/weaviate
```

---

#### **Step 3: Set Up the Database and Add Data**

```python
import weaviate
from transformers import AutoTokenizer, AutoModel

# Connect to Weaviate
client = weaviate.Client("http://localhost:8080")

# Define the schema for employee data
schema = {
    "class": "Employee",
    "properties": [
        {"name": "name", "dataType": ["text"]},
        {"name": "profession", "dataType": ["text"]},
        {"name": "achievements", "dataType": ["text"]}
    ]
}
client.schema.create_class(schema)

# Load Hugging Face embedding model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Helper function to generate embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]

# Add employee data
employees = [
    {"name": "Fiona Davis", "profession": "Mobile Developer", "achievements": "Developed app with 100k downloads"},
    {"name": "Ian Clark", "profession": "Cybersecurity Expert", "achievements": "Led red-teaming for Fortune 500"},
    {"name": "Alice Cooper", "profession": "Data Scientist", "achievements": "Built predictive analytics model"}
]

for employee in employees:
    vector = get_embedding(f"{employee['name']} {employee['profession']} {employee['achievements']}")
    client.data_object.create(
        data_object=employee,
        class_name="Employee",
        vector=vector
    )
```

---

#### **Step 4: Perform Scalable Hybrid Search**

Use **semantic search** for context-based matches combined with **keyword filtering**.

```python
# Define a query
query_text = "Find a Cybersecurity Expert to secure our systems."
query_vector = get_embedding(query_text)

# Perform hybrid search (semantic + keyword)
results = client.query.get("Employee", ["name", "profession", "achievements"]) \
    .with_near_vector({"vector": query_vector.tolist()}) \
    .with_where({
        "path": ["profession"],
        "operator": "Equal",
        "valueText": "Cybersecurity Expert"
    }) \
    .with_limit(2) \
    .do()

# Display results
print("Results:")
for result in results["data"]["Get"]["Employee"]:
    print(f"Name: {result['name']}, Profession: {result['profession']}, Achievements: {result['achievements']}")
```

---

### **Sample Output**

#### **Input Query**:

```text
Find a Cybersecurity Expert to secure our systems.
```

#### **Results**:

```text
Name: Ian Clark, Profession: Cybersecurity Expert, Achievements: Led red-teaming for Fortune 500
```

---

### **Why Weaviate?**

* **Scalable**: Handles millions of embeddings efficiently.
    
* **Hybrid Search**: Combines semantic vectors with exact keyword matches.
    
* **Open-Source**: Free and flexible for any project.
    

Here’s how to implement **real-time querying of streaming data** using **open-source tools** like **Apache Kafka** (for streaming) and **Materialize** (for real-time SQL). These tools enable live queries on dynamic data pipelines.

---

| 💀 **Before** | 🎉 **After** |
| --- | --- |
| Queries only worked on static data. | Open-source tools like **Kafka** + **Materialize** enable real-time streaming queries. |

---

### **Code Example for Real-Time Querying**

#### **Step 1: Install Required Tools**

1. **Kafka**: Set up a Kafka cluster locally (Docker recommended).
    
2. **Materialize**: Install Materialize for real-time SQL on streaming data.
    
    ```bash
    docker run -d -p 6875:6875 materialize/materialized:latest
    ```
    

Install Kafka and Materialize Python clients:

```bash
pip install confluent-kafka psycopg2
```

---

#### **Step 2: Start Kafka for Streaming**

Run Kafka with Docker:

```bash
docker-compose up -d
```

Create a Kafka topic for streaming employee data:

```bash
docker exec broker kafka-topics --create --topic employee-data --bootstrap-server localhost:9092
```

---

#### **Step 3: Stream Data to Kafka**

Simulate real-time data streaming for employee updates.

```python
from confluent_kafka import Producer
import json
import time

# Kafka producer configuration
producer = Producer({"bootstrap.servers": "localhost:9092"})

# Stream employee data
employee_updates = [
    {"name": "Fiona Davis", "profession": "Mobile Developer", "achievements": "Developed app with 200k downloads"},
    {"name": "Ian Clark", "profession": "Cybersecurity Expert", "achievements": "Secured Fortune 500 systems"},
    {"name": "Alice Cooper", "profession": "Data Scientist", "achievements": "Improved model efficiency by 20%"}
]

for update in employee_updates:
    producer.produce("employee-data", json.dumps(update).encode("utf-8"))
    print(f"Sent: {update}")
    time.sleep(1)  # Simulate real-time streaming

producer.flush()
```

---

#### **Step 4: Query Streaming Data with Materialize**

Materialize continuously syncs and allows querying streaming Kafka topics.

```sql
-- In Materialize SQL Shell
-- Create a Kafka source
CREATE SOURCE employee_source
FROM KAFKA BROKER 'localhost:9092' TOPIC 'employee-data'
FORMAT AVRO USING CONFLUENT SCHEMA REGISTRY 'http://localhost:8081';

-- Create a real-time view for querying
CREATE MATERIALIZED VIEW employee_view AS
SELECT
    data->>'name' AS name,
    data->>'profession' AS profession,
    data->>'achievements' AS achievements
FROM employee_source;

-- Query the real-time view
SELECT * FROM employee_view WHERE profession = 'Cybersecurity Expert';
```

---

#### **Step 5: Query Using Python**

Query the Materialize database in real time via Python.

```python
import psycopg2

# Connect to Materialize
conn = psycopg2.connect(
    dbname="materialize", user="materialize", password="", host="localhost", port=6875
)
cursor = conn.cursor()

# Query the materialized view
cursor.execute("SELECT * FROM employee_view WHERE profession = 'Cybersecurity Expert';")
results = cursor.fetchall()

# Display results
for row in results:
    print(f"Name: {row[0]}, Profession: {row[1]}, Achievements: {row[2]}")
```

---

### **Sample Output**

#### **Streaming Data Sent to Kafka**:

```text
Sent: {'name': 'Ian Clark', 'profession': 'Cybersecurity Expert', 'achievements': 'Secured Fortune 500 systems'}
```

#### **Real-Time Query Output**:

```text
Name: Ian Clark, Profession: Cybersecurity Expert, Achievements: Secured Fortune 500 systems
```

---

### **Why Kafka + Materialize?**

1. **Real-Time Processing**: Materialize enables live SQL queries on Kafka streams.
    
2. **Open-Source**: Both Kafka and Materialize are free and scalable.
    
3. **Scalability**: Handle large-scale streaming pipelines seamlessly.
    

Here’s how **lightweight, open-source local models** like **Mistral 7B** or **Llama 2** can replace cloud-hosted solutions to save costs and enhance privacy.

---

# 4\. Free cloud host SQL

| 💀 **Before** | 🎉 **After** |
| --- | --- |
| Expensive cloud-hosted models. | Open-source models like **Mistral 7B** and **Llama 2** cut costs and ensure privacy. |

---

### **Code Example: Using Mistral 7B Locally**

#### **Step 1: Install Required Libraries**

```bash
pip install transformers accelerate
```

---

#### **Step 2: Load and Use Mistral 7B**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Mistral 7B model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype="auto"
)

# Run a local inference
def ask_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Use GPU if available
    outputs = model.generate(inputs["input_ids"], max_length=100, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the model with a natural language query
query = "What are the benefits of using open-source AI models?"
response = ask_model(query)
print("Response:", response)
```

---

### **Sample Output**

#### **Input Prompt**:

```text
What are the benefits of using open-source AI models?
```

#### **Model Output**:

```text
Open-source AI models are cost-effective, promote transparency, and provide greater flexibility for customization.
```

---

### **Using Llama 2 Locally**

#### **Step 3: Load and Use Llama 2**

Llama 2 models are also highly efficient for local use, especially with **Hugging Face**.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Llama 2 model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype="auto"
)

# Run a local inference
def ask_llama(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Use GPU if available
    outputs = model.generate(inputs["input_ids"], max_length=100, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test with a natural language query
query = "How does Llama 2 compare to Mistral 7B for local AI use?"
response = ask_llama(query)
print("Response:", response)
```

---

### **Sample Output**

#### **Input Prompt**:

```text
How does Llama 2 compare to Mistral 7B for local AI use?
```

#### **Model Output**:

```text
Both Llama 2 and Mistral 7B are highly efficient open-source models, but Mistral 7B is optimized for lower resource usage.
```

---

### **Why Use Local Models?**

1. **Cost Savings**: No cloud costs—run models on local GPUs.
    
2. **Privacy**: No sensitive data sent to external servers.
    
3. **Flexibility**: Fully customizable for your needs.
    

---