---
title: "Why JSON object is super useful"
datePublished: Thu Dec 12 2024 10:58:14 GMT+0000 (Coordinated Universal Time)
cuid: cm4l7hbkt002q09la8011f5gr
slug: why-json-object-is-super-useful
tags: json, sqlite, regex, jsonobject, llm

---

Here are 4 **specific before-and-after demonstrations** to clarify how JSON makes working with LLMs more manageable and efficient.

---

### **1.Scenario: Chatbot for Restaurant Reservations**

The customer sends a message:  
*"Hi, I'd like to book a table for 4 people at 7 PM this Friday at La Bella Vita. My contact number is 555-1234."*

---

### **1️⃣ Without JSON (Unstructured Text Output)**

The LLM might return something like this:

*"Got it! I’ve reserved a table for 4 people at La Bella Vita this Friday at 7 PM. Your contact number is 555-1234."*

As a developer, if you need to extract details programmatically (e.g., to send them to the restaurant's booking system), you’d have to **write custom code to parse the text**:

```python
# Unstructured output parsing
import re

response = "Got it! I’ve reserved a table for 4 people at La Bella Vita this Friday at 7 PM. Your contact number is 555-1234."

# Extract details using regex
party_size = re.search(r"for (\d+) people", response).group(1)
time = re.search(r"at ([\d\sAPMapm]+)", response).group(1)
restaurant = re.search(r"at ([\w\s]+) this", response).group(1)
contact = re.search(r"\d{3}-\d{4}", response).group(0)

print(party_size, time, restaurant, contact)
# Output: 4, 7 PM, La Bella Vita, 555-1234
```

This is:

* **Hard to maintain** if the output format changes.
    
* **Error-prone**, especially for edge cases like missing details.
    

---

### **2️⃣ With JSON (Structured Output)**

If the LLM returns JSON like this:

```json
{
  "party_size": 4,
  "time": "19:00",
  "date": "2024-12-15",
  "restaurant": "La Bella Vita",
  "contact_number": "555-1234"
}
```

Extracting the information becomes **simple and robust**:

```python
import json

# Example JSON response
llm_response = '''
{
  "party_size": 4,
  "time": "19:00",
  "date": "2024-12-15",
  "restaurant": "La Bella Vita",
  "contact_number": "555-1234"
}
'''

# Parse JSON response
reservation = json.loads(llm_response)

# Access data directly
party_size = reservation["party_size"]
time = reservation["time"]
restaurant = reservation["restaurant"]
contact = reservation["contact_number"]

print(f"Table for {party_size} at {restaurant} on {reservation['date']} at {time}. Contact: {contact}.")
```

Output:  
*"Table for 4 at La Bella Vita on 2024-12-15 at 19:00. Contact: 555-1234."*

---

### **Key Advantages of JSON**

1. **Consistency**: JSON structure is predictable, making it easy to extract fields.
    
2. **Maintainability**: No need to rewrite parsing logic if the text output changes.
    
3. **Integration**: JSON can be directly sent to other APIs or stored in a database.
    

Let’s break it down with another example to show **why JSON’s consistent data schema** makes life easier, especially for handling multiple LLM calls or integrating with databases.

---

### **2\. Scenario: Multi-step Workflow for Job Applications (Sqlite)**

You’re building an app that interacts with an LLM in multiple steps:

1. **Extract job details from a posting.**
    
2. **Generate a tailored cover letter.**
    
3. **Store results in a database for future use.**
    

---

### **1️⃣ Without JSON (Inconsistent Format)**

If the LLM outputs text with no enforced schema, each response might vary:

#### Call 1: Extracting job details

*"The job title is Data Scientist at ABC Corp. The salary is $120,000 per year. The location is remote."*

#### Call 2: Generating a cover letter

*"Dear Hiring Manager, I am excited to apply for the Data Scientist role at ABC Corp..."*

If the format changes slightly in a future call: *"Data Scientist role available at ABC Corp! Offering $120,000/year, fully remote."*

Now, you need **custom parsing logic** for each response, making your code:

* **Complex**: Different cases for extracting job title, salary, etc.
    
* **Error-prone**: If the LLM outputs something unexpected, it breaks your pipeline.
    

---

### **2️⃣ With JSON (Consistent Schema)**

By enforcing JSON output, each response follows a predictable structure, regardless of the content.

#### Call 1: Extracting job details

```json
{
  "job_title": "Data Scientist",
  "company": "ABC Corp",
  "salary": 120000,
  "location": "remote"
}
```

#### Call 2: Generating a cover letter

```json
{
  "cover_letter": "Dear Hiring Manager, I am excited to apply for the Data Scientist role at ABC Corp..."
}
```

You now have:

1. **Predictable fields** (`job_title`, `salary`, etc.) for every response.
    
2. **Seamless integration** with your database or other systems.
    

---

### **Database Integration**

Here’s how JSON makes storing and querying data easy:

#### Example: Storing the output in a database

```python
import sqlite3
import json

# Connect to the database
conn = sqlite3.connect('job_applications.db')
cursor = conn.cursor()

# Create a table
cursor.execute('''
CREATE TABLE IF NOT EXISTS jobs (
    job_title TEXT,
    company TEXT,
    salary INTEGER,
    location TEXT
)
''')

# Insert JSON data into the database
job_details = '''
{
  "job_title": "Data Scientist",
  "company": "ABC Corp",
  "salary": 120000,
  "location": "remote"
}
'''
data = json.loads(job_details)
cursor.execute('''
INSERT INTO jobs (job_title, company, salary, location)
VALUES (?, ?, ?, ?)
''', (data["job_title"], data["company"], data["salary"], data["location"]))

conn.commit()
conn.close()
```

#### Benefits:

1. **No parsing headaches**: Data is stored exactly as structured in the JSON.
    
2. **Easy queries**: Want to find all remote jobs paying $100,000+? Simple SQL query:
    
    ```sql
    SELECT * FROM jobs WHERE location = 'remote' AND salary > 100000;
    ```
    

---

### **Key Takeaways**:

* JSON enforces a predictable structure, ensuring every LLM call produces data in the same format.
    
* Consistency simplifies:
    
    * **Multi-step workflows**: No need to worry about varying outputs.
        
    * **Database integration**: Directly map JSON fields to database columns.
        
* Future-proof: If you add new features or fields, you can expand the JSON schema without breaking existing workflows.
    

Let’s explore **Chain of Thought (CoT) and Reasoning** with **JSON** to show how it simplifies building complex LLM applications.

---

### **3.Scenario: Multi-step Problem Solving (Calculating Loan Eligibility)**

Imagine you're building an app to calculate loan eligibility based on user inputs. The process requires:

1. Verifying the user's **income**.
    
2. Checking their **credit score**.
    
3. Calculating the **maximum loan amount** they can afford.
    

---

### **1️⃣ Without JSON (Unstructured Reasoning)**

The LLM might return something like this:

*"Step 1: The user's monthly income is $5,000. Step 2: The credit score is 750, which is excellent. Step 3: The maximum loan amount is calculated as $5,000 12 5 = $300,000."*

You now have to parse this text and break it into steps programmatically:

```python
import re

response = "Step 1: The user's monthly income is $5,000. Step 2: The credit score is 750, which is excellent. Step 3: The maximum loan amount is calculated as $5,000 * 12 * 5 = $300,000."

# Extract values using regex
income = int(re.search(r"monthly income is \$(\d+)", response).group(1))
credit_score = int(re.search(r"credit score is (\d+)", response).group(1))
loan_amount = int(re.search(r"maximum loan amount .*?= \$(\d+)", response).group(1))

print(income, credit_score, loan_amount)
# Output: 5000, 750, 300000
```

This approach is:

* **Error-prone**: Parsing breaks if the response format changes.
    
* **Hard to debug**: Long text makes it difficult to trace each step.
    

---

### **2️⃣ With JSON (Structured Chain of Thought)**

If the LLM outputs a JSON-formatted reasoning chain, it becomes much clearer:

```json
{
  "steps": [
    {
      "step": 1,
      "description": "Verify user's monthly income",
      "result": 5000
    },
    {
      "step": 2,
      "description": "Check user's credit score",
      "result": 750
    },
    {
      "step": 3,
      "description": "Calculate maximum loan amount",
      "formula": "monthly_income * 12 * 5",
      "result": 300000
    }
  ]
}
```

Now you can extract the reasoning process step-by-step:

```python
import json

# Example JSON response
llm_response = '''
{
  "steps": [
    {
      "step": 1,
      "description": "Verify user's monthly income",
      "result": 5000
    },
    {
      "step": 2,
      "description": "Check user's credit score",
      "result": 750
    },
    {
      "step": 3,
      "description": "Calculate maximum loan amount",
      "formula": "monthly_income * 12 * 5",
      "result": 300000
    }
  ]
}
'''

# Parse JSON
reasoning_chain = json.loads(llm_response)

# Iterate through steps
for step in reasoning_chain["steps"]:
    print(f"Step {step['step']}: {step['description']}")
    print(f"Result: {step['result']}")
```

**Output:**

```python
Step 1: Verify user's monthly income
Result: 5000

Step 2: Check user's credit score
Result: 750

Step 3: Calculate maximum loan amount
Result: 300000
```

---

### **Why JSON is Ideal for CoT Reasoning**

1. **Readable and Debuggable**: Each step is clearly documented, making it easy to follow the logic.
    
2. **Machine-Processable**: JSON allows you to access specific steps, results, or formulas directly.
    
3. **Future-Proof**: New steps can be added without breaking the structure (e.g., adding a debt-to-income check).
    

---

### **Use Case Ideas**

1. **Math Tutors**: Break down equations step-by-step for student learning.
    
2. **Code Analysis**: Debug programming errors by outputting step-by-step reasoning.
    
3. **Medical Diagnosis**: Explain reasoning behind a diagnosis in stages.
    

---

Let’s explore **Error Handling and Validation** using **JSON schemas** to ensure LLM outputs are in the correct format.

---

### **4.Scenario: Verifying LLM Outputs for a Product Catalog**

Imagine you’re building an app that uses an LLM to populate a product catalog. Each product must have:

1. A **name** (string).
    
2. A **price** (number).
    
3. A **category** (string).
    

If the LLM produces incorrect or missing fields, your app could break. JSON schemas help catch such issues early.

---

### **1️⃣ Without Validation**

Here’s what the LLM outputs:

```json
{
  "name": "Wireless Mouse",
  "price": "twenty dollars",  // Should be a number!
  "category": "Electronics"
}
```

Without validation, your app might fail when processing the `"price"` field because `"twenty dollars"` is not a number. Debugging the issue could become tedious.

---

### **2️⃣ With JSON Schema Validation**

Define a **JSON schema** that specifies the expected structure of the LLM output:

#### JSON Schema Example

```json
{
  "type": "object",
  "properties": {
    "name": { "type": "string" },
    "price": { "type": "number" },
    "category": { "type": "string" }
  },
  "required": ["name", "price", "category"]
}
```

Using a JSON validation library, you can validate the LLM’s output against this schema.

#### Python Validation Example

```python
import json
from jsonschema import validate, ValidationError

# JSON schema definition
product_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "number"},
        "category": {"type": "string"}
    },
    "required": ["name", "price", "category"]
}

# Example LLM output
llm_response = '''
{
  "name": "Wireless Mouse",
  "price": "twenty dollars",  // Error: Not a number
  "category": "Electronics"
}
'''

# Parse LLM output and validate
try:
    product_data = json.loads(llm_response)
    validate(instance=product_data, schema=product_schema)
    print("Validation passed! Data is valid.")
except ValidationError as e:
    print("Validation failed:", e.message)
```

#### Output:

```python
Validation failed: 'twenty dollars' is not of type 'number'
```

---

### **How JSON Schema Helps**

1. **Catch Issues Early**: Detect incorrect or missing fields immediately.
    
2. **Ensure Consistency**: Enforce a predictable structure for LLM outputs.
    
3. **Reduce Debugging Time**: Validation pinpoints the exact issue.
    

---

### **Use Case Ideas**

1. **Chatbots**: Validate structured responses (e.g., booking details).
    
2. **E-commerce**: Ensure product data (name, price, category) is correct before adding it to your database.
    
3. **Finance**: Validate numerical outputs (e.g., interest rates, loan amounts) to prevent errors.
    

---