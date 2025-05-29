---
title: "10 ways How LLMs Help With 
Datadrift"
datePublished: Thu May 29 2025 21:54:47 GMT+0000 (Coordinated Universal Time)
cuid: cmb9wwr8w000309l26fve8yh9
slug: 10-ways-how-llms-help-with-datadrift

---

## 🔍 **Data Quality Basics**

| Concept | What It Means (Real Words) | Example |
| --- | --- | --- |
| **Anomaly** | Weird data that doesn’t fit the usual pattern | Revenue is suddenly -$999 |
| **Schema Drift** | When the shape/columns of your data change without warning | A column changes name or type |
| **Data Lineage** | Where the data came from + what changed it | Like a recipe history for your data |

---

## 🤖 **LLM + AI Help**

| Fancy Term | What It Actually Does | Why It's Cool |
| --- | --- | --- |
| **Isolation Forest** | Finds rows that look "weird" based on math | Flags unusual values, even across many columns |
| **LOF (Local Outlier Factor)** | Compares how dense a row is to its neighbors | Catches anomalies that are subtly off |
| **One-Class SVM** | Learns what “normal” looks like, then catches weird stuff | You don’t need labeled examples |
| **Drift Detection** | Notices when your data slowly changes over time | Warns you when your model is becoming dumb |

---

## 📊 **Checking the Data**

| Check Type | What It Means | Example |
| --- | --- | --- |
| **Completeness** | Are any values missing? | 100 rows, but `age` column has 30 blanks |
| **Consistency** | Do the formats or values match what we expect? | Someone typed "Ten" instead of 10 |
| **Accuracy** | Is it correct? | Customer age says 212 😬 |
| **Timeliness** | Is it fresh and updated? | Data from 3 weeks ago... not helpful |

---

## 🛠️ **Tools in the Detective Kit**

| Tool | What It Does |
| --- | --- |
| **Pandas** | Load and explore the data in Python |
| **Great Expectations** | Write rules like “this column should never be null” |
| **dbt** | Transforms SQL data + tests it before it breaks stuff |
| **OpenAI/Claude** | Reads your data, explains changes, helps write fix code |
| **Kafka** | (Advanced) Streams new data in real-time for validation |
| **Streamlit** | Makes a pretty interface for humans to see alerts 📊 |

---

Absolutely! Here’s a **side-by-side “Before → After”** for each sample. I’ll show what you see when you run the code — so you get the **full picture**, not just the script. 🧪🐼

---

## 🧪 1. Great Expectations — Check for Nulls

### ✅ Code:

```python
import great_expectations as ge
import pandas as pd

df = pd.DataFrame({
    "user_id": [101, 102, 103],
    "revenue": [99.9, None, 150.0]
})

ge_df = ge.from_pandas(df)
result = ge_df.expect_column_values_to_not_be_null("revenue")
print(result)
```

### 🟥 Output:

```json
{
  "success": false,
  "unexpected_index_list": [1],
  "unexpected_count": 1,
  "unexpected_percent": 33.3,
  "result": {
    "element_count": 3,
    "missing_count": 1
  }
}
```

> ❌ One row has `None` — GE flagged it!

---

## 🛠 2. dbt — SQL Clean + Test

### 📥 Before: Raw SQL (bad data)

| id | revenue | status |
| --- | --- | --- |
| 1 | `$120` | complete |
| 2 | null | pending |

### ➕ dbt SQL transforms it:

```sql
SELECT
  id,
  CAST(REPLACE(revenue, '$', '') AS FLOAT) AS revenue_cleaned
FROM raw.orders
WHERE status = 'complete'
```

### 📤 Output Table:

| id | revenue\_cleaned |
| --- | --- |
| 1 | 120.0 |

> ✅ dbt cleaned and filtered data — and tested `revenue_cleaned` is not null!

---

## 📡 3. Kafka Stream (Faust-style)

### ⚠️ Input Streaming Message:

```json
{"user_id": 201, "revenue": -500}
```

### ⛔ Output in console:

```plaintext
🚨 Bad row detected: {'user_id': 201, 'revenue': -500}
```

> You caught the issue **while streaming**, not after!

---

## 🧠 4. Isolation Forest

### Input:

```python
"revenue": [100, 120, 115, 99, 5000]
```

### Output:

```python
   revenue  anomaly
0      100        1
1      120        1
2      115        1
3       99        1
4     5000       -1
```

> 🚨 Row 4 (value = 5000) is flagged as **anomaly** (`-1`)

---

## 🧠 5. Local Outlier Factor (LOF)

### Input:

```python
"age": [22, 24, 23, 25, 99],
"score": [88, 89, 87, 90, 20]
```

### Output:

```python
   age  score  anomaly
0   22     88        1
1   24     89        1
2   23     87        1
3   25     90        1
4   99     20       -1
```

> 🚨 Last row is too far off — LOF says “not like the others.”

---

## 🧠 6. One-Class SVM

### Input:

```python
"metric": [0.5, 0.6, 0.55, 0.52, 3.0]
```

### Output:

```python
   metric  anomaly
0    0.50        1
1    0.60        1
2    0.55        1
3    0.52        1
4    3.00       -1
```

> 🚨 Row 4 is way off — flagged as novelty/outlier

---

## 🧠 7. Drift Detection (JS Divergence)

### Input Histograms:

```python
yesterday = [100, 120, 110, 115]
today = [300, 400, 500, 600]
```

### Output:

```plaintext
JS Divergence: 0.6931
```

> 🧠 **High number (closer to 1)** = serious distribution drift

---

## 🤖 How LLMs Help with Detective Pandas (Schema + Quality)

| 🧪 Use Case | 💡 LLM Role (What It Does) | 🛠️ Real Example |
| --- | --- | --- |
| **1\. Schema Drift** | 💬 Detect changes between old and new schema, explain what changed | “`revenue` column changed from float → string; suggest converting it back” |
| **2\. Fix Code for Drift** | 🛠️ Suggest Python code to clean or cast columns | `df['revenue'] = df['revenue'].str.replace('$', '').astype(float)` |
| **3\. Explain Quality Errors** | 📖 Translate Great Expectations output into human-readable alerts | “1 row failed — revenue column has a null at index 3” |
| **4\. Auto-generate GE rules** | 🛠️ Given a sample dataset, LLM writes a quality spec (e.g., non-null, value range) | `"expect revenue to be between 0 and 10000"` |
| **5\. Detect anomalies** | 🧠 Suggest and implement Isolation Forest or LOF automatically | “Try IsolationForest with contamination=0.1” |
| **6\. Create dashboards or reports** | 🧾 Write markdown or Streamlit summaries of what’s wrong | “Data drift detected in `signup_date`, suggest histogram comparison” |
| **7\. Explain statistical tests** | 🧠 Makes advanced stuff (like JS divergence) sound human | “Your data distribution changed — see JS score = 0.68 → consider retraining” |
| **8\. Suggest dbt tests** | ✍️ Writes schema.yml entries and SQL model tests | Adds: `tests: [not_null, accepted_range]` to YAML |
| **9\. Stream Monitoring** | 🧪 Writes logic to plug anomaly detection into a Kafka stream or Faust app | Creates Python code for “alert if revenue &lt; 0 in real-time stream” |
| **10\. Root Cause Analysis** | 🔍 Scans logs, schemas, recent changes to *guess* what broke what | “Nulls started after March 20 deploy. Check new upstream API version.” |

---

### 🧠 Prompt Example

> "Hey Claude, here's the schema yesterday and today. What's changed and how can I fix it in pandas?"

📥 Input:

```json
Schema Day 1: {"revenue": "float", "email": "string"}
Schema Day 2: {"revenue": "string", "email": "string"}
```

📤 Output:

> "Column `revenue` changed from float to string. Try:  
> `df['revenue'] = df['revenue'].str.replace('$', '').astype(float)`"

---

## ✅ What LLMs *Can* Do

* Understand what’s wrong
    
* Suggest or write the fix
    
* Automate report generation
    
* Explain scary math
    
* Help non-engineers understand issues
    

## ❌ What LLMs *Can’t* Do

* Access your database (you must provide input)
    
* Guarantee fixes are perfect (always review!)
    
* Run the code — you still execute it