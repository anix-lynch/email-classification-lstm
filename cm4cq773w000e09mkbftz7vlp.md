---
title: "Python Automation #3: Data ingestion w/pandas, SQLAlchemy, dataprep.connector, petl"
seoTitle: "Python Automation #3: Data ingestion w/pandas, SQLAlchemy, dataprep"
seoDescription: "Python Automation #3: Data ingestion w/pandas, SQLAlchemy, dataprep.connector, petl"
datePublished: Fri Dec 06 2024 12:32:19 GMT+0000 (Coordinated Universal Time)
cuid: cm4cq773w000e09mkbftz7vlp
slug: python-automation-3-data-ingestion-wpandas-sqlalchemy-dataprepconnector-petl
tags: pandas, sqlalchemy, data-ingestion, dataprepconnector, petl

---

### 1\. **Read Data from CSV, Excel, or JSON (**[`pandas.read`](http://pandas.read)`_csv`, [`pandas.read`](http://pandas.read)`_excel`, [`pandas.read`](http://pandas.read)`_json`)

#### Read CSV:

```python
import pandas as pd

# Read data from CSV
df = pd.read_csv("data.csv")
print(df.head())
```

**Output (Sample):**

```python
   ID  Name  Age
0   1  Alice   25
1   2    Bob   30
2   3  Carol   27
```

#### Read Excel:

```python
import pandas as pd

# Read data from Excel
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
print(df.head())
```

**Output (Sample):**

```python
   ID  Name  Age
0   1  Alice   25
1   2    Bob   30
2   3  Carol   27
```

#### Read JSON:

```python
import pandas as pd

# Read data from JSON
df = pd.read_json("data.json")
print(df.head())
```

**Output (Sample):**

```python
   ID  Name  Age
0   1  Alice   25
1   2    Bob   30
2   3  Carol   27
```

---

### 2\. **Load Data from Relational Databases (**`SQLAlchemy/`[`pandas.read`](http://pandas.read)`_sql`)

```python
import pandas as pd
from sqlalchemy import create_engine

# Create database connection
engine = create_engine("sqlite:///example.db")

# Load data from a SQL table
df = pd.read_sql("SELECT * FROM users", engine)
print(df.head())
```

**Output (Sample):**

```python
   ID  Name  Age
0   1  Alice   25
1   2    Bob   30
2   3  Carol   27
```

---

### 3\. **Extract Data from APIs (**[`dataprep.connector.read`](http://dataprep.connector.read)`_connector`)

```python
from dataprep.connector import Connector

# Connect to an API
c = Connector("example_api")

# Extract data from the API
df = c.query("endpoint", limit=10)
print(df.head())
```

**Output (Sample):**

```python
   ID    Name    Age
0   1   Alice    25
1   2     Bob    30
2   3   Carol    27
```

---

### 4\. **Stream Large Datasets (**`petl.fromcsv`)

```python
import petl as etl

# Stream large datasets from CSV
table = etl.fromcsv("large_data.csv")

# Preview first few rows
print(etl.head(table, 5))
```

**Output (Sample):**

```python
+------+-------+-----+
|  ID  | Name  | Age |
+------+-------+-----+
|    1 | Alice |  25 |
|    2 | Bob   |  30 |
|    3 | Carol |  27 |
+------+-------+-----+
```

---

### 5\. **Write to Relational Databases (**`SQLAlchemy/`[`pandas.to`](http://pandas.to)`_sql`)

```python
import pandas as pd
from sqlalchemy import create_engine

# Create database connection
engine = create_engine("sqlite:///example.db")

# Sample DataFrame
df = pd.DataFrame({"ID": [1, 2], "Name": ["Alice", "Bob"], "Age": [25, 30]})

# Write data to a SQL table
df.to_sql("users", engine, if_exists="replace", index=False)
print("Data written successfully!")
```

**Output:**

```python
Data written successfully!
```

---

### 6\. **Connect to Cloud Databases (**[`dataprep.connector.read`](http://dataprep.connector.read)`_connector`)

```python
from dataprep.connector import Connector

# Connect to a cloud database (e.g., AWS RDS)
c = Connector("aws_rds")

# Query data
df = c.query("database_name.table_name", limit=10)
print(df.head())
```

**Output (Sample):**

```python
   ID  Name  Age
0   1  Alice   25
1   2    Bob   30
2   3  Carol   27
```

---

### 7\. **Transform While Loading (**`petl.transform`)

```python
import petl as etl

# Load data and apply transformation
table = etl.fromcsv("data.csv")
transformed_table = etl.addfield(table, "FullName", lambda row: row.Name.upper())
print(etl.head(transformed_table, 5))
```

**Output (Sample):**

```python
+------+-------+-----+----------+
|  ID  | Name  | Age | FullName |
+------+-------+-----+----------+
|    1 | Alice |  25 | ALICE    |
|    2 | Bob   |  30 | BOB      |
+------+-------+-----+----------+
```

---

### 8\. **Batch Processing for Ingestion (**`petl.tocsv`)

```python
import petl as etl

# Stream and write in batches
table = etl.fromcsv("large_data.csv")
etl.tocsv(table, "processed_data.csv", batchsize=100)
print("Batch processing completed!")
```

**Output:**

```python
Batch processing completed!
```

---

### 9\. **Export Data to Multiple Formats (**[`pandas.to`](http://pandas.to)`_csv`, [`pandas.to`](http://pandas.to)`_excel`, [`pandas.to`](http://pandas.to)`_json`)

#### Export to CSV:

```python
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({"ID": [1, 2], "Name": ["Alice", "Bob"], "Age": [25, 30]})

# Export to CSV
df.to_csv("output.csv", index=False)
print("Exported to CSV!")
```

#### Export to Excel:

```python
df.to_excel("output.xlsx", index=False)
print("Exported to Excel!")
```

#### Export to JSON:

```python
df.to_json("output.json", orient="records", indent=2)
print("Exported to JSON!")
```

**Output:**

```python
Exported to CSV!
Exported to Excel!
Exported to JSON!
```

---

### 10\. **Handle API Rate Limits (**`dataprep.connector.rate_limit`)

```python
from dataprep.connector import Connector

# API connection with rate-limiting
c = Connector("example_api")

# Set rate limit (e.g., 1 request per second)
df = c.query("endpoint", limit=10, rate_limit=1)
print(df.head())
```

**Output (Sample):**

```python
   ID  Name  Age
0   1  Alice  25
1   2    Bob  30
2   3  Carol  27
```

---

### 11\. **SQL Query Integration (**`SQLAlchemy/`[`pandas.read`](http://pandas.read)`_sql`)

```python
import pandas as pd
from sqlalchemy import create_engine

# Create database connection
engine = create_engine("sqlite:///example.db")

# Execute SQL query and load data
df = pd.read_sql("SELECT * FROM users WHERE Age > 25", engine)
print(df)
```

**Output:**

```python
   ID    Name  Age
0   2     Bob   30
1   3   Carol   27
```

---

### 12\. **Transform Data from APIs (**[`dataprep.connector.read`](http://dataprep.connector.read)`_connector`)

```python
from dataprep.connector import Connector

# Connect to an API
c = Connector("example_api")

# Transform and load data
df = c.query("endpoint", limit=5)
df["Transformed_Name"] = df["Name"].str.upper()
print(df)
```

**Output:**

```python
   ID    Name  Age Transformed_Name
0   1   Alice   25           ALICE
1   2     Bob   30             BOB
```

---

### 13\. **Clean and Filter During Ingestion (**[`petl.select`](http://petl.select), `petl.clean_headers`)

```python
import petl as etl

# Load and clean data
table = etl.fromcsv("data.csv")
cleaned_table = etl.clean_headers(table)
filtered_table = etl.select(cleaned_table, lambda row: int(row.Age) > 25)
print(etl.head(filtered_table, 5))
```

**Output:**

```python
+------+-------+-----+
|  id  | name  | age |
+------+-------+-----+
|    2 | Bob   |  30 |
|    3 | Carol |  27 |
+------+-------+-----+
```

---

### 14\. **Handle Flat-File Ingestion (**`petl.fromcsv`, `petl.fromtsv`)

#### CSV:

```python
import petl as etl

# Load data from CSV
table = etl.fromcsv("data.csv")
print(etl.head(table, 5))
```

**Output:**

```python
+------+-------+-----+
|  ID  | Name  | Age |
+------+-------+-----+
|    1 | Alice |  25 |
|    2 | Bob   |  30 |
+------+-------+-----+
```

#### TSV:

```python
# Load data from TSV
table_tsv = etl.fromtsv("data.tsv")
print(etl.head(table_tsv, 5))
```

**Output (TSV):**

```python
+------+-------+-----+
|  ID  | Name  | Age |
+------+-------+-----+
|    1 | Alice |  25 |
|    2 | Bob   |  30 |
+------+-------+-----+
```

---

### 15\. **Parallel Ingestion (**[`dataprep.connector.read`](http://dataprep.connector.read)`_connector`)

```python
from dataprep.connector import Connector

# Connect to API with parallel ingestion
c = Connector("example_api")

# Perform parallel queries
df = c.query("endpoint", limit=10, parallel=True)
print(df.head())
```

**Output:**

```python
   ID    Name  Age
0   1   Alice   25
1   2     Bob   30
2   3   Carol   27
```

---

### 16\. **ORM Integration (**`SQLAlchemy`)

```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Define ORM base
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)

# Create engine and session
engine = create_engine("sqlite:///example.db")
Session = sessionmaker(bind=engine)
session = Session()

# Query data using ORM
users = session.query(User).filter(User.age > 25).all()
for user in users:
    print(user.name, user.age)
```

**Output:**

```python
Bob 30
Carol 27
```

---

### 17\. **Integrate Data Pipelines (**`petl`, `SQLAlchemy`)

```python
import petl as etl
from sqlalchemy import create_engine

# Load data from CSV
table = etl.fromcsv("data.csv")

# Write to SQL database
engine = create_engine("sqlite:///example.db")
etl.todb(table, engine, "users")
print("Data pipeline integrated!")
```

**Output:**

```python
Data pipeline integrated!
```

---

### 18\. **Preview Data During Ingestion (**`petl.head`, `pandas.head`)

#### Using `petl`:

```python
import petl as etl

# Preview data using petl
table = etl.fromcsv("data.csv")
print(etl.head(table, 5))
```

**Output:**

```python
+------+-------+-----+
|  ID  | Name  | Age |
+------+-------+-----+
|    1 | Alice |  25 |
|    2 | Bob   |  30 |
+------+-------+-----+
```

#### Using `pandas`:

```python
import pandas as pd

# Preview data using pandas
df = pd.read_csv("data.csv")
print(df.head())
```

**Output:**

```python
   ID    Name  Age
0   1   Alice   25
1   2     Bob   30
2   3   Carol   27
```

---

### 19\. **Connect to Custom APIs (**`dataprep.connector`)

```python
from dataprep.connector import Connector

# Connect to custom API
c = Connector("custom_api")

# Fetch data
df = c.query("custom_endpoint", params={"key": "value"}, limit=10)
print(df.head())
```

**Output:**

```python
   ID    Name  Age
0   1   Alice   25
1   2     Bob   30
```

---

### 20\. **Write to Cloud Storage (e.g., S3) (**[`pandas.to`](http://pandas.to)`_csv`)

```python
import pandas as pd
import boto3

# Save DataFrame to CSV
df = pd.DataFrame({"ID": [1, 2], "Name": ["Alice", "Bob"], "Age": [25, 30]})
df.to_csv("output.csv", index=False)

# Upload to S3
s3 = boto3.client("s3")
s3.upload_file("output.csv", "your_bucket_name", "output.csv")
print("File uploaded to S3!")
```

**Output:**

```python
File uploaded to S3!
```

---