---
title: "Python Automation#1: 🧽Data Cleaning w/janitor, pandas-profiling, dataprep, pandas"
seoTitle: "Python Automation#1: 🧽Data Cleaning w/janitor, pandas-profiling, data"
seoDescription: "Python Automation#1: 🧽Data Cleaning w/janitor, pandas-profiling, dataprep, pandas"
datePublished: Fri Dec 06 2024 11:40:22 GMT+0000 (Coordinated Universal Time)
cuid: cm4cocdx6000509idh2fe1bj7
slug: python-automation1-data-cleaning-wjanitor-pandas-profiling-dataprep-pandas
tags: pandas, dataprep, pandas-profiling, janitor

---

### 1\. **Clean Messy Column Names (**`janitor.clean_names`)

```python
import pandas as pd
import janitor

# Sample DataFrame
df = pd.DataFrame({"Col 1 ": [1, 2], "COL@2": [3, 4]})

# Clean column names
df = janitor.clean_names(df)
print(df)
```

**Output:**

```python
   col_1  col_2
0      1      3
1      2      4
```

---

### 2\. **Remove Empty Rows/Columns (**`janitor.remove_empty`)

```python
import pandas as pd
import janitor

# Adding an empty row and column
df = pd.DataFrame({"Col 1 ": [1, 2, None], "COL@2": [3, 4, None]})
df["Empty"] = None

# Remove empty rows and columns
df = janitor.remove_empty(df)
print(df)
```

**Output:**

```python
   Col 1   COL@2
0    1.0    3.0
1    2.0    4.0
```

---

### 3\. **Handle Missing Values (**`pandas.fillna`)

```python
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({"A": [1, 2, None], "B": [None, 4, 5]})

# Handle missing values: Fill with mean
df_filled = pd.DataFrame.fillna(df, df.mean())
print(df_filled)
```

**Output:**

```python
     A    B
0  1.0  4.5
1  2.0  4.0
2  1.5  5.0
```

---

### 4\. **Generate a Quick Data Report (**`pandas_profiling.ProfileReport`)

```python
import pandas as pd
from pandas_profiling import ProfileReport

# Sample DataFrame
df = pd.DataFrame({"A": [1, 2, None], "B": [3, 4, 5]})

# Generate report
profile = ProfileReport(df, title="Quick Data Report")
profile.to_file("report.html")
```

**Output:** An HTML report (`report.html`) is generated with detailed insights.

---

### 5\. **Explore Data Structure Visually (**`dataprep.eda.create_report`)

```python
import pandas as pd
from dataprep.eda import create_report

# Sample DataFrame
df = pd.DataFrame({"A": [1, 2, None], "B": [3, 4, 5]})

# Generate visual report
report = create_report(df)
report.show_browser()
```

**Output:** A visual report is displayed in your browser.

---

### 6\. **Complex Filtering/Grouping (**`pandas.query`, `pandas.groupby`)

```python
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({"Category": ["A", "B", "A"], "Value": [10, 20, 30]})

# Group by and calculate mean
grouped_df = pd.DataFrame.groupby(df, "Category").mean()
print(grouped_df)
```

**Output:**

```python
          Value
Category       
A          20.0
B          20.0
```

---

### 7\. **Add Computed Columns (**`janitor.add_column`)

```python
import pandas as pd
import janitor

# Sample DataFrame
df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

# Add a computed column
df = janitor.add_column(df, "C", lambda x: x["A"] + x["B"])
print(df)
```

**Output:**

```python
   A  B   C
0  1  3   4
1  2  4   6
```

---

### 8\. **Merge or Join Datasets (**`pandas.merge`)

```python
import pandas as pd

# Sample DataFrames
df1 = pd.DataFrame({"ID": [1, 2], "Value1": [10, 20]})
df2 = pd.DataFrame({"ID": [1, 2], "Value2": [30, 40]})

# Merge DataFrames
merged_df = pd.merge(df1, df2, on="ID")
print(merged_df)
```

**Output:**

```python
   ID  Value1  Value2
0   1      10      30
1   2      20      40
```

---

### 9\. **Data Cleaning and Standardization (**`dataprep.clean`)

```python
import pandas as pd
from dataprep.clean import clean_headers

# Sample DataFrame
df = pd.DataFrame({"Col 1 ": [1, 2], "Col@2": [3, 4]})

# Clean headers
df = clean_headers(df)
print(df)
```

**Output:**

```python
   col_1  col_2
0      1      3
1      2      4
```

---

### 10\. **Quick Insights into Distributions (**`dataprep.eda.plot`)

```python
import pandas as pd
from dataprep.eda import plot

# Sample DataFrame
df = pd.DataFrame({"A": [1, 2, 3], "B": [3, 4, 5]})

# Plot distributions
plot(df)
```

**Output:** A distribution plot is displayed in your browser.

---