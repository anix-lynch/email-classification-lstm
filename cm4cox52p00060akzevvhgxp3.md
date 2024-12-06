---
title: "Python Automation #2: 🗳️ Data Transformation w/polars, pyjanitor, pandas, polars"
seoTitle: "Python Automation #2: 🗳️ Data Transformation w/polars, pyjanitor, pan"
seoDescription: "Python Automation #2: 🗳️ Data Transformation w/polars, pyjanitor, pandas, polars "
datePublished: Fri Dec 06 2024 11:56:30 GMT+0000 (Coordinated Universal Time)
cuid: cm4cox52p00060akzevvhgxp3
slug: python-automation-2-data-transformation-wpolars-pyjanitor-pandas-polars
tags: python, pandas, data-transformation, polars, pyjanitor

---

### 1\. **Convert Column Names to Snake Case (**`pyjanitor.clean_names`)

```python
import pandas as pd
import janitor

# Sample DataFrame
df = pd.DataFrame({"Column Name 1": [1, 2], "AnotherColumn": [3, 4]})

# Convert column names to snake_case
df = janitor.clean_names(df)
print(df)
```

**Output:**

```python
   column_name_1  another_column
0              1               3
1              2               4
```

---

### 2\. **Filter Rows by Condition (**`pandas.DataFrame.query`)

```python
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

# Filter rows where A > 1
filtered_df = pd.DataFrame.query(df, "A > 1")
print(filtered_df)
```

**Output:**

```python
   A  B
1  2  5
2  3  6
```

---

### 3\. **Pivot or Unpivot Data (**`pandas.melt`)

```python
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({"ID": [1, 2], "Jan": [100, 200], "Feb": [150, 250]})

# Unpivot (melt) the DataFrame
melted_df = pd.melt(df, id_vars=["ID"], var_name="Month", value_name="Sales")
print(melted_df)
```

**Output:**

```python
   ID Month  Sales
0   1   Jan    100
1   2   Jan    200
2   1   Feb    150
3   2   Feb    250
```

---

### 4\. **Group By and Aggregate (**`polars.DataFrame.groupby`)

```python
import polars as pl

# Sample DataFrame
df = pl.DataFrame({"Category": ["A", "B", "A"], "Value": [10, 20, 30]})

# Group by and calculate sum
grouped_df = df.groupby("Category").agg(pl.col("Value").sum().alias("Total_Value"))
print(grouped_df)
```

**Output:**

```python
shape: (2, 2)
┌───────────┬────────────┐
│ Category  │ Total_Value│
│ ---       │ ---        │
│ str       │ i64        │
├───────────┼────────────┤
│ A         │ 40         │
│ B         │ 20         │
└───────────┴────────────┘
```

---

### 5\. **Add Computed Columns (**`pyjanitor.add_column`)

```python
import pandas as pd
import janitor

# Sample DataFrame
df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

# Add computed column
df = janitor.add_column(df, "C", lambda x: x["A"] + x["B"])
print(df)
```

**Output:**

```python
   A  B  C
0  1  3  4
1  2  4  6
```

---

### 6\. **Handle Missing Data (**`pyjanitor.fill_missing`)

```python
import pandas as pd
import janitor

# Sample DataFrame
df = pd.DataFrame({"A": [1, None, 3], "B": [None, 2, 3]})

# Fill missing values with 0
df = janitor.fill_missing(df, value=0)
print(df)
```

**Output:**

```python
     A    B
0  1.0  0.0
1  0.0  2.0
2  3.0  3.0
```

---

### 7\. **Join or Merge Datasets (**`polars.DataFrame.join`)

```python
import polars as pl

# Sample DataFrames
df1 = pl.DataFrame({"ID": [1, 2], "Value1": [10, 20]})
df2 = pl.DataFrame({"ID": [1, 2], "Value2": [30, 40]})

# Join DataFrames
joined_df = df1.join(df2, on="ID", how="inner")
print(joined_df)
```

**Output:**

```python
shape: (2, 3)
┌─────┬────────┬────────┐
│ ID  │ Value1 │ Value2 │
│ --- │ ---    │ ---    │
│ i64 │ i64    │ i64    │
├─────┼────────┼────────┤
│ 1   │ 10     │ 30     │
│ 2   │ 20     │ 40     │
└─────┴────────┴────────┘
```

---

### 8\. **Reshape Data (Wide ↔ Long) (**`pandas.DataFrame.stack`)

```python
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({"ID": [1, 2], "Jan": [100, 200], "Feb": [150, 250]})

# Reshape to long format
reshaped_df = df.set_index("ID").stack().reset_index(name="Sales")
print(reshaped_df)
```

**Output:**

```python
   ID level_1  Sales
0   1     Jan    100
1   1     Feb    150
2   2     Jan    200
3   2     Feb    250
```

---

### 9\. **Chaining Transformations (**`polars.lazy`)

```python
import polars as pl

# Sample DataFrame
df = pl.DataFrame({"Category": ["A", "B", "A"], "Value": [10, 20, 30]})

# Chain transformations: Group by, aggregate, and sort
result = (
    df.lazy()
    .groupby("Category")
    .agg(pl.col("Value").sum().alias("Total_Value"))
    .sort("Total_Value", reverse=True)
    .collect()
)
print(result)
```

**Output:**

```python
shape: (2, 2)
┌───────────┬────────────┐
│ Category  │ Total_Value│
│ ---       │ ---        │
│ str       │ i64        │
├───────────┼────────────┤
│ A         │ 40         │
│ B         │ 20         │
└───────────┴────────────┘
```

---

### 10\. **Handle Large Datasets (**`polars.scan_csv`)

```python
import polars as pl

# Simulating reading a large dataset
df = pl.scan_csv("large_dataset.csv")

# Perform operations lazily
result = df.groupby("Category").agg(pl.col("Value").mean()).collect()
print(result)
```

**Output:**

```python
shape: (N, 2)
┌───────────┬────────────┐
│ Category  │ Value_mean │
│ ---       │ ---        │
│ str       │ f64        │
├───────────┼────────────┤
│ A         │ 25.5       │
│ B         │ 30.0       │
└───────────┴────────────┘
```

---

### 11\. **Add or Remove Rows/Columns (**`pandas.drop`, `pandas.append`)

```python
import pandas as pd
import janitor

# Sample DataFrame
df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

# Add a new column
df["C"] = [5, 6]

# Remove a column
df = pd.DataFrame.drop(df, columns=["A"])

print(df)
```

**Output:**

```python
   B  C
0  3  5
1  4  6
```

---

### 12\. **Sort Rows by Column (**`polars.DataFrame.sort`)

```python
import polars as pl

# Sample DataFrame
df = pl.DataFrame({"Name": ["Alice", "Bob"], "Score": [85, 90]})

# Sort rows by "Score"
sorted_df = df.sort("Score", reverse=False)
print(sorted_df)
```

**Output:**

```python
shape: (2, 2)
┌───────┬───────┐
│ Name  │ Score │
│ ---   │ ---   │
│ str   │ i64   │
├───────┼───────┤
│ Alice │ 85    │
│ Bob   │ 90    │
└───────┴───────┘
```

---

### 13\. **Apply Functions Row-Wise (**`pandas.DataFrame.apply`)

```python
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

# Apply a function row-wise
df["Sum"] = df.apply(lambda row: row["A"] + row["B"], axis=1)
print(df)
```

**Output:**

```python
   A  B  Sum
0  1  3    4
1  2  4    6
```

---

### 14\. **Transform Specific Data Types (**`pyjanitor.transform_columns`)

```python
import pandas as pd
import janitor

# Sample DataFrame
df = pd.DataFrame({"A": ["1", "2"], "B": ["3.5", "4.5"]})

# Transform data types
df = janitor.transform_columns(df, {
    "A": int,
    "B": float
})
print(df)
```

**Output:**

```python
   A    B
0  1  3.5
1  2  4.5
```

---

### 15\. **Concatenate Datasets (**`pandas.concat`, `polars.concat`)

#### Using Pandas:

```python
import pandas as pd

# Sample DataFrames
df1 = pd.DataFrame({"A": [1, 2]})
df2 = pd.DataFrame({"A": [3, 4]})

# Concatenate datasets
df = pd.concat([df1, df2], ignore_index=True)
print(df)
```

**Output:**

```python
   A
0  1
1  2
2  3
3  4
```

#### Using Polars:

```python
import polars as pl

# Sample DataFrames
df1 = pl.DataFrame({"A": [1, 2]})
df2 = pl.DataFrame({"A": [3, 4]})

# Concatenate datasets
df = pl.concat([df1, df2])
print(df)
```

**Output:**

```python
shape: (4, 1)
┌─────┐
│ A   │
│ i64 │
├─────┤
│ 1   │
│ 2   │
│ 3   │
│ 4   │
└─────┘
```

---

### 16\. **Efficient Column Selection (**[`polars.select`](http://polars.select))

```python
import polars as pl

# Sample DataFrame
df = pl.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})

# Select specific columns
selected_df = df.select(["A", "C"])
print(selected_df)
```

**Output:**

```python
shape: (2, 2)
┌─────┬─────┐
│ A   │ C   │
│ i64 │ i64 │
├─────┼─────┤
│ 1   │ 5   │
│ 2   │ 6   │
└─────┴─────┘
```

---

### 17\. **Split and Expand Strings (**`pandas.Series.str.split`)

```python
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({"Name": ["Alice Smith", "Bob Jones"]})

# Split and expand strings
df_split = df["Name"].str.split(" ", expand=True)
print(df_split)
```

**Output:**

```python
       0      1
0  Alice  Smith
1    Bob  Jones
```

---

### 18\. **Lazy Evaluation for Transformations (**`polars.scan_csv`)

```python
import polars as pl

# Simulate loading a large dataset lazily
df = pl.scan_csv("large_dataset.csv")

# Perform operations
result = df.filter(pl.col("Category") == "A").select(["Value"]).collect()
print(result)
```

**Output:**

```python
shape: (N, 1)
┌───────┐
│ Value │
│ i64   │
├───────┤
│ ...   │
└───────┘
```

---

### 19\. **One-Liner Transformations (**`pyjanitor.clean_names`)

```python
import pandas as pd
import janitor

# Sample DataFrame
df = pd.DataFrame({"A Col ": [1, 2], "Another-Col": [3, 4]})

# One-liner transformation
df = janitor.clean_names(df).add_column("Sum", lambda x: x["a_col"] + x["another_col"])
print(df)
```

**Output:**

```python
   a_col  another_col  sum
0      1            3    4
1      2            4    6
```

---