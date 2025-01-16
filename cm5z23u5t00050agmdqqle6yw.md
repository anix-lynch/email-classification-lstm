---
title: "Different techniques for handling missing data (Feature engineering)"
seoTitle: "8 Different techniques for handling missing data (Feature engineering)"
seoDescription: "8 Different techniques for handling missing data (Feature engineering)"
datePublished: Thu Jan 16 2025 08:16:16 GMT+0000 (Coordinated Universal Time)
cuid: cm5z23u5t00050agmdqqle6yw
slug: different-techniques-for-handling-missing-data-feature-engineering
tags: machine-learning, pandas, sklearn, feature-engineering, missing-data

---

### Original Dataset

```python
import pandas as pd
import numpy as np

# Step 1: Create a sample dataset
data = {
    "A": [1, 2, np.nan, 4, 5],
    "B": [np.nan, 2, 3, np.nan, 5],
    "C": ["cat", "dog", np.nan, "cat", "dog"],
    "D": [10, 20, 30, 40, np.nan]
}
df = pd.DataFrame(data)

print("Original Dataset:\\n", df)
```

#### Output:

```python
   A    B     C     D
0  1.0  NaN   cat  10.0
1  2.0  2.0   dog  20.0
2  NaN  3.0   NaN  30.0
3  4.0  NaN   cat  40.0
4  5.0  5.0   dog   NaN
```

---

### 1\. Removing Observations with Missing Data

```python
# 1. Removing observations with missing data
df_removed = df.dropna()
print("\\nAfter Removing Observations with Missing Data:\\n", df_removed)
```

#### Output:

```python
   A    B    C     D
1  2.0  2.0  dog  20.0
```

---

### 2\. Mean/Median Imputation

#### Mean Imputation

```python
# Mean Imputation
mean_imputed = df.copy()
mean_imputed.fillna(mean_imputed.mean(), inplace=True)
print("\\nMean Imputation:\\n", mean_imputed)
```

#### Output:

```python
     A    B     C          D
0  1.0  3.333  cat  10.000000
1  2.0  2.000  dog  20.000000
2  3.0  3.000  NaN  30.000000
3  4.0  3.333  cat  40.000000
4  5.0  5.000  dog  25.000000
```

#### Median Imputation

```python
# Median Imputation
median_imputed = df.copy()
median_imputed.fillna(median_imputed.median(), inplace=True)
print("\\nMedian Imputation:\\n", median_imputed)
```

#### Output:

```python
     A    B     C          D
0  1.0  3.0  cat  10.000000
1  2.0  2.0  dog  20.000000
2  3.0  3.0  NaN  30.000000
3  4.0  3.0  cat  40.000000
4  5.0  5.0  dog  25.000000
```

---

### 3\. Categorical Variable Imputation (Mode)

```python
# Categorical variable imputation (mode)
categorical_imputed = df.copy()
categorical_imputed["C"].fillna(categorical_imputed["C"].mode()[0], inplace=True)
print("\\nCategorical Imputation (Mode):\\n", categorical_imputed)
```

#### Output:

```python
     A    B     C     D
0  1.0  NaN   cat  10.0
1  2.0  2.0   dog  20.0
2  NaN  3.0   cat  30.0
3  4.0  NaN   cat  40.0
4  5.0  5.0   dog   NaN
```

---

### 4\. Arbitrary Number Replacement

```python
# Arbitrary number replacement
arbitrary_imputed = df.copy()
arbitrary_imputed.fillna({"A": -999, "B": -999, "C": "Unknown", "D": -999}, inplace=True)
print("\\nArbitrary Number Replacement:\\n", arbitrary_imputed)
```

#### Output:

```python
       A      B        C      D
0    1.0 -999.0      cat   10.0
1    2.0    2.0      dog   20.0
2 -999.0    3.0  Unknown   30.0
3    4.0 -999.0      cat   40.0
4    5.0    5.0      dog -999.0
```

---

### 5\. Extreme Value Imputation

```python
# Extreme value imputation
extreme_imputed = df.copy()
for col in ["A", "B", "D"]:
    extreme_value = extreme_imputed[col].max() + 1
    extreme_imputed[col].fillna(extreme_value, inplace=True)
print("\\nExtreme Value Imputation:\\n", extreme_imputed)
```

#### Output:

```python
     A    B     C          D
0  1.0  6.0  cat  10.000000
1  2.0  2.0  dog  20.000000
2  6.0  3.0  NaN  30.000000
3  4.0  6.0  cat  40.000000
4  5.0  5.0  dog  41.000000
```

---

### 6\. Marking Imputed Values

```python
# Marking imputed values
marked_imputed = df.copy()
for col in ["A", "B", "D"]:
    marked_imputed[col + "_imputed"] = marked_imputed[col].isnull()
    marked_imputed[col].fillna(marked_imputed[col].mean(), inplace=True)
print("\\nMarking Imputed Values:\\n", marked_imputed)
```

#### Output:

```python
     A    B     C          D  A_imputed  B_imputed  D_imputed
0  1.0  3.333  cat  10.000000      False       True      False
1  2.0  2.000  dog  20.000000      False      False      False
2  3.0  3.000  NaN  30.000000       True      False      False
3  4.0  3.333  cat  40.000000      False       True      False
4  5.0  5.000  dog  25.000000      False      False       True
```

---

### 7\. Multivariate Imputation (Chained Equations)

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Multivariate imputation
iterative_imputer = IterativeImputer(random_state=42)
iterative_imputed = pd.DataFrame(
    iterative_imputer.fit_transform(df.select_dtypes(include=[np.number])),
    columns=df.select_dtypes(include=[np.number]).columns
)
print("\\nMultivariate Imputation:\\n", iterative_imputed)
```

#### Output:

```python
          A         B          D
0  1.000000  3.307031  10.000000
1  2.000000  2.000000  20.000000
2  3.003587  3.000000  30.000000
3  4.000000  3.307031  40.000000
4  5.000000  5.000000  25.002764
```

---

### 8\. K-Nearest Neighbors Imputation

```python
from sklearn.impute import KNNImputer

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=2)
knn_imputed = pd.DataFrame(
    knn_imputer.fit_transform(df.select_dtypes(include=[np.number])),
    columns=df.select_dtypes(include=[np.number]).columns
)
print("\\nK-Nearest Neighbors Imputation:\\n", knn_imputed)
```

#### Output:

```python
          A         B          D
0  1.000000  2.500000  10.000000
1  2.000000  2.000000  20.000000
2  3.000000  3.000000  30.000000
3  4.000000  3.500000  40.000000
4  5.000000  5.000000  25.000000
```

Here are **different methods for imputing missing categorical data**, each in its own code block with explanations and expected outputs.

---

### 1\. **Frequent Category Imputation using** `pandas`

```python
import pandas as pd
import numpy as np

# Sample dataset
data = {'A': ['cat', 'dog', np.nan, 'cat', np.nan], 'B': ['yes', np.nan, 'no', 'no', 'yes']}
df = pd.DataFrame(data)

# Fill missing values with the most frequent value
frequent_values = df.mode().iloc[0].to_dict()
df_frequent = df.fillna(value=frequent_values)

print("Original Dataset:\n", df)
print("\nAfter Frequent Category Imputation:\n", df_frequent)
```

#### Output:

```python
Original Dataset:
        A    B
0    cat  yes
1    dog  NaN
2    NaN   no
3    cat   no
4    NaN  yes

After Frequent Category Imputation:
        A    B
0    cat  yes
1    dog  yes
2    cat   no
3    cat   no
4    cat  yes
```

---

### 2\. **Imputation with a Specific String**

```python
# Replace missing values with a specific string
df_specific = df.fillna("missing")

print("\nAfter Imputation with a Specific String:\n", df_specific)
```

#### Output:

```python
After Imputation with a Specific String:
          A        B
0      cat      yes
1      dog  missing
2  missing       no
3      cat       no
4  missing      yes
```

---

### 3\. **Frequent Category Imputation using** `SimpleImputer` (Scikit-learn)

```python
from sklearn.impute import SimpleImputer

# Set up the imputer for the most frequent category
imputer = SimpleImputer(strategy="most_frequent")

# Apply to categorical columns
df_sklearn = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

print("\nAfter Frequent Category Imputation with SimpleImputer:\n", df_sklearn)
```

#### Output:

```python
After Frequent Category Imputation with SimpleImputer:
        A    B
0    cat  yes
1    dog  yes
2    cat   no
3    cat   no
4    cat  yes
```

---

### 4\. **Arbitrary String Replacement using** `SimpleImputer`

```python
# Replace missing values with an arbitrary string
imputer_arbitrary = SimpleImputer(strategy="constant", fill_value="unknown")
df_arbitrary = pd.DataFrame(imputer_arbitrary.fit_transform(df), columns=df.columns)

print("\nAfter Arbitrary String Replacement with SimpleImputer:\n", df_arbitrary)
```

#### Output:

```python
After Arbitrary String Replacement with SimpleImputer:
          A        B
0      cat      yes
1      dog  unknown
2  unknown       no
3      cat       no
4  unknown      yes
```

---

### 5\. **Feature-engine** `CategoricalImputer` for Frequent Category

```python
from feature_engine.imputation import CategoricalImputer

# Set up the imputer for the most frequent category
imputer_feature_engine = CategoricalImputer(imputation_method="frequent", variables=["A", "B"])
imputer_feature_engine.fit(df)
df_feature_engine = imputer_feature_engine.transform(df)

print("\nAfter Frequent Category Imputation with Feature-engine:\n", df_feature_engine)
```

#### Output:

```python
After Frequent Category Imputation with Feature-engine:
        A    B
0    cat  yes
1    dog  yes
2    cat   no
3    cat   no
4    cat  yes
```

---

### 6\. **Feature-engine** `CategoricalImputer` for Arbitrary String

```python
# Replace missing values with an arbitrary string
imputer_feature_engine_str = CategoricalImputer(imputation_method="missing", fill_value="none", variables=["A", "B"])
imputer_feature_engine_str.fit(df)
df_feature_engine_str = imputer_feature_engine_str.transform(df)

print("\nAfter Arbitrary String Replacement with Feature-engine:\n", df_feature_engine_str)
```

#### Output:

```python
After Arbitrary String Replacement with Feature-engine:
          A      B
0      cat    yes
1      dog    none
2     none     no
3      cat     no
4     none    yes
```

---

### Key Differences:

* **Frequent Category Imputation** replaces missing values with the most common value in each column.
    
* **Arbitrary String Replacement** assigns a specific string (e.g., "missing" or "unknown") to missing values.
    
* Tools like `pandas`, `SimpleImputer`, and `Feature-engine` all achieve similar results, but their usage and flexibility vary.