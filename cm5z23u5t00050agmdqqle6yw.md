---
title: "8 Different techniques for handling missing data (Feature engineering)"
seoTitle: "8 Different techniques for handling missing data (Feature engineering)"
seoDescription: "8 Different techniques for handling missing data (Feature engineering)"
datePublished: Thu Jan 16 2025 08:16:16 GMT+0000 (Coordinated Universal Time)
cuid: cm5z23u5t00050agmdqqle6yw
slug: 8-different-techniques-for-handling-missing-data-feature-engineering
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