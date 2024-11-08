---
title: "Standardization (Z-Scores) helps make data "apple to apple🍎"?"
seoTitle: "The Importance of Z-Scores: Why Standardizing Test Scores Matters for "
seoDescription: "Discover how standardizing test scores using Z-scores enhances comparability, improves model performance, and reduces bias in educational assessments. "
datePublished: Sun Aug 04 2024 13:54:59 GMT+0000 (Coordinated Universal Time)
cuid: clzfmjvvp000309jxf9ux4mn9
slug: standardization-z-scores-helps-make-data-apple-to-apple
tags: sklearn, standardization, z-scores

---

We'll use a school grading system across different subjects as our analogy.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Example data: test scores in different subjects
data = {
    'math_score': [65, 70, 80, 90, 95],
    'history_score': [75, 80, 85, 90, 95]
}
df = pd.DataFrame(data)

# Initialize the score standardizer
score_standardizer = StandardScaler()

# Fit and transform the scores
standardized_scores = score_standardizer.fit_transform(df)

# Convert standardized scores back to DataFrame
df_standardized = pd.DataFrame(standardized_scores, columns=['math_z_score', 'history_z_score'])

print("Original Scores:")
print(df)
print("\nStandardized Scores (Z-scores):")
print(df_standardized)
```

### Explanation

In this school grading analogy:

1. We have test scores from two different subjects: math and history.
    
2. The `StandardScaler` acts like a "score standardizer" that converts the raw scores into standardized scores (Z-scores).
    
3. After standardization, each score is expressed in terms of how many standard deviations it is from the mean score for that subject.
    

### Sample Output

```xml
Original Scores:
   math_score  history_score
0          65             75
1          70             80
2          80             85
3          90             90
4          95             95

Standardized Scores (Z-scores):
   math_z_score  history_z_score
0     -1.336306        -1.414214
1     -0.890871        -0.707107
2      0.000000         0.000000
3      0.890871         0.707107
4      1.336306         1.414214
```

### Interpretation

* In the original data, math and history have different score distributions.
    
* After standardization:
    
    * The mean score for each subject becomes 0.
        
    * Positive Z-scores indicate scores above the mean.
        
    * Negative Z-scores indicate scores below the mean.
        
    * The magnitude indicates how many standard deviations away from the mean the score is.
        

This standardization allows you to compare performance across different subjects on the same scale, even if the original scoring systems were different. For example:

1. A student with a math Z-score of 0.890871 and a history Z-score of 0.707107 performed above average in both subjects, but relatively better in math.
    
2. The lowest score in both subjects (65 in math, 75 in history) both have similar Z-scores around -1.4, indicating they are similarly below average in their respective subjects.
    

Test scores between two subjects are often not directly comparable due to several reasons:

### Why test score between 2 subject is not apple to apple?  
  
1\. Different Scales and Units

* **Different Maximum Scores:** Math tests might have a maximum score of 100, whereas history tests might have a maximum score of 50.
    
* **Different Grading Systems:** One subject might use a percentage-based grading system, while another might use letter grades or a different numerical scale.
    

### 2\. Different Difficulty Levels

* **Subject Complexity:** The inherent difficulty of the subjects might differ. For example, a score of 80 in math might represent a higher level of achievement than a score of 80 in history, depending on the curriculum and complexity of the material.
    
* **Test Design:** The design and difficulty of the tests themselves might vary, making direct comparisons unfair.
    

### 3\. Different Distributions

* **Score Distribution:** The distribution of scores can be different for each subject. For example, math scores might be normally distributed, while history scores might be skewed.
    
* **Variability in Scores:** The variability (standard deviation) of scores can differ, indicating that one subject might have more spread out scores than another.
    

### 4\. Different Learning Objectives

* **Skills Assessed:** Different subjects assess different skills and knowledge areas. Math might test problem-solving and analytical skills, while history might test memorization and understanding of historical events.
    
* **Educational Goals:** The goals of education in each subject might differ, making it inappropriate to compare scores directly.
    

### 5\. Different Student Populations

* **Student Strengths:** Students might have different strengths and weaknesses across subjects. A student might excel in math but struggle in history, or vice versa.
    
* **Classroom Environment:** The teaching methods and classroom environments might differ, affecting student performance in each subject.
    

### Why Standardization Helps

Standardization (converting scores to Z-scores) addresses these issues by:

* **Removing Units:** Standardization removes the units of measurement, putting all scores on a common scale.
    
* **Centering and Scaling:** It centers the scores around the mean and scales them by the standard deviation, allowing for direct comparison of relative performance.
    
* **Handling Different Distributions:** It accounts for different score distributions, making comparisons fairer.
    

### Example

Consider two students with the following scores:

* Student A: 80 in math, 85 in history
    
* Student B: 90 in math, 75 in history
    

Without standardization, it's hard to say who performed better overall because the scores are not directly comparable. After standardization, you might find:

* Student A: Z-score of 0.5 in math, 0.7 in history
    
* Student B: Z-score of 1.0 in math, -0.3 in history
    

Now, it's easier to see that Student B performed exceptionally well in math but below average in history, while Student A performed above average in both subjects.

# 10 other scenarios what might go wrong if we don't use Z-scores?

I understand now. You're asking for examples of what might go wrong if we don't use Z-scores (or standardization) in each of these 10 scenarios. Here are some examples:

1. Comparability across features:  
    Drama: A company uses height (in cm) and weight (in kg) to predict health risks. Without standardization, weight dominates the model, leading to incorrect risk assessments for tall, lean individuals.
    
    ```python
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    # Example data: height (in cm) and weight (in kg)
    data = {'height_cm': [160, 170, 180, 190], 'weight_kg': [55, 70, 85, 100]}
    df = pd.DataFrame(data)
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit and transform the data
    scaled_data = scaler.fit_transform(df)
    
    # Convert scaled data back to DataFrame
    df_scaled = pd.DataFrame(scaled_data, columns=['height_cm', 'weight_kg'])
    print(df_scaled)
    ```
    
2. Improved model performance:  
    Drama: A neural network struggles to learn from a dataset with salary (in thousands) and age (in years). The model focuses mainly on salary due to its larger scale, completely missing important age-related patterns.
    
    ```python
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    # Example data: salary (in thousands) and age (in years)
    data = {'salary_k': [50, 60, 70, 80], 'age_years': [25, 35, 45, 55]}
    df = pd.DataFrame(data)
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit and transform the data
    scaled_data = scaler.fit_transform(df)
    
    # Convert scaled data back to DataFrame
    df_scaled = pd.DataFrame(scaled_data, columns=['salary_k', 'age_years'])
    print(df_scaled)
    ```
    
3. Reducing bias: Drama: In a credit scoring model, income (in thousands) overshadows other important factors like credit history, unfairly disadvantaging lower-income applicants with excellent credit.
    
4. Normalization of distributions: Drama: A machine learning model for predicting house prices performs poorly because it assumes normal distributions. Without standardization, the skewed distribution of house prices leads to inaccurate predictions.
    
5. Faster convergence: Drama: A gradient descent algorithm takes weeks to converge on a large dataset with widely varying feature scales, causing significant delays in product development and missed deadlines.
    
6. Improved interpretability: Drama: Analysts misinterpret the importance of different factors in a linear regression model for customer churn, leading to misguided retention strategies and increased customer loss.
    
7. Handling outliers:  
    Drama: A few extremely high-value transactions in a fraud detection system cause the model to ignore smaller, more common fraudulent activities, resulting in significant financial losses.
    
    ```python
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    # Example data: transaction amounts (in dollars) with outliers
    data = {'transaction_amount': [10, 20, 30, 1000]}
    df = pd.DataFrame(data)
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit and transform the data
    scaled_data = scaler.fit_transform(df)
    
    # Convert scaled data back to DataFrame
    df_scaled = pd.DataFrame(scaled_data, columns=['transaction_amount'])
    print(df_scaled)
    ```
    
8. Consistency across datasets:
    
    Drama: Researchers combining medical data from multiple hospitals reach incorrect conclusions about treatment effectiveness due to inconsistent scaling across datasets.
    
    ```python
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    # Example data: dataset1 and dataset2 with different scales
    data1 = {'feature1': [1, 2, 3, 4], 'feature2': [10, 20, 30, 40]}
    data2 = {'feature1': [5, 6, 7, 8], 'feature2': [50, 60, 70, 80]}
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit and transform the data
    scaled_data1 = scaler.fit_transform(df1)
    scaled_data2 = scaler.fit_transform(df2)
    
    # Convert scaled data back to DataFrame
    df_scaled1 = pd.DataFrame(scaled_data1, columns=['feature1', 'feature2'])
    df_scaled2 = pd.DataFrame(scaled_data2, columns=['feature1', 'feature2'])
    print(df_scaled1)
    print(df_scaled2)
    ```
    
9. Prerequisite for certain algorithms:  
    Drama: A data scientist attempts to use PCA for dimensionality reduction on non-standardized data, resulting in a completely distorted representation of the dataset and misleading insights.
    
    ```python
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    # Example data: features for PCA
    data = {'feature1': [1, 2, 3, 4], 'feature2': [100, 200, 300, 400]}
    df = pd.DataFrame(data)
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit and transform the data
    scaled_data = scaler.fit_transform(df)
    
    # Convert scaled data back to DataFrame
    df_scaled = pd.DataFrame(scaled_data, columns=['feature1', 'feature2'])
    print(df_scaled)
    ```
    
10. Improved numerical stability: Drama: A financial model dealing with both large monetary values and small percentage changes experiences numerical overflow, leading to critical errors in investment decisions and substantial financial losses.
    

These examples illustrate how failing to standardize data can lead to various problems in data analysis and machine learning applications.