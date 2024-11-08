---
title: "Sklearn.metrics💯: Evaluating Model Performance with Accuracy, Precision, and Recall"
seoTitle: "Sklearn.metrics: Evaluating Model Performance with Accuracy, Precisi"
seoDescription: "Unlock the power of Sklearn.metrics to evaluate your machine learning models with accuracy, precision, and recall. Learn how to use these essential metrics "
datePublished: Sun Aug 04 2024 14:21:20 GMT+0000 (Coordinated Universal Time)
cuid: clzfnhrr1000i09jx9g3mgmob
slug: sklearnmetrics-evaluating-model-performance-with-accuracy-precision-and-recall
tags: data-science, machine-learning, model-evaluation, sklearn

---

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score
```

Imagine you run a clothing store and are trying to predict whether a customer will buy a certain type of clothing item based on their income and age.

* **Income:** This represents how much money a customer has available to spend on clothing. Higher income might suggest they are more likely to buy more expensive items.
    
* **Age:** This represents the age of the customer, which can influence their style preferences and purchasing behavior.
    

In this context, you want to build a model that predicts whether a customer will purchase a specific type of clothing item (let's say, a trendy jacket) based on their income and age.

### Model Prediction

In your model:

* **0** could represent that the customer is **not likely to purchase** the jacket.
    
* **1** could represent that the customer is **likely to purchase** the jacket.
    

### Scoring Metrics

After training your model, you evaluate its performance using scoring metrics:

1. **Accuracy:** This tells you the overall percentage of correct predictions made by the model. For example, if the model predicted correctly for 80 out of 100 customers, the accuracy would be 80%.
    
2. **Precision:** This measures how many of the customers predicted to buy the jacket actually did buy it. If the model predicted 10 customers would buy the jacket and 8 actually did, the precision would be 80%.
    
3. **Recall:** This measures how many of the actual buyers were correctly predicted by the model. If there were 10 actual buyers and the model correctly identified 8 of them, the recall would be 80%.
    

### Example Code

Here’s the adjusted code with the clothing analogy in mind:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Example data: features and target variable (purchase: 0 - not likely, 1 - likely)
data = {'income': [50000, 60000, 70000, 80000, 90000],
        'age': [25, 35, 45, 55, 65],
        'purchase': [0, 1, 0, 1, 0]}  # 0 = not likely to buy, 1 = likely to buy
df = pd.DataFrame(data)

# Define features and target variable
X = df[['income', 'age']]
y = df['purchase']

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict on the same data (for demonstration purposes)
y_pred = model.predict(X)

# Calculate scoring metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
```

### Expected Output

The output will display the model's accuracy, precision, and recall:

```python
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
```

# Is 1.0 high or low?

Let's break this down:

Accuracy: 1.0 Precision: 1.0 Recall: 1.0

These scores are all at the maximum possible value of 1.0 (or 100% if expressed as a percentage). In the context of model evaluation:

1. This is extremely high. In fact, it's perfect.
    
2. These scores indicate that the model is performing exceptionally well, at least on the data it was evaluated on.
    

Here's what each score means at 1.0:

* Accuracy (1.0): The model correctly predicted every single instance in the dataset.
    
* Precision (1.0): Every time the model predicted a positive result (likely to purchase), it was correct.
    
* Recall (1.0): The model correctly identified all actual positive instances (all customers who were likely to purchase).
    

However, it's important to note:

1. Perfect scores like this are rare in real-world scenarios and can be a red flag.
    
2. These scores might indicate overfitting, especially if they're based on the same data used to train the model.
    
3. It's crucial to evaluate the model on separate test data to get a more realistic assessment of its performance.
    

In conclusion, while these scores suggest a "good" model in terms of performance, they are suspiciously perfect. In practice, you would want to:

1. Test the model on new, unseen data.
    
2. Consider whether the model is too complex for the problem (potentially overfitting).
    
3. Ensure there are no data leakage issues in your preprocessing or training pipeline.
    

Remember, in real-world applications, slightly lower but consistent performance across different datasets is often more desirable than perfect scores on a single dataset.