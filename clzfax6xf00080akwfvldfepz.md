---
title: "The One Where Dr. Drake Ramoray Gets It Wrong: Confusion Matrix, Ross/Rachel case study"
seoTitle: "Understanding Medical Test Accuracy: A Fun Guide for Friends Fans"
seoDescription: "Learn how doctors can make mistakes in medical tests using a fun analogy with Ross and Rachel from Friends. Discover how confusion matrices help evaluate te"
datePublished: Sun Aug 04 2024 08:29:25 GMT+0000 (Coordinated Universal Time)
cuid: clzfax6xf00080akwfvldfepz
slug: the-one-where-dr-drake-ramoray-gets-it-wrong-confusion-matrix-rossrachel-case-study
tags: statistics, confusion-matrix

---

In the world of statistics, a confusion matrix is a powerful tool used to evaluate the performance of a classification model. But what if we took this concept and applied it to a scenario straight out of "Friends"? Imagine a doctor trying to determine if Ross and Rachel (and a few others) are pregnant based on their test results. Let's dive into this fun analogy to understand confusion matrices better.

#### The Scenario

In our scenario, the doctor is conducting pregnancy tests and providing results. We have the actual conditions (whether someone is pregnant or not) and the doctor's predictions. Here's how we can break it down:

* **True Positive (TP)**: The doctor correctly identifies someone as pregnant.
    
* **True Negative (TN)**: The doctor correctly identifies someone as not pregnant.
    
* **False Positive (FP)**: The doctor incorrectly identifies someone as pregnant (e.g., saying Ross is pregnant).
    
* **False Negative (FN)**: The doctor incorrectly identifies someone as not pregnant (e.g., saying Rachel is not pregnant when she actually is).
    

Let's see how we can use a confusion matrix to evaluate the doctor's performance.

#### Example Code

```python
import pandas as pd
from sklearn.metrics import confusion_matrix

# Example data: true labels and predicted labels
# 0: Not Pregnant, 1: Pregnant
y_true = [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]  # Actual condition (ground truth)
y_pred = [0, 0, 0, 1, 0, 1, 0, 0, 1, 1]  # Doctor's prediction

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"True Negatives: {tn}")  # Doctor correctly says not pregnant
print(f"False Positives: {fp}")  # Doctor incorrectly says pregnant (e.g., Ross)
print(f"False Negatives: {fn}")  # Doctor incorrectly says not pregnant (e.g., Rachel)
print(f"True Positives: {tp}")  # Doctor correctly says pregnant
```

#### Explanation

1. **True Negatives (TN)**:
    
    * These are cases where the doctor correctly identified someone as not pregnant.
        
    * Example: The doctor correctly says Ross is not pregnant.
        
2. **False Positives (FP)**:
    
    * These are cases where the doctor incorrectly identified someone as pregnant.
        
    * Example: The doctor incorrectly says Ross is pregnant.
        
3. **False Negatives (FN)**:
    
    * These are cases where the doctor incorrectly identified someone as not pregnant.
        
    * Example: The doctor incorrectly says Rachel is not pregnant when she actually is.
        
4. **True Positives (TP)**:
    
    * These are cases where the doctor correctly identified someone as pregnant.
        
    * Example: The doctor correctly says Rachel is pregnant.
        

#### Sample Output

```xml
True Negatives: 5
False Positives: 1
False Negatives: 1
True Positives: 3
```

### Interpretation

Based on the confusion matrix, we can draw several conclusions about the doctor's performance:

* **True Negatives (5)**: The doctor correctly identified 5 people as not pregnant.
    
* **False Positives (1)**: The doctor incorrectly identified 1 person as pregnant (e.g., saying Ross is pregnant).
    
* **False Negatives (1)**: The doctor incorrectly identified 1 person as not pregnant (e.g., saying Rachel is not pregnant when she was).
    
* **True Positives (3)**: The doctor correctly identified 3 people as pregnant.
    

### Accuracy Calculation

To calculate the accuracy of the doctor's predictions, we can use the formula:

$$\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Predictions}}$$

In this case:

$$\text{Accuracy} = \frac{3 + 5}{10} = 0.8$$

So, the doctor was correct 8 times out of 10, giving an accuracy of 80%.

### Conclusion

The confusion matrix provides a detailed breakdown of the doctor's performance in diagnosing pregnancy. It helps us understand the number of correct and incorrect predictions, which is essential for evaluating the accuracy and reliability of the test results. In this analogy, it helps us see how often the doctor correctly identifies pregnancy and how often mistakes are made, such as incorrectly saying Ross is pregnant or missing Rachel's pregnancy.

By using a confusion matrix, we can gain valuable insights into the performance of classification models, whether in medical diagnostics or any other field. So next time you watch "Friends," remember that even Ross and Rachel can help you understand complex statistical concepts!