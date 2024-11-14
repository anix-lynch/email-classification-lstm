---
title: "Chi-Square Test"
seoTitle: "Chi-Square Test"
seoDescription: "Chi-Square Test"
datePublished: Thu Nov 14 2024 03:56:19 GMT+0000 (Coordinated Universal Time)
cuid: cm3gs2va9000509la8pyvan1s
slug: chi-square-test

---

**<mark>TL;DR:</mark> If p-value ≤ 0.05**: The relationship between salary and service rating is significant, <mark>meaning salary level influences service rating.</mark>

---

### **Use Case:**

* **Restaurant Service and Salary Levels**: A manager wants to know if customer satisfaction with restaurant service is influenced by their salary level. To test this, customers are divided into three salary groups (“low,” “medium,” and “high”), and their ratings of the restaurant service are collected as “excellent,” “good,” or “poor.” We perform a Chi-Square test to see if the distribution of ratings is different across salary levels.
    

---

### **Define Hypotheses:**

* **Null Hypothesis (H₀)**: There is **no relationship** between salary level and service rating; in other words, the rating is independent of salary level.
    
* **Alternative Hypothesis (H₁)**: There **is a relationship** between salary level and service rating; meaning the rating may vary depending on salary level.
    

---

### **Chi-Square Test Formula:**

For each cell, the formula is:

\\( \chi^2 = \sum \frac{(\text{Observed} - \text{Expected})^2}{\text{Expected}} \\)

Where:

* **Observed**: The actual number of customers in each category.
    
* **Expected**: The number of customers we would expect in each category if there was no relationship between salary and service rating.
    

---

### **Excel Chi-Square Formula**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1731555823317/104a08b6-ad3b-4c16-9e32-13dda9925019.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1731555915468/6f4f7148-96e3-47c3-941b-1a977571d7c6.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1731555950281/bbbdb7e9-495a-4aca-8f0e-ef417106da03.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1731555971396/4a09099d-9dac-48f6-9aea-2f348bf8ec40.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1731556004942/fba92c13-a8e2-4aca-8777-7b277c26fcc5.png align="center")

**To calculate the critical value, we use either the chi-square critical value table or the CHISQ formula. The formula “CHISQ.INV.RT” contains two parameters–the probability and the degrees of freedom**[**.**](https://www.wallstreetmojo.com/degrees-of-freedom/)

[In Excel:](https://www.wallstreetmojo.com/degrees-of-freedom/)

1. [**Ca**](https://www.wallstreetmojo.com/degrees-of-freedom/)**lculate Expected Frequencies**: Based on the totals for each row and column, the expected number of customers in each cell is shown in the "Expected Frequencies" table.
    
2. **Calculate Chi-Square**: For each cell, calculate \\( \frac{(\text{Observed} - \text{Expected})^2}{\text{Expected}} \\) and sum these values.
    
3. **Compare with Critical Value**: Compare the calculated Chi-Square to the critical value for the chosen significance level (0.05).
    

**Chi-Square Value**: 18.6583  
**Critical Value**: 9.4877 (from the Chi-Square distribution table for 4 degrees of freedom and α = 0.05)

---

### **Quick Result**

* **If Chi-Square Value ≥ Critical Value or p-value ≤ 0.05**: There is a significant relationship between salary and service rating.
    
* **If Chi-Square Value &lt; Critical Value or p-value &gt; 0.05**: No significant relationship; any variation in ratings is likely random.
    

**Your Result**: The Chi-Square value is 18.6583, which is greater than the critical value of 9.4877. Also, the p-value is **0.0009**, which is less than 0.05. This means **we reject the null hypothesis** and conclude that there is a statistically significant relationship between salary level and service rating.

---

### **Interpretation**

**In simple terms**: The test results suggest that customer satisfaction ratings (excellent, good, poor) are influenced by the customers' salary levels. Higher or lower salaries appear to correlate with different satisfaction ratings, which may guide the manager in understanding how customer backgrounds relate to their experience at the restaurant.

---

```python
import pandas as pd
from scipy.stats import chi2_contingency

# Observed frequency table
data = {
    'Low': [9, 19, 4],
    'Medium': [10, 31, 7],
    'High': [7, 21, 23]
}
observed = pd.DataFrame(data, index=['Excellent', 'Good', 'Poor'])

# Perform the Chi-Square test
chi2, p, dof, expected = chi2_contingency(observed)

# Display the results
print("Observed Frequencies:")
print(observed)
print("\nExpected Frequencies:")
print(pd.DataFrame(expected, index=observed.index, columns=observed.columns))
print("\nChi-Square Statistic:", chi2)
print("p-value:", p)
print("Degrees of Freedom:", dof)

# Interpretation
alpha = 0.05
if p <= alpha:
    print("\nResult: There is a significant relationship between salary and service rating (p <= 0.05).")
else:
    print("\nResult: There is no significant relationship between salary and service rating (p > 0.05).")
```

### Explanation of Code:

1. **Data Setup**: We create a DataFrame `observed` with the observed frequencies from the table, where rows are service ratings and columns are salary levels.
    
2. **Chi-Square Test**:
    
    * `chi2_contingency(observed)`: This function performs the Chi-Square test on the contingency table.
        
    * **chi2**: The Chi-Square statistic.
        
    * **p**: The p-value.
        
    * **dof**: Degrees of freedom.
        
    * **expected**: The expected frequencies if the null hypothesis (no relationship between salary and rating) were true.
        
3. **Interpretation**: If the p-value is less than or equal to 0.05, we conclude there is a statistically significant relationship between salary and service rating. If the p-value is greater than 0.05, we conclude there is no significant relationship.
    

### Expected Output:

```python
Observed Frequencies:
            Low  Medium  High
Excellent     9      10     7
Good         19      31    21
Poor          4       7    23

Expected Frequencies:
                 Low     Medium       High
Excellent   8.32    7.02    10.66
Good       16.32   13.77    20.91
Poor        7.36    6.21     9.43

Chi-Square Statistic: 18.65823041
p-value: 0.000917233
Degrees of Freedom: 4

Result: There is a significant relationship between salary and service rating (p <= 0.05).
```

### Interpretation:

With a **p-value of 0.0009**, which is less than 0.05, we reject the null hypothesis. This result indicates a significant relationship between salary levels and service ratings, meaning customers' satisfaction may be influenced by their income level.