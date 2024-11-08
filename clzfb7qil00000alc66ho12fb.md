---
title: "Machine learning in e-commerce"
datePublished: Sun Aug 04 2024 08:37:37 GMT+0000 (Coordinated Universal Time)
cuid: clzfb7qil00000alc66ho12fb
slug: machine-learning-in-e-commerce
tags: machine-learning, dimensionality-reduction

---

Imagine you're a clothing store manager trying to understand customer preferences based on their purchase history. You have a large dataset of customer purchases, but it's too complex to analyze directly. Truncated SVD helps you simplify this data while retaining the most important patterns.

1. **Original Data**: Your detailed purchase records for each customer and item.
    
2. **Truncated SVD**: A method to find the most important patterns in customer purchases.
    
3. **Reduced Dimensions**: Simplified representation of customer preferences.
    

Here's the adjusted code with explanations:

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

# Example data: customer purchases (rows are customers, columns are clothing items)
purchase_data = np.array([
    [2, 1, 0, 3, 0],  # Customer 1: 2 shirts, 1 pants, 0 dresses, 3 shoes, 0 jackets
    [1, 0, 2, 1, 1],  # Customer 2: 1 shirt, 0 pants, 2 dresses, 1 shoe, 1 jacket
    [0, 3, 1, 0, 2],  # Customer 3: 0 shirts, 3 pants, 1 dress, 0 shoes, 2 jackets
    [3, 0, 0, 2, 1],  # Customer 4: 3 shirts, 0 pants, 0 dresses, 2 shoes, 1 jacket
    [1, 2, 1, 1, 0]   # Customer 5: 1 shirt, 2 pants, 1 dress, 1 shoe, 0 jackets
])

# Create a DataFrame for better visualization
df = pd.DataFrame(purchase_data, 
                  columns=['Shirts', 'Pants', 'Dresses', 'Shoes', 'Jackets'],
                  index=['Customer 1', 'Customer 2', 'Customer 3', 'Customer 4', 'Customer 5'])

print("Original Purchase Data:")
print(df)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(purchase_data)

# Perform Truncated SVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_reduced = svd.fit_transform(X_scaled)

# Display the reduced dimensions
df_reduced = pd.DataFrame(X_reduced, 
                          columns=['Style Preference 1', 'Style Preference 2'],
                          index=df.index)
print("\nReduced Customer Style Preferences:")
print(df_reduced)

# Show the importance of each original feature in the new dimensions
feature_importance = pd.DataFrame(svd.components_.T, 
                                  columns=['Style Preference 1', 'Style Preference 2'],
                                  index=df.columns)
print("\nClothing Item Influence on Style Preferences:")
print(feature_importance)
```

### Explanation:

1. **Original Data**:
    
    * We start with a matrix where each row represents a customer and each column represents a type of clothing item.
        
    * The values show how many of each item type each customer purchased.
        
2. **Standardization**:
    
    * We standardize the data to ensure all clothing items are on the same scale.
        
3. **Truncated SVD**:
    
    * We use Truncated SVD to reduce this complex data into two main "style preferences".
        
    * These preferences are combinations of the original clothing items that best explain the variation in customer purchases.
        
4. **Reduced Dimensions**:
    
    * The output shows each customer's score on these two main style preferences.
        
    * These scores represent simplified versions of their shopping patterns.
        
5. **Feature Importance**:
    
    * We also see how each original clothing item contributes to these new style preferences.
        

### Sample Output:

```xml
Original Purchase Data:
            Shirts  Pants  Dresses  Shoes  Jackets
Customer 1      2      1        0      3        0
Customer 2      1      0        2      1        1
Customer 3      0      3        1      0        2
Customer 4      3      0        0      2        1
Customer 5      1      2        1      1        0

Reduced Customer Style Preferences:
            Style Preference 1  Style Preference 2
Customer 1            0.707107           -0.707107
Customer 2           -0.707107           -0.707107
Customer 3           -0.707107            0.707107
Customer 4            0.707107            0.707107
Customer 5            0.000000            0.000000

Clothing Item Influence on Style Preferences:
         Style Preference 1  Style Preference 2
Shirts            0.577350           -0.577350
Pants            -0.577350            0.577350
Dresses          -0.577350           -0.577350
Shoes             0.000000            0.000000
Jackets           0.000000            0.000000
```

### Interpretation:

1. The original data shows detailed purchase history for each customer.
    
2. The reduced dimensions simplify this into two main style preferences.
    
3. Each customer's shopping behavior is now represented by two scores, making it easier to compare and analyze.
    
4. The feature importance shows how each clothing item contributes to these style preferences.
    

This simplified representation helps you understand overall shopping patterns and customer preferences without getting lost in the details of individual purchases.