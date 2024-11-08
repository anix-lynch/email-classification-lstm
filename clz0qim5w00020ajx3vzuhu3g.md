---
title: "How to increase sales with data 📈 Fitness business case"
datePublished: Thu Jul 25 2024 03:49:26 GMT+0000 (Coordinated Universal Time)
cuid: clz0qim5w00020ajx3vzuhu3g
slug: how-to-increase-sales-with-data-fitness-business-case
tags: exploratory-data-analysis

---

[Dataset and Jupyterfile download](https://github.com/anix-lynch/medium_jupyter/blob/main/treadmill_buyer_profile/treadmill.ipynb)

In this analysis, we aim to understand the profile of treadmill buyers to tailor our marketing strategies and improve sales. By analyzing the demographics and behaviors of our customers, we can derive actionable insights to drive business growth.

## Key Insights

### 1\. Best-Selling Product

**Problem**: Identify the best-selling treadmill model to focus marketing efforts.

**Insight**: Model KP281 is the best-selling product, accounting for 46.6% of all treadmill sales.

**Visualization**:

```python
sns.countplot(x='Product', data=aerofit_df, palette='viridis')
plt.title('Product Distribution')
plt.show()
```

![Product Distribution](link_to_your_visualization_image align="left")

**Action**: Focus marketing efforts on highlighting the features and benefits of KP281 to maintain its sales momentum.

### 2\. Income Distribution

**Problem**: Determine the income distribution of treadmill buyers to tailor marketing campaigns.

**Insight**: The majority of treadmill customers fall within the $45,000 - $60,000 income bracket. 83% of purchases are made by individuals with incomes between $35,000 and $85,000, while only 8% are from customers with incomes below $35,000.

**Visualization**:

```python
sns.histplot(aerofit_df['Income'], kde=True, color='purple')
plt.title('Income Distribution')
plt.show()
```

![Income Distribution](link_to_your_visualization_image align="left")

**Action**: Tailor marketing campaigns to target individuals within the $35,000 to $85,000 income range. Develop more affordable models to attract lower-income customers.

### 3\. Age Distribution

**Problem**: Understand the age distribution of treadmill buyers for targeted marketing.

**Insight**: 88% of treadmill buyers are aged between 20 to 40 years.

**Visualization**:

```python
sns.histplot(aerofit_df['Age'], kde=True, color='blue')
plt.title('Age Distribution')
plt.show()
```

![Age Distribution](link_to_your_visualization_image align="left")

**Action**: Develop marketing campaigns that appeal to the lifestyle and preferences of young adults. Utilize social media and digital marketing channels to reach this demographic effectively.

### 4\. Correlation Insights

**Problem**: Identify correlations between fitness levels and treadmill usage to enhance product positioning.

**Insight**: Higher fitness levels are associated with increased treadmill usage. Customers with high fitness levels tend to use treadmills more frequently.

**Visualization**:

```python
plt.figure(figsize=(18, 10))
sns.heatmap(aerofit_df.corr(), annot=True, linewidths=0.8, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
```

![Correlation Heatmap](link_to_your_visualization_image align="left")

**Action**: Highlight fitness benefits and advanced features in marketing materials to attract fitness enthusiasts. Offer fitness programs or partnerships with gyms to enhance customer engagement.

### 5\. High-End Model KP781

**Problem**: Identify the target market for high-end treadmill models.

**Insight**: KP781 is the only model purchased by customers with more than 20 years of education and an income over $85,000.

**Visualization**:

```python
sns.boxplot(data=aerofit_df, x='Gender', y='Income', hue='Product')
plt.title('Income vs. Product by Gender')
plt.show()
```

![Income vs. Product by Gender](link_to_your_visualization_image align="left")

**Action**: Position KP781 as a premium model and market it to high-income, highly educated individuals. Emphasize its advanced features and superior quality.

### 6\. Fitness Levels and Product Preference

**Problem**: Understand the preferences of customers with different fitness levels.

**Insight**: Customers with fitness levels 4 and 5 prefer high-end treadmills and use them extensively, averaging over 150 miles per week.

**Visualization**:

```python
sns.boxplot(data=aerofit_df, x='Fitness', y='Miles', hue='Product')
plt.title('Miles vs. Fitness by Product')
plt.show()
```

![Miles vs. Fitness by Product](link_to_your_visualization_image align="left")

**Action**: Promote high-end models like KP781 to customers with high fitness levels. Consider offering loyalty programs or exclusive deals for frequent users.

## Recommendations to business owner

### 1\. Affordable Models

**Recommendation**: Develop and market affordable treadmill models for customers with incomes below $35,000. This can help expand our customer base and increase market penetration.

### 2\. Premium Model Marketing

**Recommendation**: Focus on marketing premium models like KP781 to high-income, highly educated individuals. Use targeted advertising and personalized marketing strategies to reach this segment.

### 3\. Engaging Young Adults

**Recommendation**: Utilize social media, influencers, and digital content to engage young adults aged 20-40. Create campaigns that resonate with their fitness and lifestyle goals.

### 4\. Fitness Programs and Partnerships

**Recommendation**: Partner with gyms and fitness programs to offer exclusive deals and fitness plans to treadmill buyers. This can increase customer loyalty and encourage higher usage of our products.

## Conclusion

By leveraging these insights and implementing the recommended strategies, we can better meet the needs of our customers, enhance our marketing effectiveness, and drive sales growth. Our focus should be on targeting the right demographics, offering products that meet their needs, and creating value through strategic marketing initiatives.

---