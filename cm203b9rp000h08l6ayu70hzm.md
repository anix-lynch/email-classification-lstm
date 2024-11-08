---
title: "26 Type of Seaborn Plots  With Samples"
seoTitle: "26 Type of Seaborn Plots with Samples"
seoDescription: "26 Type of Seaborn Plots with Samples"
datePublished: Tue Oct 08 2024 06:58:59 GMT+0000 (Coordinated Universal Time)
cuid: cm203b9rp000h08l6ayu70hzm
slug: 26-type-of-seaborn-plots-with-samples
tags: python, data-science, data-analysis, visualization, seaborn

---

### 1\. **Relationship Plots** (Visualize relationships between variables)

* `relplot`: High-level function to draw scatter or line plots.
    
* `scatterplot`: Plot individual data points with optional grouping.
    
* `lineplot`: Show trends by connecting data points with a line.
    
* `lmplot`: Plot data with linear regression fit.
    
* `regplot`: Plot data with optional linear regression model.
    
* `residplot`: Show residuals from linear regression.
    

### 2\. **Distribution Plots** (Understand how data is distributed)

* `displot`: High-level function for histograms or kernel density estimates.
    
* `histplot`: Plot the frequency of data in bins.
    
* `kdeplot`: Show the probability density function.
    
* `ecdfplot`: Empirical cumulative distribution function plot.
    
* `rugplot`: Add small vertical ticks on the plot to show distribution.
    
* `distplot` *(deprecated)*: Used to combine histograms and kde plots.
    

### 3\. **Categorical Data Plots** (Compare categories)

* `catplot`: High-level function for categorical plots.
    
* `stripplot`: Plot individual points for each category.
    
* `swarmplot`: Spread points out within categories to avoid overlap.
    
* `boxplot`: Show the distribution of data based on quartiles.
    
* `violinplot`: Combine boxplot and kde to show data distribution.
    
* `boxenplot`: Extended box plot for larger datasets.
    
* `pointplot`: Plot point estimates with confidence intervals.
    
* `barplot`: Show means or other aggregates for each category.
    
* `countplot`: Count the occurrences of each category.
    

### 4\. **Heatmaps and Cluster Maps** (Visualize matrices and clustering)

* `heatmap`: Display matrix data with color mapping.
    
* `clustermap`: Perform hierarchical clustering and plot the data matrix.
    

### 5\. **Grids and Pair Plots** (Subplots and multiple plots)

* `FacetGrid`: Build a grid of subplots to show data relationships across subsets.
    
* `pairplot`: Plot pairwise relationships in a dataset.
    
* `PairGrid`: Fine-tuned pairwise plot creation.
    
* `jointplot`: Combine scatterplot and marginal plots.
    
* `JointGrid`: Fine-tuned creation of joint plots.
    

### **Best to Use When**:

* **Relationship plots**: When exploring correlations (when you have **only 2 variables)** or <mark>trends.</mark>
    
* **Distribution plots**: To understand data <mark>distribution.</mark>
    
* **Categorical plots**: To <mark>compare</mark> data across different categories.
    
* **Heatmaps/Cluster maps**: For visualizing <mark>matrices, correlations(when you have </mark> **<mark>more than 2 variables)</mark>** <mark>, or clustering (</mark>**<mark>grouping</mark>** <mark>or </mark> **<mark>clustering</mark>** <mark>similar variables).</mark>
    
* **Grid/Pair plots**: To visualize <mark>multiple relationships</mark> or distributions at once.
    

---

### 1\. **Relationship Plots**

**(Visualize relationships between variables)**

* `relplot`
    
    * **Data**: Monthly sales and marketing budget.
        
    * **Use**: Compare sales <mark>trends </mark> based on budget.
        
    * **Example**:
        
        ```python
        sns.relplot(x="month", y="sales", hue="budget", data=sales_data, kind="line")
        ```
        
* `scatterplot`
    
    * **Data**: Weight and height of athletes.
        
    * **Use**: Visualize <mark>correlation</mark> between weight and height.
        
    * **Example**:
        
        ```python
        sns.scatterplot(x="height", y="weight", data=athletes_data)
        ```
        
* `lineplot`
    
    * **Data**: Daily temperature over a year.
        
    * **Use**: Show temperature <mark>trends.</mark>
        
    * **Example**:
        
        ```python
        sns.lineplot(x="day", y="temperature", data=weather_data)
        ```
        
* `lmplot`
    
    * ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728369315807/7a7f14cc-b6fc-44e4-a2ed-e2be7cd04c63.png align="center")
        
        **Data**: Salary vs. years of experience.
        
    * **Use**: Show <mark>linear relationship</mark> between salary and experience.  
        Yes, this `lmplot` shows two linear regression lines, likely representing two different groups of employees (blue and red) based on some categorical variable, like gender, department, or job level.  
        
        Here's how to interpret the lmplot:
        
        1. **Dots (Scatter Points)**:
            
            * Each **dot** represents an individual employee.
                
            * The **x-axis** shows the **years of experience**.
                
            * The **y-axis** shows the **salary** (in thousands, most likely).
                
        2. **Regression Lines**:
            
            * The **blue line** and the **red line** are the regression lines for the two groups.
                
            * These lines show the overall **trend** or **relationship** between years of experience and salary for each group:
                
                * **Upward slope**: Both lines are going upwards, meaning that as employees gain more experience, their salary tends to increase.
                    
                * **Blue vs. Red Line**: The **blue group** has a steeper slope, suggesting a stronger relationship between experience and salary for this group compared to the **red group**.
                    
        3. **<mark>Confidence Intervals</mark>**<mark>:</mark>
            
            * <mark>The </mark> **<mark>shaded areas</mark>** <mark> around each line represent the </mark> **<mark>confidence intervals</mark>** <mark> (likely 95% confidence). This shows the uncertainty or variability in the salary predictions:</mark>
                
                * <mark>Narrower intervals (less shaded area) indicate more confidence in the prediction.</mark>
                    
                * <mark>Wider intervals (more shaded area) indicate greater uncertainty in the salary prediction.</mark>
                    
                * <mark>The </mark> **<mark>blue group's model</mark>** <mark> is likely more </mark> **<mark>accurate</mark>** <mark> or </mark> **<mark>reliable</mark>** <mark> because the shade area is narrower.</mark>
                    
        4. **Comparison**:
            
            * The **blue group** seems to experience a **higher salary growth** per year of experience than the **red group**, as indicated by the steeper slope of the blue line.
                
            * The **red group** has a shallower line, suggesting a weaker relationship between experience and salary.
                
        
        Conclusion:
        
        * Both groups show a positive correlation between **experience** and **salary**.
            
        * The blue group shows a **steeper salary increase** with experience compared to the red group.
            
        * There's more variability in the red group, indicated by the wider confidence interval.
            
        * <mark>The </mark> **<mark>blue group's model</mark>** <mark> is likely more </mark> **<mark>accurate</mark>** <mark> or </mark> **<mark>reliable</mark>** <mark> because the shade area is narrower.</mark>
            
        
    * **Example**:
        
        ```python
        sns.lmplot(x="experience", y="salary", data=employee_data)
        ```
        
* `regplot`
    
    * ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728369360370/97471c23-6d7d-46fa-8a85-b444e877ca31.png align="center")
        
        **Data**: Advertising spend vs. revenue.
        
    * **Use**: Add <mark>regression line </mark> to show impact of advertising on revenue. same as lmplot on how to interpret but it is just for one group. Relationship between **2 variables**, where `lmplot` = relationship between **2 variables** with an option to compare across **multiple groups**.
        
    * **Example**:
        
        ```python
        sns.regplot(x="ad_spend", y="revenue", data=business_data)
        ```
        
* `residplot`
    
    * ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728369415516/4a565de9-f577-42e9-98a9-5878c0209c46.png align="center")
        
        **Data**: Predictions vs. residuals in a sales model.
        
    * **Use**: Visualize residuals to check for model accuracy.
        
          
        How to read?  
        **Flat and scattered** = Good model fit.
        
    * **Curve or pattern** = Try a more complex model (e.g., add polynomial terms, interaction terms).
        
    * **Clusters or separate groups** = There might be a categorical variable or need for separate models.
        
    * **Fanning out or funneling in** = Possible heteroscedasticity; consider transforming your variables or using a model that accounts for this.
        
    * **Example**:
        
        ```python
        sns.residplot(x="predictions", y="residuals", data=model_data)
        ```
        

---

### 2\. **Distribution Plots**

**(Understand how data is distributed)**

* `displot`
    
    * ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728369961017/ded68fb5-1353-4c2b-a6d1-d97cf1a9495e.png align="center")
        
        **Data**: Height of people in a survey.
        
    * **Use**: Show height distribution.  
        **Basketball players** (likely the taller group) will have a peak in the higher range (e.g., 190-220 cm).
        
    * **Football players** will be more spread out, with a moderate peak.
        
    * **Tennis players** might have a lower peak around the shorter heights (e.g., 160-185 cm).
        
    * **Example**:
        
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Sample data: heights of people in three groups (e.g., sports)
        data = {
            'height': [175, 180, 165, 178, 182, 190, 210, 220, 185, 188, 195, 198, 205, 172, 160, 178, 182, 185, 192, 195, 199],
            'group': ['Basketball', 'Basketball', 'Basketball', 'Basketball', 'Basketball', 
                      'Basketball', 'Basketball', 'Basketball', 'Football', 'Football', 
                      'Football', 'Football', 'Football', 'Football', 'Football', 'Tennis', 
                      'Tennis', 'Tennis', 'Tennis', 'Tennis', 'Tennis']
        }
        
        # Plot a displot with kde
        sns.displot(data=data, x="height", hue="group", kind="kde")
        
        plt.title("Height Distribution of Athletes by Sport")
        plt.xlabel("Height (cm)")
        plt.ylabel("Density")
        plt.show()
        ```
        
* `histplot`
    
    * **Data**: Test scores of students.
        
    * **Use**: Show frequency of scores in bins.
        
    * **Example**:
        
        ```python
        sns.histplot(student_scores, bins=20)
        ```
        
* `kdeplot`
    
    * **Data**: House prices in a city.
        
    * **Use**: Smooth out distribution of house prices.
        
    * **Example**:
        
        ```python
        sns.kdeplot(data=house_prices, x="price")
        ```
        
* 1\. **Using Exact Values (Histogram)**:
    
    * **Focus**: You want to know the **precise count** of houses in each price range.
        
    * **Typical Questions**:
        
        * *"How many houses are priced between $150,000 and $200,000?"*
            
        * *"What is the most common price range?"*
            
        * *"How many houses are listed at exactly $250,000?"*
            
    * **Example**:
        
        ```python
        pythonCopy codesns.histplot(data=house_prices, x="price", bins=20)
        ```
        
    * **Answer**:
        
        * "There are **50 houses** priced between $150,000 and $200,000."
            
        * "The **most common price range** is $175,000 to $200,000, with exactly **20 houses** listed in that range."
            
        * "There are **3 houses** listed exactly at $250,000."
            
    * **Use Case**: Exact values are ideal for answering questions when you need a **specific count of houses** in various price ranges or at specific price points.
        
    
    ---
    
    2\. **Using Smoothed Distribution (KDE Plot)**:
    
    * **Focus**: You want to understand the **overall shape** or **trend** of house prices, without focusing on the precise counts.
        
    * **Typical Questions**:
        
        * *"What is the general price range where most houses are concentrated?"*
            
        * *"Does the price distribution look like it’s skewed (e.g., more houses on the lower/higher end)?"*
            
        * *"Where do house prices tend to cluster, and how spread out are they?"*
            
    * **Example**:
        
        ```python
        pythonCopy codesns.kdeplot(data=house_prices, x="price")
        ```
        
    * **Answer**:
        
        * "Most house prices are **concentrated** between $150,000 and $200,000."
            
        * "The price distribution is **right-skewed**, meaning there are more houses priced on the **lower end**, with fewer high-priced houses."
            
        * "House prices tend to **cluster** around $175,000, and the spread is **moderate**, with prices ranging from $100,000 to $250,000."
            
    * **Use Case**: KDE is ideal when you want a **big-picture** view of the **distribution** and trends, focusing on **general patterns** rather than exact counts.  
          
        `ecdfplot` **(Empirical Cumulative Distribution Function)**
        
    
    * ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728369175110/4bb3e9a5-a7d9-461c-a001-a1c7ae43c3e6.png align="center")
        
        **Data**: Time spent on different tasks.
        
    * **Use**: Show the proportion of tasks completed within a certain time
        
    * **Example**:
        
        ```python
        sns.ecdfplot(data=task_times, x="time_spent")
        ```
        
    
    Got it! Let's simplify by thinking about something more relatable than tasks—let’s use **homework completion** as an example.  
    
    You’re tracking how long it takes for students to finish their homework. Each student takes a different amount of time, and you want to see how quickly **different students** finish.
    
    What the **ECDF Plot** Does:
    
    * **X-axis**: Shows the **time** (in minutes) each student took to finish their homework.
        
    * **Y-axis**: Shows the **percentage** of students who finished their homework by a certain time.
        
    
    Example:
    
    1. If the **x-value is 30 minutes** and the **y-value is 0.5 (50%)**, it means that **50% of the students** finished their homework in **30 minutes or less**.
        
    2. If the **x-value is 60 minutes** and the **y-value is 0.9 (90%)**, it means that **90% of the students** finished in **60 minutes or less**.
        
    
    Easy Breakdown:
    
    * **At 30 minutes (x = 30)**: Half of the students are done.
        
    * **At 60 minutes (x = 60)**: Almost all students (90%) are done.
        
    
    ---
    
    Questions the ECDF Helps Answer:
    
    * *"How many students finished within 20 minutes?"*  
        → Look at **x = 20** and find the **y-value** (percentage).
        
    * *"How long does it take for 75% of students to finish?"*  
        → Look at **y = 0.75 (75%)** and find the **x-value** (time).
        
    
    ---
    
* `rugplot`
    
    * ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728369224379/bd94ac13-1fbd-4089-9395-3529e97b3303.png align="center")
        
        **Data**: Sales amounts.
        
    * **Use**: Add tick marks for each sale to show data points along the axis.
        
    * **Example**:
        
        ```python
        sns.rugplot(data=sales_data, x="amount")
        ```
        
* `distplot` *(deprecated)*
    
    * ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728369261941/3e9b9e33-a3e0-41ac-8a0f-1bc7527b3375.png align="center")
        
        **Data**: Product prices in a store.
        
    * **Use**: Plot <mark>combined histogram and kde of prices.</mark>
        
    * **Example**:
        
        ```python
        sns.distplot(store_data["prices"])
        ```
        

---

### 3\. **Categorical Data Plots**

**(Compare categories)**

* `catplot`
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728369980715/da22fc6c-3ffb-4a13-bd6f-c8c5289321db.png align="center")
    
    * **Data**: Average salary by job position.
        
    * **Use**: Compare salary distributions across different jobs.
        
    * **Example**:
        
        ```python
        sns.catplot(x="position", y="salary", data=employee_data, kind="bar")
        ```
        
* `stripplot`
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728370006484/f0b13dc2-b8b9-42c7-8748-6fb9b18094d2.png align="center")
    
    * **Data**: Sales for different regions.**blue** dots representing **Males** and **orange** dots representing **Females**.  
          
        **Which group has higher sales amounts**: Look at the spread and concentration of dots. If the **blue dots** (males) extend further to the right, it suggests that males are making **larger sales**.
        
    * **Which group has more small sales**: If **orange dots** (females) are clustered more around the lower sales amounts, it shows females tend to make **smaller sales**.
        
    * **Use**: Show individual data points across regions.
        
    * **Example**:
        
        ```python
        sns.stripplot(x="region", y="sales", data=sales_data)
        ```
        
* `swarmplot`
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728370059951/4733dac2-c4af-4ed1-bb17-140a5d5efe95.png align="center")
    
    * **What the Plot Shows**:
        
        * **X-Axis**: The **days of the week** (Thursday, Friday, Saturday, and Sunday).
            
        * **Y-Axis**: The **tips** received (likely in dollars).
            
        * **Dots**: Each dot represents an **individual tip** given by a customer on that day.
            
        
        **Key Points to Notice**:
        
        1. **No Overlapping Points**:
            
            * <mark>Unlike a stripplot where points might stack on top of each other, </mark> **<mark>swarmplot spreads the dots</mark>** <mark> horizontally to avoid overlap.</mark>
                
            * <mark>This gives a </mark> **<mark>clearer picture</mark>** <mark> of how many tips fall in the same range, especially when there are a lot of tips at similar amounts.</mark>
                
        2. **Distribution by Day**:
            
            * Each day shows a **different spread** of tips.
                
            * For example, **Saturday** seems to have a wider range of tips (from $2 up to $10), indicating that the tips are more variable that day.
                
            * **Thursday and Friday** have more **clustered tips**, mainly between $2 and $4, with fewer higher tips.
                
        3. **Comparison Across Days**:
            
            * You can easily compare **which day** has **higher tips** on average. For example, **Saturday** seems to have more **high-value tips** compared to other days, especially since there are multiple dots higher up on the y-axis.
                
            * **Sunday** also shows some higher tips (around $6), while **Friday** has more concentrated smaller tips.
                
        4. **Outliers**:
            
            * **Saturday** has a **clear outlier** with a tip above $8, which stands out compared to the rest of the days.
                
        
        **How to Interpret This**:
        
        * **Thursday and Friday** have more consistent tips that are **concentrated** around the lower amounts (mostly between $2 and $4).
            
        * **Saturday** has a **wider distribution**, with tips ranging from $2 to $10, indicating a broader range of generosity.
            
        * **Sunday** has some **higher tips** around $5 or $6 but is still somewhat clustered like Friday.
            
        
    * **Example**:
        
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Sample data: tips given by customers on different days
        tips_data = sns.load_dataset("tips")  # Built-in dataset from seaborn
        
        # Plot the swarmplot to show the distribution of tips across different days
        sns.swarmplot(x="day", y="tip", data=tips_data)
        
        # Title and labels for better understanding
        plt.title("Distribution of Tips Across Different Days")
        plt.xlabel("Day of the Week")
        plt.ylabel("Tip Amount (in dollars)")
        
        plt.show()
        ```
        
* `boxplot`
    
    * **Data**: Monthly expenses.
        
    * **Use**: Show distribution and outliers for expenses.
        
    * **Example**:
        
        ```python
        sns.boxplot(x="month", y="expense", data=expense_data)
        ```
        
* `violinplot`
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728375853125/3868cb59-8a70-4c89-8e84-60f43ad29438.png align="center")
    
    * **Data**: Titanic Dataset.
        
    * ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728376666663/95128795-9146-4279-bef7-34108b772d25.png align="center")
        
        **Detailed Explanation of Titanic Violin Plot**:
        
        #### **1\. Median (Q2)**:
        
        * The **white dot** in the center of each violin represents the **median age** of passengers within each class.
            
        * **Median** is the middle value, meaning 50% of the passengers are younger and 50% are older than this age.
            
        * For example, in **First Class**, the median age of passengers might be around **40 years**, while in **Third Class**, the median age could be lower, around **20-25 years**.
            
        
        #### **2\. Quartiles (Q1, Q3)**:
        
        * The **thick black bar** within the violin represents the **interquartile range (IQR)**, which spans from the **first quartile (Q1, 25%)** to the **third quartile (Q3, 75%)**.
            
        * This shows the range of ages where the **middle 50% of passengers** fall.
            
        * For instance, in **Second Class**, you might see that 50% of the passengers are aged between **20 and 50**.
            
        
        #### **3\. Minimum and Maximum**:
        
        * The **ends of the violin** represent the **minimum (Q0, 0%)** and **maximum (Q4, 100%)** ages for each group.
            
        * For example, the youngest passengers (minimum) in **Third Class** might be infants, while the oldest passengers (maximum) in **First Class** could be in their 70s or 80s.
            
        
        #### **4\. Density Plot (Width of the Violin)**:
        
        * The **width** of the violin at any point shows the **density of data points** (ages in this case). A **wider section** means more passengers are in that age range.
            
        * For example, in **First Class**, you might see that the violin is **wider** around the ages of **40 to 50**, meaning many passengers in First Class were in this age group.
            
        * In contrast, in **Third Class**, the violin might be **wider** around the **20 to 30** age range, showing many younger passengers.
            
        
        #### **5\. Comparing Survival Status (Alive vs. Not Alive)**:
        
        * The violin plot is **split** to compare two groups:
            
            * **Left side**: Passengers who **did not survive** (blue).
                
            * **Right side**: Passengers who **survived** (orange).
                
        * In **First Class**, you might notice that more survivors are clustered around the **40-50** age range, whereas the **non-survivors** are spread across a wider age range.
            
        * In **Third Class**, there might be a **greater proportion of non-survivors** (blue) who are younger, and the **survivors** (orange) are fewer in this age range.
            
        
        #### **6\. Outliers**:
        
        * Outliers are shown as **small individual points** outside the main body of the violin. These points represent **passengers who fall outside the normal age range** for their class.
            
        * For example, you might see a **very old passenger** (around 80) in **First Class** as an outlier, or a **young child** as an outlier in **Third Class**.
            
        
    * **Example Code**:
        
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Sample data: Sales distribution by product and demand level
        # Here, we're using the built-in 'tips' dataset to simulate a sales scenario.
        # In practice, replace 'tips' with your actual sales_data.
        sales_data = sns.load_dataset("titanic")
        
        # Plot violinplot with sales (age in this case) split by 'alive' status (e.g., high vs. low demand)
        sns.violinplot(x="class", y="age", hue="alive", data=sales_data, split=True)
        
        # Title and labels
        plt.title("Sales Distribution by Product and Demand Level")
        plt.xlabel("Product Type (class)")
        plt.ylabel("Sales (age)")
        
        plt.show()
        ```
        
* `boxenplot`
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728370094308/f8a4d3d4-e577-479e-96b8-13170288933e.png align="center")
    
    * **Data**: Tips per day
        
    * **Thursday**:
        
        * **Navy blue and light blue boxes** show that tips are **spread** in the upper percentiles, with **greater variation** in higher tips.
            
        * **Outliers** exist, but the overall range is moderate, with most tips between **$10 and $30**.
            
    * **Friday**:
        
        * Only an **orange box** shows, meaning tips are **clustered** within a **narrower range** (less variability).
            
        * Data points **above** and **below the orange box** are outside this range:
            
            * **<mark>Above</mark>** <mark> the box: The top 25% of the data points (75th to 100th percentile).</mark>
                
            * **<mark>Below</mark>** <mark> the box: The bottom 25% of the data points (0th to 25th percentile).</mark>
                
            * **Why No Light Orange Box?**
                
                * The light orange box would represent the **next layer** of data (like the 75th to 90th percentile), but there isn't enough variability or spread in the Friday data to generate this additional box.
                    
        * Tips are mostly **consistent** between **$10 and $25**, with fewer extreme values or outliers.
            
    * **Saturday**:
        
        * **Multiple light green layers** indicate that tips are **widely spread** across high values.
            
        * <mark>The </mark> **<mark>core box</mark>** <mark> (dark green or dark red) still represents the </mark> **<mark>IQR</mark>** <mark> (25th to 75th percentile).</mark>
            
        * <mark>The </mark> **<mark>light green/red boxes</mark>** <mark> extend beyond the IQR:</mark>
            
            * **<mark>Next layer</mark>**<mark>: Covers </mark> **<mark>75th to 87.5th percentile</mark>** <mark> of the data (upper middle).</mark>
                
            * **<mark>Another layer</mark>**<mark>: Covers </mark> **<mark>87.5th to 93.75th percentile</mark>** <mark> (upper-upper middle).</mark>
                
            * **<mark>Top layer</mark>**<mark>: Represents the </mark> **<mark>very top percentiles</mark>** <mark> of the data (e.g., 93.75th to 100th percentile).</mark>
                
        * There is **significant variation** in tips, with amounts reaching up to **$50**. Higher tips are more frequent.
            
        * **More outliers** and **larger spread** compared to other days.
            
    * **Sunday**:
        
        * **Multiple light red layers** show tips are **highly variable**, similar to Saturday.
            
        * A broad range of tips, with many values in the **upper percentiles**. Some tips reach **$50**, and there are many **outliers**.
            
    * **Example**:
        
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Load the sample 'tips' dataset from seaborn
        tips_data = sns.load_dataset("tips")
        
        # Create the boxenplot showing the distribution of total bill by day
        sns.boxenplot(x="day", y="total_bill", data=tips_data)
        
        # Add title and labels for clarity
        plt.title("Distribution of Total Bill by Day")
        plt.xlabel("Day of the Week")
        plt.ylabel("Total Bill (in dollars)")
        
        # Show the plot
        plt.show()
        ```
        
* `pointplot`
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728379834847/fead60a1-7b23-42a0-8ede-ec3396fa4862.png align="center")
    
    **January**:
    
    * **Point**: The **dot** for January represents the **average number of passengers** for January across all years in the dataset.
        
        * The dot is around **250 passengers**, meaning that, on average, there were about **250 passengers** flying in January across all the years in the dataset.
            
            ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728382652580/ad193069-17fe-4db9-87ad-4400b1968301.png align="center")
            
              
            
    * **<mark>Error Bar</mark>**<mark>: The </mark> **<mark>vertical line</mark>** extending from the dot is the **error bar**, which shows the **variation** in the **average number of passengers** across different years.
        
        * The **bottom** of the error bar for January is close to **200 passengers**, while the **top** is closer to **300 passengers**. This means that in some years, <mark>the </mark> **<mark>average number of passengers</mark>** in January was closer to **200**, and in other years, it was closer to **300**.
            
        * <mark>The </mark> **<mark>error bar</mark>** <mark> reflects how much the </mark> **<mark>average</mark>** <mark> could vary from year to year, not the exact number of passengers.</mark>
            
    
    ---
    
    **February**:
    
    * **Point**: The **dot** for February represents the **average number of passengers** for February across all years in the dataset.
        
        * The dot is slightly **lower** than January’s dot, around **230 passengers** on average.
            
    * **Error Bar**: T<mark>he error bar for February is </mark> **<mark>shorter</mark>** <mark> than January's, indicating </mark> **<mark>less variation</mark>** <mark> in the </mark> **<mark>average number of passengers</mark>** <mark> across different years.</mark>
        
        * The error bar ranges roughly from **220 to 250 passengers**, meaning that in most years, the **average number of passengers** in February was between **220 and 250**.
            
        * The shorter error bar suggests that the average number of passengers in February was **more consistent** from year to year compared to January.
            
    
    ---
    
    ### **Summary**:
    
    * **January**: The **average** number of passengers is around **250**, with **more variation** (error bar ranging from 200 to 300).
        
    * **February**: The **average** number of passengers is around **230**, with **less variation** (error bar ranging from 220 to 250).
        
    
    This revised explanation accurately reflects the role of the **error bars** in showing the **variation** in **average values** across years, rather than indicating the exact number of passengers.  
    
    * **Example**:
        
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Load the 'flights' dataset from seaborn
        flights_data = sns.load_dataset("flights")
        
        # Pivot the data to get it in a wide format (years as rows, months as columns)
        flights_wide = flights_data.pivot(index="year", columns="month", values="passengers")
        
        # Create a point plot using the wide-format data
        sns.pointplot(data=flights_wide)
        
        # Add labels and title
        plt.title("Monthly Number of Passengers (Averaged Across Years)")
        plt.xlabel("Month")
        plt.ylabel("Number of Passengers")
        
        # Show the plot
        plt.show()
        ```
        
* `barplot`
    
    * **Data**: Average income by education level.
        
    * **Use**: Compare income means for each education level.
        
    * **Example**:
        
        ```python
        sns.barplot(x="education", y="income", data=income_data)
        ```
        
* `countplot`
    
    * **Data**: Count of people in different age groups.
        
    * **Use**: Count occurrences of each category.
        
    * **Example**:
        
        ```python
        sns.countplot(x="age_group", data=people_data)
        ```
        

---

### 4\. **Heatmaps and Cluster Maps**

**(Visualize matrices and clustering)**

* `heatmap`
    
    * **Data**: Correlation matrix of financial indicators.
        
    * **Use**: Visualize correlations between financial metrics.
        
    * **Example**:
        
        ```python
        sns.heatmap(corr_matrix, annot=True)
        ```
        
* `clustermap`
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728370186971/e55275b7-fae3-4d7c-ba08-37b33ecdd533.png align="center")
    
    * **Data**: Customer purchase patterns.
        
    * **Use**: Perform hierarchical clustering to group similar customers.
        
    * **Example**:
        
        ```python
        sns.clustermap(purchase_data)
        ```
        

---

### 5\. **Grids and Pair Plots**

**(Subplots and multiple plots)**

* `FacetGrid`
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728370379225/8c2f8b37-3e6c-4010-97a9-0ccc61a4f9d3.png align="center")
    
    * **Data**: Sales by region and product category.
        
    * **Use**: Create subplots for each region.
        
    * **Example**:
        
        ```python
        g = sns.FacetGrid(sales_data, col="region", row="category")
        g.map(sns.scatterplot, "day", "sales")
        ```
        
* `pairplot`
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728370260236/58db6a4c-4e4f-4d2d-bb71-ed98003096bc.png align="center")
    
    * **Data**: Features of cars (price, engine size, horsepower).
        
    * **Use**: Show relationships between all features.
        
    * **Example**:
        
        ```python
        sns.pairplot(car_data)
        ```
        
* `PairGrid`
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728370330677/786c9e2d-bd17-43c3-a6d8-317d7a007420.png align="center")
    
    * **Data**: Different student scores (math, reading, writing).
        
    * **Use**: Fine-tune pairwise plots between all scores.
        
    * **Example**:
        
        ```python
        g = sns.PairGrid(student_data)
        g.map_diag(sns.histplot)
        g.map_offdiag(sns.scatterplot)
        ```
        
* `jointplot`
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728370411685/cc333c1e-9981-49b5-92d5-3db9d2b2cb9e.png align="center")
    
    * **Data**: Hours of study vs. exam score.
        
    * **Use**: Combine scatterplot and distribution.
        
    * **Example**:
        
        ```python
        sns.jointplot(x="study_hours", y="exam_score", data=student_data)
        ```
        
* `JointGrid`
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1728370611616/11d6a065-71f8-4ff0-b23c-7020f7c8f700.png align="center")
    
    * **Data**: Weight vs. height.
        
    * **Use**: Fine-tune joint plots with additional control.
        
    * **Example**:
        
        ```python
        g = sns.JointGrid(data=people_data, x="height", y="weight")
        g.plot(sns.scatterplot, sns.histplot)
        ```
        

---