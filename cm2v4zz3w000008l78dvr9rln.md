---
title: "How to get started with Gen AI, SQL AI on Google Cloud for free"
seoTitle: "How to get started with Gen AI, SQL AI on Google Cloud for free"
seoDescription: "How to get started with Gen AI, SQL AI on Google Cloud for free"
datePublished: Wed Oct 30 2024 00:27:03 GMT+0000 (Coordinated Universal Time)
cuid: cm2v4zz3w000008l78dvr9rln
slug: how-to-get-started-with-gen-ai-sql-ai-on-google-cloud-for-free
tags: ai, machine-learning, google-cloud, sql, bigquery-ml

---

Google Vertex AI's integration with **BigQuery** allows you to **bring AI into SQL workflows** in a seamless manner. This capability effectively enables data scientists, data engineers, and analysts to leverage the power of **machine learning** directly within SQL-based environments, which is really exciting because it allows AI capabilities to be accessible to those who primarily work with SQL and structured data. Here's a deeper dive into what this means:

# BigQuery ML capability

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1730247418486/2137fa52-9418-469a-90a0-d3d376f7750f.png align="center")

# How **BQML** allows analysts to work on different types of ML models, all with SQL?

It may seem a bit unfair that **machine learning (ML) models**, which traditionally required extensive knowledge of Python, data wrangling, and ML frameworks, can now be built by analysts using **just SQL**. The development of tools like **BigQuery ML (BQML)** simplifies this process for data analysts by extending their existing SQL skills into the realm of machine learning, effectively blurring the lines between **data analysts** and **data scientists**.

Here's how **BQML** allows analysts to work on different types of ML models, all with SQL. I'll show a few **boilerplate code samples** for different machine learning models so you can get an idea of why SQL-based ML lowers the barrier.

### **1\. Linear Regression Model**

Linear regression is used to predict continuous outcomes. Analysts can use SQL to create a model to, for example, predict **total sales** based on customer spending history.

**SQL Code for Linear Regression:**

```sql
-- Create a Linear Regression model to predict Total Sales
CREATE OR REPLACE MODEL `my_dataset.sales_prediction_model`
OPTIONS(
  model_type='linear_reg',
  input_label_cols=['TotalSales']  -- The column we want to predict
) AS
SELECT
  CustomerID,
  TotalSpent,
  TotalOrders,
  TotalSales
FROM
  `my_dataset.sales_data`;
```

**Explanation:**

* In this code, the analyst defines a **linear regression model** using `CREATE MODEL`.
    
* The target column (`TotalSales`) is the one we want the model to predict.
    
* The model uses the features (`TotalSpent`, `TotalOrders`) to predict `TotalSales`.
    

### **2\. Logistic Regression Model**

Logistic regression is used to predict binary outcomes, such as whether a customer will **churn or not churn**.

**SQL Code for Logistic Regression:**

```sql
-- Create a Logistic Regression model to predict customer churn
CREATE OR REPLACE MODEL `my_dataset.customer_churn_model`
OPTIONS(
  model_type='logistic_reg',
  input_label_cols=['churned']  -- The column indicating whether the customer churned (0 or 1)
) AS
SELECT
  CustomerID,
  TotalSpent,
  LastOrderDate,
  NumOrders,
  churned
FROM
  `my_dataset.customer_data`;
```

**Explanation:**

* Here, analysts are building a **churn prediction model** to determine if customers will churn (binary `churned` column).
    
* This is useful in **customer retention strategies** without the need for in-depth programming skills.
    

### **3\. K-Means Clustering Model**

K-Means clustering is used for **unsupervised learning** to group similar records, such as segmenting customers into different categories.

**SQL Code for Clustering:**

```sql
-- Create a K-Means clustering model to segment customers
CREATE OR REPLACE MODEL `my_dataset.customer_segmentation_model`
OPTIONS(
  model_type='kmeans',
  num_clusters=3  -- Specify the number of clusters we want
) AS
SELECT
  TotalSpent,
  AvgOrderValue,
  NumOrders
FROM
  `my_dataset.customer_behavior_data`;
```

**Explanation:**

* This **clustering model** groups customers into 3 different segments based on spending (`TotalSpent`), order value (`AvgOrderValue`), and number of orders (`NumOrders`).
    
* Analysts can use clustering results to **personalize marketing strategies** for different customer segments.
    

### **4\. Time Series Model (ARIMA)**

Time series models like **ARIMA** are used to predict future values based on historical data. For example, predicting **monthly sales**.

**SQL Code for Time Series Forecasting:**

```sql
-- Create a Time Series ARIMA model for sales forecasting
CREATE OR REPLACE MODEL `my_dataset.sales_forecasting_model`
OPTIONS(
  model_type='arima',
  time_series_timestamp_col='date',
  time_series_data_col='sales'
) AS
SELECT
  date,
  sales
FROM
  `my_dataset.sales_data`;
```

**Explanation:**

* This ARIMA model is built to predict future `sales` based on the `date` column.
    
* No need for extensive ARIMA model expertise—**BigQuery ML** takes care of the time series analysis, and the analyst simply needs to prepare the historical data.
    

### **5\. Deep Neural Network (DNN) for Classification**

If the problem is more complex, such as predicting product recommendations, a **deep neural network (DNN)** can be used.

**SQL Code for a Deep Neural Network:**

```sql
-- Create a DNN Classifier to predict customer purchasing behavior
CREATE OR REPLACE MODEL `my_dataset.purchase_prediction_model`
OPTIONS(
  model_type='dnn_classifier',
  hidden_units=[32, 16, 8],  -- Define the structure of the DNN
  input_label_cols=['will_purchase']
) AS
SELECT
  CustomerID,
  TotalSpent,
  LastOrderDate,
  NumOrders,
  will_purchase  -- 0 or 1, whether the customer will make a purchase
FROM
  `my_dataset.customer_data`;
```

**Explanation:**

* In this example, analysts can create a **deep neural network** with different layers (`hidden_units=[32, 16, 8]`), all within SQL.
    
* This type of model allows predictions that require learning **complex patterns** from data, but the analyst only has to provide the features.
    

### **Key Benefits and Why Analysts Don’t Need Python**

1. **SQL Familiarity**:
    
    * **Data Analysts** are already skilled in **SQL** for querying and analyzing data. By allowing them to **create ML models directly in SQL**, BigQuery ML removes the need to learn **Python**, which would be a significant barrier to entry for many analysts.
        
2. **Automatic Feature Handling and Model Tuning**:
    
    * When working with **Python and ML frameworks** like TensorFlow or scikit-learn, you need to do tasks like **data preprocessing**, **feature engineering**, and **hyperparameter tuning**. In **BigQuery ML**, much of this process is **automated**.
        
    * For example, normalization and feature scaling are automatically handled by BigQuery, saving a lot of effort that data scientists usually need to do manually.
        
3. **End-to-End Process in SQL**:
    
    * Analysts don’t have to **switch environments** or learn how to use Python libraries to get their data into shape, train a model, and evaluate it. This **one-stop solution** saves a lot of time and eliminates the complexity of moving data between different tools.
        
4. **Democratizing Machine Learning**:
    
    * BigQuery ML aims to **democratize machine learning** by making it accessible to those who have business domain expertise but aren’t necessarily ML experts. The goal is to enable these users to **generate valuable insights** without having to wait for data scientists to build the models for them.
        

### **Why It May Feel "Unfair" to Data Scientists**:

* **Data Scientists** spend a lot of time learning about **algorithms, hyperparameter tuning, feature engineering, and model evaluation**. They work on making models **more accurate** and understanding the **inner workings** of each model type.
    
* However, tools like BQML are not designed to replace data scientists. Instead, they:
    
    * **Empower analysts** to work on **simpler models** themselves, freeing up data scientists to focus on **more complex problems** that require a deeper understanding of ML.
        
    * Allow data scientists to **prototype** models quickly using SQL and then move to more sophisticated frameworks for **advanced models**.
        

### **Summary**:

By integrating ML capabilities into SQL, BQML **enables data analysts** to create, train, and evaluate machine learning models without needing programming skills. For data scientists, this means they can focus on the **nuances and complexities** of more advanced ML problems. While it may seem "unfair" in terms of learning effort, the **roles and contributions** of data analysts and data scientists are still distinct but now more complementary.

If you'd like to see more specific ML examples or want guidance on transitioning from analyst-level ML with SQL to more advanced data science, feel free to ask!  
What sample projects we can run in BigQuery ML for free?

Certainly! Here are **five specific projects** for analysts to try with **BigQuery ML (BQML)**. These projects use **public datasets** or easily available datasets, making it feasible to test them with **limited budget** or even within a **free trial** offered by Google Cloud.

### **1\. Customer Churn Prediction**

* **Algorithm**: Logistic Regression
    
* **Dataset**: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
    
* **Project Description**:
    
    * Predict whether customers are likely to **churn** (i.e., stop using services).
        
    * Use features such as **monthly charges**, **contract type**, and **tenure**.
        
* **Steps**:
    
    * Import the dataset into **BigQuery**.
        
    * Train a **logistic regression** model with BQML to classify whether a customer will churn (`Yes/No`).
        

**Suggested SQL Code**:

```sql
CREATE OR REPLACE MODEL `telco.customer_churn_model`
OPTIONS(model_type='logistic_reg', input_label_cols=['Churn']) AS
SELECT *
FROM `your_project.your_dataset.telco_customer_churn`;
```

### **2\. House Price Prediction**

* **Algorithm**: Linear Regression
    
* **Dataset**: [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) (Kaggle)
    
* **Project Description**:
    
    * Predict the **median house value** for California districts based on factors such as **population**, **median income**, and **proximity to the ocean**.
        
* **Steps**:
    
    * Import the dataset to BigQuery.
        
    * Use **linear regression** to predict the `median_house_value` for given features.
        

**Suggested SQL Code**:

```sql
CREATE OR REPLACE MODEL `housing.price_prediction_model`
OPTIONS(model_type='linear_reg', input_label_cols=['median_house_value']) AS
SELECT *
FROM `your_project.your_dataset.california_housing_data`;
```

### **3\. Customer Segmentation (K-Means Clustering)**

* **Algorithm**: K-Means Clustering
    
* **Dataset**: [Northwind Traders Dataset](https://github.com/microsoft/sql-server-samples/tree/master/samples/databases/northwind-pubs) (GitHub)
    
* **Project Description**:
    
    * Segment customers based on purchasing behavior—use the **Northwind dataset** (a popular retail sales dataset).
        
    * Features can include **total purchase amount**, **average order size**, and **order frequency**.
        
* **Steps**:
    
    * Import the **Northwind dataset**.
        
    * Use **K-means clustering** to group customers into different segments to understand purchasing patterns.
        

**Suggested SQL Code**:

```sql
CREATE OR REPLACE MODEL `northwind.customer_segmentation`
OPTIONS(model_type='kmeans', num_clusters=4) AS
SELECT
  CustomerID,
  SUM(UnitPrice * Quantity) AS TotalSpent,
  COUNT(OrderID) AS TotalOrders
FROM
  `your_project.your_dataset.orders`
GROUP BY
  CustomerID;
```

### **4\. Sales Forecasting (Time Series with ARIMA)**

* **Algorithm**: Time Series Forecasting (ARIMA)
    
* **Dataset**: [Retail Sales Dataset](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) (Kaggle)
    
* **Project Description**:
    
    * Forecast **monthly sales** for a retail store.
        
    * Use the historical sales data to train a **time series ARIMA** model to predict future sales.
        
* **Steps**:
    
    * Load the dataset into BigQuery.
        
    * Use ARIMA to forecast future sales.
        

**Suggested SQL Code**:

```sql
CREATE OR REPLACE MODEL `retail.sales_forecasting`
OPTIONS(model_type='arima', time_series_timestamp_col='date', time_series_data_col='sales') AS
SELECT
  date,
  sales
FROM
  `your_project.your_dataset.retail_sales_data`;
```

### **5\. Predict Loan Default (Deep Neural Network)**

* **Algorithm**: Deep Neural Network (Basic)
    
* **Dataset**: [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (Kaggle)
    
* **Project Description**:
    
    * Predict whether a loan applicant will **default**.
        
    * Use customer features like **income**, **loan amount**, and **credit history** to predict the probability of loan default.
        
* **Steps**:
    
    * Import the **credit risk dataset** to BigQuery.
        
    * Use a basic **Deep Neural Network** (DNN) to train a classification model that predicts `default`.
        

**Suggested SQL Code**:

```sql
CREATE OR REPLACE MODEL `credit_risk.loan_default_prediction`
OPTIONS(model_type='dnn_classifier', hidden_units=[32, 16, 8], input_label_cols=['default']) AS
SELECT *
FROM
  `your_project.your_dataset.credit_risk_data`;
```

### **Summary of Projects**

| Project | Dataset | Algorithm | Model Goal |
| --- | --- | --- | --- |
| Customer Churn Prediction | Telco Customer Churn | Logistic Regression | Predict if customer will churn |
| House Price Prediction | California Housing Prices | Linear Regression | Predict house prices |
| Customer Segmentation | Northwind Traders | K-Means Clustering | Segment customers |
| Sales Forecasting | Retail Sales Dataset | Time Series (ARIMA) | Forecast future sales |
| Predict Loan Default | Credit Risk Dataset | Deep Neural Network | Predict loan default |

# Limitations?

Absolutely, there are indeed limitations to **SQL-based ML tools** like **BigQuery ML (BQML)**, and these tools are not capable of completely replacing the depth of knowledge and skill required by data scientists—particularly when it comes to **deep learning (DL)** and more complex **machine learning (ML)** projects. Let’s explore these **limitations** to better understand why the role of a data scientist remains essential, and why expertise in Python, machine learning, and deep learning is still very much needed.

### **1\. Complexity of Advanced Models**

* **Deep Learning Architectures**:
    
    * SQL-based ML tools like **BQML** can create simple models like linear regression or even some neural networks, but when it comes to **advanced deep learning architectures** like **transformers, CNNs (Convolutional Neural Networks), or RNNs (Recurrent Neural Networks)**, these tools simply aren’t capable.
        
    * Deep learning requires frameworks like **TensorFlow** or **PyTorch**, where you need granular control over model design, activation functions, layer structures, and training behavior. These things cannot be achieved through SQL.
        
* **Custom Models and Architectures**:
    
    * Data scientists often need to create **custom models** that are specifically designed for a particular problem. For instance, building a **custom object detection model** or a **GAN (Generative Adversarial Network)** for image generation requires a deep understanding of neural network architectures and extensive experimentation. SQL tools do not offer the **flexibility** needed to build, tweak, or experiment with such advanced architectures.
        

### **2\. Feature Engineering and Data Preprocessing**

* **Advanced Feature Engineering**:
    
    * SQL can handle **basic feature engineering**—such as aggregations, transformations, or scaling—but advanced feature engineering often requires specialized transformations that go beyond simple SQL capabilities.
        
    * For example, tasks like **text preprocessing** (e.g., tokenization, stopword removal) or **feature extraction** from images require specialized tools and libraries in Python, such as **NLTK**, **spaCy**, or **OpenCV**.
        
* **Complex Data Types**:
    
    * SQL-based ML systems are great with **structured data** (tables and rows), but data scientists often work with **unstructured data** like text, images, and videos. Processing and feature extraction from such data types cannot be done in SQL and requires a detailed understanding of **natural language processing (NLP)**, **computer vision**, and other specialized areas that are best handled by Python and ML libraries.
        

### **3\. Hyperparameter Tuning and Model Optimization**

* **Automatic vs. Custom Tuning**:
    
    * SQL tools like **BQML** do provide **some automated hyperparameter tuning**, but this is often limited and not customizable. In complex ML and DL tasks, **hyperparameter tuning** can make a significant difference in model performance. Data scientists use tools like **Grid Search**, **Random Search**, or **Bayesian Optimization** to carefully tweak the parameters of a model.
        
    * For deep learning, selecting the **right optimizer**, setting **learning rates**, **dropout rates**, and other **hyperparameters** can be the difference between a mediocre model and a state-of-the-art model. These cannot be handled by SQL-based ML systems.
        

### **4\. Limited to General Algorithms**

* **Lack of Algorithm Diversity**:
    
    * BQML provides access to **basic models** like linear regression, logistic regression, clustering (K-means), and deep neural networks, but it lacks support for advanced ML algorithms, such as:
        
        * **Random Forests** or **Gradient Boosting Machines** (e.g., XGBoost) which are frequently used in data science competitions like **Kaggle** due to their power in handling tabular data.
            
        * **Reinforcement Learning (RL)** models that require agents to learn policies by interacting with environments, which cannot be handled by SQL tools.
            
        * **Anomaly Detection** methods that use complex statistical or ensemble techniques to detect outliers in data require custom algorithms, which aren’t available in SQL tools.
            

### **5\. Experimentation and Custom Pipelines**

* **Full Experimentation Control**:
    
    * Data scientists spend a lot of time in the **experimental phase**, testing out different approaches, architectures, preprocessing techniques, and evaluation metrics. Tools like **Jupyter Notebooks** allow interactive exploration, visualization, and debugging that SQL lacks.
        
    * Building a **custom ML pipeline** where data preprocessing, modeling, evaluation, and post-processing steps are well-coordinated is best handled by tools like **Scikit-learn**’s `Pipeline` or **Kubeflow Pipelines**.
        
* **Model Explainability and Debugging**:
    
    * In Python, data scientists can use **explainability tools** like **SHAP** or **LIME** to understand how features impact model predictions. Debugging why a model is behaving in a certain way requires intricate tools and visualization methods, something SQL is not equipped for.
        
    * **TensorBoard**, for example, is used extensively in deep learning projects to visualize training progress, understand model behavior, and debug issues. These tools are simply not available in SQL environments.
        

### **6\. Deployment and Production-Grade Solutions**

* **Custom Inference Logic**:
    
    * Deploying ML models is not just about generating predictions but also involves writing **custom inference logic**, such as how to handle **missing values**, **retraining schedules**, or integrating the model with a **backend API**.
        
    * Python frameworks like **FastAPI**, **Flask**, or tools like **Docker** are needed to deploy models with custom logic in production environments. SQL-based ML stops short of giving you the full flexibility needed for robust, production-grade deployments.
        

### **7\. Data Science Research and Novel Approaches**

* **Pushing State-of-the-Art (SOTA)**:
    
    * Data scientists who work on cutting-edge models—such as improving the performance of NLP models like **BERT** or image classification models like **ResNet**—are continuously **experimenting with novel approaches**. This requires a deep understanding of ML algorithms, advanced mathematics, and the ability to customize every layer of a model.
        
    * The **research domain** of data science and ML involves a lot of trial and error, iterative experimentation, and creativity. SQL-based ML tools are not suitable for pushing the boundaries of what ML can achieve.
        

### **Summary: Where SQL-Based ML Falls Short**

* **Limited to Simple and General Models**: SQL-based ML works well for simple linear regression, logistic regression, and basic clustering but falls short for advanced models like CNNs, transformers, and RL.
    
* **Lack of Flexibility in Experimentation**: No control over custom architectures, hyperparameter tuning, or the ability to experiment with different strategies in real time.
    
* **Limited Feature Engineering and Data Handling**: SQL is limited to structured data and cannot process unstructured data like text, images, or videos.
    
* **Limited Tools for Production and Debugging**: Data scientists need tools like TensorBoard, SHAP, FastAPI, etc., to properly test, debug, and deploy their models in production.
    

### **Roles and Complementarity**

The role of data scientists has evolved and continues to be crucial for:

* **Complex modeling** and building advanced **DL models**.
    
* **Pushing innovations** in machine learning.
    
* **End-to-end solution building**, from preprocessing to model deployment.
    

Tools like **BQML** or **AutoML** are there to **democratize access** to simpler machine learning models, allowing **data analysts** to perform predictive analytics themselves, freeing up data scientists to focus on solving more challenging and **non-standard problems**. Rather than replacing data scientists, SQL-based ML helps distribute workloads more efficiently, ensuring that each role can focus on what they do best.

The beauty is that there’s **room for everyone**—data analysts can solve simpler business problems with ease using SQL-based ML, while data scientists take on more complex challenges that push the boundaries of AI and ML. This synergy is what moves the field forward.

# Example: Northwind Customer Segmentation Project with BQML

Take the **customer segmentation** example using the **Northwind dataset**:

* This is a **retail dataset** that contains tables like `Orders`, `Customers`, `Products`, etc.
    
* Traditionally, you would use SQL to:
    
    * Get insights like **total sales** or **average order value**.
        
    * Prepare data, and then export it to train a model using Python.
        

With **BigQuery ML**, the **ML model training** can happen directly within the SQL environment:

1. You would use a **SQL query** to preprocess the data:
    
    ```python
    sqlCopy codeSELECT
      CustomerID,
      SUM(UnitPrice * Quantity) AS TotalSpent,
      COUNT(OrderID) AS TotalOrders
    FROM
      `your_project.your_dataset.orders`
    GROUP BY
      CustomerID;
    ```
    
2. Then, you use another SQL command to create a **K-means clustering model**:
    
    ```python
    sqlCopy codeCREATE OR REPLACE MODEL `northwind.customer_segmentation`
    OPTIONS(model_type='kmeans', num_clusters=4) AS
    SELECT
      CustomerID,
      TotalSpent,
      TotalOrders
    FROM
      `your_project.your_dataset.orders_data`;
    ```
    
3. Finally, you can make **predictions** or see which segment a customer falls into—all using **SQL**:
    
    ```python
    sqlCopy codeSELECT
      CustomerID,
      predicted_cluster
    FROM
      ML.PREDICT(MODEL `northwind.customer_segmentation`, 
      (SELECT CustomerID, TotalSpent, TotalOrders FROM `your_project.your_dataset.orders_data`));
    ```
    

### **How It Benefits Analysts**

* **Ease of Use**: Analysts only need to work in SQL—no need for Python, Jupyter notebooks, or ML libraries.
    
* **Simplified Integration**: Predictions are available **immediately** in BigQuery and can be used directly in other SQL-based processes, reports, and dashboards.
    
* **Speed**: Training a model on **BigQuery's infrastructure** can be much faster than local environments, especially with large datasets.
    

### **Summary of the SQL + ML Combination in BQML**:

* BQML **adds machine learning functionality** to SQL, allowing ML workflows to happen inside **BigQuery**.
    
* Analysts can use SQL-based commands to **train models, make predictions, and evaluate models**—tasks traditionally reserved for Python or R environments.
    
* **Machine learning with SQL** greatly simplifies the workflow, especially for people who are already familiar with data analytics using SQL, but not necessarily with data science tools like TensorFlow.
    

In short, these projects **appear ML-heavy**, but thanks to **BQML**, everything from **data preprocessing** to **model training** can be done in SQL, which makes them accessible even to those without Python skills. This is what makes BigQuery ML powerful: it extends the capabilities of traditional SQL-based environments to include predictive analytics and machine learning, empowering more data practitioners to solve advanced problems.

# GitHub repositories that demonstrate the integration of BigQuery, Vertex AI, and BigQuery ML

1. Vertex AI MLOps Repository: [https://github.com/statmike/vertex-ai-mlops](https://github.com/statmike/vertex-ai-mlops)
    

This repository provides comprehensive end-to-end workflows for machine learning using Vertex AI. It covers various aspects including BigQuery ML, TensorFlow, and scikit-learn. The repository is being updated to focus more on MLOps approaches for both predictive and generative AI operations\[1\].

2. BigQuery ML Training Tutorial: [https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/bigquery\_ml/get\_started\_with\_bqml\_training.ipynb](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/bigquery_ml/get_started_with_bqml_training.ipynb)
    

This notebook demonstrates how to use Vertex AI in production, specifically focusing on getting started with BigQuery ML training. It's part of the official Vertex AI samples provided by Google Cloud Platform\[2\].

3. Forecasting Retail Demand with Vertex AI and BigQuery ML: [https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/demand\_forecasting/forecasting-retail-demand.ipynb](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/demand_forecasting/forecasting-retail-demand.ipynb)
    

This notebook shows how to train and evaluate a BigQuery ML model for demand forecasting datasets and extract actionable future insights. It combines Vertex AI and BigQuery ML for a practical use case in retail demand forecasting\[3\].

These repositories provide excellent starting points for working with BigQuery, Vertex AI, and BigQuery ML. They offer practical examples and tutorials that can help you gain hands-on experience with these technologies.

I would recommend starting with the "BigQuery ML Training Tutorial" (#2) as it specifically focuses on getting started with BigQuery ML training in the context of Vertex AI. This aligns well with your goal of testing and gaining experience with these technologies.

As for a database name, you could use something descriptive like "retail\_demand\_forecast" or "customer\_segmentation" depending on the specific use case you want to explore. Remember to keep your dataset small (around 5 GB) to stay within the free tier limits as discussed earlier.

Sources \[1\] GitHub - statmike/vertex-ai-mlops [https://github.com/statmike/vertex-ai-mlops](https://github.com/statmike/vertex-ai-mlops) \[2\] Get started with BigQuery ML Training - GitHub [https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/bigquery\_ml/get\_started\_with\_bqml\_training.ipynb](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/bigquery_ml/get_started_with_bqml_training.ipynb) \[3\] Forecasting retail demand with Vertex AI and BigQuery ML - GitHub [https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/demand\_forecasting/forecasting-retail-demand.ipynb](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/workbench/demand_forecasting/forecasting-retail-demand.ipynb)

# How to keep it free?

1. **BigQuery Free Tier**:
    
    * 10 GB of free storage per month\[1\]
        
    * 1 TB of free query processing per month\[1\]
        
2. **Vertex AI Free Tier and Credits**:
    
    * $300 in free credits for new Google Cloud users\[2\]
        
    * 300 hours of free Notebooks usage per month\[3\]
        
3. **BigQuery ML Training Costs**:
    
    * No additional cost for training if the query processing stays within the 1 TB free tier\[4\]
        
4. **Vertex AI Training and Deployment Costs**:
    
    * Varies based on hardware and duration, but can be minimized using CPU-only training and free credits\[3\]
        
5. **Google Cloud Storage Free Tier**:
    
    * 5 GB of free storage\[5\]
        
    * $0.026 per GB per month beyond the free tier\[5\]
        

To get started, you can follow these steps:

1. Go to the Google Cloud Console: [https://console.cloud.google.com/](https://console.cloud.google.com/)
    
2. Create a new project or select an existing one.
    
3. Enable the BigQuery and Vertex AI APIs for your project.
    
4. Open the BigQuery console: [https://console.cloud.google.com/bigquery](https://console.cloud.google.com/bigquery)
    
5. Create a new dataset for your project. You can name it something like `customer_segmentation_dataset`.
    
6. Load your sample customer data into a table within this dataset. You can use the BigQuery web UI, the `bq` command-line tool, or the BigQuery API to load data.
    
7. Follow the BigQuery ML tutorials to train a k-means clustering model on your data: [https://cloud.google.com/bigquery-ml/docs/kmeans-tutorial](https://cloud.google.com/bigquery-ml/docs/kmeans-tutorial)
    
8. Export your trained model to Vertex AI for further refinement or deployment: [https://cloud.google.com/vertex-ai/docs/export/export-model-tabular](https://cloud.google.com/vertex-ai/docs/export/export-model-tabular)
    

By following these steps and keeping your dataset small (around 5 GB), you should be able to complete this project within the free tier limits, incurring minimal or no costs.

\[1\] [https://cloud.google.com/bigquery/pricing#free-tier](https://cloud.google.com/bigquery/pricing#free-tier) \[2\] [https://cloud.google.com/free/docs/gcp-free-tier#free-trial](https://cloud.google.com/free/docs/gcp-free-tier#free-trial) \[3\] [https://cloud.google.com/vertex-ai/pricing](https://cloud.google.com/vertex-ai/pricing) \[4\] [https://cloud.google.com/bigquery-ml/pricing](https://cloud.google.com/bigquery-ml/pricing) \[5\] [https://cloud.google.com/storage/pricing](https://cloud.google.com/storage/pricing)

Sources \[1\] Pricing | BigQuery: Cloud Data Warehouse [https://cloud.google.com/bigquery/pricing?gclsrc=aw.ds](https://cloud.google.com/bigquery/pricing?gclsrc=aw.ds) \[2\] Google Cloud Vertex AI Pricing Review 2024: Plans & Costs - Tekpon [https://tekpon.com/software/google-cloud-vertex-ai/pricing/](https://tekpon.com/software/google-cloud-vertex-ai/pricing/) \[3\] Vertex AI Tutorial: A Comprehensive Guide For Beginners - DataCamp [https://www.datacamp.com/tutorial/vertex-ai-tutorial](https://www.datacamp.com/tutorial/vertex-ai-tutorial) \[4\] Getting Started with BigQuery ML | Google Cloud Skills Boost [https://www.cloudskillsboost.google/focuses/2157?parent=catalog](https://www.cloudskillsboost.google/focuses/2157?parent=catalog) \[5\] Vertex AI pricing - Google Cloud [https://cloud.google.com/vertex-ai/pricing?authuser=0](https://cloud.google.com/vertex-ai/pricing?authuser=0) \[6\] Create machine learning models in BigQuery ML - Google Cloud [https://cloud.google.com/bigquery/docs/create-machine-learning-model](https://cloud.google.com/bigquery/docs/create-machine-learning-model)