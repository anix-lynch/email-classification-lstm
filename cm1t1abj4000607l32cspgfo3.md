---
title: "20 Sklearn concepts with Before-and-After Examples"
seoTitle: "20 Sklearn concepts with Before-and-After Examples"
seoDescription: "20 Sklearn concepts with Before-and-After Examples"
datePublished: Thu Oct 03 2024 08:27:52 GMT+0000 (Coordinated Universal Time)
cuid: cm1t1abj4000607l32cspgfo3
slug: from-sklearn-import-what-learn-20-key-sklearn-modules-with-before-and-after-examples
tags: ai, python, data-science, machine-learning, sklearn

---

### 1\. **Model Selection (Splitting)** 📝

**Boilerplate Code**:

```python
from sklearn.model_selection import train_test_split
```

**Use Case**: Split your data into two groups: one for **training** the model and another for **testing** how well it performs. 📚🎓

**Goal**: Ensure the model doesn’t overfit and performs well on unseen data. 🧠🤖

**Sample Code**:

```python
# Splitting the features (X) and labels (y) into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 80% of the data for training, 20% for testing!
```

**Before Example**: You’ve got a big stack of documents 📚 and don’t know which part to use for learning and which for testing.

```python
codeData: [document1, document2, document3, ..., document100]
```

**After Example**: Split the data into **training** and **testing** sets (80% for training, 20% for testing). 📊

```python
Training data: [document1, document2, ..., document80]
Testing data: [document81, document82, ..., document100]
```

**Challenge**: 🤔 Try changing `test_size` to 0.3 and see how it affects the split!

---

### 2\. **Preprocessing (Data Preparation)** 🧹

**Boilerplate Code**:

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
```

**Use Case**: You need to prepare your data by **scaling** numbers to a consistent range or **encoding** categories into numbers, so your model understands it better. 📏🔢

**Goal**: Ensure all data is in the same format and scale, so the model doesn’t get confused. 🤯

**Sample Code**:

```python
# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode categorical labels into numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Now your data is ready for the model to digest!
```

**Before Example**: The intern has raw data, with heights in centimeters and animal types in words. 😵

```python
Height: [150cm, 160cm, 170cm]
Animal: ["cat", "dog", "bird"]
```

**After Example**: Now, heights are scaled, and animals are encoded into numbers. 🔧

```python
Height (scaled): [0.2, 0.5, 0.8]
Animal (encoded): [0, 1, 2]
```

**Challenge**: 🌟 Try using `MinMaxScaler` instead of `StandardScaler` to scale values between 0 and 1, and see the difference!

### 3\. **Metrics (Evaluation)** 🏅

**Boilerplate Code**:

```python
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error
```

**Use Case**: After training your model, you need to check how well it performed by evaluating **accuracy**, **precision**, or **error**. 🏆

**Goal**: Measure how well the model’s predictions match the actual values. 📊

**Sample Code**:

```python
# Check how many predictions were correct
accuracy = accuracy_score(y_test, y_pred)

# Measure precision for binary classification
precision = precision_score(y_test, y_pred)

# Calculate the error for regression models
mse = mean_squared_error(y_test, y_pred)

# Now you know how well your model performed!
```

**Before Example**: The intern’s model makes predictions, but they don’t know how well it did. 😬

```python
Predictions: [Yes, No, Yes]
Actual: [Yes, No, No]
```

**After Example**: The accuracy score tells them how many predictions were correct! 💯

```python
Accuracy Score: 67% (2 out of 3 correct)
```

**Challenge**: 🔍 Try calculating the `f1_score` to balance precision and recall!

---

### 4\. **Linear Models** 📐

**Boilerplate Code**:

```python
from sklearn.linear_model import LogisticRegression, LinearRegression
```

**Use Case**: Use a **linear model** to predict outcomes, whether for **classification** (Logistic) or **regression** (Linear). ✏️

**Goal**: Use simple equations to make predictions about future outcomes! 📈

**Sample Code**:

```python
# Predict binary outcomes with logistic regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict continuous values with linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# You're all set to make predictions!
```

**Before Example**: We want to predict salary based on years of experience but doesn’t have a formula. 💼

```python
Data: [Years: 1, Salary: $30k], [Years: 2, Salary: $40k]
```

**After Example**: We now has a simple linear equation to make predictions! 🧮

```python
Linear Equation: Salary = 10k * (Years of Experience)
```

**Challenge**: 🔥 Try using `Ridge` or `Lasso` regression to handle overfitting and compare results!

### 5\. **Ensemble Methods** 🌲🌳

**Boilerplate Code**:

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
```

**Use Case**: Combine the predictions of multiple models to improve accuracy. 🧑‍🤝‍🧑

**Goal**: Boost your model’s performance by having multiple models "vote" on the final prediction. 🗳️

**Sample Code**:

```python
# Random forest for classification
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Gradient boosting for classification
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# Multiple models working together for a better result!
```

**Before Example**: We have one model making predictions but is unsure of its accuracy. 🤷‍♂️

```python
Model 1: "Yes"
```

**After Example**: <mark>Now, multiple models vot</mark>e, and the majority wins! 🗳️

```python
Model 1: "Yes", Model 2: "No", Model 3: "Yes" → Final prediction: "Yes"
```

**Challenge**: 🌟 Try using `AdaBoostClassifier` for a different boosting technique and compare results!

---

### 6\. **Support Vector Machines (SVM)** 🚧

**Boilerplate Code**:

```python
from sklearn.svm import SVC, SVR
```

**Use Case**: Use **Support Vector Machines** to create **decision boundaries** for classification or prediction. 🛤️  
The difference between **SVC** (Support Vector Classifier) and **SVR** (Support Vector Regressor) is in what kind of task they are used for:

1. **SVC (Support Vector Classifier)**:
    
    * **Task**: **Classification**
        
    * **Use**: Separates data points into **discrete categories** or classes.
        
    * **Example**: Classifying images as either **cats 🐱** or **dogs 🐶**.
        
    * **Goal**: Draw a decision boundary that **separates classes** as clearly as possible.
        
2. **SVR (Support Vector Regressor)**:
    
    * **Task**: **Regression**
        
    * **Use**: Predicts **continuous values**, like numbers or measurements.
        
    * **Example**: Predicting the **price of a house** based on features like size, location, etc.
        
    * **Goal**: Fit a curve that predicts continuous values with **minimal error**.
        

In both cases, they find a boundary or line (in SVC) or a curve (in SVR) that optimizes the separation or prediction based on the data.

**Goal**: Separate different classes with a **boundary** that maximizes the gap between them. 🧱

**Sample Code**:

```python
# Classify with Support Vector Classifier
svc = SVC()
svc.fit(X_train, y_train)

# Predict continuous values with Support Vector Regressor
svr = SVR()
svr.fit(X_train, y_train)

# Now you're ready to separate classes with clear boundaries!
```

**Before Example**: We work with a mixed dataset of cats 🐱 and dogs 🐶.

```python
Data: [cats, dogs, mixed up]
```

**After Example**: A decision boundary now neatly separates them into groups. 🧱

```python
Boundary: [cats on one side, dogs on the other]
```

**Challenge**: 🧠 Try changing the `kernel` parameter (e.g., 'linear', 'rbf') and see how it changes the decision boundary!

---

### 7\. **Nearest Neighbors** 🧭

**Boilerplate Code**:

```python
from sklearn.neighbors import KNeighborsClassifier
```

**Use Case**: Classify new data points based on the **nearest neighbors**. 🏡

**Goal**: Make predictions by finding the closest examples and **using them as a guide**. 👯‍♂️

**Sample Code**:

```python
# Classify with K-nearest neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Now your model will classify based on the nearest neighbors!
```

**Before Example**: We are unsure how to classify new data. 🤔

```python
New Data: ? | Neighbors: [cat, cat, dog]
```

**After Example**: The model classifies based on the majority of the nearest neighbors! 🗳️

```python
Prediction: "cat" (since most neighbors are cats)
```

**Challenge**: 🧩 Try changing the number of neighbors (e.g., `n_neighbors=3` or `7`) and see how it affects the predictions!

Here’s the next set! Let’s keep the momentum going! 😊

---

### 8\. **Decision Trees** 🌳

**Boilerplate Code**:

```python
from sklearn.tree import DecisionTreeClassifier
```

**Use Case**: Use **if-then rules** to classify data step-by-step, creating a decision-making flow. 📜

**Goal**: Build a tree of decisions to arrive at a prediction based on different features. 🌿

**Sample Code**:

```python
# Classify using a decision tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Your decision tree is ready to make predictions based on rules!
```

**Before Example**: We don’t know how to make classification decisions without a structure. 🤷‍♀️

```python
Is it furry? Yes → Is it a pet? Yes → ?
```

**After Example**: The decision tree now helps classify data based on clear rules! ✅

```python
If furry → yes, If pet → yes → classify as "cat"
```

**Challenge**: 🌟 Try adjusting the `max_depth` of the tree and see how it changes performance. Can you avoid overfitting?

---

### 9\. **Cross-Validation** 🔄

**Boilerplate Code**:

```python
from sklearn.model_selection import cross_val_score
```

**Use Case**: Test the model multiple times by splitting the data differently each time to get more reliable performance metrics. 🧪

**Goal**: Ensure your model’s performance isn’t dependent on a single data split. 🔄

**Sample Code**:

```python
# Evaluate model using cross-validation
scores = cross_val_score(model, X, y, cv=5)

# Average score across 5 different splits of the data
average_score = scores.mean()
```

**Before Example**: We test the model just once and gets a performance score, but it's hard to know if the score is reliable. 😕

```python
Score from one test: 85%
```

**After Example**: By testing across multiple splits, the intern gets a more reliable average score! 📊

```python
Scores: [85%, 90%, 80%, 88%, 84%] → Average: 85%
```

**Challenge**: 🔍 Experiment with different `cv` values (e.g., `cv=3`, `cv=10`) and see how it impacts the results!

---

### 10\. **Hyperparameter Tuning** 🎛️

**Boilerplate Code**:

```python
from sklearn.model_selection import GridSearchCV
```

**Use Case**: Optimize your model by finding the best hyperparameters through an automated search. 🔍

**Goal**: Fine-tune your model by testing different hyperparameters and selecting the best combination. 🎯

**Sample Code**:

```python
# Define the hyperparameters to search
param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7]}

# Use grid search to find the best parameters
grid_search = GridSearchCV(RandomForestClassifier(), param_grid)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
```

**Before Example**: We use default hyperparameters for the model and doesn’t know if they’re optimal. 🤷

```python
Parameters: n_estimators = 100, max_depth = 3
```

**After Example**: Now we run a grid search and finds the best parameters for the model! 🎯

```python
Best Parameters: n_estimators = 200, max_depth = 5
```

**Challenge**: 🤔 Try changing the range of hyperparameters in `param_grid` and see how the best parameters change!

---

### 11\. **Pipelines (Sequential Workflow)** 🔄

**Boilerplate Code**:

```python
from sklearn.pipeline import Pipeline
```

**Use Case**: Combine multiple steps into one single workflow. You can connect data preprocessing, feature engineering, and model training in a single pipeline! 🔗

**Goal**: Automate and streamline your machine learning workflow by creating a **sequential process** that ties multiple steps together. 🚂

**Sample Code**:

```python
# Create a pipeline with a scaler and a logistic regression model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Now the pipeline scales the data and trains the model in one go!
```

**Before Example**: manually scale the data and then pass it to the model. 🏋️‍♂️

```python
Manually scale → Manually pass to the model → Manually predict
```

**After Example**: With a pipeline, it’s all automatic! 🚂

```python
Pipeline automatically scales, trains, and predicts in sequence!
```

**Challenge**: 🌟 Try adding `PCA` into the pipeline between the scaler and the model and see how it impacts the model’s performance!

---

### 12\. **Polynomial Features (Feature Engineering)** 🧮

**Boilerplate Code**:

```python
from sklearn.preprocessing import PolynomialFeatures
```

**Use Case**: Expand your feature set by adding **polynomial terms** (e.g., squares, cubes), creating more complex relationships between features. 🔧

**Goal**: Enrich the dataset by transforming simple features into polynomial ones to capture more complex patterns. 🎯

**Sample Code**:

```python
# Expand features by adding polynomial terms (e.g., x², x³)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# You've created new polynomial features!
```

**Before Example**: has simple features but knows there are hidden patterns to capture. 🤔

```python
Features: [x1, x2]
```

**After Example**: After expanding with polynomial features, new patterns are created! 🧮

```python
Expanded Features: [1, x1, x2, x1², x1*x2, x2²]
```

**Challenge**: 🔍 Try increasing the degree (e.g., `degree=3`) and see how it affects the complexity of the model.

---

### 13\. **Feature Selection (Selecting Important Features)** ✂️

**Boilerplate Code**:

```python
from sklearn.feature_selection import SelectKBest
```

**Use Case**: Select only the most important features based on statistical tests, filtering out the unnecessary ones. 🔍

**Goal**: Narrow down your data to the **top K features** that are most relevant for the model. ✂️

**Sample Code**:

```python
# Select the top 5 best features
selector = SelectKBest(k=5)
X_selected = selector.fit_transform(X, y)

# Now you've narrowed down to the most important features!
```

**Before Example**: too many features, but not all are relevant for the model. 😵

```python
Features: [A, B, C, D, E, F]
```

**After Example**: <mark>After using feature selection, we keep only the top 5 features! </mark> ✂️

```python
Selected Features: [A, B, D, E, F]
```

**Challenge**: 🌟 Try selecting a different number of features (e.g., `k=3`) and see how it impacts model performance!

---

### 14\. **Scaling Data (MinMaxScaler)** 📏

**Boilerplate Code**:

```python
from sklearn.preprocessing import MinMaxScaler
```

**Use Case**: Scale your data to fit between a given range (usually 0 and 1) to avoid any feature dominating due to scale differences. 📊

**Goal**: Ensure all features are within the same range for better model performance. 🎯

**Sample Code**:

```python
# Scale features to the range [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Now your data is scaled to the same range!
```

**Before Example**: data has values with a wide range (e.g., heights in cm and incomes in thousands). 📏💰

```python
Height: [150, 160, 170]
Income: [30k, 50k, 60k]
```

**After Example**: After scaling, all the data is within the range \[0, 1\]. 📏

```python
Height (scaled): [0.2, 0.5, 0.8]
Income (scaled): [0.4, 0.6, 1.0]
```

**Challenge**: 🔍 Try using `StandardScaler` and compare the results with `MinMaxScaler`.

---

### 15\. **Outlier Detection (Isolation Forest)** 🚨

**Boilerplate Code**:

```python
from sklearn.ensemble import IsolationForest
```

**Use Case**: Detect **outliers** in your dataset—those rare and unusual points that don’t fit the pattern. 🕵️

**Goal**: Identify and remove outliers so they don’t skew your model. 🚨

**Sample Code**:

```python
# Use Isolation Forest to detect outliers
iso_forest = IsolationForest()
iso_forest.fit(X_train)

# Predict outliers (-1 means it's an outlier, 1 means it's not)
outlier_predictions = iso_forest.predict(X_test)
```

**Before Example**: data with hidden outliers that could throw off the model’s performance. 😬

```python
Data: [1, 2, 2, 3, 100]
```

**After Example**: The outlier is detected and can be dealt with! 🚨

```python
Outlier detected: 100
```

**Challenge**: 🌟 Try adjusting `contamination` (e.g., `contamination=0.05`) to change how sensitive the model is to outliers.

---

### 16\. **Dimensionality Reduction (PCA)** 📉

**Boilerplate Code**:

```python
from sklearn.decomposition import PCA
```

**Use Case**: Reduce the number of features while keeping most of the important information. Perfect when you have too many features! 🧠

**Goal**: Compress your dataset into fewer dimensions without losing too much information. 🎯

**Sample Code**:

```python
# Reduce to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# You've reduced the data to just 2 dimensions!
```

**Before Example**: a dataset with too many features, making analysis difficult. 😰

```python
Features: [Feature1, Feature2, ..., Feature100]
```

**After Example**: We use **PCA** to reduce the data to fewer dimensions. 📊

```python
Reduced Features: [PrincipalComponent1, PrincipalComponent2]
```

**Challenge**: 🔍 Try using different numbers of components (e.g., `n_components=3`) and see how much variance is retained!

---

### 17\. **Clustering (KMeans)** 🧲

**Boilerplate Code**:

```python
from sklearn.cluster import KMeans
```

**Use Case**: Automatically group your data into clusters based on similarity. 📊

**Goal**: Discover hidden patterns by grouping similar data points together—no labels needed! 🧲

**Sample Code**:

```python
# Cluster the data into 3 groups
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Get the cluster labels for each data point
labels = kmeans.predict(X)

# Now you’ve grouped the data into clusters!
```

**Before Example**: unlabeled data and no way to classify it into meaningful groups. 🤔

```python
Data: [Data Point 1, Data Point 2, Data Point 3, ...]
```

**After Example**: group the data into clusters, revealing hidden patterns! 📊

```python
Cluster 1: [Data Point 1, Data Point 3]
Cluster 2: [Data Point 2]
```

**Challenge**: 🧠 Try changing the number of clusters (`n_clusters=5`) and see how it affects the groupings.

---

### 18\. **Model Calibration (Calibrated Classifier)** 🎯

In the context of machine learning, **model calibration** is the process of adjusting a model’s **predicted probabilities** to make them more **realistic**.

For example, if a model predicts a **70%** chance of rain but it only rains 40% of the time when the model says 70%, <mark>then the probabilities aren't well-calibrated(or well-adjusted). Calibration fixes this</mark> so that the probabilities reflect reality more closely.  
**Boilerplate Code**:

```python
from sklearn.calibration import CalibratedClassifierCV
```

**Use Case**: Adjust your model’s probability predictions to make them more reliable and realistic. 📉

**Goal**: Fine-tune predicted probabilities so they’re accurate and trustworthy. 🎯

**Sample Code**:

```python
# Calibrate a classifier
calibrated_model = CalibratedClassifierCV(base_estimator=SVC())
calibrated_model.fit(X_train, y_train)

# Predict probabilities for test data
calibrated_probs = calibrated_model.predict_proba(X_test

)

# Now your probabilities are calibrated and more accurate!
```

**Before Example**: The model gives probabilities, but they’re not very reliable. 🤷‍♂️

```python
Predicted Probabilities: [0.7, 0.4, 0.9]
```

**After Example**: With calibration, the probabilities are more realistic and trustworthy! 📉

```python
Calibrated Probabilities: [0.6, 0.5, 0.8]
```

**Challenge**: 🌟 Try calibrating different classifiers (e.g., `RandomForestClassifier`) and compare the changes in probability predictions!

---

### 19\. **Nearest Centroid Classifier** 🧭

**Boilerplate Code**:

```python
from sklearn.neighbors import NearestCentroid
```

**Use Case**: Classify new data points based on the **centroid** of the closest class, a simple but effective classifier. 🧭

**Goal**: Assign a class to each point by calculating the **centroid** (center) of each class and finding which class’s centroid is closest. 🎯

**Sample Code**:

```python
# Use nearest centroid classifier
nc = NearestCentroid()
nc.fit(X_train, y_train)

# Predict labels for the test data
y_pred = nc.predict(X_test)

# Now your data is classified based on the nearest centroid!
```

**Before Example**: unsure how to classify new data and needs a simple approach. 🤷‍♂️

```python
New Data: [Unknown] | Centroids: [Centroid1, Centroid2]
```

**After Example**: <mark>classify based on which </mark> **<mark>centroid</mark>** <mark> is closest! 🎯</mark>

```python
Assigned Class: [Class 1]
```

**Challenge**: 🌟 Try running `NearestCentroid` on a dataset with more than two classes and see how it handles multiple centroids.

---

### 20\. **Voting Classifier (Combining Classifiers)** 🗳️

**Boilerplate Code**:

```python
from sklearn.ensemble import VotingClassifier
```

**Use Case**: Combine multiple models to make a final prediction based on the majority vote. Perfect for improving accuracy! 🗳️

**Goal**: Improve prediction performance by having multiple models vote on the outcome, and the majority wins! 🎯

**Sample Code**:

```python
# Define multiple classifiers
clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = SVC()

# Use VotingClassifier to combine them
voting_clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)], voting='hard')
voting_clf.fit(X_train, y_train)

# Now your final predictions are based on the majority vote!
```

**Before Example**: use one model, but they’re unsure of its accuracy. 🤔

```python
Model 1: "Yes"
```

**After Example**: Multiple models vote, and the majority rules! 🗳️

```python
Model 1: "Yes", Model 2: "No", Model 3: "Yes" → Final prediction: "Yes"
```

**Challenge**: 🌟 Try using `voting='soft'` to combine the probabilities instead of the hard majority and compare the results.

---

Bonus point:  
  
All of these methods—**Cross-Validation**, **Voting Classifier**, **Boosting**, **Stacking**, and **Ensemble Methods**—use multiple models, but they do so in different ways. Here's a simplified breakdown of the differences:

1\. **Cross-Validation** 📊

* **Purpose**: It's a **validation technique**, not a model combination method.
    
* **How it works**: You split your dataset into **multiple parts (folds)** and train the model on different subsets to evaluate its performance on unseen data. This helps to ensure your model generalizes well.
    
* **Key Point**: You're using **one model** but evaluating it in different ways on different data splits.
    

2\. **Voting Classifier** 🗳️

* **Purpose**: Combines **multiple models** and makes a decision by "voting."
    
* **How it works**: You train several models and let each one "vote" on the prediction. There are two types:
    
    * **Hard Voting**: Each model gives a class prediction, and the most common class wins.
        
    * **Soft Voting**: Each model gives a probability for each class, and the probabilities are averaged.
        
* **Key Point**: It aggregates decisions from multiple models to improve accuracy by combining their strengths.
    

3\. **Boosting** 🚀

* **Purpose**: Sequentially trains models, with each new model **trying to fix the errors** of the previous one.
    
* **How it works**: Models are trained one after another, and each new model focuses on **correcting mistakes** made by the previous models. Examples include **AdaBoost** and **Gradient Boosting**.
    
* **Key Point**: It's a **sequential process** where each model builds on the mistakes of the previous ones, making the final model stronger.
    

4\. **Stacking** 📚

* **Purpose**: Combines predictions from multiple models by using another model (called a **meta-model**) to make the final prediction.
    
* **How it works**: Multiple base models are trained, and their predictions are used as inputs to a **meta-model**. The meta-model then makes the final prediction.
    
* **Key Point**: It **layers** models, using the predictions of several models as inputs for another model to make a final decision.
    

5\. **Ensemble Methods** 🏗️

* **Purpose**: General term for using multiple models to make better predictions.
    
* **How it works**: Any technique that **combines multiple models** (e.g., Voting, Boosting, Stacking, etc.) is called an **ensemble method**. The idea is to **reduce error** and **increase accuracy** by using the strengths of different models.
    
* **Key Point**: An umbrella term that includes techniques like **bagging** (e.g., Random Forests), **boosting**, and **stacking**.
    

---

**Summary of Differences**:

* **Cross-Validation**: Splits data to evaluate the performance of a single model.
    
* **Voting Classifier**: Combines models by letting them vote on the final decision.
    
* **Boosting**: Sequentially builds models, where each new model corrects the previous one’s errors.
    
* **Stacking**: Uses the predictions of multiple models as inputs to a new model.
    
* **Ensemble Methods**: General term for techniques that combine multiple models to improve performance.
    

Each of these techniques has a different way of leveraging multiple models to improve accuracy or robustness!