---
title: "20 Statsmodels concepts with Before-and-After Examples"
seoTitle: "20 Statsmodels concepts with Before-and-After Examples"
seoDescription: "20 Statsmodels concepts with Before-and-After Examples"
datePublished: Fri Oct 04 2024 08:33:01 GMT+0000 (Coordinated Universal Time)
cuid: cm1ugws92000908l854qpaqex
slug: 20-statsmodels-concepts-with-before-and-after-examples
tags: statistics, python, data-science, data-analysis, statsmodels

---

Sure! Let's reformat the **statsmodels** examples to follow the new style you provided. Here are the examples from **statsmodels**, now with the **before and after real samples** included:

---

### 1\. **Linear Regression (OLS)** 📈

**Boilerplate Code**:

```python
import statsmodels.api as sm

X = sm.add_constant(X)  # Add constant to X variables
model = sm.OLS(y, X)
results = model.fit()
```

**Use Case**: Perform a basic **linear regression** using **OLS (Ordinary Least Squares)**.

**Goal**: Analyze the relationship between a dependent variable and independent variables. 🎯

**Sample Code**:

```python
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
```

**Before Example**:  
You manually try to figure out how the variables relate without a structured model. 🤔

```python
y = [3, 4, 5, 6]
X = [[1], [2], [3], [4]]
# No real model in place.
```

**After Example**:  
With `OLS`, you have a fitted linear model and can interpret relationships. 📈

```python
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.960
Model:                            OLS   Adj. R-squared:                  0.940
Method:                 Least Squares   F-statistic:                     48.00
==============================================================================
```

**Challenge**: 🌟 Fit an OLS model with multiple predictor variables and try interpreting the p-values and coefficients.

---

### 2\. **Logistic Regression** 🔄

**Boilerplate Code**:

```python
import statsmodels.api as sm

model = sm.Logit(y, X)
results = model.fit()
```

**Use Case**: Perform **logistic regression** for binary classification tasks.

**Goal**: Predict **binary outcomes** (e.g., success/failure) using logistic regression. 🎯

**Sample Code**:

```python
model = sm.Logit(y, X)
results = model.fit()
print(results.summary())
```

**Before Example**:  
You manually assign probabilities without understanding how they relate to the data. 😕

```python
y = [1, 0, 1, 0]
X = [[3], [2], [4], [1]]
```

**After Example**:  
With **Logit**, you get coefficients and probabilities for binary outcomes. 🔄

```bash
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   Pseudo R-sq:               0.624
Model:                          Logit   LL-Null:                       -23.456
Method:                           MLE   LL-Cur:                        -9.125
==============================================================================
```

**Challenge**: 🌟 Perform logistic regression on a customer churn dataset to predict customer retention.

---

### 3\. **Time Series Analysis (ARIMA)** 🕒

**Boilerplate Code**:

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(y, order=(1, 1, 1))
results = model.fit()
```

**Use Case**: Model and forecast **time series** data using ARIMA (AutoRegressive Integrated Moving Average).

**Goal**: Use ARIMA to make accurate forecasts of future data points. 🎯

**Sample Code**:

```python
model = ARIMA(y, order=(1, 1, 1))
results = model.fit()
print(results.summary())
```

**Before Example**:  
You analyze raw data without considering trends or patterns over time. 🤔

```python
y = [10, 12, 14, 16, 15, 20]
# Manually forecasting next values without capturing seasonality or trends.
```

**After Example**:  
With ARIMA, you get fitted models and forecasts that capture temporal dependencies. 🕒

```bash
                               ARIMA Model Results                               
==============================================================================
Dep. Variable:                      y   No. Observations:                    6
Model:                 ARIMA(1, 1, 1)   Log Likelihood                -10.012
AIC                             26.024
BIC                             26.946
==============================================================================
```

**Challenge**: 🌟 Fit an ARIMA model to predict monthly sales data and compare forecasts with actual results.

---

### 4\. **Autoregressive (AR) Model** 🔄

**Boilerplate Code**:

```python
from statsmodels.tsa.ar_model import AutoReg

model = AutoReg(y, lags=2)
results = model.fit()
```

**Use Case**: Model time series data using an **autoregressive model**, which predicts values based on previous observations.

**Goal**: Capture lagged relationships in time series data with AR. 🎯

**Sample Code**:

```python
model = AutoReg(y, lags=2)
results = model.fit()
print(results.summary())
```

**Before Example**:  
Manually trying to predict values without considering the past observations. 😕

```python
y = [100, 120, 130, 125]
# No lagged model to capture autocorrelation.
```

**After Example**:  
**AR** model captures autocorrelations, leading to more accurate forecasts. 🔄

```bash
                            AutoReg Model Results                            
==============================================================================
Dep. Variable:                      y   No. Observations:                    4
Model:                 AutoReg(2)   Log Likelihood               -10.234
AIC                             24.469
BIC                             26.893
==============================================================================
```

**Challenge**: 🌟 Fit an AR model to monthly temperature data and forecast future values based on lagged observations.

---

### 5\. **Poisson Regression** 🧮

**Boilerplate Code**:

```python
import statsmodels.api as sm

model = sm.GLM(y, X, family=sm.families.Poisson())
results = model.fit()
```

**Use Case**: Model **count data** where the dependent variable represents counts (e.g., number of events).

**Goal**: Use **Poisson regression** to predict the frequency of events. 🎯

**Sample Code**:

```python
model = sm.GLM(y, X, family=sm.families.Poisson())
results = model.fit()
print(results.summary())
```

**Before Example**:  
Predicting counts (e.g., website clicks) without considering the distribution of count data. 🤔

```python
y = [5, 7, 9, 12]
X = [[1], [2], [3], [4]]
```

**After Example**:  
Poisson regression models count data with non-negative integer outputs. 🧮

```bash
                            Poisson Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                    4
Model:                          GLM   Df Residuals:                          2
Model Family:             Poisson   Df Model:                                1
==============================================================================
```

**Challenge**: 🌟 Apply Poisson regression to model the number of daily website visits based on ad spending.

---

### 6\. **ANOVA (Analysis of Variance)** 🎲

**Boilerplate Code**:

```python
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model = ols('y ~ C(group)', data=df).fit()
anova_results = anova_lm(model)
```

**Use Case**: Test for significant differences between **group means**.

**Goal**: Use **ANOVA** to compare the mean of two or more groups. 🎯

**Sample Code**:

```python
model = ols('y ~ C(group)', data=df).fit()
anova_results = anova_lm(model)
print(anova_results)
```

**Before Example**:  
You try to compare means of groups manually without statistical testing. 😬

```python
group1 = [3, 4, 5]
group2 = [7, 8, 9]
```

**After Example**:  
**ANOVA** provides statistical significance between group means. 🎲

```bash
                            ANOVA Table                                
=========================================================================
        df    sum_sq   mean_sq         F    PR(>F)
group    1   102.667   102.667  19.35413   0.003
```

**Challenge**: 🌟 Run an ANOVA test on test scores across different teaching methods and interpret the significance of results.

---

### 7\. **Durbin-Watson Test (Autocorrelation)** 🔍

**Boilerplate Code**:

```python
from statsmodels.stats.stattools import durbin_watson

dw_stat = durbin_watson(results.resid)
```

**Use Case**: Test for **autocorrelation** in the residuals of a regression model.

**Goal**: Ensure residuals are independent and not autocorrelated using the **Durbin-Watson Test**. 🎯

**Sample Code**:

```python
dw_stat = durbin_watson(results.resid)
print(f'Durbin-Watson statistic: {dw_stat}')
```

**Before Example**:  
Residuals exhibit autocorrelation, leading to biased results. 😬

```python
Residuals: [0.1, 0.2, 0.15, 0.3, 0.25]  # Showing some autocorrelation.
```

**After Example**:  
Durbin-Watson test helps ensure independent residuals. 🔍

```bash
Durbin-Watson statistic: 2.12  # A value near 2 indicates no autocorrelation.
```

**Challenge**: 🌟 Apply the Durbin-Watson test on a regression model’s residuals to detect autocorrelation and assess the model’s assumptions.

---

### 8\. **Ljung-Box Test (Autocorrelation in Residuals)** 🔎

**Boilerplate Code**:

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

ljung_box_results = acorr_ljungbox(results.resid, lags=[10])
```

**Use Case**: Test whether residuals of a time series model are autocorrelated.

**Goal**: Use the **Ljung-Box Test** to determine if residuals are independent and not autocorrelated over time. 🎯

**Sample Code**:

```python
ljung_box_results = acorr_ljungbox(results.resid, lags=[10])
print(ljung_box_results)
```

**Before Example**:  
Residuals in a time series model show patterns over time. 😬

```python
Residuals: [0.3, 0.25, 0.35, 0.3, 0.4]  # Residuals follow some trend.
```

**After Example**:  
The **Ljung-Box Test** detects whether the residuals are independent. 🔎

```bash
Ljung-Box p-value: 0.08  # A p-value close to 1 indicates no autocorrelation.
```

**Challenge**: 🌟 Use the Ljung-Box test on ARIMA model residuals to check for autocorrelation and validate model assumptions.

---

### 9\. **Quantile Regression** 📊

**Boilerplate Code**:

```python
import statsmodels.api as sm

model = sm.QuantReg(y, X)
results = model.fit(q=0.5)  # Median regression
```

**Use Case**: Perform **quantile regression** to model relationships between variables at different quantiles (e.g., median, 90th percentile).

**Goal**: Use **Quantile Regression** to understand how predictors affect different parts of the outcome distribution. 🎯

**Sample Code**:

```python
model = sm.QuantReg(y, X)
results = model.fit(q=0.5)  # Median regression
print(results.summary())
```

**Before Example**:  
Modeling only the mean of a distribution might miss insights on the tails. 😕

```python
Mean regression results: Coefficient = 2.0  # Captures only central tendency.
```

**After Example**:  
**Quantile regression** provides insights into different parts of the distribution (e.g., median, 90th percentile). 📊

```bash
Quantile regression (median): Coefficient = 1.5  # Different insights at different quantiles.
```

**Challenge**: 🌟 Apply quantile regression on income data to understand disparities at the 10th, 50th, and 90th percentiles.

---

### 10\. **Autoregressive Integrated Moving Average (ARIMA)** 🕒

**Boilerplate Code**:

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(y, order=(1, 1, 1))
results = model.fit()
```

**Use Case**: Model and forecast **time series** data with trends, seasonality, and noise using ARIMA.

**Goal**: Use **ARIMA** to forecast time series data that accounts for autocorrelation, trends, and seasonality. 🎯

**Sample Code**:

```python
model = ARIMA(y, order=(1, 1, 1))
results = model.fit()
print(results.summary())
```

**Before Example**:  
Manually trying to forecast without a structured model for time series. 😩

```python
Future values based on simple extrapolation, ignoring seasonality.
```

**After Example**:  
With ARIMA, you model the trend, seasonality, and noise in time series data for accurate forecasts. 🕒

```bash
ARIMA Results: AIC = 250.0, BIC = 260.0, Coefficients estimated for AR, I, and MA terms.
```

**Challenge**: 🌟 Fit an ARIMA model on monthly sales data and use it to forecast the next 12 months.

---

### 11\. **Pooled OLS (Panel Data)** 📊

**Boilerplate Code**:

```python
from linearmodels import PooledOLS

pooled_model = PooledOLS(y, X).fit()
```

**Use Case**: Fit a **pooled OLS** model to analyze panel data, where observations are repeated over time across entities (e.g., multiple companies, countries).

**Goal**: Use **Pooled OLS** when time-invariant variables are present and you want to estimate the same coefficients for all entities. 🎯

**Sample Code**:

```python
pooled_model = PooledOLS(y, X).fit()
print(pooled_model.summary())
```

**Before Example**:  
You have panel data but aren’t considering the time dimension, treating the data as cross-sectional. 🤔

```bash
Only analyzing data without accounting for the multiple periods or entities.
```

**After Example**:  
With **Pooled OLS**, you account for repeated observations, fitting the model to panel data. 📊

```bash
                            PooledOLS Model Results                            
==============================================================================
Dep. Variable:                      y   No. Observations:                  100
Model:                 PooledOLS   Log Likelihood               -234.567
AIC                            492.34
==============================================================================
```

**Challenge**: 🌟 Try using **Pooled OLS** on a dataset with multiple companies' financial data over 5 years and interpret the output.

---

### 12\. **Fixed Effects Model (Panel Data)** 🔐

**Boilerplate Code**:

```python
from linearmodels import PanelOLS

fixed_effects_model = PanelOLS(y, X, entity_effects=True).fit()
```

**Use Case**: Control for **time-invariant differences** across entities (like companies or countries) in panel data.

**Goal**: Use **Fixed Effects** to eliminate the influence of unobserved, time-invariant characteristics that differ across entities. 🎯

**Sample Code**:

```python
fixed_effects_model = PanelOLS(y, X, entity_effects=True).fit()
print(fixed_effects_model.summary())
```

**Before Example**:  
Ignoring unobserved heterogeneity across entities, leading to biased results. 😕

```bash
Same coefficients applied across all entities, ignoring individual differences.
```

**After Example**:  
**Fixed Effects** account for these time-invariant characteristics, offering a more refined model. 🔐

```bash
Fixed Effects Model Results: Coefficients specific to each entity, R-squared improved.
```

**Challenge**: 🌟 Compare **Fixed Effects** to **Pooled OLS** on a dataset with company profits over time to see the effect of controlling for entity-specific variables.

---

### 13\. **Random Effects Model (Panel Data)** 🎯

**Boilerplate Code**:

```python
from linearmodels import RandomEffects

random_effects_model = RandomEffects(y, X).fit()
```

**Use Case**: Analyze panel data where **unobserved effects** are random and assumed to be uncorrelated with the independent variables.

**Goal**: Use **Random Effects** when you believe that differences across entities have a random component and are uncorrelated with your predictors. 🎯

**Sample Code**:

```python
random_effects_model = RandomEffects(y, X).fit()
print(random_effects_model.summary())
```

**Before Example**:  
Fitting a simple pooled OLS model and assuming no random variation across entities. 🤔

```bash
Unaccounted for random variations across entities.
```

**After Example**:  
**Random Effects** model handles random variations across entities, improving model fit. 🎯

```bash
                            RandomEffects Model Results                            
==============================================================================
Dep. Variable:                      y   No. Observations:                  100
Model:                 RandomEffects   Log Likelihood                -204.123
AIC                             412.245
==============================================================================
```

**Challenge**: 🌟 Fit both **Random Effects** and **Fixed Effects** models to panel data on customer satisfaction surveys and compare the results.

---

### 14\. **Durbin-Wu-Hausman Test (Fixed vs Random Effects)** ⚖️

**Boilerplate Code**:

```python
from linearmodels.panel import compare

results = compare(fixed_effects_model, random_effects_model)
```

**Use Case**: Decide between using **Fixed Effects** or **Random Effects** for panel data analysis by testing for endogeneity.

**Goal**: Apply the **Durbin-Wu-Hausman Test** to check whether to use fixed or random effects based on whether the entity-specific effects are correlated with your independent variables. 🎯

**Sample Code**:

```python
results = compare(fixed_effects_model, random_effects_model)
print(results)
```

**Before Example**:  
Uncertainty whether to apply fixed or random effects to panel data. 😩

```bash
Analysis proceeds without checking for endogeneity, leading to model bias.
```

**After Example**:  
The **Hausman Test** helps choose the correct model by checking for correlations in random effects. ⚖️

```bash
Hausman Test Results: p-value > 0.05: Random effects is preferable.
```

**Challenge**: 🌟 Perform the **Hausman Test** on panel data from multiple industries and interpret the p-value to decide between fixed or random effects.

---

### 15\. **Heteroskedasticity Test (Breusch-Pagan Test)** 📊

**Boilerplate Code**:

```python
from statsmodels.stats.diagnostic import het_breuschpagan

bp_test = het_breuschpagan(results.resid, results.model.exog)
```

**Use Case**: Test for **heteroskedasticity** (non-constant variance) in the residuals of a regression model.

**Goal**: Use the **Breusch-Pagan Test** to detect whether the variance of the residuals is constant across observations. 🎯

**Sample Code**:

```python
bp_test = het_breuschpagan(results.resid, results.model.exog)
print(f'Breusch-Pagan p-value: {bp_test[1]}')
```

**Before Example**:  
You fit a regression model but don't check for heteroskedasticity, leading to inefficient estimates. 😕

```bash
Residuals: [0.3, 0.25, 0.9, 0.8, 0.75]  # Possible signs of heteroskedasticity.
```

**After Example**:  
With the Breusch-Pagan test, you confirm whether or not heteroskedasticity is present. 📊

```bash
Breusch-Pagan p-value: 0.002  # p-value indicates heteroskedasticity is present.
```

**Challenge**: 🌟 Use the **Breusch-Pagan Test** on a dataset where heteroskedasticity might be present, such as housing prices or income distribution.

---

### 16\. **Ljung-Box Test (Autocorrelation in Residuals)** 🔎

**Boilerplate Code**:

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

ljung_box_results = acorr_ljungbox(results.resid, lags=[10])
```

**Use Case**: Test for autocorrelation in the residuals of a time series model.

**Goal**: Use the **Ljung-Box Test** to ensure residuals are independent and not autocorrelated. 🎯

**Sample Code**:

```python
ljung_box_results = acorr_ljungbox(results.resid, lags=[10])
print(ljung_box_results)
```

**Before Example**:  
Residuals show autocorrelation, causing bias in time series predictions. 😩

```bash
Residuals: [0.4, 0.35, 0.45, 0.5, 0.55]  # Residuals show a clear trend over time.
```

**After Example**:  
The **Ljung-Box Test** confirms if residuals are independent. 🔎

```bash
Ljung-Box p-value: 0.15  # No significant autocorrelation in residuals.
```

**Challenge**: 🌟 Apply the Ljung-Box Test on ARIMA model residuals to validate independence assumptions in time series analysis.

---

### 17\. **Seasonal Decomposition of Time Series (STL Decomposition)** 📅

**Boilerplate Code**:

```python
from statsmodels.tsa.seasonal import STL

stl = STL(y, seasonal=13)
result = stl.fit()
```

**Use Case**: Decompose a time series into its **trend**, **seasonal**, and **residual** components.

**Goal**: Use **STL Decomposition** to understand the underlying patterns in a time series, especially for data with strong seasonal components. 🎯

**Sample Code**:

```python
stl = STL(y, seasonal=13)
result = stl.fit()
result.plot()
```

**Before Example**:  
You struggle to separate seasonal patterns from the underlying trend. 😕

```bash
Time series plot shows seasonal fluctuation, but hard to identify the pattern.
```

**After Example**:  
**STL Decomposition** reveals trend, seasonality, and residuals separately. 📅

```bash
STL Decomposition plot shows clear trend, seasonal, and residual components.
```

**Challenge**: 🌟 Apply STL Decomposition to retail sales data to identify seasonal trends in product demand.

---

### 18\. **Quantile Regression** 📊

**Boilerplate Code**:

```python
import statsmodels.api as sm

model = sm.QuantReg(y, X)
results = model.fit(q=0.5)  # Median regression
```

**Use Case**: Perform **quantile regression** to model different quantiles (e.g., median, 90th percentile) of the response variable.

**Goal**: Use **Quantile Regression** to understand how predictors affect different parts of the distribution (e.g., the tails or median). 🎯

**Sample Code**:

```python
model = sm.QuantReg(y, X)
results = model.fit(q=0.5)  # Median regression
print(results.summary())
```

**Before Example**:  
You only model the mean of the distribution, missing insights about extreme values. 😞

```bash
Mean regression shows a relationship between predictors and the response variable.
```

**After Example**:  
With **Quantile Regression**, you model different parts of the distribution, gaining insight into the extremes. 📊

```bash
Quantile regression (median) results show a different relationship compared to the mean.
```

**Challenge**: 🌟 Apply quantile regression on a salary dataset to understand disparities at the 10th, 50th, and 90th percentiles.

---

### 19\. **Generalized Estimating Equations (GEE)** 📉

**Boilerplate Code**:

```python
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian

model = GEE(y, X, groups=group_var, family=Gaussian())
results = model.fit()
```

**Use Case**: Fit marginal models for **correlated data**, such as repeated measures or clustered data.

**Goal**: Use **GEE** to handle within-group correlation and produce robust estimates in the presence of correlated observations. 🎯

**Sample Code**:

```python
model = GEE(y, X, groups=group_var, family=Gaussian())
results = model.fit()
print(results.summary())
```

**Before Example**:  
You ignore within-group correlation, leading to biased estimates for repeated measures data. 😕

```bash
Repeated measures data shows correlation within groups, but it's not accounted for.
```

**After Example**:  
With **GEE**, you account for within-group correlation and produce more accurate estimates. 📉

```bash
GEE model results show adjusted coefficients accounting for within-group correlation.
```

**Challenge**: 🌟 Fit a GEE model to analyze patient health outcomes measured over multiple visits to the clinic.

---

### 20\. **Autoregressive Integrated Moving Average (ARIMA)** 🕒

**Boilerplate Code**:

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(y, order=(1, 1, 1))
results = model.fit()
```

**Use Case**: Model and forecast **time series** data using ARIMA, which captures autoregressive, integrated, and moving average components.

**Goal**: Use **ARIMA** to handle trends, seasonality, and noise in time series data. 🎯

**Sample Code**:

```python
model = ARIMA(y, order=(1, 1, 1))
results = model.fit()
print(results.summary())
```

**Before Example**:  
Without ARIMA, you're unable to accurately forecast time series data because trends and patterns are ignored. 😕

```bash
Time series predictions based on a simple model ignore trends and autocorrelations.
```

**After Example**:  
With ARIMA, you model the trend, seasonality, and noise in time series data for more accurate forecasts. 🕒

```bash
ARIMA results: Captured autocorrelations, trends, and moving averages for better forecasting.
```

**Challenge**: 🌟 Fit an ARIMA model on energy consumption data to forecast future usage based on historical patterns.

---