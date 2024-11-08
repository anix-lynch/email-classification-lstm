---
title: "Discrete Probability Distribution vs Continuous Probability Distribution. 🛍️ e-commerce case study"
seoTitle: "Discrete Probability Distribution vs Continuous Probability Distributi"
seoDescription: "Discrete Probability Distribution vs Continuous Probability Distribution. 🛍️ e-commerce case study  "
datePublished: Mon Aug 05 2024 17:00:15 GMT+0000 (Coordinated Universal Time)
cuid: clzh8lzsl000008l8hc6e92jt
slug: discrete-probability-distribution-vs-continuous-probability-distribution-e-commerce-case-study
tags: statistics, discrete-probability-distribution, continuous-probability-distribution

---

| Concept | Real-world Application | Common Libraries/Tools | Related Concepts |
| --- | --- | --- | --- |
| Discrete Probability Distribution | Modeling the number of defects in a batch of products | `numpy`, `scipy`, `pandas` | Probability mass function (PMF), Binomial distribution, Poisson distribution |
| Continuous Probability Distribution | Modeling the time between arrivals of customers at a store | `numpy`, `scipy`, `pandas` | Probability density function (PDF), Normal distribution, Exponential distribution |

### Code

#### Discrete Probability Distribution

```python
import numpy as np
import pandas as pd
import scipy.stats as stats

# Example: Binomial distribution for number of defects
n = 10  # Number of trials
p = 0.3  # Probability of defect

# Generate a sample
sample = np.random.binomial(n, p, 1000)

# Calculate probability mass function
pmf = stats.binom.pmf(k=np.arange(n+1), n=n, p=p)

print("Sample of number of defects:", sample[:10])
print("Probability Mass Function:", pmf)
```

#### Continuous Probability Distribution

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Example: Normal distribution for time between arrivals
mean = 10  # Mean time
std_dev = 2  # Standard deviation

# Generate a sample
sample = np.random.normal(mean, std_dev, 1000)

# Calculate probability density function
x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
pdf = stats.norm.pdf(x, mean, std_dev)

# Plot the PDF
plt.plot(x, pdf, label='PDF')
plt.hist(sample, bins=30, density=True, alpha=0.6, color='g')
plt.xlabel('Time between arrivals')
plt.ylabel('Probability Density')
plt.title('Normal Distribution')
plt.legend()
plt.show()

print("Sample of time between arrivals:", sample[:10])
```

### Explanation

#### Discrete Probability Distribution

* **Definition**: Describes the probability of occurrence of each value of a discrete random variable.
    
* **Example**: Number of defects in a batch of products (modeled using a binomial distribution).
    

#### Continuous Probability Distribution

* **Definition**: Describes the probabilities of the possible values of a continuous random variable.
    
* **Example**: Time between arrivals of customers at a store (modeled using a normal distribution).
    

### Key Concepts

* **PMF (Probability Mass Function)**: Function that gives the probability that a discrete random variable is exactly equal to some value.
    
* **PDF (Probability Density Function)**: Function that describes the likelihood of a continuous random variable to take on a particular value.
    

# Application to real life case

Certainly! Let's explore discrete and continuous probability distributions in the context of a niche e-commerce store selling crop tops for people with six-packs.

### Discrete Probability Distribution

In this context, a discrete probability distribution could model events that have distinct, countable outcomes.

Example: Number of crop tops sold per day

Imagine your store sells a limited edition crop top designed for people with six-packs. You want to model the number of these crop tops sold each day.

1. **Scenario**:
    
    * You typically sell between 0 to 5 crop tops per day.
        
    * The probability of selling each number of crop tops varies.
        
2. **Probability Mass Function (PMF)**: Let's say the probabilities of selling different numbers of crop tops are:
    
    * 0 crop tops: 10%
        
    * 1 crop top: 20%
        
    * 2 crop tops: 30%
        
    * 3 crop tops: 25%
        
    * 4 crop tops: 10%
        
    * 5 crop tops: 5%
        
3. **Application**:
    
    * Inventory management: Understanding the likelihood of selling a certain number of crop tops helps in stocking decisions.
        
    * Sales forecasting: You can predict the most likely sales outcomes for any given day.
        
4. **Code Example**:
    

```python
import numpy as np
import matplotlib.pyplot as plt

# Define the probability mass function
x = np.arange(6)  # 0 to 5 crop tops
pmf = [0.10, 0.20, 0.30, 0.25, 0.10, 0.05]

# Plot the PMF
plt.bar(x, pmf)
plt.title('Discrete Probability Distribution: Crop Tops Sold Per Day')
plt.xlabel('Number of Crop Tops Sold')
plt.ylabel('Probability')
plt.show()

# Simulate sales for 30 days
sales = np.random.choice(x, size=30, p=pmf)
print("Simulated daily sales for a month:", sales)
print("Average daily sales:", np.mean(sales))
```

### Continuous Probability Distribution

A continuous probability distribution could model events that can take any value within a range.

Example: Time spent browsing the crop top page before purchase

Consider the time customers spend browsing your crop top page before making a purchase.

1. **Scenario**:
    
    * Browsing time can be any positive real number.
        
    * The average browsing time is 5 minutes, with most customers spending between 2 to 8 minutes.
        
2. **Probability Density Function (PDF)**: We can model this using a normal distribution with a mean of 5 minutes and a standard deviation of 1.5 minutes.
    
3. **Application**:
    
    * User experience optimization: Understanding browsing patterns helps in designing the website layout.
        
    * Marketing strategies: Tailoring promotional content based on typical browsing durations.
        
4. **Code Example**:
    

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Define the parameters of the normal distribution
mean = 5  # average browsing time
std_dev = 1.5  # standard deviation

# Generate a range of values for x-axis
x = np.linspace(0, 10, 100)

# Calculate the PDF
pdf = stats.norm.pdf(x, mean, std_dev)

# Plot the PDF
plt.plot(x, pdf)
plt.title('Continuous Probability Distribution: Browsing Time Before Purchase')
plt.xlabel('Time (minutes)')
plt.ylabel('Probability Density')
plt.show()

# Simulate browsing times for 100 customers
browsing_times = np.random.normal(mean, std_dev, 100)
print("Sample browsing times:", browsing_times[:10])
print("Average browsing time:", np.mean(browsing_times))
```

### Key Differences and Applications:

1. **Discrete vs. Continuous**:
    
    * Discrete: Models countable outcomes (number of crop tops sold)
        
    * Continuous: Models measurable quantities that can take any value within a range (browsing time)
        
2. **Business Insights**:
    
    * Discrete distribution helps in inventory management and sales forecasting.
        
    * Continuous distribution aids in understanding customer behavior and optimizing user experience.
        
3. **Decision Making**:
    
    * Discrete: Helps decide how many crop tops to stock each day.
        
    * Continuous: Guides decisions on website design and content placement based on typical browsing durations.
        

By understanding these distributions, you can make data-driven decisions to optimize your niche e-commerce store's operations and customer experience.