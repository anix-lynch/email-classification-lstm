---
title: "Conditional Probability 💉and Bayes' Theorem in HIV case study"
datePublished: Mon Aug 05 2024 01:36:11 GMT+0000 (Coordinated Universal Time)
cuid: clzgblmce000909mi4t420ud7
slug: conditional-probability-and-bayes-theorem-in-hiv-case-study
tags: conditional-probability, bayes-theorem

---

### HIV Transmission Risks with unprotected sex

1. **Receptive Anal Sex**:
    
    * **Risk**: 1.38% per exposure (1 in 72 chance) for unprotected receptive anal sex.
        
    * **With Undetectable Viral Load**: The risk drops to 0%, meaning if the HIV-positive partner is on effective treatment and has an undetectable viral load, there is no risk of transmission.
        
2. **Insertive Anal Sex**:
    
    * **Risk**: 0.11% per exposure (1 in 909 chance) for unprotected insertive anal sex.
        
    * **With Undetectable Viral Load**: The risk also drops to 0% under the same conditions as above.
        
3. **Receptive Penile-Vaginal Sex**:
    
    * **Risk**: 0.08% per exposure (8 in 10,000 chance) for unprotected receptive vaginal sex.
        
    * **With Undetectable Viral Load**: The risk is 0% if the HIV-positive partner is effectively treated.
        
4. **Insertive Penile-Vaginal Sex**:
    
    * **Risk**: 0.04% per exposure (4 in 10,000 chance) for unprotected insertive vaginal sex.
        
    * **With Undetectable Viral Load**: The risk is also 0% under effective treatment conditions.
        

### Case Study: George and His Partner

**Background**: George was in a relationship with his Brazilian partner, who later died of AIDS. They were together for several months before George learned about his partner's HIV status. George took care of him until his partner's passing, which raised questions about George's own health status, especially regarding whether he contracted HIV during that time.

### Understanding the Risks

1. **HIV Transmission Risks**:
    
    * **Receptive Anal Sex** (if George was the receptive partner): The risk of contracting HIV without a condom is approximately **1.38%** per exposure (1 in 72 chance).
        
    * **Insertive Anal Sex** (if George was the insertive partner): The risk is about **0.11%** per exposure (1 in 909 chance).
        
    * **Vaginal Sex**: The risk for receptive vaginal sex is **0.08%** (8 in 10,000) and for insertive vaginal sex is **0.04%** (4 in 10,000).
        
2. **Using Condoms**:
    
    * Condoms can reduce the risk of HIV transmission by about **85%** when used consistently and correctly.
        

### Applying Conditional Probability and Bayes' Theorem

Let's define the events:

* **A**: George contracts HIV.
    
* **B**: George had unprotected sexual contact with his HIV-positive partner.
    

We want to find the probability of George contracting HIV given that he had unprotected sexual contact with his partner (P(A|B)).

### **Statistical Analysis:**

1. **Understanding the Risk:**
    
    * For receptive anal sex (highest risk activity), the risk of HIV transmission without a condom is approximately 1.38% per exposure.
        
    * With condom use, which reduces the risk by 85%, the risk per exposure drops to approximately 0.207%.
        
2. **Conditional Probability:**  
    Let's define some events:
    
    * **A**: George Michael contracts HIV.
        
    * **B**: George Michael had unprotected sexual contact with his HIV-positive partner.
        
    
    We want to find the probability of George Michael contracting HIV given that he had unprotected sexual contact with his partner (P(A|B)).
    
3. **Bayes' Theorem:**  
    Bayes' Theorem helps us calculate conditional probabilities. It is expressed as:
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1722821646298/55ff0d1f-8823-4143-9a33-6023ab0792c4.png align="center")
    
    Let's assume:
    
    * **P(A)**: Probability of contracting HIV in the general population = 0.001 (0.1%)
        
    * **P(B)**: Probability of having unprotected sexual contact with an HIV-positive person = 0.05 (5%)
        
    * **P(B|A)**: Probability of having unprotected sexual contact with an HIV-positive person given that one has contracted HIV = 0.8 (80%)
        
    * **P(B|A)** specific to "receptive anal sex" = 0.0138 (1.38%)
        
    
    Using Bayes' Theorem:
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1722821688422/43ef0bf0-a225-4b7b-aae9-e38a15a99ddc.png align="center")
    
      
    
    So, the probability of contracting HIV given unprotected sexual contact with an HIV-positive person in this specific scenario is 0.0276% or 0.000276.
    
4. **Probability of Not Contracting HIV:**
    
    * Probability of not contracting HIV in one encounter: 1−0.0138=0.98621−0.0138=0.9862 or 98.62%
        
    * Probability of not contracting HIV in 10 encounters: (0.9862)10=0.8703(0.9862)10=0.8703 or 87.03%
        
5. **With Condom Use:**
    
    * Risk per encounter with a condom: 0.207%
        
    * Probability of not contracting HIV in one encounter with a condom: 1−0.00207=0.997931−0.00207=0.99793 or 99.793%
        
    * Probability of not contracting HIV in 100 encounters with condoms: (0.99793)100=0.8137(0.99793)100=0.8137 or 81.37%
        

### Interpretation

1. **Low Probability**: The calculated probability indicates that while there was a risk, it was relatively low. This suggests that it is statistically possible for George not to have contracted HIV despite the close contact.
    
2. **Risk Factors**: Other factors could have influenced this outcome, such as:
    
    * The frequency of unprotected sexual encounters.
        
    * The viral load of his partner (if they were on effective treatment and had an undetectable viral load, the risk would be significantly reduced).
        
    * Any preventive measures George may have taken after learning about his partner's status.
        

### Conclusion

In this case study, we used conditional probability and Bayes' Theorem to analyze George's situation. The low probability of contracting HIV suggests that he could have been fortunate or that the transmission did not occur despite the exposure. This analysis highlights the importance of understanding risks and making informed decisions regarding sexual health.

By incorporating factual probabilities and statistical reasoning, we gain a clearer understanding of the dynamics of HIV transmission in real-life scenarios.

Citations: \[1\] [https://www.aidsmap.com/about-hiv/vaginal-sex-and-risk-hiv-transmission](https://www.aidsmap.com/about-hiv/vaginal-sex-and-risk-hiv-transmission) \[2\] [https://www.healthline.com/health/hiv/do-condoms-prevent-hiv](https://www.healthline.com/health/hiv/do-condoms-prevent-hiv) \[3\] [https://www.aidsmap.com/about-hiv/do-condoms-work](https://www.aidsmap.com/about-hiv/do-condoms-work) \[4\] [https://www.aidsmap.com/about-hiv/estimated-hiv-risk-exposure](https://www.aidsmap.com/about-hiv/estimated-hiv-risk-exposure) \[5\] [https://www.usaid.gov/sites/default/files/2022-05/condomfactsheet.pdf](https://www.usaid.gov/sites/default/files/2022-05/condomfactsheet.pdf)

Citations: \[1\] [https://hivrisk.cdc.gov/risk-estimator-tool/](https://hivrisk.cdc.gov/risk-estimator-tool/) \[2\] [https://aidsetc.org/sites/default/files/resources\_files/etres-307.pdf](https://aidsetc.org/sites/default/files/resources_files/etres-307.pdf) \[3\] [https://www.hiv.gov/hiv-basics/hiv-prevention/reducing-sexual-risk/preventing-sexual-transmission-of-hiv](https://www.hiv.gov/hiv-basics/hiv-prevention/reducing-sexual-risk/preventing-sexual-transmission-of-hiv) \[4\] [https://www.aidsmap.com/about-hiv/vaginal-sex-and-risk-hiv-transmission](https://www.aidsmap.com/about-hiv/vaginal-sex-and-risk-hiv-transmission) \[5\] [https://www.aidsmap.com/about-hiv/estimated-hiv-risk-exposure](https://www.aidsmap.com/about-hiv/estimated-hiv-risk-exposure)