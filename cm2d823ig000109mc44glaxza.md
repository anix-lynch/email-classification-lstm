---
title: "Advanced Machine Learning Q&A (3/3)"
seoTitle: "Advanced Machine Learning Q&A (2/3)"
seoDescription: "Advanced Machine Learning Q&A (2/3)"
datePublished: Thu Oct 17 2024 11:32:49 GMT+0000 (Coordinated Universal Time)
cuid: cm2d823ig000109mc44glaxza
slug: advanced-machine-learning-qa-23-1
tags: ai, data-science, machine-learning, deep-learning

---

**Q: What is the main goal of using interpretability techniques like LIME and SHAP in machine learning?**

Explaining model predictions.

* **LIME (Local Interpretable Model-agnostic Explanations):**
    
    LIME breaks down a model's prediction into smaller, understandable parts by making small changes to the input data. It builds a simple model (surrogate) around a single prediction to explain why the model made that decision for a specific instance (Single data point). For example, LIME can show why a person was denied a loan based on their individual details like age or income.
    
    **SHAP (SHapley Additive exPlanations):**
    
    SHAP explains how each feature contributed to a model’s prediction by testing all possible combinations of features. It fairly distributes the contribution of each feature, considering interactions between them. SHAP helps explain the overall patterns of the model, showing how features like income and age affect loan decisions across many applicants.
    
    Both LIME and SHAP improve model transparency and help users understand **why** a model made certain predictions, building trust and accountability.
    

**Q: What is the difference between SHAP and kernel SHAP**

SHAP tests **all possible combinations** of features to calculate exact contributions. It looks at every single way to combine the features, so it provides a precise answer but can be slow for complex models.

Kernel SHAP (Approximate SHAP): Instead of testing every combination (which can be too slow for complex models), Kernel SHAP uses **only a subset** of combinations and estimates the contributions based on those samples. It doesn't try every combination, which makes it faster but less precise.

**Q: Which common technique is used to assess the fair- ness of a machine learning model across different demographic groups?**

Disparate Impact Analysis

Imagine a company uses a machine learning model to decide who gets a loan. If the model is giving loans to **70% of men** but only **50% of women**, even though both groups have similar credit histories, this might be unfair. **Disparate Impact Analysis** would highlight this difference and help the company see that the model is giving men an advantage.

This analysis helps check if the model is **unintentionally biased** and ensures that decisions are not favoring one group unfairly over another.

**Q: To detect and mitigate bias in machine learning models, practitioners may use fairness auditing tools like \_, which provide insights into how models behave across different demographic groups.**

Fairlearn

**Q: You are building a credit scoring model and want to ensure that the model does not discriminate against applicants based on gender. What steps would you take to assess and ensure fairness in the model?**

Ignore gender during training; Use disparate impact analysis

**Q: A healthcare organization is using machine learning to predict patient outcomes. They are concerned about potential bias in the model affecting minority eth- nic groups. How would you approach the evaluation of bias in this scenario, and what methods could you apply to mitigate it?**

Collect more diverse data; Implement fairness-aware algorithm

**Q: What is an adversarial attack in the context of machine learning?**

Manipulating input to produce incorrect output

An adversarial attack refers to the deliberate manipulation of input data to mislead a machine learning model and produce incorrect predictions or classifications.

**Q: Why deep learning are typically most vulnerable to adversarial attacks?**

* **Complexity:** Neural networks have many layers and parameters, which makes them highly flexible. This flexibility allows them to capture complex patterns in data but also makes it easier for small, carefully crafted changes (adversarial examples) to fool the model without the model noticing these changes.
    
* **Non-linear decision boundaries:** Neural networks use non-linear functions to make predictions. These non-linear boundaries can be manipulated with tiny changes in the input, causing the model to classify the input incorrectly. Even slight changes that are not noticeable to humans (like tweaking a few pixels in an image) can push the input across these non-linear boundaries, tricking the model.
    

**Q: What are type of attack and defense.**

| **Attack/Defense** | **Type** | **Description** | **Analogy** |
| --- | --- | --- | --- |
| **Fast Gradient Sign Method (FGSM)** | Attack | Quickly adds noise based on the model's gradient to trick the model. | Imagine quickly adding a small, invisible scratch to a painting that confuses an art detector. |
| **Jacobian-based Saliency Map Attack (JSMA)** | Attack | Targets specific parts of the input to fool the model with small, calculated changes. | Like changing just the eyes in a photo to fool facial recognition. |
| **L2 Attack** | Attack | Modifies the input by adding minimal changes that are optimized to fool the model (measured by L2 norm). | Imagine slightly adjusting the brightness of a photo, barely noticeable but confusing a scanner. |
| **Black-Box Attack** | Attack | Creates adversarial examples using a different model and applies them to another model without access. | Like learning how to pick one car’s lock and using the same trick to open a different car. |
| **Carlini & Wagner (C&W) Attack** | Attack | Optimizes small, subtle changes to fool the model, focusing on finding the minimal change needed. | Like adding the tiniest smudge to a painting to make an expert misidentify it. |

| **Defense** | **Type** | **Description** | **Analogy** |
| --- | --- | --- | --- |
| **Adversarial Training** | Defense | Trains the model with adversarial examples to make it more robust against future attacks. | Like training a boxer by having them spar against different types of opponents to prepare. |
| **Gradient Masking** | Defense | Tries to hide or confuse the gradients used in attacks, making it harder for attackers to exploit them. | Like putting up a false trail in a maze to confuse someone trying to follow the correct path. |
| **Defensive Distillation** | Defense | Smooths the model’s predictions to make it less sensitive to adversarial inputs, reducing overconfidence. | Imagine teaching someone to make decisions slowly and carefully so they don’t get tricked easily. |

**Question:** A bank is implementing a machine learning model for loan approval. Stakeholders demand that the model's decisions be transparent and easily understood. What strategies and methods should be used to meet this requirement?

**Answer:**  
Use **white-box models** and **local explanation techniques**.

**Explanation**  
White-box models are like **clear boxes**, where you can see exactly how decisions are made. By combining them with **local explanation techniques**, you help stakeholders understand the reasoning behind each decision.

**Q: Which variant of attention mechanism allows the model to focus on different parts of the input for different parts of the output in sequence-to-sequence tasks?**

**Global Attention:** Looks at the entire input sentence for each word in the output sentence.

* **Example**: When translating the English sentence "The cat sits on the mat" into French, for every word it generates in the output (like "Le", "chat", etc.), the model looks at all the words in the English input ("The", "cat", "sits", "on", "the", "mat"). This helps the model understand the whole context for each word it predicts.
    
* **Imagine**: Like when you're writing a summary of a book, you keep the entire book open in front of you and look at every part of it before writing each sentence.
    

**Local Attention:** Focuses on only a small part of the input sentence around the current word in the output.

* **Example**: When translating "The cat sits on the mat", the model only looks at a few nearby words, like just "The cat sits", when generating the output word "Le" or "chat". It doesn’t focus on "mat" at this stage, because it's only interested in a local context.
    
* **Imagine**: Like reading a book, but instead of skimming through the whole book at once, you only read one page at a time and refer to it when summarizing that specific section.
    

**Self-Attention:** Lets the model look at all the words in a sentence, including the word it’s currently working on, to better understand relationships between words.

* **Example**: If the model is processing the word "apple" in the sentence "The apple is sweet," it looks not only at "apple", but also at "sweet", to understand that "sweet" is describing "apple", which helps improve understanding.
    
* **Imagine**: Like when you're trying to understand a conversation, you pay attention to each word individually but also look at how the words interact with each other within the same conversation.
    

**Additive Attention:** Combines different pieces of information by adding them together to decide how much weight to give each part of the input when generating the output.

* **Example**: When translating "The cat is sitting", the model might combine the importance of "cat" and "sitting" when deciding how to translate "is sitting" to the correct French phrase ("est assis"). It’s slower but helps ensure accuracy in aligning words across languages.
    
* **Imagine**: Like when you're deciding what to pack for a trip, you add together different factors (like weather, activities, etc.) to make a well-balanced decision on what clothes to take.
    

These examples break down how different attention mechanisms work when translating languages, but the idea can be applied to other sequence-to-sequence tasks like text summarization or speech recognition.

**Q: What is the key difference between semantic segmentation and instance segmentation in the context of computer vision?**

Semantic segmentation and instance segmentation are both techniques used to label objects in images, but they do it in different ways:

* **Semantic segmentation**: Groups pixels into object classes, like labeling all pixels that belong to a car or a tree. It doesn’t separate different instances of the same object. For example, all cars in an image would be labeled the same way.
    
* **Instance segmentation**: Not only identifies the object class (like cars or trees) but also separates each instance of that object. So if there are two cars in the image, it will label each car separately as its own instance.
    
* Semantic segmentation tells you, "There are cars here."
    
* Instance segmentation tells you, "There are two separate cars here."
    

**Q: What is single forward pass.**

In a "single forward pass," the model processes the entire image at once, rather than analyzing small parts of the image one by one. This allows models like YOLO (You Only Look Once) to detect objects quickly, making it great for real-time applications like video analysis.

**Q: To design an image search feature for an e-commerce site, you need to?**

* Build a feature extraction model that recognizes important image details.
    
* Make sure the system works regardless of image size or orientation.
    
* Protect user privacy and ensure data security.
    

All of these factors are important to create a flexible and safe system for users to upload and search for products with images.

**Q: What is CTC loss function?**

CTC (Connectionist Temporal Classification) loss helps match audio with text without knowing the exact timing. It's useful for speech recognition when the timing of the words isn't clear.

**Q: what is WaveNet?**

WaveNet stands out by creating high-quality sound using raw audio waveforms, producing more natural-sounding voices compared to traditional methods. It achieves this through a deep model that generates audio at a higher quality.

**Q: What is edge AI and explanable AI**

**Edge AI**:  
Edge AI refers to running AI algorithms directly on devices (like smartphones, cameras, or IoT sensors) without relying on a central cloud server. This approach enables real-time decision-making, improves privacy by keeping data locally, and reduces latency since data doesn’t need to be sent to the cloud for processing.

**Explainable AI (XAI)**:  
Explainable AI focuses on making AI systems more understandable and transparent to humans. It provides explanations for how a model made a decision, helping build trust and ensuring accountability. This is especially important in fields like healthcare or finance, where understanding why a decision was made is crucial.

Q: How does edge AI different from **Federated Learning**

**Federated Learning** and **Edge AI** are related but serve different purposes:

* **Federated Learning** is about training models across multiple devices (like smartphones, IoT devices) without moving the data to a central server. The data stays on the devices, and only model updates are shared with a central server.
    
* **Edge AI** refers to running AI algorithms directly on local devices (like sensors, smartphones, or IoT devices) rather than in the cloud or a central server. The focus here is on making fast, real-time decisions directly on the device without needing to connect to a server.
    

**Key Difference**:

* **Federated Learning**: Focuses on how models are trained across many devices while keeping data private.
    
* **Edge AI**: Focuses on running AI models on local devices for fast decision-making.
    

Analogy:

* **Federated Learning** is like students in different cities studying the same subject but only sharing their notes with a teacher.
    
* **Edge AI** is like each student solving problems on their own without needing to consult the teacher for every answer.
    

**Q: How does Meta-Learning differ from traditional Machine Learning, and why is it considered a promising direction for future research?**

It enables models to learn how to learn

**Q: What is the main advantage of using Neuromorphic Computing in the context of Machine Learning, and how might it shape future developments?**

Mimics the Human Brain

Neuromorphic Computing mimics the human brain's structure and functioning, providing a novel approach that might lead to more efficient and biologically inspired algorithms.