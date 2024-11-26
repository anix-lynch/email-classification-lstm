---
title: "How to train your model to talk like Chandler Bing😅"
seoTitle: "How to train your model to talk like Chandler Bing😅"
seoDescription: "How to train your model to talk like Chandler Bing😅"
datePublished: Tue Nov 26 2024 12:50:16 GMT+0000 (Coordinated Universal Time)
cuid: cm3ygfr9u001r09jyfse188na
slug: how-to-train-your-model-to-talk-like-chandler-bing
tags: openai

---

The key to making the model respond in a **Chandler Bing-like tone** (or any character's tone) is setting the **role** properly, especially the **system role**. The **system role** is where you give the model explicit instructions on how to behave, and it allows you to control the **style**, **tone**, and **approach** of the model's responses.

### **Breakdown of the Roles:**

1. **System Role**: This is where you define **how the assistant should behave**. For example, for Chandler Bing, you specify that the assistant should respond with **sarcasm**, **humor**, and **dry wit** in a **Chandler Bing** style.
    
    Example:
    
    ```python
    pythonCopy code{'role': 'system', 'content': """You are an assistant who responds in the sarcastic and humorous tone of Chandler Bing from Friends. Your responses should include sarcasm, self-deprecating humor, and occasional dry jokes. Always be playful and sound like you might be breaking the fourth wall with a punchline."""}
    ```
    
2. **User Role**: This is the part where the **user** (or the script) provides input for the model to respond to. For example, you ask Chandler about **Joey's acting career** or **Monica's cleaning habits**, and Chandler responds accordingly.
    
    Example:
    
    ```python
    pythonCopy code{'role': 'user', 'content': """What do you think about Joey’s acting career?"""}
    ```
    
3. **Assistant Role**: This is where the model generates the response based on the **system** and **user** messages. You don’t set this role directly, but the model’s output is part of this role in the conversation.
    

### **Full Code with Chandler Bing's Realistic Topics:**

```python
#!/usr/bin/env python
# coding: utf-8

# ## Setup
# #### Load the API key and relevant Python libraries.
import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # Load environment variables (API key)

openai.api_key = os.environ['OPENAI_API_KEY']

# #### Helper function to interact with OpenAI
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=1,  # Chandler's tone requires a bit more creativity
    )
    return response.choices[0].message["content"]

# #### Get Chandler Bing's response in his usual tone
messages =  [  
    {'role':'system', 
     'content':"""You are an assistant who responds in the sarcastic, witty, and humorous tone of Chandler Bing from Friends. 
     Your responses should include sarcasm, self-deprecating humor, and occasional dry jokes. Always be playful and make sure to 
     sound like you might be breaking the fourth wall with a punchline. Use Chandler’s iconic quips like:
     "Could I BE any more...?", "I'm not great at the advice. Can I interest you in a sarcastic comment?"."""},    
    {'role':'user', 
     'content':"""How do you feel about Joey’s acting career?"""},  
] 

# Get the response with Chandler's tone
response = get_completion_from_messages(messages, temperature=1)
print(response)

# ### Example: Chandler talking about Monica's cleaning habits
response = get_completion("How do you feel about Monica's obsession with cleaning?")
print(response)

# ### Example: Chandler sarcastic response about Ross' relationships
response = get_completion("What do you think about Ross and Rachel getting back together?")
print(response)

# #### Chandler’s response with additional constraints (One sentence)
messages =  [  
    {'role':'system', 'content':'All your responses must be one sentence long.'},    
    {'role':'user', 'content':'What do you think about Joey eating a sandwich with his feet?'},  
] 
response = get_completion_from_messages(messages, temperature=1)
print(response)

# #### Chandler-style response with sentence length constraints (sarcasm)
messages =  [  
    {'role':'system', 'content':"""You are an assistant who responds in the sarcastic and humorous tone of Chandler Bing. 
    All your responses must be one sentence long."""},    
    {'role':'user', 'content':"""What do you think about Ross' dinosaur obsession?"""},
] 
response = get_completion_from_messages(messages, temperature=1)
print(response)

# #### Helper function to get completion and token count (for tracking tokens used)
def get_completion_and_token_count(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens,
    )
    
    content = response.choices[0].message["content"]
    token_dict = {
        'prompt_tokens':response['usage']['prompt_tokens'],
        'completion_tokens':response['usage']['completion_tokens'],
        'total_tokens':response['usage']['total_tokens'],
    }
    return content, token_dict

# #### Chandler-style response with token count tracking
messages = [
    {'role':'system', 
     'content':"""You are an assistant who responds in the sarcastic and humorous tone of Chandler Bing."""},    
    {'role':'user', 
     'content':"""What do you think about Joey getting a role in the soap opera?"},  
] 
response, token_dict = get_completion_and_token_count(messages)

# Print the response and token usage details
print(response)
print(token_dict)
```

### **Changes to Make Chandler's Responses Realistic**:

1. **Topic Change to Real Friends Discussions**:
    
    * Chandler **would actually talk** about **Joey's acting career**, **Monica's obsession with cleaning**, **Ross's dinosaur obsession**, and **Ross and Rachel's relationship**.
        
    * For example:
        
        * **"What do you think about Joey’s acting career?"**  
            Chandler might respond:  
            `"Oh, Joey’s career? Yeah, because *Days of Our Lives* is clearly the pinnacle of acting achievement. Next stop, an Oscar!"`
            
        * **"How do you feel about Monica's obsession with cleaning?"**  
            Chandler might say:  
            `"Monica cleaning? Oh, you mean her daily ritual of scrubbing the world until it’s as spotless as her self-esteem?"`
            
        * **"What do you think about Ross and Rachel getting back together?"**  
            Chandler's response might be:  
            `"Well, Ross and Rachel back together? It’s like a never-ending season finale. Could this BE any more dramatic?"`
            
2. **Add Specific Situations**:
    
    * We’ve changed the prompts to questions Chandler would typically be asked, or the topics he’d comment on, like Joey's feet-eating sandwich or Ross's dinosaur obsession.
        
3. **Tone**: The responses are delivered in Chandler’s **sarcastic** and **humorous** tone with **one-liners**, using his **iconic sarcasm** and **dry humor**:
    
    * `"Could I BE any more sarcastic?"`
        
    * `"I'm not great at advice. Can I interest you in a sarcastic comment?"`
        
4. **Constraints**: We maintain constraints like:
    
    * **One sentence responses** (as Chandler tends to deliver his humor in quick, punchy lines).
        

---

### **Expected Chandler-style Output Examples:**

1. **How do you feel about Joey’s acting career?**
    
    ```python
    "Oh, Joey’s career? Yeah, because *Days of Our Lives* is clearly the pinnacle of acting achievement. Next stop, an Oscar!"
    ```
    
2. **How do you feel about Monica’s obsession with cleaning?**
    
    ```python
    "Monica cleaning? Oh, you mean her daily ritual of scrubbing the world until it’s as spotless as her self-esteem?"
    ```
    
3. **What do you think about Ross and Rachel getting back together?**
    
    ```python
    "Well, Ross and Rachel back together? It’s like a never-ending season finale. Could this BE any more dramatic?"
    ```
    
4. **What do you think about Joey eating a sandwich with his feet?**
    
    ```python
    "Oh yeah, Joey eating a sandwich with his feet—because that’s exactly how you get a *star* role in a soap opera."
    ```
    
5. **What do you think about Ross' dinosaur obsession?**
    
    ```python
    "Ross and dinosaurs? Sure, why not. I mean, he’s *really* fitting in with the rest of us adults here. Just don’t ask him about any of his relationships."
    ```
    

### **Key Points**:

1. **System Role**: This is where you define **Chandler’s character** and how he should respond (sarcastic, witty, etc.).
    
2. **User Role**: This is where you ask a **question or prompt** for Chandler to respond to.
    
3. **Assistant Role**: This is the **model's response**, which will use the tone defined in the system role.
    

By setting the **system role** to define Chandler’s tone, the model will understand how to **respond appropriately** to the input while maintaining the **correct character**!

Now you have Chandler answering questions *like Chandler Bing* from *Friends*! Enjoy! 😎