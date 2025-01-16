---
title: "Agentic AI w/ Neo4j & Langchain for graph database"
seoTitle: "Agentic AI w/ Neo4j & Langchain for graph database"
seoDescription: "Agentic AI w/ Neo4j & Langchain for graph database"
datePublished: Thu Jan 16 2025 00:27:42 GMT+0000 (Coordinated Universal Time)
cuid: cm5yld9ey000809mdf7wfedbs
slug: agentic-ai-w-neo4j-langchain-for-graph-database
tags: neo4j, openai, knowledge-graph, langchain, many-to-many

---

**Goal**: Use **Neo4j** to build a graph of job roles and required skills. The agent maps relationships, answers queries like “What skills are linked to a Data Scientist?” or “Which roles need Python?” and provides context-aware insights.

---

### **1\. Install Required Libraries**

```bash
!pip install neo4j langchain openai
```

* **Purpose**: This installs:
    
    * `neo4j`: Python library to connect and interact with the Neo4j graph database.
        
    * `langchain`: Framework to integrate LLMs (Large Language Models) with tools like Neo4j.
        
    * `openai`: Library to interact with OpenAI's LLMs like GPT-3.5 or GPT-4.
        
* **What Happens**: Libraries are installed so we can connect to Neo4j, use LangChain for agents, and interact with OpenAI models.
    

---

### **2\. Import Required Libraries**

```python
from neo4j import GraphDatabase
from langchain.agents import Tool, initialize_agent
from langchain.llms import OpenAI
```

* **Purpose**:
    
    * `GraphDatabase`: Handles connection to Neo4j.
        
    * `Tool` and `initialize_agent`: Part of LangChain, used to define tools and set up the agent.
        
    * `OpenAI`: Lets us interact with LLMs like GPT.
        
* **What Happens**: These libraries are now available for use in our script.
    

---

### **3\. Connect to Neo4j**

```python
class Neo4jDatabase:
    # Initialize Neo4j connection
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    # Close the connection
    def close(self):
        self.driver.close()

    # Run a query in the database
    def run_query(self, query):
        with self.driver.session() as session:
            result = session.run(query)  # Executes the query in Neo4j
            return [record.data() for record in result]  # Return query results as a list of dictionaries
```

* **Purpose**:
    
    * The `Neo4jDatabase` class connects to your Neo4j instance, runs queries, and returns results.
        
    * Replace `uri`, `user`, and `password` with your Neo4j credentials.
        

```python
db = Neo4jDatabase(uri="bolt://localhost:7687", user="neo4j", password="your_password")
```

* **What Happens**: A connection to your Neo4j database is established. The `run_query` method lets us send Cypher (Neo4j query language) commands to the database.
    

---

### **4\. Create and Populate a Knowledge Graph**

```python
query = """
MERGE (:JobRole {name: 'Data Scientist'})-[:REQUIRES]->(:Skill {name: 'Python'})
MERGE (:JobRole {name: 'Data Scientist'})-[:REQUIRES]->(:Skill {name: 'Machine Learning'})
MERGE (:JobRole {name: 'Data Engineer'})-[:REQUIRES]->(:Skill {name: 'SQL'})
MERGE (:JobRole {name: 'Data Engineer'})-[:REQUIRES]->(:Skill {name: 'Python'})
MERGE (:JobRole {name: 'Data Analyst'})-[:REQUIRES]->(:Skill {name: 'Excel'})
MERGE (:JobRole {name: 'Data Analyst'})-[:REQUIRES]->(:Skill {name: 'SQL'})
"""
db.run_query(query)
```

* **Explanation**:
    
    * **MERGE**: Ensures that the specified nodes (e.g., `JobRole` or `Skill`) and relationships (e.g., `REQUIRES`) exist in the graph.
        
    * Example Relationships:
        
        * "Data Scientist" requires "Python" and "Machine Learning".
            
        * "Data Engineer" requires "SQL" and "Python".
            
* **What Happens**:
    
    * A graph is created in Neo4j with job roles and their required skills.
        
    * You can visualize this graph in Neo4j Desktop or Neo4j Browser with `MATCH (n) RETURN n`.
        

---

### **5\. Define a Custom Tool for Neo4j Queries**

```python
class Neo4jTool(Tool):
    # Define the tool's name and description
    name = "query_neo4j"
    description = "Use this tool to query relationships between job roles and skills."

    # Function to run a Cypher query
    def _run(self, query: str):
        result = db.run_query(query)  # Execute the query using our Neo4jDatabase class
        return result  # Return query results as-is

    # Async functionality (not implemented here)
    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async mode not supported.")
```

* **Purpose**:
    
    * A LangChain-compatible tool for querying Neo4j.
        
    * When the agent needs information about job roles or skills, it will use this tool.
        
* **What Happens**:
    
    * This tool interacts with Neo4j and retrieves relationships (e.g., "Data Scientist requires Python").
        

---

### **6\. Initialize the LangChain Agent**

```python
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)  # Define the LLM

# Initialize the Neo4j tool
neo4j_tool = Neo4jTool()

# Create an agent with the Neo4j tool
agent = initialize_agent([neo4j_tool], llm, agent="zero-shot-react-description", verbose=True)
```

* **Purpose**:
    
    * Sets up a LangChain agent that uses the Neo4j tool to query the knowledge graph.
        
    * The `zero-shot-react-description` agent type allows the agent to choose tools autonomously based on queries.
        
* **What Happens**:
    
    * The agent is ready to handle queries and decide how to use the Neo4j tool for retrieving information.
        

---

### **7\. Query the Knowledge Graph**

```python
job_role = "Data Scientist"
query = f"""
MATCH (j:JobRole)-[:REQUIRES]->(s:Skill)
WHERE j.name = '{job_role}'
RETURN s.name AS skill
"""
response = agent.run(f"Retrieve the skills required for the job role: {job_role}. Use the query: {query}")
print(response)
```

* **Purpose**:
    
    * Queries Neo4j to find skills required for the "Data Scientist" role.
        
    * The agent runs the query and formats the response.
        
* **What Happens**:
    
    * The agent sends the query to Neo4j.
        
    * Neo4j returns: `[{ 'skill': 'Python' }, { 'skill': 'Machine Learning' }]`.
        
    * The agent processes the results and outputs a human-readable response.
        

---

### **Sample Output**

#### **Agent Logs (Verbose Mode)**

```python
> Entering new AgentExecutor chain...
Action: query_neo4j
Action Input: MATCH (j:JobRole)-[:REQUIRES]->(s:Skill) WHERE j.name = 'Data Scientist' RETURN s.name AS skill
Observation: [{'skill': 'Python'}, {'skill': 'Machine Learning'}]
Thought: I have retrieved the required skills. I will now summarize.
Final Answer: The skills required for a Data Scientist are Python and Machine Learning.
> Finished chain.
```

#### **Output:**

```python
The skills required for a Data Scientist are Python and Machine Learning.
```

---

### **What’s Happening in the Query?**

1. **MATCH**: Finds all relationships (`REQUIRES`) between the "Data Scientist" node and connected "Skill" nodes.
    
2. **RETURN**: Retrieves the skill names.
    

The agent formats this into a user-friendly response.

---

### **Summary**

* **Neo4j** stores relationships between job roles and skills in a graph database.
    
* **LangChain Agent** queries the graph, retrieves relevant data, and formats a response autonomously.
    
* This setup can be extended to more complex queries and workflows, like mapping career paths or analyzing market trends.
    

Let me know if you'd like help with further extensions or concepts! 🚀✨