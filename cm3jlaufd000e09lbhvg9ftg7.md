---
title: "Python Request with Sample Output"
seoTitle: "Python Request with Sample Output"
seoDescription: "Python Request with Sample Output"
datePublished: Sat Nov 16 2024 03:09:52 GMT+0000 (Coordinated Universal Time)
cuid: cm3jlaufd000e09lbhvg9ftg7
slug: python-request-with-sample-output
tags: python, webscraping, requests

---

### **1\. Introduction to HTTP Request**

**Explanation**: Making a request to the web server to fetch resources.

```python
response = requests.get("https://aijobs.net/")
print(response.status_code)  # Status of the request
```

**Sample Output**:

```python
200
```

---

### **2\. Python Modules**

* Relevant Python modules used here:
    
    * `requests`: Handles HTTP requests.
        
    * `BeautifulSoup`: Parses HTML (from `bs4`).
        

---

### **3\. Requests vs. urllib2**

* `requests` is simpler and more intuitive than `urllib2`. Example comparison:
    
    * `requests`: `requests.get(url)`
        
    * `urllib2`: `urllib.request.urlopen(url)`
        

For most web scraping tasks, `requests` is preferred for its clean syntax and additional features like JSON handling.

---

### **4\. Essence of Requests**

The **essence** of `requests` lies in:

* Simple syntax for making requests.
    
* Direct integration with cookies, headers, and redirection handling.
    
* Example of setting headers for a request:
    

```python
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get("https://aijobs.net/", headers=headers)
print(response.status_code)
```

**Sample Output**:

```python
200
```

---

### **5\. Making a Simple Request**

```python
response = requests.get("https://aijobs.net/")
print(response.text[:200])  # Show first 200 characters
```

**Sample Output**:

```python
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Jobs</title>
</head>
<body>
...
```

---

### **6\. Response Content**

**Explanation**: Extract specific content types.

```python
print(response.text[:100])  # HTML
```

**Sample Output**:

```python
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
...
```

---

### **7\. Different Types of Request Contents**

**Explanation**: Extract different types of data from the response.

* **HTML Content**: `response.text`
    
* **Binary Content**: `response.content`
    

```python
print(response.text[:100])  # Text output
print(response.content[:50])  # Binary output
```

**Sample Output**:

```python
<!DOCTYPE html>
<html lang="en">
...
b'<!DOCTYPE html><html lang="en">'
```

---

### **8\. Looking up Built-in Response Status Codes**

```python
print(f"Status Code: {response.status_code}")
```

**Sample Output**:

```python
Status Code: 200
```

---

### **9\. Viewing Response Headers**

```python
print(response.headers)
```

**Sample Output**:

```python
{'Content-Type': 'text/html; charset=UTF-8',
 'Content-Length': '16234',
 'Connection': 'keep-alive',
 'Server': 'nginx',
 ...
}
```

---

### **10\. Accessing Cookies with Requests**

```python
print(response.cookies.get_dict())
```

**Sample Output**:

```python
{'sessionid': 'abcd1234efgh5678'}
```

---

### **11\. Tracking Redirection of the Request**

```python
if response.history:
    print("Redirect History:")
    for res in response.history:
        print(f"Status Code: {res.status_code}, URL: {res.url}")
else:
    print("No redirection occurred.")
```

**Sample Output**:

```python
No redirection occurred.
```

---

### **12\. Using Timeout to Keep Productive Usage in Check**

```python
try:
    response = requests.get("https://aijobs.net/", timeout=5)
    print(response.status_code)
except requests.Timeout:
    print("Request timed out!")
```

**Sample Output**:

```python
200
```

Or if a timeout occurs:

```python
Request timed out!
```

---

### **13\. Errors and Exceptions**

```python
try:
    response = requests.get("https://aijobs.net/", timeout=5)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
```

**Sample Output (No Errors)**:

```python
200
```

**Sample Output (With Error)**:

```python
An error occurred: 403 Client Error: Forbidden for URL: https://aijobs.net/
```

---

Here’s a breakdown of **Digging Deep into Requests** with **sample outputs**

---

### **1\. Persisting Parameters Across Requests Using Session Objects’**

A `Session` object allows you to persist headers, cookies, and parameters across multiple requests.

Imagine **Botty McScraper 🤖** is exploring a big castle 🏰 (a website like [**AIJobs.net**](http://AIJobs.net)) with many rooms. Each room has a guard (server) that asks, "Who are you?" before letting Botty enter.

Instead of introducing itself every single time (which gets tiring), Botty uses a **Session object**. The session acts like a **magic pass** that tells every guard, "Hey, it’s me, Botty! You already know me!" 😎✨  
A **Session object** in `requests` is like a reusable ID card:

* It saves **headers**, **cookies**, and other parameters across multiple requests.
    
* Instead of providing these details each time, the session **remembers them** for you.
    

```python
import requests

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})

response1 = session.get("https://aijobs.net/")
response2 = session.get("https://aijobs.net/jobs")

print(response1.status_code, response2.status_code)
```

**Sample Output**:

```python
200 200
```

---

### **2\. Revealing the Structure of a Request and Response**

The structure includes methods like `request.method`, `request.url`, `response.status_code`, `response.headers`.

```python
response = requests.get("https://aijobs.net/")
print(f"Request Method: {response.request.method}")
print(f"Request URL: {response.request.url}")
print(f"Response Status Code: {response.status_code}")
print(f"Response Headers: {response.headers}")
```

**Sample Output**:

```python
Request Method: GET
Request URL: https://aijobs.net/
Response Status Code: 200
Response Headers: {'Content-Type': 'text/html; charset=UTF-8', ...}
```

---

### **3\. Using Prepared Requests**

Imagine **Botty McScraper 🤖** is going to visit the grand [**AIJobs.net**](http://AIJobs.net) **castle** 🏰 again. But this time, Botty wants to be extra polite and well-prepared! Instead of rushing in, Botty writes a **formal invitation** (prepared request) to the guard (server), clearly stating:

* Who it is (headers, like User-Agent).
    
* What it wants (HTTP method like GET or POST).
    
* Where it's going (URL).
    

Once everything is ready, Botty sends this polished invitation to make a perfect first impression. ✨

```python
from requests import Request, Session

session = Session()
req = Request('GET', 'https://aijobs.net', headers={'User-Agent': 'Mozilla/5.0'})
prepared = session.prepare_request(req)

response = session.send(prepared)
print(prepared.url, response.status_code)
```

**Sample Output**:

```python
https://aijobs.net 200
```

---

### **4\. Verifying an SSL Certificate with Requests**

Requests verifies SSL certificates by default. You can disable it or provide custom certificates.

Imagine **Botty McScraper 🤖** is about to visit the grand [**AIJobs.net**](http://AIJobs.net) **castle** 🏰 again. But before Botty enters, it notices the guards holding a shiny **badge of trust** (SSL certificate). This badge tells Botty that:

1. The castle (website) is **secure**.
    
2. It really is [**AIJobs.net**](http://AIJobs.net) and not a fake replica set up by hackers.
    

Botty checks the badge (verifies the SSL certificate) to ensure it’s safe to proceed. If the badge is fake or missing, Botty refuses to enter. 🚷

```python
try:
    response = requests.get("https://aijobs.net/", verify=True)
    print(response.status_code)
except requests.exceptions.SSLError as e:
    print(f"SSL Error: {e}")
```

**Sample Output**:

```python
200
```

---

### **5\. Body Content Workflow**

Imagine **Botty McScraper 🤖** wants to send a letter 📨 to the [**httpbin.org**](http://httpbin.org) server, asking for some help. Inside the letter (the request body), Botty includes a **key-value pair**:  
`key: value`

The server receives the letter, processes the request, and sends a **detailed reply** back to Botty. The reply contains a summary of everything Botty sent and how the server handled it. ✨

---

### **What Is the Body Content Workflow?**

The **body** of an HTTP request is where you send data (like a form submission or API payload). The workflow is:

1. **Botty Sends Data**: The request body includes key-value pairs, files, or JSON.
    
2. **Server Processes It**: The server extracts the information from the body.
    
3. **Server Responds**: The response contains details of what was received.
    

Demonstrates how the body content flows in requests and responses.

```python
import requests

# Data Botty sends to the server
data = {'key': 'value'}

# Botty makes a POST request with the data
response = requests.post("https://httpbin.org/post", data=data)

# Botty prints the server's JSON response
print(response.json())
```

**Sample Output**:

```python
{
  "args": {},            # Query string parameters (not used here)
  "data": "",            # Raw body data (empty because we're using form data)
  "files": {},           # Any uploaded files (not used here)
  "form": {              # Form data Botty sent
    "key": "value"
  },
  "json": null,          # JSON payload (not used here)
  ...
}
```

---

### **6\. Using Generator for Sending Chunk-Encoded Requests**

Chunk-encoded requests are useful for large files or streaming data.Botty Sends a File in Bitesized Chunks 🍪

```python
import requests

# Botty prepares the data generator
def generate_data():
    yield b'chunk1\n'  # First chunk
    yield b'chunk2\n'  # Second chunk

# Botty sends the data in chunks
response = requests.post("https://httpbin.org/post", data=generate_data())

# Botty prints the server's response
print(response.text)
```

**Sample Output**:

```python
{
  "args": {},           # Query string parameters (not used here)
  "data": "chunk1\nchunk2\n",  # Combined chunks received by the server
  "files": {},          # Any uploaded files (not used here)
  "form": {},           # Form data (not used here)
  "json": null,         # JSON payload (not used here)
  ...
}
```

---

### **7\. Getting the Request Method Arguments with Event Hooks**

Imagine **Botty McScraper 🤖** is on a treasure hunt on [**AIJobs.net**](http://AIJobs.net) 🏴‍☠️. Botty has a habit of keeping a **logbook** of all the places it visits. To avoid forgetting, Botty uses a **magical helper** called an **event hook** that records the exact URL of every page it visits.

---

### **What Are Event Hooks?**

* **Event Hooks** allow you to **monitor or modify behavior** during a request/response cycle in `requests`.
    
* You can:
    
    * Track the request’s progress.
        
    * Inspect the response before processing.
        
    * Log useful information (like URLs or headers).
        

```python
import requests

# Define a function to log the URL of each response
def print_url(response, *args, **kwargs):
    print(f"Request URL: {response.url}")

# Create an event hook dictionary
hooks = {'response': print_url}

# Botty makes a GET request with the event hook
response = requests.get("https://aijobs.net/", hooks=hooks)

# Print the response status code
print(f"Status Code: {response.status_code}")
```

**Sample Output**:

```python
Request URL: https://aijobs.net/
Status Code: 200
```

---

### **8\. Iterating Over Streaming APIs**

Useful for APIs that return data in chunks.

```python
import requests

# Botty listens to a streaming API
response = requests.get("https://httpbin.org/stream/3", stream=True)

# Botty processes each line (chunk) of data as it arrives
for chunk in response.iter_lines():
    print(chunk)
```

**Sample Output**:

```python
b'{"id": 0, "message": "foo"}'
b'{"id": 1, "message": "bar"}'
b'{"id": 2, "message": "baz"}'
```

---

### **9\. Self-Describing APIs with Link Headers**

Imagine **Botty McScraper 🤖** is exploring the **GitHub API** 📦, a treasure trove of data. To help Botty find its way, the API leaves **breadcrumbs** 🧵 in the form of **Link headers**. These breadcrumbs guide Botty to the next page of data or related resources, making navigation smooth and effortless. 🚀

**Initial Exploration**:

* Botty starts by checking if the **root endpoint** of the API provides a `Link` header.
    
* This step helps confirm if pagination or related resources are available.
    

```python
import requests

# Botty makes a request to the GitHub API
response = requests.get("https://api.github.com/")
link_header = response.headers.get('Link')
print(f"Link Header: {link_header}")
```

**Sample Output**:

### **Sample Output**

#### Case 1: No Link Header

```python
Link Header: None
```

#### Case 2: Pagination Example

```python
Link Header: <https://api.github.com/resource?page=2>; rel="next", <https://api.github.com/resource?page=5>; rel="last"
```

---

```python
#Second Block Follow Pagination Links
url = "https://api.github.com/resource?page=1"
while url:
    response = requests.get(url)
    print(f"Fetching: {url}")
    
    # Look for 'next' in Link header
    link_header = response.headers.get('Link')
    url = link_header.split(";")[0].strip("<>") if link_header and 'rel="next"' in link_header else None
```

### **Sample Output**

#### Fetching the First Page:

```python
Fetching: https://api.github.com/resource?page=1
```

#### Fetching the Second Page:

```python
Fetching: https://api.github.com/resource?page=2
```

#### Fetching the Third Page:

```python
Fetching: https://api.github.com/resource?page=3
```

#### End of Pagination (No More Pages):

```python
Fetching: https://api.github.com/resource?page=4
```

Once the last page is fetched and no `rel="next"` link exists in the `Link` header, the loop terminates.

### **10\. Transport Adapter**

**Story: Botty and the Stubborn Gatekeepers 🔄**

**Botty McScraper 🤖** is on its way to explore [**AIJobs.net**](http://AIJobs.net), but the castle's gatekeepers (servers) are a bit stubborn today. Sometimes they ignore Botty’s knocks (requests), and Botty gets no response. 😟

To solve this, Botty gets itself a **Transport Adapter** (a magical tool). With this, Botty can retry knocking **up to 3 times** if the gatekeepers don’t respond. Thanks to this persistence, Botty eventually gets inside the castle to collect all the treasure it needs! 🏰✨

### **What Are Transport Adapters?**

A **Transport Adapter** customizes how a `requests` session connects to the server. This includes:

1. **Retries**: Automatically retry requests if they fail due to network issues.
    
2. **Timeout Handling**: Specify retry strategies like delays between retries.
    
3. **Custom Behavior**: Tailor connection settings for specific protocols (e.g., HTTP, HTTPS).
    

```python
import requests
from requests.adapters import HTTPAdapter

# Botty creates a session
session = requests.Session()

# Configure the transport adapter with retry logic
adapter = HTTPAdapter(max_retries=3)  # Retry up to 3 times
session.mount("https://", adapter)  # Apply to all HTTPS requests

# Botty makes a request to AIJobs.net
response = session.get("https://aijobs.net/")

# Botty checks the status code
print(f"Status Code: {response.status_code} 🛡️")
```

### **Sample Output**

#### First Attempt Succeeds:

```python
Status Code: 200 🛡️
```

#### If the First Attempt Fails (Retry in Action):

```python
Retry 1: Failed
Retry 2: Failed
Retry 3: Success!
Status Code: 200 🛡️
```

---

### **1\. Basic Authentication**

Basic authentication uses a username and password encoded in base64.

```python
from requests.auth import HTTPBasicAuth

url = "https://httpbin.org/basic-auth/user/pass"
response = requests.get(url, auth=HTTPBasicAuth('user', 'pass'))

print(f"Status Code: {response.status_code}")
print(response.json())
```

**Sample Output**:

```python
Status Code: 200
{'authenticated': True, 'user': 'user'}
```

---

### **2\. Digest Authentication**

Digest authentication hashes the credentials, adding security.

```python
from requests.auth import HTTPDigestAuth

url = "https://httpbin.org/digest-auth/auth/user/pass"
response = requests.get(url, auth=HTTPDigestAuth('user', 'pass'))

print(f"Status Code: {response.status_code}")
print(response.json())
```

**Sample Output**:

```python
Status Code: 200
{'authenticated': True, 'user': 'user'}
```

---

### **3\. Kerberos Authentication**

Kerberos is used for secure single sign-on (SSO). You’ll need the `requests_kerberos` library.

### **What Is Kerberos Authentication?**

* **Kerberos** is a secure protocol for **single sign-on (SSO)**.
    
* Often used in corporate environments, Kerberos allows users (or bots like Botty) to access multiple services after logging in once.
    
* With Kerberos, authentication is handled via a **secure ticketing system**.
    

```bash
pip install requests-kerberos
```

```python
from requests_kerberos import HTTPKerberosAuth

url = "https://example.com/kerberos-auth"
kerberos_auth = HTTPKerberosAuth()

response = requests.get(url, auth=kerberos_auth)
print(f"Status Code: {response.status_code}")
```

**Sample Output**: This depends on the Kerberos setup. If not configured, you’ll see:

```python
Status Code: 401  # Unauthorized
```

---

### **4\. Token Authentication**

Use tokens for API authentication.

Instead of sending username/password repeatedly, the server issues a **token** (like a key) after login.

```python
import requests

# Botty's secret API token
headers = {'Authorization': 'Bearer your_token_here'}

# Botty makes a request to the protected API
url = "https://api.example.com/protected"
response = requests.get(url, headers=headers)

# Botty checks the response
print(f"Status Code: {response.status_code}")
print(response.text)
```

**Sample Output**:

```python
Status Code: 200
{"data": "protected resource"}
```

---

#### **Case 2: Token is Invalid or Missing**

```python
Status Code: 401
{"error": "Unauthorized"}
```

### **5\. Custom Authentication**

**Story: Botty Designs Its Own VIP Badge 🛠️**

**Botty McScraper 🤖** is invited to a unique API party 🎉 where the standard authentication methods (like tokens or basic auth) don’t work. Instead, Botty needs a **custom badge** 🛡️ that includes a unique secret called `X-Custom-Auth`.

```python
import requests
from requests.auth import AuthBase

# Define Botty's custom authentication class
class CustomAuth(AuthBase):
    def __call__(self, r):
        # Add a custom header for authentication
        r.headers['X-Custom-Auth'] = 'my_custom_token'
        return r

# Botty makes a request with custom authentication
url = "https://httpbin.org/headers"
response = requests.get(url, auth=CustomAuth())

# Botty checks the response
print(f"Status Code: {response.status_code}")
print(response.json())
```

**Sample Output**:

```python
Status Code: 200
{
  "headers": {
    "X-Custom-Auth": "my_custom_token",
    ...
  }
}
```

---

### Summary

* **Basic Authentication**: Easy, but less secure.
    
* **Digest Authentication**: More secure than Basic.
    
* **Kerberos Authentication**: Advanced, used in enterprise settings.
    
* **Token Authentication**: Common for modern APIs.
    
* **Custom Authentication**: Flexibility for unique authentication needs.
    

Here’s a step-by-step guide to **Mocking HTTP Requests Using HTTPretty**, with sample outputs for each step:

---

### **1\. Understanding HTTPretty**

**Botty McScraper 🤖** wants to practice talking to servers before the big adventure. Instead of making real requests (which can be slow or expensive), Botty uses a magical mirror called **HTTPretty** 🪞. This mirror pretends to be a server, responding with predefined answers every time Botty asks it something. It’s like a rehearsal for Botty’s big day! 🎉

---

### **What Is HTTPretty?**

* **HTTPretty** is a Python library that mocks HTTP requests.
    
* It intercepts real HTTP calls and returns **predefined responses** without contacting the actual server.
    
* **Why Use It?**:
    
    1. **Test Without Real Servers**: Great for simulating APIs.
        
    2. **Save Time and Resources**: Avoid unnecessary network requests.
        
    3. **Reproducible Tests**: Control the response and environment.
        

---

### **2\. Installing HTTPretty**

Install HTTPretty via `pip`:

```bash
pip install HTTPretty
```

**Sample Output**:

```python
Successfully installed httpretty
```

---

### **3\. Working with HTTPretty**

Mock an HTTP request and return a predefined response.

```python
import httpretty
import requests

# Enable HTTPretty
httpretty.enable()

# Register a mock endpoint
httpretty.register_uri(
    httpretty.GET,
    "https://mockapi.example.com/users",
    body='[{"id": 1, "name": "Alice"}]',
    content_type="application/json"
)

# Make a request
response = requests.get("https://mockapi.example.com/users")
print(f"Status Code: {response.status_code}")
print(f"Response Body: {response.json()}")

# Disable HTTPretty
httpretty.disable()
httpretty.reset()
```

**Sample Output**:

```python
Status Code: 200
Response Body: [{'id': 1, 'name': 'Alice'}]
```

---

### **4\. Setting Headers**

You can mock responses with custom headers.

```python
httpretty.register_uri(
    httpretty.GET,
    "https://mockapi.example.com/headers",
    body="Headers mock",
    adding_headers={"X-Custom-Header": "HTTPretty"}
)

response = requests.get("https://mockapi.example.com/headers")
print(f"Headers: {response.headers}")
```

**Sample Output**:

```python
Headers: {'X-Custom-Header': 'HTTPretty', ...}
```

---

### **5\. Working with Responses**

You can define status codes and more complex responses.

```python
httpretty.register_uri(
    httpretty.POST,
    "https://mockapi.example.com/login",
    status=201,
    body='{"message": "User created"}',
    content_type="application/json"
)

response = requests.post("https://mockapi.example.com/login", data={"username": "test"})
print(f"Status Code: {response.status_code}")
print(f"Response Body: {response.json()}")
```

**Sample Output**:

```python
Status Code: 201
Response Body: {'message': 'User created'}
```

---

### Summary

* **Understanding HTTPretty**: Used for mocking HTTP requests.
    
* **Installing HTTPretty**: Install with `pip`.
    
* **Working with HTTPretty**: Register URIs to return mock responses.
    
* **Setting Headers**: Define custom headers in responses.
    
* **Working with Responses**: Control body, headers, and status codes.
    

HTTPretty is powerful for testing HTTP-based code without hitting real endpoints. Let me know if you’d like a deeper dive into any of these steps!

Here’s how you can interact with **social media platforms** using the `requests` library.

---

### **1\. API Introduction**

Social media platforms provide APIs to interact with their services. You need:

* **API keys/tokens**: Authentication credentials.
    
* **Endpoints**: URLs for specific actions like posting or retrieving data.
    
* **Rate limits**: Restrictions on the number of API calls.
    

For example:

* Facebook: Graph API
    
* Reddit: REST API with OAuth
    

```python
import requests

# Example of a generic API call
url = "https://api.example.com/data"
headers = {"Authorization": "Bearer YOUR_ACCESS_TOKEN"}
response = requests.get(url, headers=headers)
print(response.json())
```

---

### **2\. Interacting with Facebook**

Facebook uses the **Graph API** to interact with its platform. You can retrieve and post data like user details or page posts.

#### **Example: Fetching Facebook Page Data**

Replace `YOUR_ACCESS_TOKEN` with a valid token.

```python
import requests

url = "https://graph.facebook.com/v12.0/YOUR_PAGE_ID/posts"
params = {
    "access_token": "YOUR_ACCESS_TOKEN",
    "fields": "id,message,created_time"
}
response = requests.get(url, params=params)

print(f"Status Code: {response.status_code}")
print(f"Response JSON: {response.json()}")
```

**Sample Output**:

```python
Status Code: 200
Response JSON: {
    "data": [
        {"id": "12345", "message": "Hello, world!", "created_time": "2024-11-16T12:34:56+0000"},
        ...
    ]
}
```

#### **Posting to a Page**

```python
url = "https://graph.facebook.com/v12.0/YOUR_PAGE_ID/feed"
data = {
    "access_token": "YOUR_ACCESS_TOKEN",
    "message": "This is a post from the Graph API!"
}
response = requests.post(url, data=data)

print(f"Status Code: {response.status_code}")
print(f"Response JSON: {response.json()}")
```

**Sample Output**:

```python
Status Code: 200
Response JSON: {"id": "12345_67890"}
```

---

### **3\. Interacting with Reddit**

Reddit's API requires OAuth authentication. You can use `requests` to authenticate and interact with posts, comments, and subreddits.

#### **Authentication with Reddit**

First, get access tokens using your Reddit app credentials.

```python
import requests

auth = requests.auth.HTTPBasicAuth("CLIENT_ID", "SECRET")
data = {
    "grant_type": "password",
    "username": "YOUR_USERNAME",
    "password": "YOUR_PASSWORD"
}
headers = {"User-Agent": "YourApp/0.0.1"}
response = requests.post("https://www.reddit.com/api/v1/access_token", auth=auth, data=data, headers=headers)

print(f"Status Code: {response.status_code}")
print(f"Access Token: {response.json().get('access_token')}")
```

**Sample Output**:

```python
Status Code: 200
Access Token: abc123def456ghi789
```

#### **Fetching Subreddit Posts**

Use the token to fetch subreddit data.

```python
headers = {"Authorization": "Bearer YOUR_ACCESS_TOKEN", "User-Agent": "YourApp/0.0.1"}
url = "https://oauth.reddit.com/r/python/hot"
response = requests.get(url, headers=headers)

print(f"Status Code: {response.status_code}")
print(f"Response JSON: {response.json()}")
```

**Sample Output**:

```python
Status Code: 200
Response JSON: {
    "data": {
        "children": [
            {"data": {"title": "How to learn Python?", "score": 1234}},
            ...
        ]
    }
}
```

#### **Posting to a Subreddit**

```python
url = "https://oauth.reddit.com/api/submit"
headers = {"Authorization": "Bearer YOUR_ACCESS_TOKEN", "User-Agent": "YourApp/0.0.1"}
data = {
    "title": "Learning Python",
    "sr": "python",
    "kind": "self",
    "text": "What are the best resources to learn Python?"
}
response = requests.post(url, headers=headers, data=data)

print(f"Status Code: {response.status_code}")
print(f"Response JSON: {response.json()}")
```

**Sample Output**:

```python
Status Code: 200
Response JSON: {"json": {"data": {"id": "t3_abc123"}}}
```

---

### Summary

* **API Introduction**: APIs are the backbone for interacting with social media.
    
* **Interacting with Facebook**: Use Graph API for posts and data retrieval.
    
* **Interacting with Reddit**: Authenticate using OAuth, then interact with subreddits and posts.
    

Here’s a breakdown of **Web Scraping with Python Requests and BeautifulSoup**, with explanations and practical examples.

---

### **1\. Types of Data**

Web scraping allows you to extract various types of data:

* **Text**: Articles, product descriptions, blog posts.
    
* **Tables**: Financial data, sports stats.
    
* **Images**: Product photos, graphs.
    
* **Links**: URLs for navigation or crawling.
    

---

### **2\. What is Web Scraping?**

Web scraping is the process of automating the extraction of information from websites. It involves sending HTTP requests to a server, retrieving the HTML, and parsing it for data.

**Use Cases**:

* Price monitoring.
    
* Job listing aggregation.
    
* News or blog updates.
    
* Data collection for analytics.
    

---

### **3\. Key Web Scraping Tasks**

1. **Sending HTTP Requests**: Use the `requests` library to fetch web pages.
    
2. **Parsing HTML**: Use `BeautifulSoup` to navigate and extract desired elements.
    
3. **Handling Pagination**: Scrape multiple pages of data.
    
4. **Data Cleaning**: Process raw data into usable formats.
    
5. **Storing Data**: Save to files or databases.
    

---

### **4\. What is BeautifulSoup?**

BeautifulSoup is a Python library for parsing HTML and XML. It provides methods to navigate, search, and modify the parse tree.

**Key Features**:

* Search elements by tags, attributes, or text.
    
* Extract structured data like tables or lists.
    
* Handle poorly formatted HTML.
    

Install it via `pip`:

```bash
pip install beautifulsoup4
```

---

### **5\. Building a Web Scraping Bot - A Practical Example**

Let’s build a bot to scrape job postings from [AIJobs.net](http://AIJobs.net).

#### **Step 1: Send an HTTP Request**

```python
import requests
from bs4 import BeautifulSoup

url = "https://aijobs.net/"
response = requests.get(url)

print(f"Status Code: {response.status_code}")
```

**Sample Output**:

```python
Status Code: 200
```

---

#### **Step 2: Parse HTML with BeautifulSoup**

```python
soup = BeautifulSoup(response.text, 'html.parser')
print(soup.title.text)  # Extract the page title
```

**Sample Output**:

```python
AI Jobs
```

---

#### **Step 3: Extract Specific Data**

Assume job titles are in `<h2>` tags.

```python
job_titles = [title.text.strip() for title in soup.find_all('h2')]
print("Job Titles:")
for idx, title in enumerate(job_titles, start=1):
    print(f"{idx}: {title}")
```

**Sample Output**:

```python
Job Titles:
1: Senior AI Engineer
2: Machine Learning Engineer
3: Data Scientist
```

---

#### **Step 4: Extract Additional Details**

Scrape job titles, companies, and links.

```python
jobs = soup.find_all('div', class_='job-card')  # Adjust the class name as per website
for idx, job in enumerate(jobs, start=1):
    title = job.find('h2').text.strip()
    company = job.find('span', class_='company-name').text.strip()  # Example tag
    link = job.find('a', href=True)['href']
    print(f"Job {idx}:")
    print(f"  Title: {title}")
    print(f"  Company: {company}")
    print(f"  Link: {link}")
```

**Sample Output**:

```python
Job 1:
  Title: Senior AI Engineer
  Company: OpenAI
  Link: https://aijobs.net/job/senior-ai-engineer

Job 2:
  Title: Data Scientist
  Company: Google
  Link: https://aijobs.net/job/data-scientist
```

---

#### **Step 5: Handle Pagination**

For multi-page scraping, update the URL and loop through pages.

```python
for page in range(1, 4):  # Example: 3 pages
    url = f"https://aijobs.net/jobs?page={page}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract job titles as before
    job_titles = [title.text.strip() for title in soup.find_all('h2')]
    print(f"Page {page}:")
    print(job_titles)
```

**Sample Output**:

```python
Page 1:
['Senior AI Engineer', 'Machine Learning Engineer']

Page 2:
['AI Product Manager', 'Data Analyst']

Page 3:
['Deep Learning Researcher', 'AI Strategist']
```

---

#### **<mark>Step 6: Save Data</mark>**

<mark>Save scraped data to a CSV file.</mark>

```python
import csv

with open('jobs.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Title', 'Company', 'Link'])
    for job in jobs:
        title = job.find('h2').text.strip()
        company = job.find('span', class_='company-name').text.strip()
        link = job.find('a', href=True)['href']
        writer.writerow([title, company, link])

print("Data saved to jobs.csv")
```

**Output File (jobs.csv)**:

```python
Title,Company,Link
Senior AI Engineer,OpenAI,https://aijobs.net/job/senior-ai-engineer
Data Scientist,Google,https://aijobs.net/job/data-scientist
```

---

### **Key Takeaways**

1. Use `requests` to fetch HTML.
    
2. Use `BeautifulSoup` to parse and extract elements.
    
3. Handle multi-page scraping with loops.
    
4. Save extracted data to structured formats like CSV.
    

Here’s a step-by-step guide to **implementing a web application with Python using Flask**, with explanations and practical examples.

---

### **1\. What is Flask?**

Flask is a lightweight web framework for Python. It’s minimal yet powerful, allowing developers to build web applications quickly.

* **Key Features**:
    
    * Simple and flexible.
        
    * Built-in development server.
        
    * Extensible with plugins.
        

---

### **2\. Getting Started with Flask**

#### Basic Flask Application:

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

if __name__ == '__main__':
    app.run(debug=True)
```

**How it works**:

* `Flask`: Creates the app.
    
* `@app.route('/')`: Maps the root URL (`/`) to the `home` function.
    
* [`app.run`](http://app.run)`(debug=True)`: Starts the development server.
    

**Run it**:

```bash
python app.py
```

**Sample Output (in browser)**:

```python
Hello, Flask!
```

---

### **3\. Installing Flask**

Install Flask using `pip`:

```bash
pip install flask
```

**Sample Output**:

```python
Successfully installed flask
```

---

### **4\. Survey - A Simple Voting Application Using Flask**

We’ll build a voting app where users can vote on a survey.

---

#### **4.1 Application Structure**

Create a directory for the app:

```python
survey_app/
│
├── app.py            # Main application
├── templates/        # HTML files
│   ├── index.html
│   ├── result.html
├── static/           # Static files (CSS, JS, images)
│   ├── style.css
```

---

#### **4.2 Views**

Define routes in [`app.py`](http://app.py):

```python
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Survey data
survey_options = {"Option A": 0, "Option B": 0, "Option C": 0}

@app.route('/')
def index():
    return render_template('index.html', options=survey_options.keys())

@app.route('/vote', methods=['POST'])
def vote():
    option = request.form.get('option')
    if option in survey_options:
        survey_options[option] += 1
    return redirect(url_for('result'))

@app.route('/result')
def result():
    return render_template('result.html', results=survey_options)

if __name__ == '__main__':
    app.run(debug=True)
```

---

#### **4.3 Templates**

Create HTML files in the `templates/` folder.

**index.html**:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Survey</title>
</head>
<body>
    <h1>Survey Application</h1>
    <form method="POST" action="/vote">
        {% for option in options %}
            <label>
                <input type="radio" name="option" value="{{ option }}" required> {{ option }}
            </label><br>
        {% endfor %}
        <button type="submit">Vote</button>
    </form>
</body>
</html>
```

**result.html**:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Survey Results</title>
</head>
<body>
    <h1>Survey Results</h1>
    <ul>
        {% for option, votes in results.items() %}
            <li>{{ option }}: {{ votes }} votes</li>
        {% endfor %}
    </ul>
    <a href="/">Back to Survey</a>
</body>
</html>
```

---

#### **4.4 Running the Survey Application**

Run the app:

```bash
python app.py
```

Visit:

* **Survey**: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
    
* **Results**: Redirected to `/result` after voting.
    

---

### **5\. Writing Unit Tests for the Survey Application**

Write tests in a separate file, `test_`[`app.py`](http://app.py):

```python
import unittest
from app import app

class SurveyAppTest(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Survey Application", response.data)

    def test_vote(self):
        response = self.app.post('/vote', data={"option": "Option A"})
        self.assertEqual(response.status_code, 302)  # Redirect to results

    def test_results(self):
        response = self.app.get('/result')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Survey Results", response.data)

if __name__ == '__main__':
    unittest.main()
```

Run tests:

```bash
python test_app.py
```

**Sample Output**:

```python
...
----------------------------------------------------------------------
Ran 3 tests in 0.001s

OK
```

---

### **Summary**

1. **What is Flask?**: A lightweight web framework for Python.
    
2. **Getting Started with Flask**: Set up routes and run a basic app.
    
3. **Installing Flask**: Install with `pip`.
    
4. **Survey Application**: Built a voting app with views and templates.
    
5. **Writing Unit Tests**: Ensure the app behaves as expected.