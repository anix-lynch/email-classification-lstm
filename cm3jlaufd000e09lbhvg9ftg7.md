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

### **1\. Persisting Parameters Across Requests Using Session Objects**

A `Session` object allows you to persist headers, cookies, and parameters across multiple requests.

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

Prepared requests let you construct and inspect the request before sending.

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

Demonstrates how the body content flows in requests and responses.

```python
data = {'key': 'value'}
response = requests.post("https://httpbin.org/post", data=data)
print(response.json())
```

**Sample Output**:

```python
{
  "args": {},
  "data": "",
  "files": {},
  "form": {"key": "value"},
  "json": null,
  ...
}
```

---

### **6\. Using Generator for Sending Chunk-Encoded Requests**

Chunk-encoded requests are useful for large files or streaming data.

```python
def generate_data():
    yield b'chunk1\n'
    yield b'chunk2\n'

response = requests.post("https://httpbin.org/post", data=generate_data())
print(response.text)
```

**Sample Output**:

```python
{
  "args": {},
  "data": "chunk1\nchunk2\n",
  ...
}
```

---

### **7\. Getting the Request Method Arguments with Event Hooks**

Event hooks allow you to monitor or modify request/response behavior.

```python
def print_url(response, *args, **kwargs):
    print(f"Request URL: {response.url}")

hooks = {'response': print_url}
response = requests.get("https://aijobs.net/", hooks=hooks)
```

**Sample Output**:

```python
Request URL: https://aijobs.net/
```

---

### **8\. Iterating Over Streaming APIs**

Useful for APIs that return data in chunks.

```python
response = requests.get("https://httpbin.org/stream/3", stream=True)
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

APIs often use link headers to provide navigation or pagination information.

```python
response = requests.get("https://api.github.com/")
print(response.headers.get('Link'))
```

**Sample Output**:

```python
None  # Link header depends on the API being called.
```

---

### **10\. Transport Adapter**

Transport adapters allow customization of the connection process.

```python
from requests.adapters import HTTPAdapter

session = requests.Session()
adapter = HTTPAdapter(max_retries=3)
session.mount("https://", adapter)

response = session.get("https://aijobs.net/")
print(response.status_code)
```

**Sample Output**:

```python
200
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

```python
headers = {'Authorization': 'Bearer your_token_here'}
url = "https://api.example.com/protected"
response = requests.get(url, headers=headers)

print(f"Status Code: {response.status_code}")
print(response.text)
```

**Sample Output**:

```python
Status Code: 200
{"data": "protected resource"}
```

---

### **5\. Custom Authentication**

You can define custom authentication schemes by creating a class.

```python
from requests.auth import AuthBase

class CustomAuth(AuthBase):
    def __call__(self, r):
        # Add a custom header for authentication
        r.headers['X-Custom-Auth'] = 'my_custom_token'
        return r

url = "https://httpbin.org/headers"
response = requests.get(url, auth=CustomAuth())

print(f"Status Code: {response.status_code}")
print(response.json())
```

**Sample Output**:

```python
Status Code: 200
{'headers': {'X-Custom-Auth': 'my_custom_token', ...}}
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

HTTPretty is a Python library that mocks HTTP requests by intercepting them and returning predefined responses. This is useful for testing HTTP interactions without making actual network calls.

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

#### **Step 6: Save Data**

Save scraped data to a CSV file.

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