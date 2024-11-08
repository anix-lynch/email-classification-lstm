---
title: "20 Scrapy concepts with Before-and-After Examples"
seoTitle: "20 Scrapy concepts with Before-and-After Examples"
seoDescription: "20 Scrapy concepts with Before-and-After Examples"
datePublished: Thu Oct 03 2024 15:09:15 GMT+0000 (Coordinated Universal Time)
cuid: cm1tfmhx1001s09jo3xao3r69
slug: 20-scrapy-concepts-with-before-and-after-examples
tags: python, data-science, data-analysis, scrapy, web-scraping

---

### 1\. **Creating a Scrapy Project** 📁

**Boilerplate Code**:

```bash
scrapy startproject myproject
```

**Use Case**: Initialize a new **Scrapy project**. 📁

**Goal**: Set up the basic structure for your Scrapy project. 🎯

**Sample Command**:

```bash
scrapy startproject myproject
```

**Before Example**:  
You need to scrape data but don’t have a project structure. 🤔

```bash
No project directory.
```

**After Example**:  
With **scrapy startproject**, you get a fully scaffolded project directory! 📁

```bash
myproject/
    ├── myproject/
    ├── scrapy.cfg
    └── ...
```

**Challenge**: 🌟 Try creating multiple Scrapy projects and see how the project structure varies with different settings.

---

### 2\. **Creating a Spider** 🕷️

**Boilerplate Code**:

```bash
scrapy genspider spider_name domain.com
```

**Use Case**: Create a **Spider** to scrape a specific website. 🕷️

**Goal**: Set up a spider that defines how to crawl and parse a website. 🎯

**Sample Command**:

```bash
scrapy genspider myspider example.com
```

**Before Example**:  
You want to scrape a website but don’t have a spider defined. 🤔

```bash
No spider available.
```

**After Example**:  
With **scrapy genspider**, you generate a spider file ready to customize! 🕷️

```bash
myproject/spiders/myspider.py
```

**Challenge**: 🌟 Try creating spiders for multiple domains and define the rules for each.

---

### 3\. **Running a Spider** 🏃‍♂️

**Boilerplate Code**:

```bash
scrapy crawl spider_name
```

**Use Case**: Use **crawl** to run your Scrapy spider. 🏃‍♂️

**Goal**: Execute the spider to crawl and scrape data from the target site. 🎯

**Sample Command**:

```bash
scrapy crawl myspider
```

**Before Example**:  
You’ve written your spider but don’t know how to execute it. 🤔

```bash
Spider exists, but no data collected.
```

**After Example**:  
With **scrapy crawl**, the spider runs, scrapes, and collects data! 🏃‍♂️

```bash
Data is collected and printed or stored.
```

**Challenge**: 🌟 Run the spider with the `-o` option to save scraped data into a file (e.g., `json`, `csv`).

---

### 4\. **Parsing Responses (parse method)** 🔍

**Boilerplate Code**:

```python
def parse(self, response):
    # Extract data here
    pass
```

**Use Case**: Define the **parse** method to handle the data extracted from responses. 🔍

**Goal**: Extract data from the HTML content of the page. 🎯

**Sample Code**:

```python
def parse(self, response):
    title = response.css('title::text').get()
    yield {'title': title}
```

**Before Example**:  
You have a spider that crawls pages but doesn’t extract specific data. 🤔

```python
HTML response is received but no data extracted.
```

**After Example**:  
With **parse**, you extract specific elements from the page! 🔍

```python
Extracted data: {"title": "Example Title"}
```

**Challenge**: 🌟 Try extracting multiple fields like headers, paragraphs, or links using CSS or XPath selectors.

---

### 5\. **CSS Selectors (response.css)** 🌐

**Boilerplate Code**:

```python
response.css('css_selector')
```

**Use Case**: Use **CSS selectors** to locate elements within the HTML response. 🌐

**Goal**: Select and extract data using CSS-like syntax. 🎯

**Sample Code**:

```python
title = response.css('title::text').get()
```

**Before Example**:  
You have an HTML response but can’t efficiently extract specific elements. 🤔

```python
Data: <title>Example Title</title>
```

**After Example**:  
With **CSS selectors**, you can easily extract the desired text or attributes! 🌐

```python
Output: "Example Title"
```

**Challenge**: 🌟 Use CSS selectors to extract different elements such as images (`img::attr(src)`), links (`a::attr(href)`), or text.

---

### 6\. **XPath Selectors (response.xpath)** 🧭

**Boilerplate Code**:

```python
response.xpath('xpath_expression')
```

**Use Case**: Use **XPath selectors** to extract elements from the HTML response. 🧭

**Goal**: Use powerful XPath expressions for more flexible or complex queries. 🎯

**Sample Code**:

```python
title = response.xpath('//title/text()').get()
```

**Before Example**:  
You need to extract elements but CSS selectors are not flexible enough. 🤔

```python
Data: <title>Example Title</title>
```

**After Example**:  
With **XPath**, you can extract data using more complex queries! 🧭

```python
Output: "Example Title"
```

**Challenge**: 🌟 Try using XPath to extract nested elements or multiple attributes in a single query.

---

### 7\. **Extracting Links (response.follow)** 🔗

**Boilerplate Code**:

```python
response.follow(link, callback)
```

**Use Case**: Use **follow** to navigate to links and scrape multiple pages. 🔗

**Goal**: Extract links from a page and follow them to scrape additional pages. 🎯

**Sample Code**:

```python
for href in response.css('a::attr(href)').getall():
    yield response.follow(href, self.parse)
```

**Before Example**:  
Your spider scrapes a single page but doesn’t navigate to other linked pages. 🤔

```python
Only the first page is scraped.
```

**After Example**:  
With **response.follow**, you can follow links and scrape multiple pages! 🔗

```python
The spider navigates and scrapes linked pages.
```

**Challenge**: 🌟 Try following only specific links, such as those that contain certain keywords or paths.

---

### 8\. **Storing Data (Item Pipeline)** 📊

**Boilerplate Code**:

```python
class MyItemPipeline:
    def process_item(self, item, spider):
        # Process and store the item
        return item
```

**Use Case**: Use **item pipelines** to store or process the scraped data. 📊

**Goal**: Define how scraped data should be processed and stored after extraction. 🎯

**Sample Code**:

```python
class MyItemPipeline:
    def process_item(self, item, spider):
        # Save item to a file or database
        with open('output.txt', 'a') as f:
            f.write(f"{item}\n")
        return item
```

**Before Example**:  
You’ve extracted data but have no way to store or process it. 🤔

```python
Scraped data is printed but not saved.
```

**After Example**:  
With **pipelines**, you can process and store data in files, databases, etc.! 📊

```python
Output: Data is saved to a file or database.
```

**Challenge**: 🌟 Try implementing pipelines to save data in formats like `CSV` or `JSON`.

---

### 9\. **Defining Items (Item Class)** 📋

**Boilerplate Code**:

```python
from scrapy import Item, Field
```

**Use Case**: Define a structured **Item** to represent the data you are scraping. 📋

**Goal**: Organize the scraped data into a structured format. 🎯

**Sample Code**:

```python
class MyItem(Item):
    title = Field()
    link = Field()
```

**Before Example**:  
You’ve scraped data but don’t have a structured format to represent it. 🤔

```python
Unstructured data extraction.
```

**After Example**:  
With **Item**, your data is organized into fields for better structure and processing! 📋

```python
Structured data: {"title": "Example", "link": "https://example.com"}
```

**Challenge**: 🌟 Try defining multiple fields and extract values for each one using CSS or XPath.

---

### 10\. **Handling Pagination (next page)** 🔄

**Boilerplate Code**:

```python
next_page = response.css('a.next::attr(href)').get()
if next_page:
    yield response.follow(next_page, self.parse)
```

**Use Case**: Handle **pagination** to scrape data across multiple pages. 🔄

**Goal**: Automatically navigate through paginated content to collect more data. 🎯

**Sample Code**:

```python
def parse(self, response):
    # Extract data from the current page
    yield {'title': response.css('title::text').get()}



    # Follow the pagination link
    next_page = response.css('a.next::attr(href)').get()
    if next_page:
        yield response.follow(next_page, self.parse)
```

**Before Example**:  
Your spider scrapes only the first page of a paginated website. 🤔

```python
Data is limited to the first page.
```

**After Example**:  
With **pagination handling**, the spider follows links and scrapes additional pages! 🔄

```python
Data collected from multiple pages.
```

**Challenge**: 🌟 Try handling pagination where the "next" button has different forms (e.g., buttons, JavaScript events).

---

### 11\. **Configuring Settings (Settings Module)** ⚙️

**Boilerplate Code**:

```python
from scrapy.utils.project import get_project_settings
```

**Use Case**: Use the **settings module** to configure how Scrapy runs. ⚙️

**Goal**: Adjust settings like user-agent, download delays, and more. 🎯

**Sample Code**:

```python
settings = get_project_settings()
settings.set('USER_AGENT', 'Mozilla/5.0 (compatible; MyScrapyBot/1.0)')
```

**Before Example**:  
Your spider runs with default settings, like a default user-agent, causing potential blocking. 🤔

```bash
Scrapy default settings in use.
```

**After Example**:  
With custom **settings**, you can fine-tune spider behavior like user-agent and download delays! ⚙️

```bash
Custom user-agent or settings applied.
```

**Challenge**: 🌟 Try adding download delays to prevent being blocked by websites (`DOWNLOAD_DELAY = 2`).

---

### 12\. **Handling Cookies (COOKIES\_ENABLED)** 🍪

**Boilerplate Code**:

```python
settings.set('COOKIES_ENABLED', True)
```

**Use Case**: Enable or disable **cookies** in your Scrapy project. 🍪

**Goal**: Control how your spider handles cookies for session-based scraping. 🎯

**Sample Code**:

```python
settings.set('COOKIES_ENABLED', True)
```

**Before Example**:  
Your spider struggles to maintain a session because cookies are not handled. 🤔

```bash
Session information is lost.
```

**After Example**:  
With **cookies enabled**, your spider maintains sessions correctly across requests! 🍪

```bash
Session data maintained via cookies.
```

**Challenge**: 🌟 Try scraping a website that requires login using cookies to maintain the session.

---

### 13\. **Customizing Request Headers (headers)** 📜

**Boilerplate Code**:

```python
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept-Language': 'en'
}
yield scrapy.Request(url, headers=headers)
```

**Use Case**: Customize **headers** in your requests to mimic real browser behavior. 📜

**Goal**: Avoid detection by websites and mimic genuine users. 🎯

**Sample Code**:

```python
# Send a request with custom headers
headers = {'User-Agent': 'Mozilla/5.0', 'Accept-Language': 'en'}
yield scrapy.Request(url="https://example.com", headers=headers)
```

**Before Example**:  
Your spider is blocked due to a missing or default user-agent. 🤔

```bash
Request blocked by server.
```

**After Example**:  
With **custom headers**, your spider mimics a real browser request! 📜

```bash
Request accepted with custom headers.
```

**Challenge**: 🌟 Experiment with different headers like `Referer` and `Accept-Encoding` to bypass bot detection.

---

### 14\. **Downloading Files (media)** 📂

**Boilerplate Code**:

```python
yield scrapy.Request(url, callback=self.save_file)
```

**Use Case**: Use Scrapy to **download files** like images or PDFs from the web. 📂

**Goal**: Automate the process of downloading media files from web pages. 🎯

**Sample Code**:

```python
def save_file(self, response):
    filename = response.url.split("/")[-1]
    with open(filename, 'wb') as f:
        f.write(response.body)
```

**Before Example**:  
You manually download files, which is time-consuming. 🤔

```bash
Files manually downloaded.
```

**After Example**:  
With **Scrapy**, files are automatically downloaded and saved! 📂

```bash
Files automatically saved to your system.
```

**Challenge**: 🌟 Try downloading multiple file types (e.g., images, PDFs, audio) from a website.

---

### 15\. **Using CrawlSpider (CrawlSpider Class)** 🕸️

**Boilerplate Code**:

```python
from scrapy.spiders import CrawlSpider, Rule
```

**Use Case**: Use **CrawlSpider** to handle more complex crawling, with automatic link extraction. 🕸️

**Goal**: Define rules to crawl a website efficiently, automatically following links. 🎯

**Sample Code**:

```python
from scrapy.linkextractors import LinkExtractor

class MySpider(CrawlSpider):
    name = 'my_crawler'
    start_urls = ['https://example.com']
    rules = [Rule(LinkExtractor(allow=('category/',)), callback='parse_item')]

    def parse_item(self, response):
        # Extract data
        yield {'title': response.css('title::text').get()}
```

**Before Example**:  
Your spider requires manual coding to follow links and extract data. 🤔

```bash
Manually coded link following.
```

**After Example**:  
With **CrawlSpider**, link extraction and crawling are automated! 🕸️

```bash
Automatic crawling and data extraction based on rules.
```

**Challenge**: 🌟 Define multiple rules for different types of links and customize crawling behavior.

---

### 16\. **Throttling Requests (AUTOTHROTTLE)** ⏳

**Boilerplate Code**:

```python
settings.set('AUTOTHROTTLE_ENABLED', True)
```

**Use Case**: Enable **AutoThrottle** to control the speed of requests dynamically. ⏳

**Goal**: Prevent being blocked by websites by adjusting request rates. 🎯

**Sample Code**:

```python
settings.set('AUTOTHROTTLE_ENABLED', True)
settings.set('AUTOTHROTTLE_START_DELAY', 1)
settings.set('AUTOTHROTTLE_MAX_DELAY', 10)
```

**Before Example**:  
Your spider sends too many requests too quickly, getting blocked by websites. 🤔

```bash
Website blocks requests due to high volume.
```

**After Example**:  
With **AutoThrottle**, your spider automatically adjusts request speed to avoid detection! ⏳

```bash
Spider adapts to avoid being blocked.
```

**Challenge**: 🌟 Try combining `AutoThrottle` with a proxy or user-agent rotation to further avoid detection.

---

### 17\. **Handling Redirects (REDIRECT\_ENABLED)** 🔄

**Boilerplate Code**:

```python
settings.set('REDIRECT_ENABLED', False)
```

**Use Case**: Control how your spider handles **redirects** (enable/disable). 🔄

**Goal**: Decide whether to follow redirects or handle them manually. 🎯

**Sample Code**:

```python
settings.set('REDIRECT_ENABLED', False)  # Prevent following redirects
```

**Before Example**:  
Your spider follows redirects, leading to pages you don't want to scrape. 🤔

```bash
Unwanted redirects followed.
```

**After Example**:  
With **redirects disabled**, your spider stays on the original page and handles redirects manually! 🔄

```bash
Redirects are not automatically followed.
```

**Challenge**: 🌟 Try enabling redirects and handling specific redirects programmatically.

---

### 18\. **Rotating User Agents (FAKE USER AGENT)** 🔄

**Boilerplate Code**:

```python
from fake_useragent import UserAgent
```

**Use Case**: Rotate **user agents** to avoid detection by websites. 🔄

**Goal**: Prevent being blocked by websites that monitor for bots with static user agents. 🎯

**Sample Code**:

```python
from fake_useragent import UserAgent

def start_requests(self):
    ua = UserAgent()
    headers = {'User-Agent': ua.random}
    yield scrapy.Request(url='https://example.com', headers=headers)
```

**Before Example**:  
You use the same user-agent for all requests, making it easy for websites to detect you as a bot. 🤔

```bash
Static user-agent leads to detection.
```

**After Example**:  
With **rotating user agents**, you reduce the chance of being detected! 🔄

```bash
User-agent rotated for each request.
```

**Challenge**: 🌟 Try using multiple user-agent strings and test different websites to see which are most effective.

---

### 19\. **Logging (LOG\_LEVEL)** 📝

**Boilerplate Code**:

```python
settings.set('LOG_LEVEL', 'INFO')
```

**Use Case**: Set the **log level** to control the verbosity of Scrapy’s logging. 📝

**Goal**: Adjust the level of logging (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`). 🎯

**Sample Code**:

```python
settings.set('LOG_LEVEL', 'DEBUG')  # Show detailed logging info
```

**Before Example**:  
Your logs are too verbose or too quiet, making it hard to debug or monitor the spider. 🤔

```bash
Irrelevant or missing log data.
```

**After Example**:  
With **log level control**, you see only the logs you need! 📝

```bash
Logs set to "DEBUG" for detailed information.
```

**Challenge**: 🌟 Experiment with different log levels and monitor how your spider behaves in each case.

---

### 20\. **Middleware (Custom Middleware)** ⚙️

**Boilerplate Code**:

```python

python
class MyCustomMiddleware:
    def process_request(self, request, spider):
        # Custom request processing logic
        return None
```

**Use Case**: Write **custom middleware** to modify requests or responses before/after they are handled. ⚙️

**Goal**: Intercept and modify requests or responses dynamically during scraping. 🎯

**Sample Code**:

```python
class MyCustomMiddleware:
    def process_request(self, request, spider):
        # Add a custom header to all requests
        request.headers['Custom-Header'] = 'MyValue'
        return None
```

**Before Example**:  
You need to modify requests/responses dynamically, but there’s no built-in feature for your use case. 🤔

```bash
Static request handling.
```

**After Example**:  
With **middleware**, you can intercept and modify requests or responses as needed! ⚙️

```bash
Custom headers added to all requests.
```

**Challenge**: 🌟 Try using middleware to retry failed requests or handle custom error conditions.

---