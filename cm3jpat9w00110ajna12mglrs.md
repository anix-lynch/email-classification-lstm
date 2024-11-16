---
title: "Scrapy with Sample Output"
seoTitle: "Scrapy with Sample Output"
seoDescription: "Scrapy with Sample Output"
datePublished: Sat Nov 16 2024 05:01:49 GMT+0000 (Coordinated Universal Time)
cuid: cm3jpat9w00110ajna12mglrs
slug: scrapy-with-sample-output
tags: data-science, scrapy, webscraping

---

# **1\. Introduction to Scrapy**

---

#### **What is Scrapy?**

Scrapy is an open-source web scraping framework for Python. It is designed for fast and efficient extraction of data from websites, handling large-scale scraping projects with ease.

---

#### **Features and Advantages**

1. **Fast and Efficient**:
    
    * Built-in support for asynchronous requests, making it faster than most scraping libraries.
        
2. **Structured Data Handling**:
    
    * Uses `Items` and `Pipelines` to process and store data efficiently.
        
3. **Flexible and Extensible**:
    
    * Customize behavior with middlewares and settings.
        
4. **Built-in Support for Pagination**:
    
    * Handles multi-page scraping and recursive crawling.
        
5. **Export Options**:
    
    * Save data in formats like JSON, CSV, XML, or directly into databases.
        

---

#### **Use Cases for Scrapy in Job Scraping**

* Extracting job titles, descriptions, and companies from websites like Indeed, LinkedIn, and [AIJobs.net](http://AIJobs.net).
    
* Automating data collection for job analysis and trend insights.
    
* Collecting job board data for market research or career planning.
    

---

#### **Scrapy Workflow**

1. **Create a Project**:
    
    * Initialize a Scrapy project with the necessary files and folders.
        
2. **Write a Spider**:
    
    * Define the logic for scraping, including target URLs and data extraction.
        
3. **Run the Spider**:
    
    * Execute the spider to collect data.
        
4. **Store the Data**:
    
    * Save extracted data into structured formats like JSON or CSV.
        

---

### **Example: Setting Up and Running Scrapy**

#### **1\. Install Scrapy**

Install Scrapy using pip:

```bash
pip install scrapy
```

Verify the installation:

```bash
scrapy --version
```

---

#### **2\. Create a Scrapy Project**

Run the following command to create a Scrapy project:

```bash
scrapy startproject job_scraper
```

**Directory Structure:**

```python
job_scraper/
    scrapy.cfg
    job_scraper/
        __init__.py
        items.py
        middlewares.py
        pipelines.py
        settings.py
        spiders/
            __init__.py
```

---

#### **3\. Create a Spider**

Navigate to the `spiders` folder and create a new spider file, e.g., `job_`[`spider.py`](http://spider.py).

**Code Example: Basic Spider**

```python
import scrapy

class JobSpider(scrapy.Spider):
    name = "job_spider"
    start_urls = ["https://example.com/jobs"]

    def parse(self, response):
        # Extract job titles
        job_titles = response.css('h2.job-title::text').getall()
        for title in job_titles:
            yield {"Job Title": title}
```

---

#### **4\. Run the Spider**

Navigate to the project directory and run the spider:

```bash
scrapy crawl job_spider -o jobs.json
```

**Sample Output (jobs.json):**

```json
[
    {"Job Title": "Data Scientist"},
    {"Job Title": "Machine Learning Engineer"},
    {"Job Title": "AI Researcher"}
]
```

---

#### **5\. Features Demonstrated**

* **Asynchronous Requests**: Multiple pages are crawled simultaneously.
    
* **CSS Selectors**: Used to extract job titles from the page.
    
* **Data Export**: Saved in JSON format for analysis.
    

---

# **2\. Installing Scrapy**

---

### **1\. Scrapy Installation on macOS**

#### **Step 1: Install Prerequisites**

Scrapy requires Python and some system dependencies.

1. **Install Homebrew (if not already installed):**
    
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```
    
2. **Install Python:**
    
    ```bash
    brew install python
    ```
    
3. **Install** `libxml2` and `libxslt`:
    
    ```bash
    brew install libxml2 libxslt
    ```
    
4. **Ensure** `pip` is up to date:
    
    ```bash
    python3 -m pip install --upgrade pip
    ```
    

---

#### **Step 2: Install Scrapy**

Install Scrapy using `pip`:

```bash
pip install scrapy
```

If you want to install it in a virtual environment (recommended), create and activate a virtual environment first:

```bash
python3 -m venv scrapy_env
source scrapy_env/bin/activate
pip install scrapy
```

---

### **2\. Verifying the Installation**

Run the following command to verify that Scrapy is installed successfully:

```bash
scrapy --version
```

**Expected Output:**

```plaintext
Scrapy 2.x.x
```

---

### **3\. Setting Up the Environment**

#### **Step 1: Create a New Scrapy Project**

1. Navigate to the directory where you want to create your project:
    
    ```bash
    cd ~/projects
    ```
    
2. Create a new Scrapy project:
    
    ```bash
    scrapy startproject job_scraper
    ```
    
3. Navigate to the newly created project:
    
    ```bash
    cd job_scraper
    ```
    

**Directory Structure:**

```python
job_scraper/
    scrapy.cfg
    job_scraper/
        __init__.py
        items.py
        middlewares.py
        pipelines.py
        settings.py
        spiders/
            __init__.py
```

---

#### **Step 2: Install Additional Dependencies**

Install any additional dependencies your project might need, such as:

* `playwright` for JavaScript-rendered content:
    
    ```bash
    pip install scrapy-playwright
    playwright install
    ```
    
* `pandas` for data processing:
    
    ```bash
    pip install pandas
    ```
    

---

### **4\. Testing the Installation**

Run the default Scrapy command to test the environment:

```bash
scrapy list
```

**Expected Output:**

```plaintext
No spiders found
```

This indicates Scrapy is installed and the project is set up correctly. You can now create your first spider.

---

### **Quick Troubleshooting**

1. **Issue: Command** `scrapy` not found.
    
    * Ensure Scrapy is installed in the correct Python environment.
        
    * Activate the virtual environment if used:
        
        ```bash
        source scrapy_env/bin/activate
        ```
        
2. **Issue: Missing dependencies (**`libxml2`, `libxslt`).
    
    * Install using Homebrew:
        
        ```bash
        brew install libxml2 libxslt
        ```
        

---

# **3\. Creating a Scrapy Project**

---

### **1\. Directory Structure of a Scrapy Project**

After creating a Scrapy project, it follows this structure:

```python
job_scraper/
    scrapy.cfg           # Project configuration file
    job_scraper/         # Main module for your Scrapy project
        __init__.py
        items.py         # Define data models for scraped items
        middlewares.py   # Middleware for processing requests/responses
        pipelines.py     # Define how to process/store scraped data
        settings.py      # Project-wide settings
        spiders/         # Folder containing spider definitions
            __init__.py
```

---

### **2\. Key Components of a Scrapy Project**

#### **Spiders**

* The spiders define the logic for crawling websites and extracting data.
    
* They specify starting URLs, how to parse responses, and how to follow links.
    

#### **Items**

* Items are Python classes that structure the scraped data.
    
* They define fields such as job title, company, and location.
    

#### **Pipelines**

* Pipelines process scraped data before storage.
    
* Examples: Cleaning data, saving to a database, or exporting to files.
    

#### **Middlewares**

* Middlewares process requests and responses.
    
* Examples: Rotating proxies, adding custom headers.
    

#### **Settings**

* Settings configure the Scrapy project (e.g., download delays, concurrent requests).
    
* Custom settings can be applied for each spider.
    

---

### **3\. Creating Your First Scrapy Project**

#### **Step 1: Create a New Project**

Run the following command to create a Scrapy project:

```bash
scrapy startproject job_scraper
```

Navigate to the project directory:

```bash
cd job_scraper
```

---

#### **Step 2: Define an Item**

Open [`items.py`](http://items.py) and define fields for the job data you want to scrape:

```python
import scrapy

class JobScraperItem(scrapy.Item):
    title = scrapy.Field()
    company = scrapy.Field()
    location = scrapy.Field()
    link = scrapy.Field()
```

---

#### **Step 3: Create a Spider**

Inside the `spiders/` directory, create a file named `job_`[`spider.py`](http://spider.py):

```python
import scrapy
from job_scraper.items import JobScraperItem

class JobSpider(scrapy.Spider):
    name = "job_spider"
    start_urls = ["https://example.com/jobs"]

    def parse(self, response):
        # Loop through job cards
        for job in response.css('div.job-card'):
            item = JobScraperItem()
            item['title'] = job.css('h2.job-title::text').get()
            item['company'] = job.css('span.company-name::text').get()
            item['location'] = job.css('span.job-location::text').get()
            item['link'] = response.urljoin(job.css('a::attr(href)').get())
            yield item
```

---

#### **Step 4: Configure Settings**

Open [`settings.py`](http://settings.py) and configure as needed:

* Enable JSON or CSV export:
    
    ```python
    FEEDS = {
        'jobs.json': {'format': 'json'},
    }
    ```
    
* Set a download delay to prevent bans:
    
    ```python
    DOWNLOAD_DELAY = 2
    ```
    

---

#### **Step 5: Run the Spider**

Run your spider:

```bash
scrapy crawl job_spider
```

Output will be saved to `jobs.json` as configured in [`settings.py`](http://settings.py).

**Sample Output (**`jobs.json`):

```json
[
    {"title": "Data Scientist", "company": "TechCorp", "location": "San Francisco, CA", "link": "https://example.com/jobs/123"},
    {"title": "Machine Learning Engineer", "company": "AI Corp", "location": "New York, NY", "link": "https://example.com/jobs/456"}
]
```

---

### **4\. Running Your Project and Verifying Outputs**

#### **Check the Data**

Ensure the scraped data matches the structure defined in [`items.py`](http://items.py).

#### **Debugging**

* Use the Scrapy Shell to debug selectors:
    
    ```bash
    scrapy shell "https://example.com/jobs"
    ```
    
    Example command in shell:
    
    ```python
    response.css('h2.job-title::text').getall()
    ```
    

---

# **4\. Writing and Running Spiders**

---

### **1\. Creating a Spider**

#### **What is a Spider?**

A spider in Scrapy is a class that defines how a website will be scraped:

* The URLs to scrape.
    
* How to extract data.
    
* How to follow links.
    

#### **Code Example: Creating a Spider**

Navigate to the `spiders/` directory and create a file, e.g., `job_`[`spider.py`](http://spider.py):

```python
import scrapy

class JobSpider(scrapy.Spider):
    name = "job_spider"  # Unique name for the spider
    start_urls = ["https://example.com/jobs"]  # Initial URLs to scrape

    def parse(self, response):
        pass  # Logic for extracting data goes here
```

---

### **2\. Defining** `start_urls`

#### **Purpose of** `start_urls`:

* It lists the initial URLs the spider will visit.
    

You can add multiple URLs:

```python
start_urls = [
    "https://example.com/jobs",
    "https://example.com/remote-jobs",
]
```

---

### **3\. Sending Requests and Receiving Responses**

Scrapy automatically sends GET requests to `start_urls` and passes the response to the `parse` method.

#### **Code Example: Extracting Response Status**

```python
def parse(self, response):
    print(f"Visited {response.url}, Status: {response.status}")
```

**Sample Output:**

```plaintext
Visited https://example.com/jobs, Status: 200
```

---

### **4\. Extracting Data from Job Listings**

#### **Using CSS Selectors**

CSS selectors allow you to extract elements based on tags, classes, or IDs.

**Example: Extracting Job Titles**

```python
def parse(self, response):
    job_titles = response.css('h2.job-title::text').getall()  # Get all job titles
    for title in job_titles:
        yield {"Job Title": title}
```

---

#### **Using XPath**

XPath is a more powerful alternative for selecting elements.

**Example: Extracting Job Titles with XPath**

```python
def parse(self, response):
    job_titles = response.xpath('//h2[@class="job-title"]/text()').getall()
    for title in job_titles:
        yield {"Job Title": title}
```

---

#### **Extracting Multiple Fields**

You can scrape job titles, companies, locations, and links in one loop.

**Code Example:**

```python
def parse(self, response):
    for job in response.css('div.job-card'):
        yield {
            "Job Title": job.css('h2.job-title::text').get(),
            "Company": job.css('span.company-name::text').get(),
            "Location": job.css('span.job-location::text').get(),
            "Link": response.urljoin(job.css('a::attr(href)').get()),
        }
```

**Sample Output:**

```json
[
    {"Job Title": "Data Scientist", "Company": "TechCorp", "Location": "San Francisco, CA", "Link": "https://example.com/job/123"},
    {"Job Title": "AI Engineer", "Company": "AI Corp", "Location": "New York, NY", "Link": "https://example.com/job/456"}
]
```

---

### **5\. Running Spiders Using** `scrapy crawl`

#### **Step 1: Run the Spider**

Run the following command to execute the spider:

```bash
scrapy crawl job_spider
```

#### **Step 2: Export Data to a File**

Export data directly to JSON, CSV, or XML:

```bash
scrapy crawl job_spider -o jobs.json  # Export to JSON
scrapy crawl job_spider -o jobs.csv   # Export to CSV
```

#### **Step 3: Debugging in Scrapy Shell**

Use the Scrapy Shell to debug selectors:

```bash
scrapy shell "https://example.com/jobs"
```

Example commands in shell:

```python
response.css('h2.job-title::text').getall()  # Get all job titles
response.xpath('//h2[@class="job-title"]/text()').getall()  # Using XPath
```

---

### **Key Takeaways**

1. **Spider Basics**:
    
    * Name your spider uniquely.
        
    * Use `start_urls` for the initial requests.
        
2. **Data Extraction**:
    
    * Use CSS and XPath selectors to extract structured data.
        
3. **Running and Exporting**:
    
    * Use `scrapy crawl` to run spiders and export data in desired formats.
        

# **5\. Data Extraction with Scrapy**

---

### **1\. CSS Selectors vs. XPath**

#### **How to Choose Between CSS and XPath**

* **CSS Selectors**:
    
    * Simple and readable syntax.
        
    * Ideal for extracting data based on tag names, classes, or IDs.
        
    * Supported in most browsers for quick testing.
        
* **XPath**:
    
    * More powerful and flexible.
        
    * Ideal for selecting elements based on advanced conditions (e.g., siblings, specific attributes).
        
    * Useful when elements are deeply nested or don’t have unique classes/IDs.
        

---

#### **Combining CSS and XPath**

You can mix both in a spider for robust data extraction.

**Code Example: Combining CSS and XPath**

```python
def parse(self, response):
    job_cards = response.css('div.job-card')  # CSS Selector for job cards
    for job in job_cards:
        title = job.css('h2.job-title::text').get()  # CSS for title
        company = job.xpath('.//span[@class="company-name"]/text()').get()  # XPath for company name
        yield {"Title": title, "Company": company}
```

---

### **2\. CSS Selectors**

#### **Selecting Elements**

* **By Tag Name**: Select elements by their HTML tag:
    
    ```css
    h2
    ```
    
* **By Class**: Select elements with a specific class:
    
    ```css
    .job-title
    ```
    
* **By ID**: Select elements with a specific ID:
    
    ```css
    #main-header
    ```
    
* **By Attribute**: Select elements with a specific attribute:
    
    ```css
    [type="text"]
    ```
    

#### **Extracting Text**

* Use `::text` to extract the text content of an element:
    
    ```css
    h2.job-title::text
    ```
    

#### **Extracting Attributes**

* Use `::attr(attribute)` to extract the value of an attribute:
    
    ```css
    a::attr(href)
    ```
    

**Code Example: CSS Selectors**

```python
def parse(self, response):
    for job in response.css('div.job-card'):
        yield {
            "Title": job.css('h2.job-title::text').get(),
            "Company": job.css('span.company-name::text').get(),
            "Location": job.css('span.job-location::text').get(),
            "Link": response.urljoin(job.css('a::attr(href)').get()),
        }
```

---

### **3\. XPath Selectors**

#### **Basic XPath Syntax**

* **By Tag Name**: Select all `<h2>` elements:
    
    ```python
    //h2
    ```
    
* **By Class or Attribute**: Select elements with a specific class or attribute:
    
    ```python
    //div[@class="job-card"]
    ```
    
* **Text Content**: Select elements based on their text:
    
    ```python
    //h2[text()="Data Scientist"]
    ```
    
* **Partial Match**: Use `contains()` for partial matches:
    
    ```python
    //h2[contains(@class, "title")]
    ```
    

#### **Extracting Text**

Use `text()` to extract the text content of an element:

```python
//h2[@class="job-title"]/text()
```

#### **Extracting Attributes**

Use `@attribute` to extract the value of an attribute:

```python
//a[@class="apply-link"]/@href
```

**Code Example: XPath Selectors**

```python
def parse(self, response):
    for job in response.xpath('//div[@class="job-card"]'):
        yield {
            "Title": job.xpath('.//h2[@class="job-title"]/text()').get(),
            "Company": job.xpath('.//span[@class="company-name"]/text()').get(),
            "Location": job.xpath('.//span[@class="job-location"]/text()').get(),
            "Link": response.urljoin(job.xpath('.//a[@class="apply-link"]/@href').get()),
        }
```

---

### **4\. Using** `text()` and `@attribute` in XPath

#### **Extracting Direct Text**

* Extract text directly inside a tag:
    
    ```python
    //h2[@class="job-title"]/text()
    ```
    

#### **Extracting Attribute Values**

* Extract attributes like `href`:
    
    ```python
    //a[@class="apply-link"]/@href
    ```
    

#### **Combining Conditions**

* Extract elements that satisfy multiple conditions:
    
    ```python
    //h2[@class="job-title" and contains(text(), "Engineer")]
    ```
    

---

### **5\. Comparison of CSS and XPath**

| **Feature** | **CSS** | **XPath** |
| --- | --- | --- |
| **Ease of Use** | Simple syntax | More complex but flexible |
| **Partial Matches** | Limited support | Fully supported with `contains()` |
| **Hierarchy** | Limited parent-child selection | Robust parent-child and sibling selection |
| **Attributes** | Simple with `::attr()` | Flexible with `@attribute` |
| **Testing** | Can test in browser DevTools | Requires XPath testing tools |

---

### **Key Takeaways**

1. **CSS Selectors**:
    
    * Best for straightforward structures.
        
    * Use `::text` and `::attr()` for text and attribute extraction.
        
2. **XPath**:
    
    * Best for complex conditions and nested elements.
        
    * Use `text()` and `@attribute` for fine-grained control.
        
3. **Combination**:
    
    * Use both CSS and XPath selectors in a spider for optimal results.
        

# **6\. Handling Pagination in Scrapy**

---

### **1\. Identifying Pagination Patterns**

#### **Types of Pagination**

1. **Next Button**:
    
    * A link or button to the next page, e.g., `<a href="/jobs?page=2">Next</a>`.
        
2. **Numbered Links**:
    
    * Links for specific pages, e.g., `<a href="/jobs?page=1">1</a>`.
        

#### **How to Identify Pagination**

1. Inspect the website's pagination element using browser DevTools.
    
2. Look for a `href` attribute that changes across pages.
    
3. Verify if URLs are relative or absolute.
    

---

### **2\. Extracting** `href` Attributes for the Next Page

#### **Example: Extracting the Next Page Link**

If the pagination uses a "Next" button:

```python
def parse(self, response):
    # Extract the next page link
    next_page = response.css('a.next::attr(href)').get()  # Update CSS selector as per website
    if next_page:
        yield response.follow(next_page, self.parse)
```

**Using XPath:**

```python
next_page = response.xpath('//a[@class="next"]/@href').get()
```

---

### **3\. Sending Requests to Subsequent Pages**

Scrapy's `response.follow()` handles relative and absolute URLs seamlessly.

**Example: Following Pagination Links**

```python
def parse(self, response):
    # Extract job data
    for job in response.css('div.job-card'):
        yield {
            "Title": job.css('h2.job-title::text').get(),
            "Company": job.css('span.company-name::text').get(),
            "Location": job.css('span.job-location::text').get(),
            "Link": response.urljoin(job.css('a::attr(href)').get()),
        }

    # Follow the next page
    next_page = response.css('a.next::attr(href)').get()
    if next_page:
        yield response.follow(next_page, self.parse)
```

---

### **4\. Implementing Recursive Crawling**

#### **Using** `start_requests`

You can override the `start_requests` method to handle initial requests or pagination logic.

**Example: Crawling Multiple Pages from Start**

```python
def start_requests(self):
    base_url = "https://example.com/jobs?page="
    for page in range(1, 6):  # Crawl pages 1 to 5
        yield scrapy.Request(url=f"{base_url}{page}", callback=self.parse)
```

#### **Using a Custom Callback Function**

Custom callbacks help when the pagination logic is complex or requires separate parsing for links.

**Example: Callback for Parsing Subsequent Pages**

```python
def parse(self, response):
    # Extract job data
    for job in response.css('div.job-card'):
        yield {
            "Title": job.css('h2.job-title::text').get(),
            "Company": job.css('span.company-name::text').get(),
            "Location": job.css('span.job-location::text').get(),
            "Link": response.urljoin(job.css('a::attr(href)').get()),
        }

    # Follow pagination
    next_page = response.css('a.next::attr(href)').get()
    if next_page:
        yield response.follow(next_page, callback=self.parse_next_page)

def parse_next_page(self, response):
    # Extract data from the next page
    for job in response.css('div.job-card'):
        yield {
            "Title": job.css('h2.job-title::text').get(),
            "Company": job.css('span.company-name::text').get(),
            "Location": job.css('span.job-location::text').get(),
            "Link": response.urljoin(job.css('a::attr(href)').get()),
        }

    # Continue following pagination
    next_page = response.css('a.next::attr(href)').get()
    if next_page:
        yield response.follow(next_page, callback=self.parse_next_page)
```

---

### **5\. Handling Numbered Pagination Links**

When a website lists page numbers, iterate over all available links.

**Example: Extracting All Pagination Links**

```python
def parse(self, response):
    # Extract job data
    for job in response.css('div.job-card'):
        yield {
            "Title": job.css('h2.job-title::text').get(),
            "Company": job.css('span.company-name::text').get(),
            "Location": job.css('span.job-location::text').get(),
            "Link": response.urljoin(job.css('a::attr(href)').get()),
        }

    # Follow numbered pagination links
    pagination_links = response.css('a.page-link::attr(href)').getall()
    for link in pagination_links:
        yield response.follow(link, callback=self.parse)
```

---

### **6\. Full Example: Handling Pagination**

**Spider Example: Scraping a Job Website**

```python
import scrapy

class JobSpider(scrapy.Spider):
    name = "job_spider"
    start_urls = ["https://example.com/jobs"]

    def parse(self, response):
        # Extract job data
        for job in response.css('div.job-card'):
            yield {
                "Title": job.css('h2.job-title::text').get(),
                "Company": job.css('span.company-name::text').get(),
                "Location": job.css('span.job-location::text').get(),
                "Link": response.urljoin(job.css('a::attr(href)').get()),
            }

        # Follow the next page
        next_page = response.css('a.next::attr(href)').get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)
```

**Command to Run Spider:**

```bash
scrapy crawl job_spider -o jobs.json
```

**Sample Output (**`jobs.json`):

```json
[
    {"Title": "Data Scientist", "Company": "TechCorp", "Location": "San Francisco, CA", "Link": "https://example.com/job/123"},
    {"Title": "AI Engineer", "Company": "AI Corp", "Location": "New York, NY", "Link": "https://example.com/job/456"},
    ...
]
```

---

### **Key Takeaways**

1. **Identify Pagination Patterns**:
    
    * Inspect next buttons or numbered links using browser DevTools.
        
2. **Follow Pagination Links**:
    
    * Use `response.follow()` for seamless crawling across pages.
        
3. **Recursive Crawling**:
    
    * Implement callbacks to handle dynamic or complex pagination logic.
        
4. **Use** `start_requests`:
    
    * For iterating over predefined URL patterns.
        

# **7\. Storing and Exporting Data in Scrapy**

---

### **1\. Structuring Scraped Data Using** `Items`

#### **Why Use** `Items`?

* Provides a structured way to define fields for your scraped data.
    
* Easier to maintain and process data consistently.
    

#### **Defining** `Items`

Open [`items.py`](http://items.py) in your Scrapy project and define fields for the data you want to scrape.

**Code Example: Defining Job Fields**

```python
import scrapy

class JobScraperItem(scrapy.Item):
    title = scrapy.Field()
    company = scrapy.Field()
    location = scrapy.Field()
    link = scrapy.Field()
```

---

### **2\. Using** `ItemLoaders` for Data Pre-Processing

#### **Why Use** `ItemLoaders`?

* Pre-process scraped data before storing it.
    
* Clean and format data (e.g., stripping whitespace, converting cases).
    

#### **Example: Using** `ItemLoaders`

Modify your spider to use `ItemLoaders`:

**Code Example: Applying Pre-Processing**

```python
from scrapy.loader import ItemLoader
from job_scraper.items import JobScraperItem

class JobSpider(scrapy.Spider):
    name = "job_spider"
    start_urls = ["https://example.com/jobs"]

    def parse(self, response):
        for job in response.css('div.job-card'):
            loader = ItemLoader(item=JobScraperItem(), selector=job)
            loader.add_css('title', 'h2.job-title::text')
            loader.add_css('company', 'span.company-name::text')
            loader.add_css('location', 'span.job-location::text')
            loader.add_css('link', 'a::attr(href)')
            yield loader.load_item()
```

**Output Example After Pre-Processing:**

```json
{
    "title": "Data Scientist",
    "company": "TechCorp",
    "location": "San Francisco, CA",
    "link": "https://example.com/job/123"
}
```

---

### **3\. Exporting Data**

Scrapy supports exporting data directly in formats like JSON, CSV, and XML.

#### **Exporting to JSON**

Run the spider and specify the output format:

```bash
scrapy crawl job_spider -o jobs.json
```

**Sample Output (**`jobs.json`):

```json
[
    {"title": "Data Scientist", "company": "TechCorp", "location": "San Francisco, CA", "link": "https://example.com/job/123"},
    {"title": "Machine Learning Engineer", "company": "AI Corp", "location": "New York, NY", "link": "https://example.com/job/456"}
]
```

---

#### **Exporting to CSV**

Run the spider with CSV format:

```bash
scrapy crawl job_spider -o jobs.csv
```

**Sample Output (**`jobs.csv`):

```python
title,company,location,link
Data Scientist,TechCorp,San Francisco,CA,https://example.com/job/123
Machine Learning Engineer,AI Corp,New York,NY,https://example.com/job/456
```

---

#### **Exporting to Excel via Pipelines**

You can save data in Excel format (`.xlsx`) by installing and using `openpyxl`.

**Step 1: Install** `openpyxl`

```bash
pip install openpyxl
```

**Step 2: Modify** [`pipelines.py`](http://pipelines.py) Update the pipeline to save data to Excel.

**Code Example: Excel Pipeline**

```python
import openpyxl

class ExcelPipeline:
    def open_spider(self, spider):
        # Create a new workbook and sheet
        self.workbook = openpyxl.Workbook()
        self.sheet = self.workbook.active
        self.sheet.title = "Jobs"
        # Define headers
        self.sheet.append(["Title", "Company", "Location", "Link"])

    def process_item(self, item, spider):
        # Append data row
        self.sheet.append([item['title'], item['company'], item['location'], item['link']])
        return item

    def close_spider(self, spider):
        # Save the workbook
        self.workbook.save("jobs.xlsx")
```

**Step 3: Enable the Pipeline in** [`settings.py`](http://settings.py)

```python
ITEM_PIPELINES = {
    'job_scraper.pipelines.ExcelPipeline': 300,
}
```

**Run the Spider:**

```bash
scrapy crawl job_spider
```

**Output:** `jobs.xlsx` with structured job data.

---

### **4\. Appending Data to Existing Files**

To append data without overwriting:

#### **For JSON and CSV**

* Use the `FEEDS` setting with `append` mode in [`settings.py`](http://settings.py):
    

```python
FEEDS = {
    'jobs.json': {
        'format': 'json',
        'overwrite': False,
        'encoding': 'utf8',
    },
    'jobs.csv': {
        'format': 'csv',
        'overwrite': False,
        'encoding': 'utf8',
    },
}
```

#### **For Excel**

Modify the pipeline to open and append to an existing file.

**Code Example: Append Data to Excel**

```python
import openpyxl

class ExcelPipeline:
    def open_spider(self, spider):
        try:
            # Try to load existing workbook
            self.workbook = openpyxl.load_workbook("jobs.xlsx")
            self.sheet = self.workbook.active
        except FileNotFoundError:
            # Create a new workbook if file doesn't exist
            self.workbook = openpyxl.Workbook()
            self.sheet = self.workbook.active
            self.sheet.append(["Title", "Company", "Location", "Link"])

    def process_item(self, item, spider):
        self.sheet.append([item['title'], item['company'], item['location'], item['link']])
        return item

    def close_spider(self, spider):
        self.workbook.save("jobs.xlsx")
```

---

### **Key Takeaways**

1. **Structured Data with** `Items`:
    
    * Define fields to ensure consistent data structure.
        
    * Use `ItemLoaders` for pre-processing.
        
2. **Exporting Data**:
    
    * Easily export data to JSON, CSV, or Excel.
        
3. **Appending Data**:
    
    * Use `FEEDS` settings or modify pipelines to prevent overwriting.
        
4. **Excel Integration**:
    
    * Utilize `openpyxl` for storing data in `.xlsx` files.
        

# **8\. Working with APIs**

---

### **1\. Identifying API Endpoints from Job Websites**

#### **Why Use APIs?**

* APIs provide structured data (e.g., JSON or XML) directly, eliminating the need for HTML parsing.
    
* Faster and more efficient compared to scraping web pages.
    

#### **Steps to Identify API Endpoints**

1. **Inspect Network Requests:**
    
    * Open browser DevTools (F12 in Chrome/Firefox).
        
    * Navigate to the **Network** tab and perform an action (e.g., searching for jobs).
        
    * Look for API calls in the **XHR** or **Fetch** filters.
        
2. **Analyze Request and Response:**
    
    * Examine the URL, HTTP method (GET/POST), headers, and payload.
        
    * Check if the response contains JSON data relevant to your needs.
        
3. **Validate Endpoint Accessibility:**
    
    * Copy the API URL and try sending a request using a tool like Postman or `curl`.
        

---

### **2\. Sending Authenticated API Requests with Scrapy**

#### **Step 1: Understand Authentication Requirements**

* **No Authentication**: Some APIs are open and can be accessed directly.
    
* **Token-Based Authentication**: APIs like LinkedIn require an API key or OAuth token in the headers.
    

#### **Step 2: Include Headers for Authentication**

Add required headers to your Scrapy request.

**Example: Adding API Key to Headers**

```python
class JobApiSpider(scrapy.Spider):
    name = "job_api_spider"
    start_urls = ["https://api.example.com/jobs"]

    def start_requests(self):
        headers = {
            "Authorization": "Bearer YOUR_API_KEY",
            "User-Agent": "Scrapy/1.0 (+https://example.com)",
        }
        for url in self.start_urls:
            yield scrapy.Request(url, headers=headers, callback=self.parse)

    def parse(self, response):
        data = response.json()  # Extract JSON data from the API response
        for job in data.get('results', []):
            yield {
                "Title": job.get('title'),
                "Company": job.get('company'),
                "Location": job.get('location'),
                "Link": job.get('url'),
            }
```

---

### **3\. Extracting Structured JSON Data**

#### **Step 1: Parse the JSON Response**

Scrapy provides a `response.json()` method to parse JSON directly.

**Example: Extracting Data**

```python
def parse(self, response):
    data = response.json()  # Parse JSON response
    for job in data['jobs']:
        yield {
            "Title": job['title'],
            "Company": job['company']['name'],
            "Location": job['location'],
            "Link": job['link'],
        }
```

#### **Sample API Response:**

```json
{
    "jobs": [
        {
            "title": "Data Scientist",
            "company": {"name": "TechCorp"},
            "location": "San Francisco, CA",
            "link": "https://example.com/job/123"
        },
        {
            "title": "AI Engineer",
            "company": {"name": "AI Corp"},
            "location": "New York, NY",
            "link": "https://example.com/job/456"
        }
    ]
}
```

#### **Output:**

```json
{
    "Title": "Data Scientist",
    "Company": "TechCorp",
    "Location": "San Francisco, CA",
    "Link": "https://example.com/job/123"
}
```

---

### **4\. Storing API Results in CSV or JSON Formats**

#### **Export to JSON**

Run the spider and save the output directly:

```bash
scrapy crawl job_api_spider -o jobs.json
```

#### **Export to CSV**

Save the output in CSV format:

```bash
scrapy crawl job_api_spider -o jobs.csv
```

**Sample CSV Output (**`jobs.csv`):

```python
Title,Company,Location,Link
Data Scientist,TechCorp,San Francisco, CA,https://example.com/job/123
AI Engineer,AI Corp,New York, NY,https://example.com/job/456
```

---

### **5\. Full Example: Using LinkedIn API**

#### **Step 1: Get API Access**

* Create a LinkedIn Developer Account.
    
* Register an application to get your **client ID** and **client secret**.
    
* Authenticate using OAuth to obtain an access token.
    

#### **Step 2: Use LinkedIn API for Job Data**

LinkedIn API requires an access token in the headers.

**Spider Example:**

```python
class LinkedInJobSpider(scrapy.Spider):
    name = "linkedin_jobs"
    start_urls = ["https://api.linkedin.com/v2/jobSearch"]

    def start_requests(self):
        headers = {
            "Authorization": "Bearer YOUR_ACCESS_TOKEN",
            "Content-Type": "application/json",
        }
        for url in self.start_urls:
            yield scrapy.Request(url, headers=headers, callback=self.parse)

    def parse(self, response):
        data = response.json()
        for job in data['elements']:
            yield {
                "Title": job['title'],
                "Company": job['companyName'],
                "Location": job.get('location', 'Remote'),
                "Link": job['applyUrl'],
            }
```

**Command to Run Spider:**

```bash
scrapy crawl linkedin_jobs -o linkedin_jobs.json
```

---

### **6\. Key Considerations for API Scraping**

#### **Advantages of Using APIs**

* **Structured Data**: Easier to extract and process.
    
* **Stability**: Less prone to breakage compared to scraping HTML.
    
* **Efficiency**: Faster data retrieval.
    

#### **Challenges**

* **Rate Limits**: APIs often limit the number of requests per minute/hour.
    
* **Authentication**: Some APIs require complex OAuth authentication.
    
* **Paid Access**: Certain APIs charge for higher usage tiers.
    

#### **Best Practices**

1. **Respect Rate Limits**:
    
    * Use the `DOWNLOAD_DELAY` setting in Scrapy:
        
        ```python
        DOWNLOAD_DELAY = 2  # Delay of 2 seconds between requests
        ```
        
2. **Retry Failed Requests**:
    
    * Enable the Retry Middleware in [`settings.py`](http://settings.py):
        
        ```python
        RETRY_ENABLED = True
        RETRY_TIMES = 3
        ```
        
3. **Store Data Efficiently**:
    
    * Use pipelines to save data directly into databases (e.g., MongoDB or PostgreSQL).
        

---

### **Key Takeaways**

1. **Identify APIs**:
    
    * Use DevTools to find API endpoints.
        
    * Test API calls with tools like Postman.
        
2. **Authentication**:
    
    * Add required headers for token-based APIs.
        
3. **Data Extraction**:
    
    * Use `response.json()` to parse structured JSON responses.
        
4. **Exporting Results**:
    
    * Save results in JSON or CSV for analysis.
        
5. **Best Practices**:
    
    * Respect rate limits and handle retries gracefully.
        

# **9\. Scrapy Shell**

---

### **1\. What is the Scrapy Shell and How to Use It?**

#### **What is the Scrapy Shell?**

The Scrapy Shell is an interactive command-line environment for testing and debugging scraping logic. It allows you to:

* Fetch web pages.
    
* Inspect HTML responses.
    
* Experiment with CSS and XPath selectors.
    
* Test data extraction logic before implementing it in spiders.
    

---

#### **How to Start the Scrapy Shell**

Run the Scrapy Shell for a specific URL:

```bash
scrapy shell "https://example.com/jobs"
```

---

### **2\. Using the** `fetch()` Command to Inspect Responses

#### **Fetch a URL**

If you’re already in the shell, use the `fetch()` command to load a webpage:

```python
fetch("https://example.com/jobs")
```

#### **Inspect the Response**

After fetching a URL, the response is stored in the `response` object. You can inspect the response attributes:

* **Check the status code:**
    
    ```python
    response.status
    ```
    
    **Output:**
    
    ```plaintext
    200
    ```
    
* **View the response URL:**
    
    ```python
    response.url
    ```
    
    **Output:**
    
    ```plaintext
    https://example.com/jobs
    ```
    
* **Check the HTML content:**
    
    ```python
    print(response.text[:500])  # Display the first 500 characters
    ```
    

---

### **3\. Experimenting with CSS and XPath in the Shell**

#### **Using CSS Selectors**

You can test CSS selectors directly in the shell:

* **Select all job titles:**
    
    ```python
    response.css('h2.job-title::text').getall()
    ```
    
    **Output:**
    
    ```python
    ['Data Scientist', 'AI Engineer', 'Machine Learning Engineer']
    ```
    
* **Extract a specific attribute:**
    
    ```python
    response.css('a.apply-link::attr(href)').getall()
    ```
    
    **Output:**
    
    ```python
    ['/job/123', '/job/456', '/job/789']
    ```
    

#### **Using XPath**

Test XPath expressions for the same results:

* **Select all job titles:**
    
    ```python
    response.xpath('//h2[@class="job-title"]/text()').getall()
    ```
    
    **Output:**
    
    ```python
    ['Data Scientist', 'AI Engineer', 'Machine Learning Engineer']
    ```
    
* **Extract specific attributes:**
    
    ```python
    response.xpath('//a[@class="apply-link"]/@href').getall()
    ```
    
    **Output:**
    
    ```python
    ['/job/123', '/job/456', '/job/789']
    ```
    

---

#### **Using Both CSS and XPath**

You can use both CSS and XPath to locate elements and test which works better for your target website.

**Example: Get Job Titles with CSS and XPath**

```python
# CSS Selector
response.css('h2.job-title::text').getall()

# XPath
response.xpath('//h2[@class="job-title"]/text()').getall()
```

---

### **4\. Debugging Using Scrapy Shell**

#### **Inspecting Issues**

If your spider isn’t working as expected, use the shell to debug:

1. Fetch the URL that’s causing the issue:
    
    ```python
    fetch("https://example.com/jobs")
    ```
    
2. Inspect the structure of the HTML:
    
    ```python
    print(response.text)
    ```
    
3. Test your selector logic:
    
    * CSS:
        
        ```python
        response.css('div.job-card h2.job-title::text').getall()
        ```
        
    * XPath:
        
        ```python
        response.xpath('//div[@class="job-card"]/h2[@class="job-title"]/text()').getall()
        ```
        

#### **Check Element Presence**

Sometimes, elements are missing or incorrectly targeted:

```python
response.css('div.nonexistent-class').get()
```

**Output:**

```python
None
```

This confirms that the element is not present in the HTML.

#### **Check for Errors in Responses**

If a request fails, you can inspect the response:

```python
response.status  # HTTP status code
response.headers  # Response headers
```

---

### **5\. Example Workflow in Scrapy Shell**

#### **Step 1: Start the Shell**

```bash
scrapy shell "https://example.com/jobs"
```

#### **Step 2: Inspect the Response**

* View the HTML content:
    
    ```python
    print(response.text[:500])
    ```
    

#### **Step 3: Experiment with Selectors**

* Extract job titles:
    
    ```python
    response.css('h2.job-title::text').getall()
    ```
    
* Extract links:
    
    ```python
    response.css('a.apply-link::attr(href)').getall()
    ```
    

#### **Step 4: Debug Complex Selectors**

* Test XPath for deeply nested elements:
    
    ```python
    response.xpath('//div[@class="job-card"]/a/@href').getall()
    ```
    

#### **Step 5: Verify Pagination**

* Test the next page selector:
    
    ```python
    response.css('a.next::attr(href)').get()
    ```
    

---

### **Key Takeaways**

1. **Scrapy Shell Basics**:
    
    * Use the shell for quick debugging and experimentation.
        
2. `fetch()` Command:
    
    * Fetch and inspect responses dynamically.
        
3. **Testing Selectors**:
    
    * Experiment with CSS and XPath to ensure data extraction works.
        
4. **Debugging**:
    
    * Inspect missing elements or incorrect logic before modifying your spider.
        

# **10\. Advanced Techniques for Job Scraping**

---

### **1\. Handling Dynamic JavaScript-Rendered Websites**

Some job websites use JavaScript to load content dynamically. Scrapy alone cannot render JavaScript, but libraries like `scrapy-playwright` can help.

---

#### **1.1 Using** `scrapy-playwright` for Rendering JavaScript

**Step 1: Install** `scrapy-playwright`

```bash
pip install scrapy-playwright
```

**Step 2: Enable** `scrapy-playwright` in Settings Add the following to your [`settings.py`](http://settings.py):

```python
DOWNLOADER_MIDDLEWARES = {
    'scrapy_playwright.middleware.ScrapyPlaywrightDownloadHandler': 543,
}

PLAYWRIGHT_BROWSER_TYPE = "chromium"  # Options: chromium, firefox, webkit
```

**Step 3: Use** `playwright` in Your Spider

```python
import scrapy

class DynamicJobSpider(scrapy.Spider):
    name = "dynamic_job_spider"

    def start_requests(self):
        yield scrapy.Request(
            url="https://example.com/jobs",
            meta={"playwright": True},  # Enable Playwright for this request
            callback=self.parse,
        )

    def parse(self, response):
        # Extract job data from JavaScript-rendered content
        for job in response.css('div.job-card'):
            yield {
                "Title": job.css('h2.job-title::text').get(),
                "Company": job.css('span.company-name::text').get(),
                "Location": job.css('span.job-location::text').get(),
                "Link": response.urljoin(job.css('a::attr(href)').get()),
            }
```

---

#### **1.2 Scraping Job Pages with Infinite Scrolling**

For websites with infinite scrolling:

1. Use Playwright to scroll and load content dynamically.
    
2. Capture the full page content.
    

**Code Example: Infinite Scrolling**

```python
from scrapy_playwright.page import PageCoroutine

class InfiniteScrollSpider(scrapy.Spider):
    name = "infinite_scroll"

    def start_requests(self):
        yield scrapy.Request(
            url="https://example.com/infinite-jobs",
            meta={
                "playwright": True,
                "playwright_page_coroutines": [
                    PageCoroutine("evaluate", "window.scrollTo(0, document.body.scrollHeight)"),
                    PageCoroutine("wait_for_timeout", 5000),  # Wait for more content to load
                ],
            },
            callback=self.parse,
        )

    def parse(self, response):
        for job in response.css('div.job-card'):
            yield {
                "Title": job.css('h2.job-title::text').get(),
                "Company": job.css('span.company-name::text').get(),
                "Location": job.css('span.job-location::text').get(),
                "Link": response.urljoin(job.css('a::attr(href)').get()),
            }
```

---

### **2\. Logging into Websites to Scrape Hidden Job Listings**

Many job websites require user authentication to access full job details.

---

#### **2.1 Using** `FormRequest` for Login Forms

**Code Example: Submitting a Login Form**

```python
from scrapy.http import FormRequest

class LoginSpider(scrapy.Spider):
    name = "login_spider"
    start_urls = ["https://example.com/login"]

    def parse(self, response):
        # Send a POST request to login
        return FormRequest.from_response(
            response,
            formdata={"username": "your_username", "password": "your_password"},
            callback=self.after_login,
        )

    def after_login(self, response):
        # Scrape job listings after successful login
        if "Welcome" in response.text:
            yield scrapy.Request(
                url="https://example.com/hidden-jobs",
                callback=self.parse_jobs,
            )

    def parse_jobs(self, response):
        for job in response.css('div.job-card'):
            yield {
                "Title": job.css('h2.job-title::text').get(),
                "Company": job.css('span.company-name::text').get(),
                "Location": job.css('span.job-location::text').get(),
            }
```

---

#### **2.2 Extracting CSRF Tokens Dynamically**

Many websites use CSRF tokens for form submissions.

**Code Example: Extract CSRF Tokens**

```python
def parse(self, response):
    csrf_token = response.css('input[name="csrf_token"]::attr(value)').get()
    return FormRequest.from_response(
        response,
        formdata={
            "username": "your_username",
            "password": "your_password",
            "csrf_token": csrf_token,
        },
        callback=self.after_login,
    )
```

---

### **3\. Extracting Job Listings from Tables**

Some websites display job data in HTML tables. Scraping these requires parsing rows (`<tr>`) and columns (`<td>`).

---

#### **3.1 Scraping Tabular HTML Structures**

**HTML Example:**

```html
<table class="job-table">
    <tr>
        <th>Title</th>
        <th>Company</th>
        <th>Location</th>
    </tr>
    <tr>
        <td>Data Scientist</td>
        <td>TechCorp</td>
        <td>San Francisco, CA</td>
    </tr>
    <tr>
        <td>AI Engineer</td>
        <td>AI Corp</td>
        <td>New York, NY</td>
    </tr>
</table>
```

**Code Example: Scraping Table Data**

```python
class TableJobSpider(scrapy.Spider):
    name = "table_job_spider"
    start_urls = ["https://example.com/jobs"]

    def parse(self, response):
        rows = response.css('table.job-table tr')[1:]  # Skip header row
        for row in rows:
            yield {
                "Title": row.css('td:nth-child(1)::text').get(),
                "Company": row.css('td:nth-child(2)::text').get(),
                "Location": row.css('td:nth-child(3)::text').get(),
            }
```

**Output:**

```json
[
    {"Title": "Data Scientist", "Company": "TechCorp", "Location": "San Francisco, CA"},
    {"Title": "AI Engineer", "Company": "AI Corp", "Location": "New York, NY"}
]
```

---

#### **3.2 Parsing and Cleaning Table Data**

Clean and normalize data before storing it.

**Code Example: Data Cleaning**

```python
def parse(self, response):
    rows = response.css('table.job-table tr')[1:]  # Skip header row
    for row in rows:
        title = row.css('td:nth-child(1)::text').get().strip()
        company = row.css('td:nth-child(2)::text').get().strip()
        location = row.css('td:nth-child(3)::text').get().strip()
        yield {
            "Title": title,
            "Company": company,
            "Location": location,
        }
```

---

### **Key Takeaways**

1. **Dynamic Content**:
    
    * Use `scrapy-playwright` for JavaScript-rendered pages and infinite scrolling.
        
2. **Authentication**:
    
    * Use `FormRequest` to log in and access hidden job listings.
        
    * Dynamically extract CSRF tokens for secure login forms.
        
3. **Tabular Data**:
    
    * Extract structured data from HTML tables and clean it for consistent storage.
        

# **11\. Managing Scrapy Settings**

Efficient Scrapy settings can improve scraping performance while reducing the risk of being blocked by target websites.

---

### **1\. Configuring Settings for Efficient Scraping**

#### **1.1 Controlling Concurrency and Delays**

* `CONCURRENT_REQUESTS`: Specifies the number of concurrent requests sent by Scrapy. The default is 16.
    
* `DOWNLOAD_DELAY`: Sets a delay between requests to the same domain to avoid triggering rate limits.
    

**Example Settings in** [`settings.py`](http://settings.py):

```python
CONCURRENT_REQUESTS = 8  # Reduce concurrency for sensitive websites
DOWNLOAD_DELAY = 2  # Add a 2-second delay between requests
```

---

#### **1.2 Enabling** `AUTOTHROTTLE` for Dynamic Rate Limiting

* **What is** `AUTOTHROTTLE`?
    
    * Dynamically adjusts request rates based on the server’s response times.
        
    * Helps reduce the chances of getting banned by websites.
        

**Example Settings in** [`settings.py`](http://settings.py):

```python
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1  # Initial download delay
AUTOTHROTTLE_MAX_DELAY = 10  # Maximum delay in case of high latency
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0  # Average concurrent requests
AUTOTHROTTLE_DEBUG = True  # Enable debug logging for auto-throttling
```

**How It Works:**

* Scrapy monitors server response times and adjusts the request rate accordingly.
    
* Ensures efficient scraping while reducing server load.
    

---

### **2\. Setting Custom Headers and User Agents**

#### **2.1 Why Set Custom Headers?**

* Default headers can trigger bans if the website identifies Scrapy as a bot.
    
* Mimic real user behavior with custom headers and user agents.
    

---

#### **2.2 Setting Custom Headers**

**Example Settings in** [`settings.py`](http://settings.py):

```python
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en',
}
```

**Add Custom Headers in the Spider:**

```python
class JobSpider(scrapy.Spider):
    name = "job_spider"
    start_urls = ["https://example.com/jobs"]

    def start_requests(self):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36',
        }
        for url in self.start_urls:
            yield scrapy.Request(url, headers=headers, callback=self.parse)
```

---

#### **2.3 Rotating User Agents**

Install the `scrapy-user-agents` library to easily rotate user agents:

```bash
pip install scrapy-user-agents
```

**Enable in** [`settings.py`](http://settings.py):

```python
DOWNLOADER_MIDDLEWARES = {
    'scrapy_user_agents.middlewares.RandomUserAgentMiddleware': 400,
}
```

---

### **3\. Rotating Proxies for Avoiding Bans**

#### **3.1 Why Rotate Proxies?**

* Prevent websites from blocking your IP address during large-scale scraping.
    
* Proxies can mask your IP and distribute requests across multiple addresses.
    

---

#### **3.2 Adding Proxies to Scrapy**

**Option 1: Manually Specify Proxies**

```python
class ProxySpider(scrapy.Spider):
    name = "proxy_spider"
    start_urls = ["https://example.com/jobs"]

    def start_requests(self):
        proxies = ["http://proxy1:8080", "http://proxy2:8080"]
        for url in self.start_urls:
            proxy = proxies[0]  # Rotate proxies manually
            yield scrapy.Request(url, meta={"proxy": proxy}, callback=self.parse)
```

---

**Option 2: Using a Proxy Middleware**

Install `scrapy-rotating-proxies`:

```bash
pip install scrapy-rotating-proxies
```

**Enable in** [`settings.py`](http://settings.py):

```python
ROTATING_PROXY_LIST = [
    'http://proxy1:8080',
    'http://proxy2:8080',
    'http://proxy3:8080',
]

DOWNLOADER_MIDDLEWARES = {
    'rotating_proxies.middlewares.RotatingProxyMiddleware': 610,
    'rotating_proxies.middlewares.BanDetectionMiddleware': 620,
}
```

---

### **4\. Full Settings Example**

[`settings.py`](http://settings.py):

```python
# Concurrency and Delay
CONCURRENT_REQUESTS = 8
DOWNLOAD_DELAY = 2

# Auto-throttle
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0
AUTOTHROTTLE_DEBUG = True

# Custom Headers
DEFAULT_REQUEST_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en',
}

# Rotating User Agents
DOWNLOADER_MIDDLEWARES = {
    'scrapy_user_agents.middlewares.RandomUserAgentMiddleware': 400,
}

# Rotating Proxies
ROTATING_PROXY_LIST = [
    'http://proxy1:8080',
    'http://proxy2:8080',
    'http://proxy3:8080',
]

DOWNLOADER_MIDDLEWARES.update({
    'rotating_proxies.middlewares.RotatingProxyMiddleware': 610,
    'rotating_proxies.middlewares.BanDetectionMiddleware': 620,
})
```

---

### **Key Takeaways**

1. **Concurrency and Delays**:
    
    * Control request rates with `CONCURRENT_REQUESTS` and `DOWNLOAD_DELAY`.
        
    * Use `AUTOTHROTTLE` for dynamic rate limiting.
        
2. **Custom Headers and User Agents**:
    
    * Mimic browser behavior to avoid detection.
        
    * Use `scrapy-user-agents` for rotating user agents automatically.
        
3. **Rotating Proxies**:
    
    * Distribute requests across multiple proxies to prevent IP bans.
        
    * Use `scrapy-rotating-proxies` for seamless proxy management.
        

# **12\. Error Handling and Debugging in Scrapy**

---

### **1\. Common Scrapy Errors and Solutions**

#### **1.1 HTTP Errors**

* **404 (Not Found)**:
    
    * The requested page doesn't exist.
        
    * Solution: Check if the URL is correct or dynamically constructed.
        
* **403 (Forbidden)**:
    
    * The server blocked your request, likely due to bot detection.
        
    * Solution:
        
        * Rotate user agents or proxies.
            
        * Add headers that mimic real browsers.
            
        * Slow down requests using `DOWNLOAD_DELAY`.
            
* **500 (Internal Server Error)**:
    
    * The server encountered an error.
        
    * Solution:
        
        * Retry the request.
            
        * Verify if the server is overloaded.
            

**Enable HTTP Error Logging in** [`settings.py`](http://settings.py):

```python
HTTPERROR_ALLOWED_CODES = [403, 500]  # Log these codes without stopping
```

---

#### **1.2 Selector Errors**

* **Empty Results from CSS or XPath Selectors**:
    
    * The selectors don't match the target elements.
        
    * Solution:
        
        * Use the Scrapy Shell to debug selectors.
            
        * Ensure elements are loaded before scraping (e.g., JavaScript-rendered pages).
            

**Example Debugging in Shell:**

```bash
scrapy shell "https://example.com/jobs"
response.css('div.job-card').getall()
response.xpath('//div[@class="job-card"]').getall()
```

---

#### **1.3 Missing or Unexpected Data**

* The structure of the website may have changed.
    
* Solution:
    
    * Regularly update selectors and test scraping logic.
        

---

### **2\. Logging and Debugging Scrapy Spiders**

#### **2.1 Configuring Logging**

Scrapy provides built-in logging to track spider activity.

**Example Logging Configuration in** [`settings.py`](http://settings.py):

```python
LOG_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = 'scrapy_log.txt'  # Save logs to a file
```

**Log Output Example:**

```plaintext
INFO: Scrapy 2.x.x started
INFO: Spider opened
INFO: Crawled (200) <GET https://example.com/jobs> (referer: None)
WARNING: Response status code 403 for https://example.com/protected
```

---

#### **2.2 Adding Custom Logs**

Add logs within your spider to monitor specific events.

**Code Example:**

```python
class JobSpider(scrapy.Spider):
    name = "job_spider"
    start_urls = ["https://example.com/jobs"]

    def parse(self, response):
        self.logger.info(f"Visited: {response.url}")
        if response.status == 403:
            self.logger.warning(f"403 Forbidden: {response.url}")
        for job in response.css('div.job-card'):
            yield {"title": job.css('h2.job-title::text').get()}
```

---

### **3\. Retrying Failed Requests**

#### **Enable Retry Middleware**

Scrapy retries failed requests by default. Configure it in [`settings.py`](http://settings.py):

```python
RETRY_ENABLED = True
RETRY_TIMES = 3  # Number of retries
RETRY_HTTP_CODES = [500, 502, 503, 504, 403, 408]  # Retry these status codes
```

---

#### **Using Proxies for Retries**

Combine retries with rotating proxies to increase success rates.

**Example in** [`settings.py`](http://settings.py):

```python
DOWNLOADER_MIDDLEWARES.update({
    'rotating_proxies.middlewares.RotatingProxyMiddleware': 610,
    'rotating_proxies.middlewares.BanDetectionMiddleware': 620,
})
```

---

### **4\. Handling Timeouts**

#### **Timeout Settings**

Set time limits for requests to avoid hanging indefinitely.

**Configuration in** [`settings.py`](http://settings.py):

```python
DOWNLOAD_TIMEOUT = 15  # Timeout in seconds
```

---

#### **Detect and Retry Timeouts**

Enable retrying for timeout errors:

```python
RETRY_HTTP_CODES.append(408)  # Add 408 (Request Timeout) to retry codes
```

---

### **5\. Handling CAPTCHA Challenges**

#### **5.1 Detecting CAPTCHA**

If your response contains a CAPTCHA page:

* Check for elements like `<div class="captcha">` or specific keywords.
    

**Code Example: Detect CAPTCHA**

```python
def parse(self, response):
    if "captcha" in response.text.lower():
        self.logger.warning(f"CAPTCHA encountered at {response.url}")
        return
    # Proceed with scraping
```

---

#### **5.2 Bypassing CAPTCHA**

1. **Use Third-Party CAPTCHA Solvers**:
    
    * Services like [2Captcha](https://2captcha.com) or [Anti-Captcha](https://anti-captcha.com) can solve CAPTCHA challenges.
        

**Install the** `anticaptchaofficial` Python library:

```bash
pip install anticaptchaofficial
```

**Code Example: Solve CAPTCHA Using 2Captcha**

```python
from anticaptchaofficial.recaptchav2proxyless import recaptchaV2Proxyless

def solve_captcha(site_key, page_url):
    solver = recaptchaV2Proxyless()
    solver.set_verbose(1)
    solver.set_key("YOUR_2CAPTCHA_API_KEY")
    solver.set_website_url(page_url)
    solver.set_website_key(site_key)
    return solver.solve_and_return_solution()

def parse(self, response):
    site_key = response.css('div.g-recaptcha::attr(data-sitekey)').get()
    captcha_solution = solve_captcha(site_key, response.url)
    self.logger.info(f"CAPTCHA Solved: {captcha_solution}")
```

2. **Use Playwright for JavaScript CAPTCHA**:
    
    * Use `scrapy-playwright` to render CAPTCHA pages and solve simple challenges.
        

---

### **6\. Full Error Handling Example**

**Spider Code:**

```python
import scrapy
from scrapy.http import Request

class JobSpider(scrapy.Spider):
    name = "job_spider"
    start_urls = ["https://example.com/jobs"]

    def parse(self, response):
        if response.status == 403:
            self.logger.warning(f"403 Forbidden: {response.url}")
            return

        for job in response.css('div.job-card'):
            yield {
                "Title": job.css('h2.job-title::text').get(),
                "Company": job.css('span.company-name::text').get(),
            }

        # Follow pagination
        next_page = response.css('a.next::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)
```

**Settings Example (**[`settings.py`](http://settings.py)):

```python
# Retry Settings
RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 403, 408]

# Timeout
DOWNLOAD_TIMEOUT = 15

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'scrapy_log.txt'

# Throttling
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 10
```

---

### **Key Takeaways**

1. **HTTP Errors**:
    
    * Handle `403` with headers, proxies, or user agents.
        
    * Retry `500` and timeout errors using retry middleware.
        
2. **Logging**:
    
    * Use custom logs to monitor spider behavior and debug issues.
        
3. **CAPTCHA Challenges**:
    
    * Detect CAPTCHAs and use external solvers or Playwright.
        
4. **Timeouts**:
    
    * Set a `DOWNLOAD_TIMEOUT` and retry timeouts when necessary.
        

# **13\. Scrapy Item Pipelines**

---

### **1\. Enabling Pipelines in Scrapy Settings**

To use item pipelines, you must enable them in the [`settings.py`](http://settings.py) file by defining their priority. Pipelines with lower numbers are executed first.

**Example in** [`settings.py`](http://settings.py):

```python
ITEM_PIPELINES = {
    'job_scraper.pipelines.ValidationPipeline': 100,
    'job_scraper.pipelines.CleaningPipeline': 200,
    'job_scraper.pipelines.JsonPipeline': 300,  # For JSON output
    'job_scraper.pipelines.PostgreSQLPipeline': 400,  # For PostgreSQL storage
}
```

---

### **2\. Processing Scraped Data in Pipelines**

#### **Pipeline Template**

All pipelines must implement the `process_item` method:

```python
class YourPipeline:
    def process_item(self, item, spider):
        # Process and transform the item
        return item
```

---

#### **2.1 Saving Data Locally**

**a. Saving to JSON** Save each scraped item as a JSON line in a file.

**Code Example: JSON Pipeline**

```python
import json

class JsonPipeline:
    def open_spider(self, spider):
        self.file = open('jobs.json', 'w')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        line = json.dumps(dict(item)) + "\n"
        self.file.write(line)
        return item
```

**b. Saving to CSV** Save data to a CSV file.

**Code Example: CSV Pipeline**

```python
import csv

class CsvPipeline:
    def open_spider(self, spider):
        self.file = open('jobs.csv', 'w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=["title", "company", "location", "link"])
        self.writer.writeheader()

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        self.writer.writerow(dict(item))
        return item
```

---

#### **2.2 Storing Data in PostgreSQL**

**Step 1: Install PostgreSQL Python Driver**

```bash
pip install psycopg2-binary
```

**Step 2: Define the PostgreSQL Pipeline** This pipeline connects to a PostgreSQL database and inserts scraped data into a table.

**Code Example: PostgreSQL Pipeline**

```python
import psycopg2

class PostgreSQLPipeline:
    def open_spider(self, spider):
        self.conn = psycopg2.connect(
            dbname="job_database",
            user="your_username",
            password="your_password",
            host="localhost",
            port="5432"
        )
        self.cursor = self.conn.cursor()

        # Create table if it doesn't exist
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                title TEXT,
                company TEXT,
                location TEXT,
                link TEXT
            )
        """)

    def close_spider(self, spider):
        self.conn.commit()
        self.conn.close()

    def process_item(self, item, spider):
        self.cursor.execute("""
            INSERT INTO jobs (title, company, location, link)
            VALUES (%s, %s, %s, %s)
        """, (item['title'], item['company'], item['location'], item['link']))
        return item
```

---

### **3\. Writing Custom Pipelines for Validation and Cleaning**

#### **3.1 Validation Pipeline**

Ensure that mandatory fields (e.g., `title`, `company`) are not empty. Drop items if required fields are missing.

**Code Example: Validation Pipeline**

```python
class ValidationPipeline:
    def process_item(self, item, spider):
        if not item.get('title'):
            raise scrapy.exceptions.DropItem(f"Missing title in {item}")
        if not item.get('company'):
            item['company'] = "Unknown"  # Assign default value
        return item
```

---

#### **3.2 Cleaning Pipeline**

Standardize and clean the scraped data, such as removing extra spaces or formatting fields.

**Code Example: Cleaning Pipeline**

```python
class CleaningPipeline:
    def process_item(self, item, spider):
        # Strip whitespace
        item['title'] = item['title'].strip()
        item['company'] = item['company'].strip()
        item['location'] = item['location'].strip()

        # Normalize location format
        if "remote" in item['location'].lower():
            item['location'] = "Remote"
        return item
```

---

### **4\. Full Example**

Here’s an example setup with pipelines for validation, cleaning, and saving to JSON and PostgreSQL.

#### **Pipeline Code (**[`pipelines.py`](http://pipelines.py))

```python
import scrapy
import json
import psycopg2

class ValidationPipeline:
    def process_item(self, item, spider):
        if not item.get('title'):
            raise scrapy.exceptions.DropItem(f"Missing title in {item}")
        if not item.get('company'):
            item['company'] = "Unknown"
        return item

class CleaningPipeline:
    def process_item(self, item, spider):
        item['title'] = item['title'].strip()
        item['company'] = item['company'].strip()
        item['location'] = item['location'].strip()
        if "remote" in item['location'].lower():
            item['location'] = "Remote"
        return item

class JsonPipeline:
    def open_spider(self, spider):
        self.file = open('jobs.json', 'w')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        line = json.dumps(dict(item)) + "\n"
        self.file.write(line)
        return item

class PostgreSQLPipeline:
    def open_spider(self, spider):
        self.conn = psycopg2.connect(
            dbname="job_database",
            user="your_username",
            password="your_password",
            host="localhost",
            port="5432"
        )
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                title TEXT,
                company TEXT,
                location TEXT,
                link TEXT
            )
        """)

    def close_spider(self, spider):
        self.conn.commit()
        self.conn.close()

    def process_item(self, item, spider):
        self.cursor.execute("""
            INSERT INTO jobs (title, company, location, link)
            VALUES (%s, %s, %s, %s)
        """, (item['title'], item['company'], item['location'], item['link']))
        return item
```

#### **Settings (**[`settings.py`](http://settings.py))

```python
ITEM_PIPELINES = {
    'job_scraper.pipelines.ValidationPipeline': 100,
    'job_scraper.pipelines.CleaningPipeline': 200,
    'job_scraper.pipelines.JsonPipeline': 300,
    'job_scraper.pipelines.PostgreSQLPipeline': 400,
}
```

---

### **5\. Output Examples**

#### `jobs.json`:

```json
[
    {"title": "Data Scientist", "company": "TechCorp", "location": "San Francisco, CA", "link": "https://example.com/job/123"},
    {"title": "AI Engineer", "company": "AI Corp", "location": "Remote", "link": "https://example.com/job/456"}
]
```

#### **PostgreSQL Table:**

```plaintext
+--------------------+------------+--------------------+-----------------------------+
| title              | company    | location           | link                        |
+--------------------+------------+--------------------+-----------------------------+
| Data Scientist     | TechCorp   | San Francisco, CA  | https://example.com/job/123 |
| AI Engineer        | AI Corp    | Remote             | https://example.com/job/456 |
+--------------------+------------+--------------------+-----------------------------+
```

---

### **Key Takeaways**

1. **Validation and Cleaning**:
    
    * Use custom pipelines to ensure data quality.
        
2. **Local Storage**:
    
    * Save data as JSON or CSV using dedicated pipelines.
        
3. **Database Storage**:
    
    * Store data in PostgreSQL for advanced querying and analysis.
        
4. **Pipeline Priority**:
    
    * Order pipelines in [`settings.py`](http://settings.py) for sequential processing.
        

# **14\. Deploying Scrapy Projects**

---

### **1\. Deploying Spiders to Scrapy Cloud**

Scrapy Cloud is a cloud-based platform by Scrapinghub for deploying, scheduling, and monitoring Scrapy spiders.

---

#### **1.1 Install the** `shub` CLI

```bash
pip install shub
```

#### **1.2 Authenticate with Scrapy Cloud**

Run the following command and enter your Scrapy Cloud API key:

```bash
shub login
```

#### **1.3 Deploy Your Spider**

In your Scrapy project directory, deploy the spider to Scrapy Cloud:

```bash
shub deploy
```

**Output Example:**

```plaintext
Packing version 1.0
Deploying to Scrapy Cloud project '12345'
Spider deployed: https://app.scrapinghub.com/p/12345
```

#### **1.4 Schedule a Spider**

After deployment, schedule the spider from the Scrapy Cloud dashboard or using the `shub` CLI:

```bash
shub schedule my_spider
```

---

### **2\. Using** `scrapyd` to Manage and Schedule Spiders

`scrapyd` is a service for running Scrapy spiders remotely, managing deployments, and scheduling jobs via an HTTP API.

---

#### **2.1 Install** `scrapyd`

```bash
pip install scrapyd
```

#### **2.2 Start the** `scrapyd` Service

Run `scrapyd` to start the server (default port: 6800):

```bash
scrapyd
```

---

#### **2.3 Deploy Your Spider to** `scrapyd`

**Step 1: Install** `scrapyd-client`

```bash
pip install scrapyd-client
```

**Step 2: Configure Deployment** Create a `scrapy.cfg` file in your project directory:

```plaintext
[settings]
default = my_project.settings

[deploy]
url = http://localhost:6800/
project = my_project
```

**Step 3: Deploy the Spider**

```bash
scrapyd-deploy
```

---

#### **2.4 Schedule Spiders with** `scrapyd`

Use `curl` or a Python script to schedule a spider on `scrapyd`.

**Command Example:**

```bash
curl http://localhost:6800/schedule.json -d project=my_project -d spider=my_spider
```

**Python Example:**

```python
import requests

response = requests.post(
    "http://localhost:6800/schedule.json",
    data={"project": "my_project", "spider": "my_spider"}
)
print(response.json())
```

---

### **3\. Running Spiders on Cloud Platforms**

You can deploy Scrapy spiders to cloud platforms like AWS or GCP for better scalability.

---

#### **3.1 Running Spiders on AWS**

**Step 1: Install AWS CLI**

```bash
pip install awscli
```

**Step 2: Set Up an EC2 Instance**

* Launch an EC2 instance with an appropriate image (e.g., Amazon Linux or Ubuntu).
    
* SSH into the instance and install Python, Scrapy, and necessary dependencies:
    
    ```bash
    sudo apt update
    sudo apt install python3 python3-pip
    pip install scrapy
    ```
    

**Step 3: Run Your Spider** Upload your Scrapy project to the instance and run the spider:

```bash
scrapy crawl my_spider
```

---

#### **3.2 Running Spiders on GCP**

**Step 1: Install Google Cloud CLI**

```bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-<version>-linux-x86_64.tar.gz
```

**Step 2: Launch a Virtual Machine**

* Create a VM instance in the Google Cloud Console.
    
* SSH into the instance and set up Python, Scrapy, and dependencies.
    

**Step 3: Run Your Spider** Transfer your Scrapy project to the VM and execute the spider:

```bash
scrapy crawl my_spider
```

---

### **4\. Scheduling Scrapy Jobs**

#### **4.1 Using** `cron`

**Step 1: Open** `crontab`

```bash
crontab -e
```

**Step 2: Add a Job** Schedule your spider to run at a specific time (e.g., every day at midnight):

```plaintext
0 0 * * * cd /path/to/scrapy/project && scrapy crawl my_spider
```

---

#### **4.2 Using** `APScheduler`

**Install** `APScheduler`:

```bash
pip install apscheduler
```

**Code Example: Schedule a Spider**

```python
from apscheduler.schedulers.blocking import BlockingScheduler
from subprocess import call

def run_spider():
    call(["scrapy", "crawl", "my_spider"])

scheduler = BlockingScheduler()
scheduler.add_job(run_spider, "interval", hours=24)  # Run every 24 hours

try:
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    pass
```

Run the script to keep the scheduler active.

---

### **Key Takeaways**

1. **Scrapy Cloud**:
    
    * Use `shub` to deploy and schedule spiders on Scrapy Cloud.
        
2. `scrapyd`:
    
    * Manage and schedule spiders remotely using `scrapyd`.
        
3. **Cloud Platforms**:
    
    * Run spiders on AWS or GCP for scalability.
        
4. **Job Scheduling**:
    
    * Use `cron` for simple scheduling or `APScheduler` for more flexibility.
        

# **15\. Tips and Tricks for Efficient Scraping**

---

### **1\. Using Spider Arguments for Flexible Spiders**

#### **Why Use Spider Arguments?**

* Spider arguments allow you to pass dynamic parameters to spiders at runtime, making them more flexible.
    

#### **Example: Define Spider with Arguments**

Modify your spider to accept arguments:

```python
import scrapy

class JobSpider(scrapy.Spider):
    name = "job_spider"

    def __init__(self, location=None, job_type=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.location = location
        self.job_type = job_type

    def start_requests(self):
        url = f"https://example.com/jobs?location={self.location}&type={self.job_type}"
        yield scrapy.Request(url, self.parse)

    def parse(self, response):
        for job in response.css('div.job-card'):
            yield {
                "title": job.css('h2.job-title::text').get(),
                "company": job.css('span.company-name::text').get(),
                "location": self.location,
                "type": self.job_type,
            }
```

#### **Run the Spider with Arguments**

```bash
scrapy crawl job_spider -a location="New York" -a job_type="Data Scientist"
```

**Output:**

```json
{
    "title": "Data Scientist",
    "company": "TechCorp",
    "location": "New York",
    "type": "Data Scientist"
}
```

---

### **2\. Custom Callback Functions for Following Links**

#### **Why Use Custom Callbacks?**

* Custom callbacks allow you to handle links and responses differently based on their context.
    

#### **Example: Following Links with Custom Callback**

```python
class JobSpider(scrapy.Spider):
    name = "job_spider"
    start_urls = ["https://example.com/jobs"]

    def parse(self, response):
        for job in response.css('div.job-card'):
            job_link = job.css('a::attr(href)').get()
            yield response.follow(job_link, self.parse_job)

    def parse_job(self, response):
        yield {
            "title": response.css('h1.job-title::text').get(),
            "company": response.css('span.company-name::text').get(),
            "description": response.css('div.job-description').get(),
        }
```

---

### **3\. Debugging CSS and XPath Selectors with the Scrapy Shell**

#### **Why Use Scrapy Shell?**

* Scrapy Shell helps you test CSS and XPath selectors interactively, making it easier to debug and refine your extraction logic.
    

#### **Example Workflow:**

1. Start the Scrapy Shell:
    
    ```bash
    scrapy shell "https://example.com/jobs"
    ```
    
2. Test CSS Selectors:
    
    ```python
    response.css('div.job-card h2.job-title::text').getall()
    ```
    
    **Output:**
    
    ```python
    ['Data Scientist', 'AI Engineer', 'ML Developer']
    ```
    
3. Test XPath Selectors:
    
    ```python
    response.xpath('//div[@class="job-card"]/h2[@class="job-title"]/text()').getall()
    ```
    
    **Output:**
    
    ```python
    ['Data Scientist', 'AI Engineer', 'ML Developer']
    ```
    
4. Inspect the Full HTML of a Node:
    
    ```python
    print(response.css('div.job-card').get())
    ```
    

---

### **4\. Writing Reusable Standalone Spiders**

#### **Why Create Standalone Spiders?**

* Standalone spiders can be reused across multiple projects with minimal changes.
    

#### **Example: Reusable Spider**

Create a base spider class with configurable attributes:

```python
class ReusableSpider(scrapy.Spider):
    name = "reusable_spider"

    def __init__(self, base_url=None, selectors=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = base_url
        self.selectors = selectors

    def start_requests(self):
        yield scrapy.Request(self.base_url, self.parse)

    def parse(self, response):
        for item in response.css(self.selectors['item']):
            yield {
                "title": item.css(self.selectors['title']).get(),
                "link": response.urljoin(item.css(self.selectors['link']).get()),
            }
```

#### **Run the Spider with Arguments:**

```bash
scrapy crawl reusable_spider -a base_url="https://example.com/jobs" -a selectors='{"item":"div.job-card","title":"h2.job-title::text","link":"a::attr(href)"}'
```

---

### **5\. Best Practices for Reusable Spiders**

* **Use Variables for Selectors**:
    
    * Define selectors in [`settings.py`](http://settings.py) or pass them as arguments.
        
* **Handle Common Scenarios**:
    
    * Add logic for pagination, error handling, and retries.
        
* **Separate Logic**:
    
    * Use item pipelines to clean data instead of adding all processing to spiders.
        

---

### **Key Takeaways**

1. **Spider Arguments**:
    
    * Add flexibility to spiders by accepting arguments at runtime.
        
2. **Custom Callbacks**:
    
    * Use callbacks to handle different pages and links contextually.
        
3. **Scrapy Shell**:
    
    * Debug and refine your selectors interactively.
        
4. **Reusable Spiders**:
    
    * Write spiders that can adapt to different use cases with minimal changes.
        

# **16\. Building a Job Scraper for** [**AIJobs.net**](http://AIJobs.net)**, Indeed, and LinkedIn**

---

### **1\. Analyzing the Structure of Job Websites**

#### **Key Data Points**

For job websites, we typically extract:

* **Job Titles**: The role being advertised.
    
* **Companies**: The employer or organization.
    
* **Locations**: Where the job is based or if it’s remote.
    
* **Links**: Direct URLs to job details.
    

#### **Identifying Pagination Patterns**

1. **Look for "Next" Buttons or Pagination Links**:
    
    * HTML example:
        
        ```html
        <a class="next" href="/jobs?page=2">Next</a>
        ```
        
2. **Dynamic APIs**:
    
    * Inspect network requests in browser DevTools for APIs providing job data.
        

---

### **2\. Writing Spiders for** [**AIJobs.net**](http://AIJobs.net)

#### **Step 1: Analyze** [**AIJobs.net**](http://AIJobs.net)

* Inspect the structure of job cards on the website:
    
    ```html
    <div class="job-card">
        <h2 class="job-title">Data Scientist</h2>
        <span class="company-name">TechCorp</span>
        <a href="/job/123" class="job-link">View Job</a>
    </div>
    ```
    

#### **Step 2: Create the Spider**

**Spider Code for** [**AIJobs.net**](http://AIJobs.net)**:**

```python
import scrapy

class AIJobsSpider(scrapy.Spider):
    name = "aijobs"
    start_urls = ["https://aijobs.net/jobs"]

    def parse(self, response):
        for job in response.css('div.job-card'):
            yield {
                "title": job.css('h2.job-title::text').get(),
                "company": job.css('span.company-name::text').get(),
                "link": response.urljoin(job.css('a.job-link::attr(href)').get()),
            }

        # Handle pagination
        next_page = response.css('a.next::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)
```

#### **Run the Spider**

```bash
scrapy crawl aijobs -o aijobs.json
```

---

### **3\. Writing Spiders for Indeed**

#### **Step 1: Analyze Indeed**

* Identify job cards and key data points:
    
    ```html
    <div class="job_seen_beacon">
        <h2 class="jobTitle">Data Scientist</h2>
        <span class="companyName">TechCorp</span>
        <div class="companyLocation">San Francisco, CA</div>
        <a href="/rc/clk?jk=abc123" class="jobTitle-link">View Job</a>
    </div>
    ```
    

#### **Step 2: Create the Spider**

**Spider Code for Indeed:**

```python
import scrapy

class IndeedSpider(scrapy.Spider):
    name = "indeed"
    start_urls = ["https://www.indeed.com/jobs?q=data+scientist&l="]

    def parse(self, response):
        for job in response.css('div.job_seen_beacon'):
            yield {
                "title": job.css('h2.jobTitle::text').get(),
                "company": job.css('span.companyName::text').get(),
                "location": job.css('div.companyLocation::text').get(),
                "link": response.urljoin(job.css('a::attr(href)').get()),
            }

        # Handle pagination
        next_page = response.css('a[aria-label="Next"]::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)
```

#### **Run the Spider**

```bash
scrapy crawl indeed -o indeed.json
```

---

### **4\. Writing Spiders for LinkedIn**

#### **Step 1: Analyze LinkedIn**

LinkedIn heavily relies on JavaScript for rendering job data. Scrapy alone cannot scrape it effectively. Use one of these approaches:

* **API**: Requires developer access and an access token.
    
* **Selenium**: Automates browser interaction for dynamic content.
    

---

#### **Step 2: Using Selenium for LinkedIn**

**Install Selenium and WebDriver**

```bash
pip install selenium
```

**Code Example: LinkedIn Spider with Selenium**

```python
from scrapy import Spider
from scrapy.http import HtmlResponse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

class LinkedInSpider(Spider):
    name = "linkedin"

    def __init__(self):
        self.driver = webdriver.Chrome()

    def start_requests(self):
        url = "https://www.linkedin.com/jobs/search/?keywords=data%20scientist"
        self.driver.get(url)
        time.sleep(5)  # Wait for the page to load
        html = self.driver.page_source
        response = HtmlResponse(url=url, body=html, encoding='utf-8')
        self.parse(response)

    def parse(self, response):
        for job in response.css('li.result-card'):
            yield {
                "title": job.css('h3.result-card__title::text').get(),
                "company": job.css('h4.result-card__subtitle::text').get(),
                "location": job.css('span.job-result-card__location::text').get(),
                "link": job.css('a.result-card__full-card-link::attr(href)').get(),
            }

    def closed(self, reason):
        self.driver.quit()
```

---

#### **Step 3: Run the Spider**

Ensure you have the appropriate WebDriver installed (e.g., ChromeDriver for Chrome).

```bash
scrapy crawl linkedin -o linkedin.json
```

---

### **5\. Key Takeaways**

* [**AIJobs.net**](http://AIJobs.net):
    
    * Pure HTML scraping with Scrapy is sufficient.
        
    * Focus on job cards and pagination.
        
* **Indeed**:
    
    * Scrape job titles, companies, and locations with pagination handling.
        
    * Handle URLs carefully as they might be relative.
        
* **LinkedIn**:
    
    * Use Selenium for JavaScript-rendered content or APIs for structured data.
        

### **Focus Areas for Your Use Case**

Here’s how to address each focus area step by step, tailored to job scraping.

---

### **1\. CSS vs. XPath Selectors**

#### **Goal**:

* Learn how to extract job-related data from real websites using both CSS and XPath.
    

#### **CSS Selectors**:

CSS is intuitive and efficient for most scraping tasks.

**Example: Extracting Job Titles**

```python
response.css('h2.job-title::text').getall()
```

**Output:**

```python
['Data Scientist', 'AI Engineer', 'ML Developer']
```

#### **XPath Selectors**:

XPath provides more power for complex HTML structures.

**Example: Extracting Job Titles**

```python
response.xpath('//h2[@class="job-title"]/text()').getall()
```

**Output:**

```python
['Data Scientist', 'AI Engineer', 'ML Developer']
```

---

### **2\. Pagination**

#### **Goal**:

* Enable spiders to navigate multi-page job listings.
    

#### **Recursive Crawling Example**:

```python
class PaginationSpider(scrapy.Spider):
    name = "pagination"
    start_urls = ["https://example.com/jobs"]

    def parse(self, response):
        # Extract job data
        for job in response.css('div.job-card'):
            yield {
                "title": job.css('h2.job-title::text').get(),
                "company": job.css('span.company-name::text').get(),
                "link": response.urljoin(job.css('a::attr(href)').get()),
            }

        # Follow pagination links
        next_page = response.css('a.next::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)
```

#### **Test Pagination Logic**:

Use Scrapy Shell to debug pagination links.

```bash
scrapy shell "https://example.com/jobs"
response.css('a.next::attr(href)').get()
```

---

### **3\. APIs and Dynamic Content**

#### **Goal**:

* Extract data from APIs or render JavaScript-heavy pages.
    

#### **Using APIs**:

APIs often return structured JSON data.

**Example: Requesting Job Data from an API**

```python
import scrapy
import json

class APISpider(scrapy.Spider):
    name = "api_spider"
    start_urls = ["https://example.com/api/jobs"]

    def parse(self, response):
        data = json.loads(response.text)
        for job in data['jobs']:
            yield {
                "title": job['title'],
                "company": job['company'],
                "location": job['location'],
                "link": job['url'],
            }
```

#### **Handling JavaScript-Rendered Pages**:

Use `scrapy-playwright` to scrape dynamic content.

**Example: Scraping with Playwright**

```python
import scrapy

class PlaywrightSpider(scrapy.Spider):
    name = "playwright_spider"
    start_urls = ["https://example.com/jobs"]

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, meta={"playwright": True})

    def parse(self, response):
        for job in response.css('div.job-card'):
            yield {
                "title": job.css('h2.job-title::text').get(),
                "company": job.css('span.company-name::text').get(),
            }
```

---

### **4\. Data Storage**

#### **Goal**:

* Save job data in structured formats (JSON, CSV).
    

#### **Saving to JSON**

```python
import json

class JsonPipeline:
    def open_spider(self, spider):
        self.file = open('jobs.json', 'w')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        line = json.dumps(dict(item)) + "\n"
        self.file.write(line)
        return item
```

#### **Saving to CSV**

```python
import csv

class CsvPipeline:
    def open_spider(self, spider):
        self.file = open('jobs.csv', 'w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=["title", "company", "location", "link"])
        self.writer.writeheader()

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        self.writer.writerow(dict(item))
        return item
```

---

### **5\. Deployment**

#### **Goal**:

* Automate spider execution using `scrapyd` or cloud platforms.
    

#### **Using** `scrapyd`:

Deploy and schedule spiders on a local or remote `scrapyd` instance.

**Deploy Spider to** `scrapyd`:

```bash
scrapyd-deploy
```

**Schedule Spider**:

```bash
curl http://localhost:6800/schedule.json -d project=my_project -d spider=my_spider
```

#### **Using Cron Jobs**:

Automate spider execution with `cron`.

**Example Cron Job**:

```bash
0 0 * * * cd /path/to/scrapy/project && scrapy crawl my_spider
```

#### **Using Cloud Platforms**:

Run spiders on AWS or GCP for scalability.

**Example on AWS EC2**:

* SSH into the instance, upload your project, and execute:
    
    ```bash
    scrapy crawl my_spider
    ```
    

---

### **Key Steps for Your Use Case**

1. **Practice CSS and XPath selectors**:
    
    * Extract job data interactively with Scrapy Shell.
        
2. **Master Pagination**:
    
    * Scrape multiple pages dynamically.
        
3. **Use APIs or** `scrapy-playwright`:
    
    * Handle modern job boards with JavaScript-rendered content.
        
4. **Save Data**:
    
    * Use pipelines to export job listings in JSON or CSV.
        
5. **Automate Deployment**:
    
    * Schedule jobs with `scrapyd` or cloud platforms.