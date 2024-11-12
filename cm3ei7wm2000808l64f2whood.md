---
title: "Playwright Python and Pytest for Web Automation Testing"
seoTitle: "Playwright Python and Pytest for Web Automation Testing"
seoDescription: "Playwright Python and Pytest for Web Automation Testing"
datePublished: Tue Nov 12 2024 13:44:45 GMT+0000 (Coordinated Universal Time)
cuid: cm3ei7wm2000808l64f2whood
slug: playwright-python-and-pytest-for-web-automation-testing
tags: playwright, web-automation

---

# Getting Start

### 1\. **Playwright Installation**

Run this in your terminal:

```bash
pip install playwright
playwright install
```

---

### 2\. **Launching the Browser**

Create a Python script and add:

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # headless=False allows you to see the browser open
    page = browser.new_page()
    page.goto("https://www.indeed.com")  # start with your target site
```

This will open a Chromium browser and navigate to Indeed. You can swap the URL to any job site.

---

### 3\. **Clicking a Link Element**

To automate clicks (like “Apply Now” buttons), add this code right after `page.goto(...)`:

```python
page.click("text=Apply Now")  # Example selector; you'll need the right one for each site
```

* **Explanation**: [`page.click`](http://page.click)`(...)` clicks the button or link. You'll adjust the `"text=..."` part based on what the job site uses (e.g., text on the button).
    

---

### 4\. **Filling Out a Form**

To fill out input fields (like your name and email), use:

```python
page.fill("input[name='name']", "Your Name")
page.fill("input[name='email']", "your.email@example.com")
```

* **Explanation**: `page.fill(...)` fills input boxes. You'll need the correct `input[name='...']` values, which you can find by inspecting the page’s HTML.
    

---

### 5\. **Script Overview**

Combine the above steps in a Python script to navigate to a job site, click “Apply Now,” and fill in details. Example script:

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://www.indeed.com")

    # Click apply button
    page.click("text=Apply Now")  # Adjust selector as needed

    # Fill form fields
    page.fill("input[name='name']", "Your Name")
    page.fill("input[name='email']", "your.email@example.com")

    # Continue with more form fields as needed...
```

---

This script opens Indeed, clicks the apply button, and fills out the form with your info. Adjust selectors (like `"text=Apply Now"` and `input[name='...']`) based on each job site’s HTML.

# Locator

---

### **1\. Playwright Python REPL**

* **What It Is**: A quick, interactive command line where you can test Playwright code.
    
* **Why It’s Useful**: You can try small snippets (like finding buttons or fields) and see immediate results, which is great for testing and troubleshooting without running the full script each time.
    
* **How to Use**:
    
    ```bash
    playwright codegen https://www.indeed.com
    ```
    
    * This command opens a browser and lets you interactively click or type while it records the code.
        

**Expected Result**:

* You’ll see a new browser window where you can interact with the page, and the code for each action will appear in the terminal. This can help in identifying the right selectors for buttons or fields.
    

---

### **2\. Locator Role**

* **What It Is**: Playwright can locate elements by their role (e.g., button, textbox).
    
* **Why It’s Useful**: Instead of specific selectors, you can identify elements by their function, helpful for locating “Apply” buttons or input fields.
    
* **Example**:
    
    ```python
    page.locator("role=button[name='Apply Now']").click()
    ```
    

**Expected Result**:

* This will try to click any button with the “Apply Now” role. If successful, you’ll see the application form or next step load.
    

---

### **3\. Locators for Input Field**

* **What It Is**: Specifically targets input fields (like name, email).
    
* **Why It’s Useful**: It helps autofill forms accurately.
    
* **Example**:
    
    ```python
    page.locator("input[name='email']").fill("your.email@example.com")
    ```
    

**Expected Result**:

* The “email” input field should fill with your email. You’ll see it appear in the text box on the page.
    

---

### **4\. Locator Text**

* **What It Is**: Finds elements based on visible text.
    
* **Why It’s Useful**: Useful for targeting links or buttons with specific words.
    
* **Example**:
    
    ```python
    page.locator("text=Submit Application").click()
    ```
    

**Expected Result**:

* If the text matches, the “Submit Application” button will be clicked, and you’ll advance to the next page or confirmation.
    

---

### **5\. Locator Alt Text**

* **What It Is**: Targets images by their `alt` text.
    
* **Why It’s Useful**: For images (like icons on some apply buttons), targeting `alt` text can ensure you click the right image or icon button.
    
* **Example**:
    
    ```python
    page.locator("img[alt='Apply Icon']").click()
    ```
    

**Expected Result**:

* If there’s an image button with that `alt` text, it will be clicked, potentially opening the application form.
    

---

### **6\. Locator Title**

* **What It Is**: Finds elements with a specific `title` attribute.
    
* **Why It’s Useful**: Some sites use `title` attributes for tooltips or hidden buttons.
    
* **Example**:
    
    ```python
    page.locator("[title='Apply']").click()
    ```
    

**Expected Result**:

* The element with that title gets clicked, opening the next step or form.
    

---

### **7\. Locating with CSS Selectors**

* **What It Is**: Targets elements based on CSS selectors.
    
* **Why It’s Useful**: Helps pinpoint exact elements, like specific buttons or fields on complicated pages.
    
* **Example**:
    
    ```python
    page.locator("button.apply-button").click()
    ```
    

**Expected Result**:

* This targets a button with the `apply-button` class. If it exists, it clicks it and loads the application form.
    

---

### **8\. CSS Selectors Hierarchy**

* **What It Is**: Use CSS selector hierarchy to locate nested elements (e.g., form &gt; div &gt; input).
    
* **Why It’s Useful**: Helps in accessing elements deep inside forms.
    
* **Example**:
    
    ```python
    page.locator("form .form-group input[name='phone']").fill("123-456-7890")
    ```
    

**Expected Result**:

* Fills the phone number field if it’s structured within a form hierarchy. You’ll see the phone number appear in the input.
    

---

### **9\. CSS Selectors Pseudo Classes**

* **What It Is**: Targets elements with specific pseudo-classes, like `:first-child` or `:last-of-type`.
    
* **Why It’s Useful**: Helpful for filling out the first field in a list, for example.
    
* **Example**:
    
    ```python
    page.locator("input:first-of-type").fill("First Name")
    ```
    

**Expected Result**:

* Fills the first input field on the page, displaying your input.
    

---

### **10\. Locators XPath**

* **What It Is**: Uses XPath to target elements (an alternative to CSS).
    
* **Why It’s Useful**: XPath is flexible for complex structures.
    
* **Example**:
    
    ```python
    page.locator("//button[text()='Submit']").click()
    ```
    

**Expected Result**:

* Finds a button by its text and clicks it. If successful, the form should submit, or the page advances.
    

---

### **11\. XPath Functions**

* **What It Is**: Uses functions in XPath (e.g., `contains()`, `starts-with()`).
    
* **Why It’s Useful**: Locate elements with partial text matches.
    
* **Example**:
    
    ```python
    page.locator("//button[contains(text(),'Apply')]").click()
    ```
    

**Expected Result**:

* Clicks any button containing “Apply” in its text, potentially leading to the application form.
    

---

### **12\. Other Locators**

* **Overview**: Playwright supports multiple locator strategies beyond CSS and XPath, including text, role, or attribute-based locators.
    
* **Why It’s Useful**: Gives you flexibility if one method doesn’t work on a site.
    

---

# Actions

### **1\. Mouse Actions**

* **What It Is**: Allows you to simulate mouse actions like hover or drag.
    
* **Why It’s Useful**: Handy if you need to hover over menus or drag elements for the application process.
    
* **Example**:
    
    ```python
    page.hover("text=More Options")  # Hover over a menu item
    page.mouse.click(200, 150)       # Click at specific coordinates if needed
    ```
    

**Expected Result**:

* The browser will hover or click at the specified location. For hovering, you might see a dropdown or tooltip appear.
    

---

### **2\. Actions for Text Input**

* **What It Is**: Simulates typing text into fields.
    
* **Why It’s Useful**: Essential for filling out form fields like name, email, and address.
    
* **Example**:
    
    ```python
    page.fill("input[name='firstName']", "Your First Name")
    page.type("input[name='lastName']", "Your Last Name")  # Simulates typing more naturally
    ```
    

**Expected Result**:

* Each field fills in with the provided text. You’ll see the text appear in the input boxes on the page.
    

---

### **3\. Radios, Checkboxes, and Switches**

* **What It Is**: Allows you to select options like radio buttons, checkboxes, or toggle switches.
    
* **Why It’s Useful**: Some applications require agreeing to terms or selecting specific options (e.g., full-time vs. part-time).
    
* **Example**:
    
    ```python
    page.check("input[type='checkbox'][name='agreeTerms']")
    page.check("input[type='radio'][value='full-time']")
    ```
    

**Expected Result**:

* The specified checkbox or radio button gets selected. You should see the checkmark or dot appear.
    

---

### **4\. Select Option**

* **What It Is**: Selects a choice from dropdowns with fixed options (e.g., country or job type).
    
* **Why It’s Useful**: Dropdowns are common in job forms.
    
* **Example**:
    
    ```python
    page.select_option("select[name='jobType']", "full-time")
    ```
    

**Expected Result**:

* The dropdown will show “full-time” as selected. You’ll see the option appear in the dropdown.
    

---

### **5\. Dropdown Menu**

* **What It Is**: Opens and selects items in dropdown menus.
    
* **Why It’s Useful**: Often needed for choosing job location, job type, etc.
    
* **Example**:
    
    ```python
    page.click("text=Choose Location")  # Click to open dropdown
    page.click("text=New York")         # Select an option
    ```
    

**Expected Result**:

* The dropdown opens, and then the “New York” option is selected.
    

---

### **6\. Upload Files**

* **What It Is**: Automates file upload, useful for CV or cover letter uploads.
    
* **Why It’s Useful**: Most applications require uploading your resume.
    
* **Example**:
    
    ```python
    page.set_input_files("input[type='file']", "/path/to/your_resume.pdf")
    ```
    

**Expected Result**:

* The file upload input will show the selected file. You’ll see the file name or an upload confirmation on the page.
    

---

### **7\. Keyboard Shortcuts**

* **What It Is**: Simulates keyboard shortcuts (e.g., `Ctrl + Enter`).
    
* **Why It’s Useful**: Some sites use shortcuts to submit forms or navigate.
    
* **Example**:
    
    ```python
    page.keyboard.press("Enter")       # Simulates pressing Enter
    page.keyboard.press("Control+S")   # Shortcut for Save or Submit, if applicable
    ```
    

**Expected Result**:

* The browser will trigger any action tied to the keypress, like submitting a form or saving data.
    

---

# Events

### **1\. Playwright Auto-Waiting**

* **What It Is**: Playwright automatically waits for elements to be ready (like buttons or input fields) before interacting with them.
    
* **Why It’s Useful**: It avoids errors by waiting until elements are fully loaded, so your script won’t fail if a page is slow.
    
* **Example**:
    
    ```python
    page.click("text=Apply Now")  # Playwright waits until "Apply Now" is visible
    ```
    

**Expected Result**:

* Playwright will click the “Apply Now” button as soon as it’s ready. If the page is slow, it will wait until the button appears.
    

---

### **2\. Auto-Waiting Navigation**

* **What It Is**: Automatically waits for a page to finish loading after navigation (like after clicking “Next” or “Submit”).
    
* **Why It’s Useful**: Ensures the next page is fully loaded before the script proceeds.
    
* **Example**:
    
    ```python
    page.click("text=Submit Application")  # Waits until the navigation completes
    page.wait_for_load_state("load")       # Ensures page has fully loaded
    ```
    

**Expected Result**:

* After clicking “Submit,” the script won’t proceed until the next page is loaded, so you won’t see errors from trying to interact too early.
    

---

### **3\. Custom Waiting**

* **What It Is**: You can manually set wait times for certain elements if auto-waiting isn’t enough.
    
* **Why It’s Useful**: Useful if an element takes extra time to load or appears conditionally.
    
* **Example**:
    
    ```python
    page.wait_for_selector("text=Confirmation Message")  # Wait for confirmation to appear
    ```
    

**Expected Result**:

* The script will pause until it sees the “Confirmation Message” element, so you know the application was submitted.
    

---

### **4\. Event Listeners**

* **What It Is**: Allows your script to listen for specific events (like page load or network requests).
    
* **Why It’s Useful**: You can monitor page events to trigger actions or logging.
    
* **Example**:
    
    ```python
    page.on("load", lambda: print("Page has loaded"))
    page.goto("https://www.indeed.com")
    ```
    

**Expected Result**:

* When the page loads, you’ll see “Page has loaded” in the terminal. This confirms that the page is fully ready.
    

---

### **5\. Handling Dialogs**

* **What It Is**: Lets you handle pop-up dialogs (e.g., alerts, prompts).
    
* **Why It’s Useful**: You might encounter pop-ups that need to be accepted or dismissed.
    
* **Example**:
    
    ```python
    page.on("dialog", lambda dialog: dialog.accept())  # Automatically accept any dialog
    ```
    

**Expected Result**:

* Any pop-up dialogs will be accepted automatically, so they don’t interrupt the script.
    

---

### **6\. Download Files**

* **What It Is**: Automates file downloading, useful if you need to save receipts or confirmation documents.
    
* **Why It’s Useful**: Allows you to download files to a specified location.
    
* **Example**:
    
    ```python
    with page.expect_download() as download_info:
        page.click("text=Download Receipt")
    download = download_info.value
    download.save_as("/path/to/save/receipt.pdf")
    ```
    

**Expected Result**:

* The file (e.g., a receipt) will download to your specified path. You should see it saved in that folder.
    

---

### **7\. What Is Sync and Async?**

* **Sync**: Code runs one line at a time, waiting for each line to complete.
    
* **Async**: Code can run multiple tasks at once, improving efficiency.
    
* **Why It’s Useful**: Playwright has both synchronous (sync) and asynchronous (async) options. For simpler scripts, sync is easier; for complex, async allows faster operations.
    

---

### **8\. Asynchronous Playwright**

* **What It Is**: The async version of Playwright, using `async` and `await` keywords to handle multiple actions at once.
    
* **Why It’s Useful**: Useful for running multiple actions in parallel, which is faster but slightly more complex.
    
* **Example**:
    
    ```python
    import asyncio
    from playwright.async_api import async_playwright
    
    async def main():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto("https://www.indeed.com")
            await page.click("text=Apply Now")
    
    asyncio.run(main())
    ```
    

**Expected Result**:

* The script runs asynchronously, which is faster for large-scale automation like handling many applications.
    

---

# Authentication

### **1\. Authentication**

* **What It Is**: Automates login processes, allowing your script to handle sites that require a username and password.
    
* **Why It’s Useful**: For job applications, many sites require you to be logged in to apply. Automating this can save time.
    
* **Example**:
    
    ```python
    page.fill("input[name='username']", "your_username")
    page.fill("input[name='password']", "your_password")
    page.click("text=Sign In")
    ```
    

**Expected Result**:

* Playwright will log you in automatically, filling in the username and password fields and clicking “Sign In.”
    

---

### **2\. Google Sign In**

* **What It Is**: Handles logging in with a Google account, which many job sites support.
    
* **Why It’s Useful**: Useful for sites that don’t have a traditional username/password login.
    
* **Example**:
    
    ```python
    page.goto("https://accounts.google.com")
    page.fill("input[type='email']", "your_email@gmail.com")
    page.click("text=Next")
    page.fill("input[type='password']", "your_password")
    page.click("text=Next")
    ```
    

**Expected Result**:

* The browser navigates to Google, fills in your email and password, and completes the login. You should be logged into the site that uses Google Sign-In.
    

---

### **3\. Reuse Authentication State**

* **What It Is**: Saves your login session (cookies, tokens), so you don’t have to log in every time you run the script.
    
* **Why It’s Useful**: Saves time by avoiding repetitive logins.
    
* **Example**:
    
    ```python
    context = browser.new_context(storage_state="auth.json")  # Load saved session
    page = context.new_page()
    page.goto("https://www.indeed.com")
    ```
    
    * **Saving State**:
        
        ```python
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.indeed.com")
        # Perform login here
        context.storage_state(path="auth.json")  # Save session to a file
        ```
        

**Expected Result**:

* After the initial login, the session is saved in `auth.json`. You can reuse it in future runs without logging in again.
    

---

### **4\. Reuse Auth State**

* **What It Is**: Similar to reusing authentication state, it loads saved session data across pages or tests.
    
* **Why It’s Useful**: Useful for multi-page applications where you need to stay logged in as you move between pages.
    
* **Example**:
    
    ```python
    context = browser.new_context(storage_state="auth.json")
    page = context.new_page()
    page.goto("https://www.aijobs.net")
    ```
    

**Expected Result**:

* With this setup, you’re already logged in across multiple job sites or pages using the same session data. This makes moving between job sites faster and more efficient.
    

---

# **Automated Mail Checker**

### **1\. Automated Mail Checker / Automatic Mail Checker**

* **What It Is**: A script to log into your email and check for new messages.
    
* **Why It’s Useful**: Helps you stay updated on new job-related emails without manually refreshing your inbox.
    
* **Example Setup**:
    
    ```python
    from playwright.sync_api import sync_playwright
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # Run in background
        page = browser.new_page()
        page.goto("https://mail.google.com")
    
        # Login steps here (filling email, password)
        # page.fill(...) and page.click(...) for authentication
    ```
    

**Expected Result**:

* This setup logs into your email in the background. You’ll add steps to identify new emails next.
    

---

### **2\. Project Setup**

* **What It Is**: Initial setup steps to organize your mail-checking project.
    
* **Why It’s Useful**: Keeps everything organized and reusable for checking mail regularly.
    
* **Suggested Structure**:
    
    ```python
    project_folder/
    ├── mail_checker.py         # Main script
    ├── requirements.txt        # Dependencies
    ├── auth.json               # Saved authentication state
    ```
    
* **Code to Install Requirements**:
    
    ```bash
    pip install playwright
    playwright install
    ```
    

**Expected Result**:

* Your project folder is organized and dependencies are installed.
    

---

### **3\. Locate New Emails**

* **What It Is**: Code to identify unread or new emails.
    
* **Why It’s Useful**: Focuses only on new messages, so you know exactly what’s fresh.
    
* **Example**:
    
    ```python
    unread_emails = page.locator("div[aria-label='Unread']")
    print("Number of new emails:", unread_emails.count())
    ```
    

**Expected Result**:

* You’ll see the number of new (unread) emails printed in your terminal.
    

---

### **4\. Locate Email Data**

* **What It Is**: Extracts details like sender, subject, and date.
    
* **Why It’s Useful**: Helps you quickly review new emails without opening each one.
    
* **Example**:
    
    ```python
    for i in range(unread_emails.count()):
        subject = unread_emails.nth(i).locator("span.subject").inner_text()
        sender = unread_emails.nth(i).locator("span.sender").inner_text()
        print(f"Email {i+1} - Subject: {subject}, From: {sender}")
    ```
    

**Expected Result**:

* Prints details of each new email’s subject and sender, so you can easily scan important messages.
    

---

### **5\. Combine Locators**

* **What It Is**: Uses multiple locators together to access nested elements (e.g., email subject within each new email).
    
* **Why It’s Useful**: Lets you accurately capture detailed information.
    
* **Example**:
    
    ```python
    for i in range(unread_emails.count()):
        email_data = unread_emails.nth(i).locator("div.email-details")
        subject = email_data.locator("span.subject").inner_text()
        sender = email_data.locator("span.sender").inner_text()
        print(f"Subject: {subject}, Sender: {sender}")
    ```
    

**Expected Result**:

* Each email’s subject and sender appear clearly. This organized data makes it easy to check details.
    

---

### **6\. Check Email from Terminal**

* **What It Is**: Runs your mail-checker script directly from the terminal, allowing you to check emails without opening a browser.
    
* **How to Use**:
    
    ```bash
    python mail_checker.py
    ```
    

**Expected Result**:

* Your terminal displays new email data like subject and sender. This setup is ideal for quick inbox checks.
    

---

# **Pytest**

### **1\. Pytest Installation**

* **Command**:
    
    ```bash
    pip install pytest
    ```
    

**Expected Result**:

* Pytest is installed and ready for testing your Indeed script.
    

---

### **2\. Testing with Pytest**

* **What It Is**: Pytest lets you write test cases to verify each part of your Indeed script.
    
* **Why It’s Useful**: Testing pieces of the Indeed automation (like logging in, clicking “Apply,” and filling forms) helps prevent unexpected issues.
    
* **Example**:
    
    ```python
    def test_sample():
        assert 1 + 1 == 2  # Basic test to confirm Pytest works
    ```
    

**Expected Result**:

* Running `pytest` will confirm Pytest is set up correctly before adding more specific tests for Indeed.
    

---

### **3\. Writing Tests for Indeed**

* **What It Is**: Creating test cases for each step of your Indeed automation.
    
* **Why It’s Useful**: Ensures each part, like finding job postings or filling out forms, works correctly.
    
* **Example**:
    
    ```python
    def test_navigate_to_indeed(browser):
        page = browser.new_page()
        page.goto("https://www.indeed.com")
        assert "Job Search" in page.title()  # Confirm page loaded correctly
    ```
    

**Expected Result**:

* Verifies that your script can navigate to Indeed and checks if the title matches. This confirms the page is loaded.
    

---

### **4\. Running Tests**

* **Command**:
    
    ```bash
    pytest
    ```
    

**Expected Result**:

* Pytest runs all tests and outputs pass/fail results for each function in your Indeed script.
    

---

### **5\. Type Hinting for Indeed Functions**

* **What It Is**: Adding type hints to help ensure the correct data types.
    
* **Why It’s Useful**: Makes it clear what inputs and outputs each function in your Indeed automation expects, reducing errors.
    
* **Example**:
    
    ```python
    def apply_to_job(job_url: str, browser) -> bool:
        page = browser.new_page()
        page.goto(job_url)
        # Code to apply to job
        return True
    ```
    

**Expected Result**:

* Type hints make the function’s purpose and requirements clear, reducing missteps while navigating Indeed job pages.
    

---

### **6\. Test State for Indeed**

* **What It Is**: Ensuring each test starts with Indeed in a predictable state.
    
* **Why It’s Useful**: Helps avoid conflicts between tests by setting a fresh, consistent start for each Indeed page load.
    
* **Example**: Use fixtures to open and close the Indeed page in each test.
    

**Expected Result**:

* Each test runs independently, loading Indeed’s homepage fresh each time to avoid session or state conflicts.
    

---

### **7\. Pytest Fixture for Indeed**

* **What It Is**: Prepares reusable browser sessions for Indeed.
    
* **Why It’s Useful**: Saves time by setting up the Indeed page once and reusing it across multiple tests.
    
* **Example**:
    
    ```python
    import pytest
    from playwright.sync_api import sync_playwright
    
    @pytest.fixture
    def browser():
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            yield browser
            browser.close()
    ```
    

**Expected Result**:

* The `browser` fixture can be used in each test function, streamlining the Indeed page setup.
    

---

### **8\. Using Fixture in Indeed Tests**

* **What It Is**: Calls the `browser` fixture in test functions to reuse the Indeed setup.
    
* **Why It’s Useful**: Avoids repetitive code, letting you focus on testing specific Indeed actions.
    
* **Example**:
    
    ```python
    def test_search_jobs(browser):
        page = browser.new_page()
        page.goto("https://www.indeed.com")
        page.fill("input[name='q']", "Data Scientist")  # Fill in job search
        page.click("text=Find Jobs")
        assert page.locator("text=Data Scientist").count() > 0
    ```
    

**Expected Result**:

* Verifies that searching for “Data Scientist” on Indeed yields results. This confirms the search functionality works.
    

---

### **9\. Fixture Scope for Indeed**

* **What It Is**: Controls how often a fixture is set up (e.g., once per session for Indeed).
    
* **Why It’s Useful**: Loading Indeed once per session saves time when running multiple tests.
    
* **Example**:
    
    ```python
    @pytest.fixture(scope="session")
    def indeed_browser():
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("https://www.indeed.com")
            yield page
            browser.close()
    ```
    

**Expected Result**:

* This `indeed_browser` fixture logs in to Indeed once and keeps the session open, saving time on repeated logins.
    

---

# P**ytest-playwright plugin**

### **1\. Install pytest-playwright Plugin**

* **Command**:
    
    ```bash
    pip install pytest-playwright
    ```
    

**Expected Result**:

* This command installs the plugin, readying your environment for Playwright tests within Pytest.
    

---

### **2\. Using Playwright Test with Pytest**

* **What It Is**: The pytest-playwright plugin provides Playwright’s `page` and `browser` fixtures for easy use in Pytest.
    
* **Why It’s Useful**: Simplifies Playwright test setup, making it easy to write and run tests specifically for Indeed.
    
* **Example Test**:
    
    ```python
    def test_navigate_to_indeed(page):
        page.goto("https://www.indeed.com")
        assert "Job Search" in page.title()  # Check if Indeed loaded correctly
    ```
    

**Expected Result**:

* Runs the test, confirming that it can navigate to Indeed and the title contains “Job Search.” This ensures the page loads correctly.
    

---

### **3\. Running Test with pytest-playwright**

* **Command**:
    
    ```bash
    pytest
    ```
    

**Expected Result**:

* All Pytest tests, including Playwright ones, will execute. You’ll see a pass/fail report in the terminal.
    

---

### **4\. Pytest Config with pytest-playwright**

* **What It Is**: Allows configuring test behavior, like setting the browser type or headless mode in a `pytest.ini` or `pyproject.toml` file.
    
* **Why It’s Useful**: Ensures consistency across tests and lets you customize settings for Indeed testing.
    
* **Example Configuration** (`pytest.ini`):
    
    ```ini
    [pytest]
    addopts = --browser chromium --headed
    ```
    

**Expected Result**:

* Sets the browser to Chromium and opens it in “headed” mode (visible). This config ensures each test uses the same setup.
    

---

### **5\. Test Hook**

* **What It Is**: Allows adding setup or teardown code that runs before or after each test, useful for initializing the Indeed page.
    
* **Why It’s Useful**: Reduces repetitive setup code by initializing common elements like the Indeed page URL.
    
* **Example Hook**:
    
    ```python
    def pytest_playwright_page_setup(page):
        page.goto("https://www.indeed.com")
    ```
    

**Expected Result**:

* This hook automatically navigates to Indeed for each test, so you don’t need to include `page.goto()` in each function.
    

---

# Playwright tools

### **1\. Take Screenshot**

* **What It Is**: Captures a screenshot of the current page, useful for verifying that elements appear as expected or for debugging.
    
* **Why It’s Useful**: Helps you visually confirm that pages on Indeed, like job search or application pages, are loading and displaying correctly.
    
* **Example**:
    
    ```python
    page.goto("https://www.indeed.com")
    page.screenshot(path="indeed_homepage.png")
    ```
    

**Expected Result**:

* A screenshot file named `indeed_homepage.png` saves in your project directory, showing the Indeed homepage as it appears during the test.
    

---

### **2\. Record Video**

* **What It Is**: Records a video of the browser session, capturing all actions taken on Indeed.
    
* **Why It’s Useful**: Helps with debugging complex workflows (e.g., clicking “Apply” and filling out forms) by letting you review each step.
    
* **Example**:
    
    ```python
    context = browser.new_context(record_video_dir="videos/")
    page = context.new_page()
    page.goto("https://www.indeed.com")
    ```
    

**Expected Result**:

* A video recording of the session saves in the `videos/` directory, showing everything that happened on the Indeed page.
    

---

### **3\. Trace Generator and Viewer**

* **What It Is**: Generates a trace file of your Playwright session, capturing detailed information about each action and response.
    
* **Why It’s Useful**: Helps you analyze and debug each step in your Indeed automation, especially useful for understanding why certain actions might fail.
    
* **How to Use**:
    
    * Start tracing:
        
        ```python
        context.tracing.start(screenshots=True, snapshots=True, sources=True)
        ```
        
    * Stop tracing and save:
        
        ```python
        context.tracing.stop(path="trace.zip")
        ```
        

**Expected Result**:

* A [`trace.zip`](http://trace.zip) file saves, which you can load in the Playwright trace viewer to see each step in detail (including screenshots and page structure).
    

---

### **4\. Playwright Codegen**

* **What It Is**: A code generator that records actions in the browser and outputs the corresponding Playwright code.
    
* **Why It’s Useful**: Quickly generates code for repetitive tasks, like clicking on job listings or filling out forms on Indeed.
    
* **How to Use**:
    
    ```bash
    playwright codegen https://www.indeed.com
    ```
    

**Expected Result**:

* A browser window opens to Indeed. As you click around (e.g., entering search terms or navigating pages), Playwright outputs the code in the terminal or a separate window, allowing you to copy the generated code directly into your project.
    

---

# **Web-First Assertions**

### **1\. Web-First Assertions**

* **What It Is**: Playwright’s “web-first” assertions automatically wait for the condition to be true (e.g., element visibility) before proceeding.
    
* **Why It’s Useful**: Helps ensure elements on Indeed are fully loaded before interacting, reducing errors caused by slow loading times.
    
* **Example**:
    
    ```python
    page.goto("https://www.indeed.com")
    page.get_by_text("Job Search").wait_for()
    ```
    

**Expected Result**:

* The script waits until the text “Job Search” is visible on Indeed’s homepage, ensuring the page is ready before further actions.
    

---

### **2\. Assertions for Page**

* **What It Is**: Checks properties of the entire page, such as the title.
    
* **Why It’s Useful**: Confirms you’re on the correct page or section within Indeed.
    
* **Example**:
    
    ```python
    assert page.title() == "Job Search | Indeed"
    ```
    

**Expected Result**:

* Verifies the page title is correct. If it’s not, the assertion fails, alerting you that you might be on the wrong page.
    

---

### **3\. Assertions for Element State**

* **What It Is**: Verifies the state of elements, like whether they are visible, enabled, or checked.
    
* **Why It’s Useful**: Ensures elements like “Apply Now” buttons are ready for interaction on Indeed.
    
* **Example**:
    
    ```python
    assert page.get_by_text("Apply Now").is_visible()
    ```
    

**Expected Result**:

* Confirms the “Apply Now” button is visible on the page, meaning it’s ready for the next step.
    

---

### **4\. Assertions for Element Text**

* **What It Is**: Checks the text content of an element, useful for verifying job titles or labels.
    
* **Why It’s Useful**: Helps confirm that Indeed job listings or form labels have the expected text.
    
* **Example**:
    
    ```python
    assert page.locator("h2.job-title").text_content() == "Data Scientist"
    ```
    

**Expected Result**:

* Verifies that the job title displayed is “Data Scientist.” If not, the assertion fails, alerting you that the job listing might not match.
    

---

### **5\. Assertions for Attribute**

* **What It Is**: Checks an element’s attribute, such as `href` for links or `src` for images.
    
* **Why It’s Useful**: Ensures links and buttons on Indeed point to the correct URLs or paths.
    
* **Example**:
    
    ```python
    assert page.locator("a.apply-link").get_attribute("href") == "https://www.indeed.com/apply"
    ```
    

**Expected Result**:

* Confirms the “Apply” link has the correct URL. If it doesn’t, the assertion fails, helping catch misdirected links.
    

---

### **6\. Assertions for Input Field**

* **What It Is**: Verifies the value of input fields.
    
* **Why It’s Useful**: Confirms fields like search inputs or form fields contain the correct prefilled or typed values on Indeed.
    
* **Example**:
    
    ```python
    assert page.locator("input[name='q']").input_value() == "Data Scientist"
    ```
    

**Expected Result**:

* Confirms the job search field contains “Data Scientist.” If it’s not there, the assertion fails, indicating the search input didn’t register correctly.
    

---

### **7\. Assertions for Checkbox**

* **What It Is**: Checks if a checkbox is selected or not.
    
* **Why It’s Useful**: Ensures options like job type filters (e.g., “Full-Time”) are selected when needed on Indeed.
    
* **Example**:
    
    ```python
    assert page.locator("input[name='full_time']").is_checked()
    ```
    

**Expected Result**:

* Confirms the “Full-Time” checkbox is selected. If not, the assertion fails, indicating a filter option might be missing.
    

---

### **8\. Assertions for Option Menu**

* **What It Is**: Verifies the selected option in dropdown menus.
    
* **Why It’s Useful**: Ensures dropdowns, like location or job type on Indeed, have the correct selection.
    
* **Example**:
    
    ```python
    assert page.locator("select[name='job_type']").input_value() == "Full-Time"
    ```
    

**Expected Result**:

* Confirms that “Full-Time” is selected in the job type dropdown. If not, the assertion fails, indicating the wrong option might be selected.
    

---

# **UI Testing**

### **1\. UI Testing Playground**

* **What It Is**: A tool or website that provides interactive elements for testing automation scripts.
    
* **Why It’s Useful**: Practice automating complex UI elements (like dynamic tables or delayed content) before applying the same techniques on Indeed.
    

---

### **2\. UI Testing Dynamic ID**

* **What It Is**: Handles elements with IDs that change each time the page loads.
    
* **Why It’s Useful**: On sites like Indeed, dynamic IDs can make finding elements tricky.
    
* **Example**:
    
    ```python
    page.locator("[id^='apply-button']").click()  # Clicks a button with an ID that starts with 'apply-button'
    ```
    

**Expected Result**:

* Playwright finds the button even if the ID changes, ensuring reliable interaction.
    

---

### **3\. UI Testing Class Attribute**

* **What It Is**: Targets elements by their class name, often more stable than IDs.
    
* **Why It’s Useful**: Useful for elements like buttons or inputs where classes are more consistent than IDs.
    
* **Example**:
    
    ```python
    page.locator(".job-title").click()
    ```
    

**Expected Result**:

* Finds and clicks the job title element reliably using the class name.
    

---

### **4\. UI Testing Hidden Layer**

* **What It Is**: Detects hidden elements or layers.
    
* **Why It’s Useful**: Ensures only visible elements are interacted with, avoiding errors when interacting with hidden elements on Indeed.
    
* **Example**:
    
    ```python
    assert page.locator(".apply-popup").is_hidden()
    ```
    

**Expected Result**:

* Verifies the “apply” popup is hidden, helping to avoid interacting with it prematurely.
    

---

### **5\. UI Testing Load Delay**

* **What It Is**: Manages elements that load with a delay.
    
* **Why It’s Useful**: Helps you avoid errors on pages where content takes time to load, like Indeed job listings.
    
* **Example**:
    
    ```python
    page.locator(".job-card").wait_for()
    ```
    

**Expected Result**:

* The script waits until job cards are fully loaded on the page before proceeding.
    

---

### **6\. UI Testing Ajax Request**

* **What It Is**: Detects elements that appear or change based on AJAX requests.
    
* **Why It’s Useful**: Ensures updated job listings or search results are fully loaded before interacting.
    
* **Example**:
    
    ```python
    page.wait_for_load_state("networkidle")  # Wait until AJAX requests are complete
    ```
    

**Expected Result**:

* Ensures all AJAX requests (like updated job listings) finish loading before proceeding.
    

---

### **7\. UI Testing Click Action**

* **What It Is**: Simulates clicks on elements.
    
* **Why It’s Useful**: Essential for clicking job links, “Apply” buttons, or other interactive elements on Indeed.
    
* **Example**:
    
    ```python
    page.click("text=Apply Now")
    ```
    

**Expected Result**:

* Clicks the “Apply Now” button and advances to the application form.
    

---

### **8\. UI Testing Input Field**

* **What It Is**: Enters data into text fields.
    
* **Why It’s Useful**: Used for filling out forms (e.g., name, email) on Indeed.
    
* **Example**:
    
    ```python
    page.fill("input[name='email']", "user@example.com")
    ```
    

**Expected Result**:

* The email field populates with the specified text, confirming the input action works.
    

---

### **9\. UI Testing Scrollbars**

* **What It Is**: Handles elements that require scrolling.
    
* **Why It’s Useful**: Ensures elements hidden by scrollbars (like footer links) are accessible.
    
* **Example**:
    
    ```python
    page.locator("footer").scroll_into_view_if_needed()
    ```
    

**Expected Result**:

* Scrolls to the footer if needed, allowing interaction with hidden elements.
    

---

### **10\. UI Testing Dynamic Table**

* **What It Is**: Interacts with tables that dynamically update.
    
* **Why It’s Useful**: Useful for checking Indeed job listings in table format, which may change with each search.
    
* **Example**:
    
    ```python
    rows = page.locator(".job-table-row")
    assert rows.count() > 0  # Ensure at least one job listing is present
    ```
    

**Expected Result**:

* Verifies that job listings are available in the table, confirming the table is loaded.
    

---

### **11\. UI Testing Verify Text**

* **What It Is**: Confirms specific text content is present.
    
* **Why It’s Useful**: Ensures you’re on the correct page or viewing the right job posting.
    
* **Example**:
    
    ```python
    assert page.locator("h2.job-title").text_content() == "Data Scientist"
    ```
    

**Expected Result**:

* Verifies the job title text matches the expected role, like “Data Scientist.”
    

---

### **12\. UI Testing Progress Bar**

* **What It Is**: Handles pages with progress bars.
    
* **Why It’s Useful**: Ensures actions complete (like loading a job page) before interacting.
    
* **Example**:
    
    ```python
    page.locator(".progress-bar").wait_for(state="hidden")
    ```
    

**Expected Result**:

* The script waits until the progress bar is hidden, indicating the page is fully loaded.
    

---

### **13\. UI Testing Visibility**

* **What It Is**: Checks if elements are visible or hidden.
    
* **Why It’s Useful**: Verifies interactive elements are displayed before clicking them.
    
* **Example**:
    
    ```python
    assert page.locator(".apply-button").is_visible()
    ```
    

**Expected Result**:

* Confirms the “Apply” button is visible, avoiding errors from clicking hidden elements.
    

---

### **14\. UI Testing App Login**

* **What It Is**: Automates logging into an application.
    
* **Why It’s Useful**: Useful if Indeed requires a login to view or apply to jobs.
    
* **Example**:
    
    ```python
    page.fill("input[name='username']", "your_username")
    page.fill("input[name='password']", "your_password")
    page.click("text=Sign In")
    ```
    

**Expected Result**:

* Logs into Indeed, ready for further interactions.
    

---

### **15\. UI Testing Mouse Hover**

* **What It Is**: Simulates hovering over elements.
    
* **Why It’s Useful**: Useful for dropdowns or revealing hidden menus.
    
* **Example**:
    
    ```python
    page.hover(".menu-item")
    ```
    

**Expected Result**:

* The dropdown menu appears, showing additional options.
    

---

### **16\. UI Testing NBSP Character**

* **What It Is**: Ensures proper handling of non-breaking spaces (`&nbsp;`) in text.
    
* **Why It’s Useful**: Some job descriptions might contain `&nbsp;`, which can interfere with text matching.
    
* **Example**:
    
    ```python
    assert page.locator("text=Job Description ").is_visible()
    ```
    

**Expected Result**:

* Confirms the element with non-breaking space text is handled correctly.
    

---

### **17\. UI Testing Overlapped**

* **What It Is**: Checks and interacts with elements that might be covered by other elements.
    
* **Why It’s Useful**: Helps avoid errors from attempting to click overlapped elements.
    
* **Example**:
    
    ```python
    page.locator(".apply-button").scroll_into_view_if_needed().click()
    ```
    

**Expected Result**:

* Scrolls to and clicks the “Apply” button, ensuring it’s not hidden by other elements.
    

---

# **Playwright Fixtures**

### **1\. Playwright Fixtures**

* **What They Are**: Predefined setups in Playwright to manage resources (like the browser or page) that can be reused across tests.
    
* **Why They’re Useful**: Save time by setting up and tearing down elements like browser instances and contexts only once per test or session.
    
* **Example**:
    
    ```python
    import pytest
    from playwright.sync_api import Page
    
    @pytest.fixture
    def setup_indeed_page(page: Page):
        page.goto("https://www.indeed.com")
        return page
    ```
    

**Expected Result**:

* The fixture `setup_indeed_page` navigates to Indeed and can be reused across multiple tests without repeating the navigation setup.
    

---

### **2\. Function Scope Fixtures**

* **What They Are**: Fixtures that initialize and tear down for each individual test function.
    
* **Why They’re Useful**: Useful when each test needs a fresh browser state, like a new Indeed search.
    
* **Example**:
    
    ```python
    @pytest.fixture(scope="function")
    def new_page_context(browser):
        context = browser.new_context()
        page = context.new_page()
        yield page
        context.close()
    ```
    

**Expected Result**:

* Each test will open a new page context, which is closed at the end of each test function, providing an isolated environment.
    

---

### **3\. Session Scope Fixtures**

* **What They Are**: Fixtures with a scope set to last for the entire test session, meaning they’re only initialized once.
    
* **Why They’re Useful**: Speeds up testing by reusing the same setup (like a logged-in state on Indeed) across multiple tests.
    
* **Example**:
    
    ```python
    @pytest.fixture(scope="session")
    def indeed_session_browser():
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("https://www.indeed.com")
            yield page
            browser.close()
    ```
    

**Expected Result**:

* The `indeed_session_browser` fixture opens Indeed once and reuses this browser session for all tests, saving login or navigation time.
    

---

### **4\. Browser Selection**

* **What It Is**: Allows you to specify which browser (Chromium, Firefox, or WebKit) to use.
    
* **Why It’s Useful**: Lets you test your Indeed automation across different browsers to ensure compatibility.
    
* **Example**:
    
    ```bash
    pytest --browser firefox  # Run tests with Firefox
    ```
    

**Expected Result**:

* The tests will run in the specified browser, helping ensure the Indeed automation works across browser types.
    

---

### **5\. Browser Launch and Context Arguments**

* **What It Is**: Custom arguments for launching the browser and creating contexts, like running in headless mode or setting viewport size.
    
* **Why It’s Useful**: Customizes how the browser is set up for each test, which can affect performance or visibility.
    
* **Example**:
    
    ```python
    @pytest.fixture
    def custom_browser():
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False, args=["--start-maximized"])
            context = browser.new_context(viewport={"width": 1920, "height": 1080})
            page = context.new_page()
            yield page
            browser.close()
    ```
    

**Expected Result**:

* The `custom_browser` fixture runs the browser in maximized, visible mode with a specific viewport, making it easier to verify visual elements on Indeed.
    

---

# **Page Object Model (POM)**

### **1\. What Is Page Object Model?**

* **What It Is**: POM separates the page structure from the test logic by representing each page (like an Indeed job search page) as a class.
    
* **Why It’s Useful**: Makes your code modular, reusable, and easier to maintain, especially helpful if Indeed’s layout changes.
    
* **Example Structure**:
    
    ```plaintext
    pages/
    ├── home_page.py
    ├── job_search_page.py
    tests/
    ├── test_job_search.py
    ```
    

**Expected Result**:

* This structure organizes the test code, making it easier to update individual page classes without changing test logic.
    

---

### **2\. Page Object Model Implementation**

* **What It Is**: Creating a class for each page, encapsulating page elements (selectors) and actions (methods).
    
* **Why It’s Useful**: Reduces duplication by centralizing all locators and actions for a page within its class.
    
* **Example** (`job_search_`[`page.py`](http://page.py)):
    
    ```python
    from playwright.sync_api import Page
    
    class JobSearchPage:
        def __init__(self, page: Page):
            self.page = page
            self.search_input = "input[name='q']"
            self.search_button = "button[aria-label='Find Jobs']"
        
        def go_to(self):
            self.page.goto("https://www.indeed.com")
        
        def search_job(self, job_title: str):
            self.page.fill(self.search_input, job_title)
            self.page.click(self.search_button)
    ```
    

**Expected Result**:

* The `JobSearchPage` class encapsulates locators and methods for searching jobs on Indeed, making the logic reusable and easy to update.
    

---

### **3\. Page Object Model Usage**

* **What It Is**: Using the page classes in your tests.
    
* **Why It’s Useful**: Simplifies test code by leveraging methods from page classes, making tests more readable and maintainable.
    
* **Example** (`test_job_`[`search.py`](http://search.py)):
    
    ```python
    import pytest
    from pages.job_search_page import JobSearchPage
    
    def test_search_data_scientist_job(page):
        search_page = JobSearchPage(page)
        search_page.go_to()
        search_page.search_job("Data Scientist")
        assert page.locator("text=Data Scientist").count() > 0
    ```
    

**Expected Result**:

* This test searches for “Data Scientist” jobs on Indeed using the `JobSearchPage` methods, keeping the test concise and readable.
    

---

### **4\. Playwright Homepage POM**

* **What It Is**: Creating a POM for Indeed’s homepage or other primary pages.
    
* **Why It’s Useful**: Organizes core actions like navigating to the site or clicking main menu links.
    
* **Example** (`home_`[`page.py`](http://page.py)):
    
    ```python
    from playwright.sync_api import Page
    
    class HomePage:
        def __init__(self, page: Page):
            self.page = page
    
        def go_to_homepage(self):
            self.page.goto("https://www.indeed.com")
        
        def click_login(self):
            self.page.click("text=Sign In")
    ```
    

**Expected Result**:

* This `HomePage` class provides quick access to Indeed’s main actions, such as loading the homepage and handling sign-in.
    

---

### **5\. POM Usage**

* **What It Is**: Integrating multiple page classes in a single test.
    
* **Why It’s Useful**: Keeps each test focused on a specific action or flow, making it modular and easy to adjust if Indeed changes.
    
* **Example Usage** (`test_apply_`[`job.py`](http://job.py)):
    
    ```python
    from pages.home_page import HomePage
    from pages.job_search_page import JobSearchPage
    
    def test_apply_to_data_scientist_job(page):
        home_page = HomePage(page)
        job_search_page = JobSearchPage(page)
        
        home_page.go_to_homepage()
        job_search_page.search_job("Data Scientist")
        # Assuming you have a method to click and apply to a job
    ```
    

**Expected Result**:

* This test reuses methods from both the `HomePage` and `JobSearchPage` classes, making each step modular and maintainable.
    

---

# **Network Events**

### **1\. Network Events**

* **What It Is**: Network events in Playwright let you monitor and handle network activity, such as API calls or page requests.
    
* **Why It’s Useful**: Useful for observing and debugging data fetched from the network, like job listings on Indeed, which may be dynamically loaded.
    
* **Example**:
    
    ```python
    def log_request(route):
        print(f"Request made: {route.request.url}")
    
    page.on("request", log_request)
    page.goto("https://www.indeed.com")
    ```
    

**Expected Result**:

* Each request URL made by the Indeed page will print to the console, allowing you to monitor what resources are being loaded.
    

---

### **2\. Handle Requests**

* **What It Is**: Allows intercepting and controlling requests, enabling modifications like blocking ads or API calls.
    
* **Why It’s Useful**: Useful if you want to block certain requests to speed up the Indeed page load or handle specific API requests for testing.
    
* **Example**:
    
    ```python
    def handle_route(route):
        if "ads" in route.request.url:
            route.abort()  # Block ads
        else:
            route.continue_()
    
    page.route("**/*", handle_route)
    page.goto("https://www.indeed.com")
    ```
    

**Expected Result**:

* The script blocks requests containing “ads” in their URL, which could speed up page loading and reduce distractions on the Indeed page.
    

---

### **3\. Modify Response**

* **What It Is**: Lets you intercept and modify responses before they reach the browser.
    
* **Why It’s Useful**: Can be used for testing scenarios by modifying response data, like changing job listings on Indeed to see how the page reacts.
    
* **Example**:
    
    ```python
    def mock_response(route):
        if "job-listings" in route.request.url:
            route.fulfill(
                status=200,
                content_type="application/json",
                body='[{"title": "Mock Job", "company": "Mock Company"}]'
            )
        else:
            route.continue_()
    
    page.route("**/job-listings", mock_response)
    page.goto("https://www.indeed.com")
    ```
    

**Expected Result**:

* The script replaces the response for job listings with a mock job title and company. This lets you test Indeed’s behavior with modified data, ideal for isolated tests.
    

---

# **API Testing**

### **1\. Making an API Call**

* **What It Is**: Playwright allows you to make HTTP requests directly, without opening a browser.
    
* **Why It’s Useful**: Useful for verifying API responses, like job search results, independently of the UI.
    
* **Example**:
    
    ```python
    from playwright.sync_api import sync_playwright
    
    with sync_playwright() as p:
        request = p.request.new_context()
        response = request.get("https://api.indeed.com/jobs?q=Data+Scientist")
        print(response.json())
    ```
    

**Expected Result**:

* This code makes a GET request to Indeed’s API (or a similar job search API) and prints the JSON response, showing job listings.
    

---

### **2\. API Request Context**

* **What It Is**: Sets up a request context, which can include headers, cookies, or authentication.
    
* **Why It’s Useful**: Allows you to test API endpoints that require authentication or custom headers.
    
* **Example**:
    
    ```python
    request_context = p.request.new_context(
        extra_http_headers={"Authorization": "Bearer YOUR_TOKEN"}
    )
    response = request_context.get("https://api.indeed.com/user/profile")
    ```
    

**Expected Result**:

* Sends a GET request with an authorization token, allowing access to secured endpoints like user profile data.
    

---

### **3\. API Query String**

* **What It Is**: Appends parameters to the URL for filtering, sorting, or customizing API results.
    
* **Why It’s Useful**: Used to refine job searches or other API data requests.
    
* **Example**:
    
    ```python
    params = {"q": "Data Scientist", "location": "New York"}
    response = request_context.get("https://api.indeed.com/jobs", params=params)
    print(response.json())
    ```
    

**Expected Result**:

* Fetches job listings for “Data Scientist” in New York, using query strings to customize the search.
    

---

### **4\. CRUD Operations**

* **What It Is**: Allows you to Create, Read, Update, and Delete data using HTTP methods like POST, GET, PUT, and DELETE.
    
* **Why It’s Useful**: Verifies the functionality of APIs that support user-generated content, such as saving or deleting job applications.
    
* **Examples**:
    
    ```python
    # Create (POST)
    response = request_context.post("https://api.indeed.com/jobs", json={"title": "New Job"})
    
    # Read (GET)
    response = request_context.get("https://api.indeed.com/jobs/1")
    
    # Update (PUT)
    response = request_context.put("https://api.indeed.com/jobs/1", json={"title": "Updated Job"})
    
    # Delete (DELETE)
    response = request_context.delete("https://api.indeed.com/jobs/1")
    ```
    

**Expected Result**:

* Each CRUD operation interacts with Indeed’s job data (or a mock API) as expected, performing operations like adding, viewing, updating, or deleting job listings.
    

---

### **5\. Mock API**

* **What It Is**: Intercepts API calls and returns custom responses, simulating real API behavior.
    
* **Why It’s Useful**: Helps test specific scenarios, like error responses or modified data, without affecting real data.
    
* **Example**:
    
    ```python
    def mock_job_response(route):
        route.fulfill(
            status=200,
            content_type="application/json",
            body='[{"title": "Mocked Job", "company": "Mock Company"}]'
        )
    
    page.route("https://api.indeed.com/jobs", mock_job_response)
    ```
    

**Expected Result**:

* This mock replaces Indeed’s actual API response with a predefined list of jobs, allowing you to test UI behavior with mock data.
    

---

# Optimizing

### **1\. Intercept Requests**

* **What It Is**: Intercepts and optionally modifies network requests, allowing you to block unnecessary resources like images, ads, or third-party scripts.
    
* **Why It’s Useful**: Reduces load times by blocking heavy or irrelevant resources, making tests faster and more stable.
    
* **Example**:
    
    ```python
    def handle_route(route):
        if route.request.resource_type in ["image", "stylesheet", "font"]:
            route.abort()  # Block images, CSS, and fonts to speed up page load
        else:
            route.continue_()
    
    page.route("**/*", handle_route)
    page.goto("https://www.indeed.com")
    ```
    

**Expected Result**:

* The page loads faster on Indeed by skipping non-essential resources, making interactions and assertions quicker.
    

---

### **2\. Disabling JavaScript**

* **What It Is**: Disables JavaScript on the page to simplify loading and interactions.
    
* **Why It’s Useful**: Ideal for testing static content or interactions that don’t rely on JavaScript, which can further speed up tests.
    
* **Example**:
    
    ```python
    context = browser.new_context(java_script_enabled=False)
    page = context.new_page()
    page.goto("https://www.indeed.com")
    ```
    

**Expected Result**:

* Indeed loads without JavaScript, eliminating dynamic elements, which can speed up loading and simplify tests focused on static content.
    

---

### **3\. Run Tests in Parallel**

* **What It Is**: Executes multiple tests simultaneously across different browser contexts or instances.
    
* **Why It’s Useful**: Significantly reduces total test runtime, especially useful for large test suites like a full Indeed job application workflow.
    
* **How to Use**:
    
    * In the `pytest.ini` file:
        
        ```ini
        [pytest]
        addopts = -n auto  # Runs tests in parallel across available CPU cores
        ```
        
    * Alternatively, use the command line:
        
        ```bash
        pytest -n auto
        ```
        

**Expected Result**:

* Tests execute in parallel across available cores, reducing runtime, especially for suites with many Indeed-related tests.
    

---

These optimizations can make your Indeed automation faster, more efficient, and cost-effective, especially in CI/CD environments. Let me know if you’d like further examples or adjustments for specific test cases!

---

# T**ips and tricks**

### **1\. Pytest CLI Arguments**

* **What It Is**: Command-line options to control Pytest behavior, like selecting specific tests, running in parallel, or controlling verbosity.
    
* **Why It’s Useful**: Gives you flexibility in how tests are run, allowing quicker iterations or targeted testing.
    
* **Common Arguments**:
    
    ```bash
    pytest -v                       # Verbose mode, shows detailed output
    pytest -k "test_search"         # Runs only tests with 'test_search' in the name
    pytest --maxfail=3              # Stops after 3 failures
    pytest --tb=short               # Shortens traceback for readability
    ```
    

**Expected Result**:

* Customizable test runs tailored to your current focus, like isolating Indeed’s search functionality or debugging a specific test.
    

---

### **2\. Python Debugger (pdb)**

* **What It Is**: A built-in Python debugger that lets you pause execution, inspect variables, and step through code.
    
* **Why It’s Useful**: Essential for diagnosing issues within complex test flows, like debugging Indeed job application interactions.
    
* **How to Use**:
    
    ```python
    import pdb; pdb.set_trace()  # Add this line where you want to start debugging
    ```
    

**Expected Result**:

* The script pauses at the `pdb.set_trace()` line, letting you inspect variables and test logic step by step in the terminal.
    

---

### **3\. Device Emulation**

* **What It Is**: Playwright can emulate various devices (e.g., mobile or tablet) by setting specific viewports, user agents, and other parameters.
    
* **Why It’s Useful**: Allows you to test how Indeed’s job search and application pages look and behave on different devices, ensuring compatibility.
    
* **Example**:
    
    ```python
    iPhone = playwright.devices["iPhone 12"]
    context = browser.new_context(**iPhone)
    page = context.new_page()
    page.goto("https://www.indeed.com")
    ```
    

**Expected Result**:

* Indeed loads as it would on an iPhone 12, helping you validate the mobile experience and detect any issues with responsiveness or layout.
    

---

### **4\. Evaluate JavaScript**

* **What It Is**: Executes JavaScript directly on the page, enabling you to access or modify DOM elements or retrieve information that might not be accessible via standard locators.
    
* **Why It’s Useful**: Useful for custom interactions or gathering specific data, especially when you need to extract or manipulate page content dynamically on Indeed.
    
* **Example**:
    
    ```python
    result = page.evaluate("document.title")
    print(result)
    ```
    

**Expected Result**:

* Outputs the title of the current page (e.g., “Job Search | Indeed”), verifying the page load or retrieving dynamic data for further validation.
    

---

These tips provide greater control over your test execution and debugging process, improving efficiency, especially for more complex Indeed workflows. Let me know if you’d like more tips on a specific area!**1\. Intercept Requests**

* **What It Is**: Intercepts and optionally modifies network requests, allowing you to block unnecessary resources like images, ads, or third-party scripts.
    
* **Why It’s Useful**: Reduces load times by blocking heavy or irrelevant resources, making tests faster and more stable.
    
* **Example**:
    
    ```python
    def handle_route(route):
        if route.request.resource_type in ["image", "stylesheet", "font"]:
            route.abort()  # Block images, CSS, and fonts to speed up page load
        else:
            route.continue_()
    
    page.route("**/*", handle_route)
    page.goto("https://www.indeed.com")
    ```
    

**Expected Result**:

* The page loads faster on Indeed by skipping non-essential resources, making interactions and assertions quicker.
    

---

### **2\. Disabling JavaScript**

* **What It Is**: Disables JavaScript on the page to simplify loading and interactions.
    
* **Why It’s Useful**: Ideal for testing static content or interactions that don’t rely on JavaScript, which can further speed up tests.
    
* **Example**:
    
    ```python
    context = browser.new_context(java_script_enabled=False)
    page = context.new_page()
    page.goto("https://www.indeed.com")
    ```
    

**Expected Result**:

* Indeed loads without JavaScript, eliminating dynamic elements, which can speed up loading and simplify tests focused on static content.
    

---

### **3\. Run Tests in Parallel**

* **What It Is**: Executes multiple tests simultaneously across different browser contexts or instances.
    
* **Why It’s Useful**: Significantly reduces total test runtime, especially useful for large test suites like a full Indeed job application workflow.
    
* **How to Use**:
    
    * In the `pytest.ini` file:
        
        ```ini
        [pytest]
        addopts = -n auto  # Runs tests in parallel across available CPU cores
        ```
        
    * Alternatively, use the command line:
        
        ```bash
        pytest -n auto
        ```
        

**Expected Result**:

* Tests execute in parallel across available cores, reducing runtime, especially for suites with many Indeed-related tests.
    

---

# Continuous Integration (CI), GitHub Actions, and Data-Driven Testing

### **1\. Continuous Integration (CI) with GitHub Actions**

CI automates testing and deployment of your code, ensuring that any changes don’t break your automation script. This is particularly useful if you’re updating your Indeed autofill script regularly or working with a team.

* **Setup Repository**: Store your automation code in a GitHub repository, which allows you to version your code and keep track of changes.
    
* **GitHub Actions for CI**:
    
    * **Why It’s Relevant**: Using GitHub Actions, you can set up a workflow to automatically run tests on your Indeed automation script each time you make a change. This helps catch any issues with form-filling, button clicking, or page navigation in real-time.
        
    * **Example Workflow**:
        
        ```yaml
        name: Test Indeed Application Automation
        
        on:
          push:
            branches:
              - main
          pull_request:
            branches:
              - main
        
        jobs:
          test:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v2
              - name: Set up Python
                uses: actions/setup-python@v2
                with:
                  python-version: '3.x'
              - name: Install dependencies
                run: |
                  pip install -r requirements.txt
              - name: Run tests
                run: pytest
        ```
        
        **Expected Outcome**: This workflow automatically runs tests on your Indeed automation script with each commit, helping catch errors before they go live.
        

---

### **2\. Data-Driven Testing**

Data-Driven Testing is especially useful for an Indeed autofill project, as it allows you to test multiple job applications with varying data inputs (like job titles, cover letters, or resume versions) without changing the main script.

* **What It Is**: Running the same test with different sets of data, such as multiple job titles or location preferences.
    
* **Pytest Parametrize**:
    
    * **Why It’s Relevant**: `@pytest.mark.parametrize` in Pytest allows you to run the autofill script multiple times with different inputs, which is useful for testing different application scenarios.
        
    * **Example**:
        
        ```python
        import pytest
        
        @pytest.mark.parametrize("job_title,location", [
            ("Data Scientist", "New York"),
            ("Machine Learning Engineer", "San Francisco"),
            ("Data Analyst", "Chicago"),
        ])
        def test_autofill_job_application(page, job_title, location):
            # Code to navigate to Indeed and search for the job title/location
            # Code to fill out application form fields
            # Assertions to verify form filling is correct
        ```
        
        **Expected Outcome**: Each test run uses a different job title and location combination, allowing you to validate the form-filling functionality across multiple inputs.
        

---

### **3\. Behavior-Driven Development (BDD)**

BDD focuses on writing tests that mirror user behaviors, which is ideal for Indeed automation since it’s heavily user-interaction-based (filling forms, clicking buttons).

* **Define Feature with BDD**:
    
    * **Why It’s Relevant**: BDD tools (like `Behave` or `pytest-bdd`) allow you to define job application scenarios in plain language (e.g., “When I click ‘Apply Now’ on a Data Scientist job, the application form should autofill and submit successfully”).
        
    * **Example Feature**:
        
        ```python
        Feature: Indeed Job Application Autofill
          Scenario: Autofill application form for a Data Scientist position
            Given I am on the Indeed job search page
            When I search for "Data Scientist" in "New York"
            And I click on "Apply Now"
            Then the form should autofill with my information
            And I should be able to submit the application
        ```
        
        **Implement Steps**:
        
        ```python
        from behave import given, when, then
        
        @given("I am on the Indeed job search page")
        def step_impl(context):
            context.page.goto("https://www.indeed.com")
        
        @when('I search for "{job_title}" in "{location}"')
        def step_impl(context, job_title, location):
            context.page.fill("input[name='q']", job_title)
            context.page.fill("input[name='l']", location)
            context.page.click("button[aria-label='Find Jobs']")
        ```
        
        **Expected Outcome**: BDD tests create clear, readable scenarios that outline the steps your automation will take on Indeed. This makes it easier to maintain and expand as requirements change.
        

---

### **Summary: Best Options for Indeed Automation**

1. **GitHub CI with GitHub Actions**: Automates testing every time you update your autofill script, ensuring reliability.
    
2. **Data-Driven Testing**: Allows you to test multiple job titles, locations, and other input variations without changing the main code.
    
3. **Behavior-Driven Development (BDD)**: Creates easy-to-read test scenarios focused on real user interactions, ideal for automation workflows.