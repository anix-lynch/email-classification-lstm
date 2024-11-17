---
title: "Webscrape API #1"
seoTitle: "Webscrape API #1"
seoDescription: "Webscrape API #1"
datePublished: Sun Nov 17 2024 13:48:00 GMT+0000 (Coordinated Universal Time)
cuid: cm3lnjc52000909iagt9lb59t
slug: webscrape-api-1

---

Getting Started with Public APIs: Extracting Currency Exchange Rates Using Python

```python
# Importing the required libraries
import requests  # For sending HTTP requests
import json      # For working with JSON data

# Define the base URL for the API
base_url = "https://api.exchangeratesapi.io/latest"  # Endpoint for latest currency exchange rates

# Send a GET request to the API endpoint
response = requests.get(base_url)  # Make an HTTP GET request
# Sample output: <Response [200]> (if successful)

# Check if the request was successful
print(response.ok)  # Returns True if status code is 200–299
# Sample output: True

# Print the status code of the response
print(response.status_code)  # Returns the HTTP status code
# Sample output: 200

# Get the raw text of the response
print(response.text)  # Returns the body of the response as a string
# Sample output: 
# {"rates":{"USD":1.1,"GBP":0.85},"base":"EUR","date":"2024-11-17"}

# Get the raw content of the response
print(response.content)  # Returns the response content as bytes
# Sample output: 
# b'{"rates":{"USD":1.1,"GBP":0.85},"base":"EUR","date":"2024-11-17"}'

# Convert the JSON response to a Python dictionary
data = response.json()  # Converts the JSON content to a Python dictionary
print(data)
# Sample output:
# {'rates': {'USD': 1.1, 'GBP': 0.85}, 'base': 'EUR', 'date': '2024-11-17'}

# Check the data type of the JSON response
print(type(data))  # Confirms the data type (should be a dictionary)
# Sample output: <class 'dict'>

# Pretty-print the JSON data
pretty_data = json.dumps(data, indent=4)  # Format the JSON with indentation for readability
print(pretty_data)
# Sample output:
# {
#     "rates": {
#         "USD": 1.1,
#         "GBP": 0.85
#     },
#     "base": "EUR",
#     "date": "2024-11-17"
# }

# Extract and print the top-level keys of the JSON
print(data.keys())  # Lists all keys at the top level of the JSON dictionary
# Sample output: dict_keys(['rates', 'base', 'date'])

# Access specific parts of the JSON data
print(data["rates"])  # Prints the exchange rates dictionary
# Sample output: {'USD': 1.1, 'GBP': 0.85}

print(data["base"])  # Prints the base currency
# Sample output: EUR

print(data["date"])  # Prints the date for the rates
# Sample output: 2024-11-17
# Step 8: Add parameters to customize the API request
param_url = base_url + "?symbols=USD,GBP"  # Request rates for USD and GBP only
response = requests.get(param_url)
data = response.json()
print(data)
# Output:
# {'rates': {'USD': 1.1, 'GBP': 0.85}, 'base': 'EUR', 'date': '2024-11-17'}

# Step 9: Change the base currency to USD
param_url = base_url + "?symbols=GBP&base=USD"
data = requests.get(param_url).json()  # Request data and parse response
print(data)
# Output:
# {'rates': {'GBP': 0.85}, 'base': 'USD', 'date': '2024-11-17'}

# Step 10: Extract and calculate specific conversion rates
usd_to_gbp = data['rates']['GBP']  # Extract the GBP rate relative to USD
print(f"1 USD is equal to {usd_to_gbp} GBP")
# Output: 1 USD is equal to 0.85 GBP

# Base URL for historical data
historical_url = "https://api.exchangeratesapi.io/2016-01-26"

# Fetch historical exchange rates
response = requests.get(historical_url)
print(response.json())
# Sample Output:
# {
#     "rates": {
#         "USD": 1.09,
#         "GBP": 0.76
#     },
#     "base": "EUR",
#     "date": "2016-01-26"
# }
# Fetch historical exchange rates over a time period
time_period_url = "https://api.exchangeratesapi.io/history?start_at=2017-04-26&end_at=2018-04-26&symbols=GBP"
data = requests.get(time_period_url).json()

# Pretty-print the data
print(json.dumps(data, indent=4, sort_keys=True))
# Sample Output:
# {
#     "base": "EUR",
#     "end_at": "2018-04-26",
#     "rates": {
#         "2017-04-26": {"GBP": 0.84},
#         "2018-04-26": {"GBP": 0.88}
#     },
#     "start_at": "2017-04-26"
# }
# Invalid date format
invalid_url = "https://api.exchangeratesapi.io/2019-13-01"
response = requests.get(invalid_url)

print(response.status_code)  # Status code for bad request
# Output: 400

print(response.json())  # Error message
# Sample Output: {'error': 'time data \'2019-13-01\' does not match format \'%Y-%m-%d\''}
```

Here’s **iTunes Search API script with inline comments and sample outputs** for clarity. This will give you a complete understanding of how the code works.

---

### Script with Comments and Outputs

```python
# Import the relevant modules
import requests  # For making HTTP requests
import json      # For handling JSON data

# Step 1: Define the base URL for the iTunes API
base_site = "https://itunes.apple.com/search"  # API endpoint for search queries

# Step 2: Make a GET request with parameters
r = requests.get(base_site, params={"term": "the beatles", "country": "us", "limit": 200})
print(r.status_code)  # Check if the request was successful
# Output: 200 (indicating the request was successful)

# Step 3: Convert the response to JSON format
info = r.json()

# Step 4: Inspect the entire JSON response
print(json.dumps(info, indent=4))  # Pretty print the JSON for readability
# Sample Output (abbreviated for clarity):
# {
#     "resultCount": 200,
#     "results": [
#         {
#             "wrapperType": "track",
#             "kind": "song",
#             "artistId": 136975,
#             "collectionId": 1441133109,
#             "trackId": 1441133114,
#             "artistName": "The Beatles",
#             "collectionName": "1 (2015 Version)",
#             "trackName": "Hey Jude",
#             "releaseDate": "1968-08-30T12:00:00Z",
#             ...
#         },
#         ...
#     ]
# }

# Step 5: Inspect only the first result in detail
print(json.dumps(info['results'][0], indent=4))  # Print details of the first result
# Sample Output (abbreviated):
# {
#     "wrapperType": "track",
#     "kind": "song",
#     "artistId": 136975,
#     "collectionId": 1441133109,
#     "trackId": 1441133114,
#     "artistName": "The Beatles",
#     "collectionName": "1 (2015 Version)",
#     "trackName": "Hey Jude",
#     "releaseDate": "1968-08-30T12:00:00Z",
#     ...
# }

# Step 6: Extract the track name of the first result
print(info['results'][0]['trackName'])  # Extract the track name
# Output: Hey Jude

# Step 7: Extract the release date of the first result
print(info['results'][0]['releaseDate'])  # Extract the release date
# Output: 1968-08-30T12:00:00Z

# Step 8: Iterate through all results and print their track names
print("\nTrack Names:")
for result in info['results']:
    print(result['trackName'])
# Sample Output:
# Hey Jude
# Let It Be
# Yesterday
# ...

# Step 9: Iterate through all results and print their release dates
print("\nRelease Dates:")
for result in info['results']:
    print(result['releaseDate'])
# Sample Output:
# 1968-08-30T12:00:00Z
# 1970-03-06T12:00:00Z
# 1965-08-06T12:00:00Z
# ...
```

```python
#!/usr/bin/env python
# coding: utf-8

# Importing the required libraries
import requests  # For making HTTP requests
import json      # For handling JSON data
import pandas as pd  # For data structuring and exporting

# ----------- Currency Exchange Rates Example -----------

# Define the base URL for the API
base_url = "https://api.exchangeratesapi.io/latest"

# Make a GET request to fetch the latest exchange rates
response = requests.get(base_url)
print(f"Status Code: {response.status_code}")  # Verify the response status
# Output: Status Code: 200

# Check if the request was successful
if response.ok:
    # Convert the response to JSON format
    data = response.json()
    print(json.dumps(data, indent=4))  # Pretty-print the JSON data
    # Output: Formatted exchange rate data
else:
    print("Request failed.")

# Extracting specific keys
print(f"Base Currency: {data['base']}")
# Output: Base Currency: EUR

print(f"Exchange Rates: {data['rates']}")
# Output: Exchange Rates: {'USD': ..., 'GBP': ..., ...}

# ----------- iTunes Search API Example -----------

# Define the base URL for the iTunes Search API
itunes_base_site = "https://itunes.apple.com/search"

# Define search parameters
params = {"term": "the beatles", "country": "us", "limit": 200}

# Make a GET request with parameters
r = requests.get(itunes_base_site, params=params)
print(f"Status Code: {r.status_code}")  # Verify the response status
# Output: Status Code: 200

# Parse the JSON response
info = r.json()
print(json.dumps(info, indent=4))  # Inspect the full JSON response
# Output: Full JSON response containing song data

# Inspect the first result in detail
first_song = info['results'][0]
print("\nFirst Song Details:")
print(f"Track Name: {first_song['trackName']}")
print(f"Release Date: {first_song['releaseDate']}")
# Output:
# Track Name: Hey Jude
# Release Date: 1968-08-30T12:00:00Z

# Iterate through all results to extract track names and release dates
print("\nAll Track Names and Release Dates:")
for result in info['results']:
    print(f"{result['trackName']} - {result['releaseDate']}")
# Output: Prints all track names and their release dates

# ----------- Structuring and Exporting Data -----------

# Create a pandas DataFrame from the results
songs_df = pd.DataFrame(info["results"])
print("\nDataFrame Sample:")
print(songs_df.head())  # Display the first few rows of the DataFrame
# Output: Table-like display of song information

# Export the DataFrame to a CSV file
songs_df.to_csv("songs_info.csv", index=False)
print("Data exported to 'songs_info.csv'.")
# Output: CSV file saved in the current directory

```

```python
#!/usr/bin/env python
# coding: utf-8

# # Pagination Example: Job Listings on GitHub Jobs API

# Importing the necessary libraries
import requests  # For sending HTTP requests
import json      # For working with JSON data

# Define the base URL for the API
base_site = "https://jobs.github.com/positions.json"

# --- Submitting a GET Request with Filters ---

# Submit a request for "data science" jobs in "Los Angeles"
r = requests.get(base_site, params={"description": "data science", "location": "los angeles"})
print(f"Status Code: {r.status_code}")
# Output: Status Code: 200

# Inspect the response data
data = r.json()
print(f"Number of Jobs Found: {len(data)}")
# Example Output: Number of Jobs Found: 12 (varies depending on available jobs)

# Print details of the first job
if len(data) > 0:
    print(json.dumps(data[0], indent=4))
# Example Output: Pretty-printed details of the first job

# --- Searching for All Jobs (No Filters) ---

# Submit a request without filters to get all jobs
r = requests.get(base_site)
print(f"Status Code: {r.status_code}")
# Output: Status Code: 200

# Inspect all jobs on the first page
data = r.json()
print(f"Number of Jobs on Page 1: {len(data)}")
# Example Output: Number of Jobs on Page 1: 50 (the API paginates results to 50 per page)

# --- Fetching the Next Page ---

# Add the "page" parameter to fetch results from page 2
r = requests.get(base_site, params={"page": 2})
print(f"Status Code for Page 2: {r.status_code}")
# Output: Status Code for Page 2: 200

# Inspect jobs on page 2
data = r.json()
print(f"Number of Jobs on Page 2: {len(data)}")
# Example Output: Number of Jobs on Page 2: 50 (if available)

# --- Handling Non-Existing Pages ---

# Make a request to a non-existing page (e.g., page 10)
r = requests.get(base_site, params={"page": 10})
print(f"Status Code for Page 10: {r.status_code}")
# Output: Status Code for Page 10: 200 (if API is working) or 204 (if no results)

# Check the response for a non-existing page
data = r.json()
print(f"Response for Page 10: {data}")
# Example Output: Response for Page 10: []

# --- Extracting Results from Multiple Pages ---

# Initialize an empty list to store results
results = []

# Iterate through the first 5 pages
for i in range(5):
    r = requests.get(base_site, params={"page": i + 1})
    
    # Break the loop if no results are found
    if len(r.json()) == 0:
        print(f"No more jobs found on Page {i + 1}. Stopping.")
        break
    else:
        # Extend the results list with jobs from the current page
        results.extend(r.json())

print(f"Total Number of Jobs Found Across Pages: {len(results)}")
# Example Output: Total Number of Jobs Found Across Pages: 250 (depends on available jobs)

```

Here's the complete example with **inline comments**, and instructions to proceed. Since this involves sensitive information, be sure to keep your **API keys private**.

---

### Complete Script with Outputs and Explanation

```python
#!/usr/bin/env python
# coding: utf-8

# # API Requiring Registration - POST Request Example

# Importing necessary libraries
import requests  # For sending HTTP requests
import json      # For working with JSON data

# --- Step 1: API Setup and Configuration ---

# Replace these placeholders with your own credentials from Edamam
APP_ID = "your_API_ID_here"  # Replace with your Edamam API ID
APP_KEY = "your_API_key_here"  # Replace with your Edamam API Key

# Define the API endpoint URL
api_endpoint = "https://api.edamam.com/api/nutrition-details"

# Construct the URL with credentials
url = f"{api_endpoint}?app_id={APP_ID}&app_key={APP_KEY}"
print(f"Constructed URL: {url}")
# Output: Constructed URL: https://api.edamam.com/api/nutrition-details?app_id=your_API_ID_here&app_key=your_API_key_here

# --- Step 2: Define the Data for the POST Request ---

# Sample recipe data to analyze
recipe_data = {
    "title": "Fresh Salad",
    "ingr": [
        "1 cup lettuce",
        "1/2 cup tomatoes",
        "1/4 cup shredded carrots",
        "1 tbsp olive oil",
        "1 tsp vinegar"
    ]
}

# Pretty-print the recipe data
print("Recipe Data Sent:")
print(json.dumps(recipe_data, indent=4))
# Output:
# Recipe Data Sent:
# {
#     "title": "Fresh Salad",
#     "ingr": [
#         "1 cup lettuce",
#         "1/2 cup tomatoes",
#         "1/4 cup shredded carrots",
#         "1 tbsp olive oil",
#         "1 tsp vinegar"
#     ]
# }

# --- Step 3: Make the POST Request ---

# Send the POST request with the JSON payload
response = requests.post(url, json=recipe_data)

# --- Step 4: Check for Errors ---

# Check if the request was successful
if not response.ok:
    print(f"Error {response.status_code}: {response.text}")
    # Output for invalid API credentials or errors
    # Error 401: Invalid app_id or app_key
else:
    # Parse the response JSON
    nutrition_data = response.json()

    # --- Step 5: Inspect the Response ---

    # Pretty-print the entire nutrition analysis response
    print("Nutrition Analysis Response:")
    print(json.dumps(nutrition_data, indent=4))
    # Output: Detailed nutritional analysis (JSON format)

    # --- Step 6: Extract Specific Fields ---

    # Example: Extract calories and nutrients
    calories = nutrition_data.get("calories", "N/A")
    print(f"Total Calories: {calories}")
    # Output: Total Calories: 150 (varies based on recipe)

    # Example: Extracting protein content
    protein = next((nutrient for nutrient in nutrition_data.get("totalNutrients", {}).values() 
                   if nutrient.get("label") == "Protein"), {})
    protein_quantity = protein.get("quantity", "N/A")
    protein_unit = protein.get("unit", "")
    print(f"Protein Content: {protein_quantity} {protein_unit}")
    # Output: Protein Content: 3.5 g
```

---

### Key Features Demonstrated in the Script

1. **API Authentication**:
    
    * Requiring an `APP_ID` and `APP_KEY` to access the API.
        
2. **POST Request**:
    
    * Sending data (a recipe) as a JSON payload to the API.
        
3. **Response Parsing**:
    
    * Handling and inspecting the JSON response for relevant nutritional details.
        
4. **Error Handling**:
    
    * Checking for invalid credentials or other errors in the request.
        

---

### Here is the script rewritten with **inline comments** and **sample outputs** clearly displayed:

```python
#!/usr/bin/env python
# coding: utf-8

# Import required libraries
import requests  # For sending HTTP requests
import json  # For handling JSON data
import pandas as pd  # For structuring and exporting data

# Section 1: Exchange Rates API (Public API)
# -------------------------------------------

# Define the base URL for the Exchange Rates API
base_url = "https://api.exchangeratesapi.io/latest"

# 1. Sending a GET request to fetch the latest exchange rates
response = requests.get(base_url)
print(f"Response Status: {response.status_code}")  # Sample Output: Response Status: 200
print(f"Response OK: {response.ok}")  # Sample Output: Response OK: True

# 2. Display the JSON response
data = response.json()
print(json.dumps(data, indent=4))
# Sample Output:
# {
#     "rates": {
#         "USD": 1.1,
#         "GBP": 0.85,
#         ...
#     },
#     "base": "EUR",
#     "date": "2024-11-17"
# }

# 3. Fetch specific rates for USD and GBP
param_url = base_url + "?symbols=USD,GBP"
response = requests.get(param_url)
print(json.dumps(response.json(), indent=4))
# Sample Output:
# {
#     "rates": {
#         "USD": 1.1,
#         "GBP": 0.85
#     },
#     "base": "EUR",
#     "date": "2024-11-17"
# }

# 4. Change the base currency to USD
param_url = base_url + "?symbols=GBP&base=USD"
response = requests.get(param_url)
usd_to_gbp = response.json()['rates']['GBP']
print(f"1 USD is equal to {usd_to_gbp} GBP")
# Sample Output: 1 USD is equal to 0.85 GBP

# Section 2: GitHub Jobs API (Public API with Pagination)
# -------------------------------------------------------

# Define the base URL for the GitHub Jobs API
github_base_url = "https://jobs.github.com/positions.json"

# 1. Fetch the first page of job listings
response = requests.get(github_base_url, params={"description": "data science", "location": "los angeles"})
jobs = response.json()
print(f"Jobs Found on Page 1: {len(jobs)}")  # Sample Output: Jobs Found on Page 1: 50

# 2. Fetch multiple pages of results
all_jobs = []
for page in range(1, 6):  # Fetch first 5 pages
    response = requests.get(github_base_url, params={"page": page})
    page_jobs = response.json()
    if not page_jobs:  # Stop if no more results
        break
    all_jobs.extend(page_jobs)
print(f"Total Jobs Found: {len(all_jobs)}")  # Sample Output: Total Jobs Found: 200

# Section 3: Edamam Nutrition API (POST Request with API Key)
# -----------------------------------------------------------

# API credentials (replace with your own)
APP_ID = "your_app_id_here"
APP_KEY = "your_app_key_here"

# Define the API endpoint and the request URL
edamam_url = f"https://api.edamam.com/api/nutrition-details?app_id={APP_ID}&app_key={APP_KEY}"

# Define headers and payload for the POST request
headers = {'Content-Type': 'application/json'}
recipe = {
    "title": "Cappuccino",
    "ingr": ["18g ground espresso", "150ml milk"]
}

# Send the POST request
response = requests.post(edamam_url, headers=headers, json=recipe)
print(f"Response Status: {response.status_code}")  # Sample Output: Response Status: 200

# Display the JSON response
nutrition_data = response.json()
print(json.dumps(nutrition_data["totalNutrients"], indent=4))
# Sample Output:
# {
#     "ENERC_KCAL": {
#         "label": "Energy",
#         "quantity": 110,
#         "unit": "kcal"
#     },
#     "SUGAR": {
#         "label": "Sugars",
#         "quantity": 8.5,
#         "unit": "g"
#     },
#     ...
# }

# Export nutrient data to a CSV
nutrients_df = pd.DataFrame(nutrition_data["totalNutrients"]).transpose()
nutrients_df.to_csv("Cappuccino_nutrients.csv")
print("Nutrients data exported to Cappuccino_nutrients.csv")  
# Sample Output: Export success message
```

### Features:

1. **Inline Comments**: Explains the purpose of each block.
    
2. **Sample Outputs**: Shows exactly what each step will print to the console.
    
3. **Scenarios Covered**:
    
    * Fetching and manipulating API data with GET requests.
        
    * Working with paginated APIs (e.g., GitHub Jobs API).
        
    * Using POST requests with payloads (e.g., Edamam Nutrition API).
        
4. **File Export**: Demonstrates how to save API data to a `.csv` file.