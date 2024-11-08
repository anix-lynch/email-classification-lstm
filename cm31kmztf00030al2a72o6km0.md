---
title: "How to manually create a table of contents (TOC) in Markdown for a GitHub README:"
seoTitle: "Creating a TOC in Markdown in Github
"
seoDescription: "Creating a TOC in Markdown  in Github

"
datePublished: Sun Nov 03 2024 12:31:28 GMT+0000 (Coordinated Universal Time)
cuid: cm31kmztf00030al2a72o6km0
slug: how-to-manually-create-a-table-of-contents-toc-in-markdown-for-a-github-readme
tags: github

---

## Creating a TOC in Markdown

1. Write your README content using Markdown syntax, including headings (e.g., ##, ###) for the sections you want to include in the TOC.
    
2. At the top of your README, add a "Table of Contents" section where you will put the TOC.
    
3. For each heading you want to include in the TOC, create a Markdown link using this syntax:
    
    ```python
    [Text to display](#link-destination)
    ```
    
    * Replace "Text to display" with the actual section title.
        
    * Replace "link-destination" with the section title in lowercase, replacing spaces with hyphens.
        
4. If you have nested headings, you can show the hierarchy by using indentation and different bullet point styles in your TOC links:
    
    ```python
    - [Top-level heading](#top-level-heading)
      - [Nested heading](#nested-heading) 
        - [Deeply nested heading](#deeply-nested-heading)
    ```
    
5. Make sure the link destinations in your TOC exactly match the headings in your content (case-sensitive).
    

Here's an example of what the Markdown TOC might look like:

```python
## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Running the app](#running-the-app)
  - [Configuration](#configuration) 
- [Contributing](#contributing)
- [License](#license)

## Installation
Content for the Installation section...

## Usage 
Content for the Usage section...

### Running the app
Content for Running the app...

### Configuration
Content for Configuration...

## Contributing
Content for the Contributing section...

## License
Content for the License section...
```

Some key things to note:

* GitHub will automatically make your TOC links clickable when rendering the Markdown.
    
* The link destinations are case-sensitive and must exactly match the heading text.
    
* Spaces in headings are replaced with hyphens in the link destinations.
    
* Indentation is used to show nesting of headings in the TOC.
    

While this process is manual, it allows you full control over what goes in your TOC and how it's formatted. Some Markdown editors and tools can help automate TOC generation as well.

Alternatively, GitHub now has built-in support for automatically generating a TOC from your Markdown headings - it will appear at the top of the rendered Markdown when there are at least 2 headings\[8\]\[7\]. So you may not need a manual TOC unless you want more control over its content and formatting.