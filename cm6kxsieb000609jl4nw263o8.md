---
title: "GitHub Automation w/Claude"
seoTitle: " GitHub Automation w/Claude"
seoDescription: " GitHub Automation w/Claude"
datePublished: Fri Jan 31 2025 15:46:25 GMT+0000 (Coordinated Universal Time)
cuid: cm6kxsieb000609jl4nw263o8
slug: github-automation-wclaude
tags: github, claudeai

---

Claude, through **MCP GitHub integration**, can perform **full development workflows** like a software engineer. With a single **English prompt**, it can:

✅ **Create repositories**  
✅ **Push code & update files**  
✅ **Create issues & track bugs**  
✅ **Make new branches**  
✅ **Submit pull requests (PRs) to merge code**

### 🔹 **How It Works**

1. **GitHub API Integration**:
    
    * Claude connects to **GitHub’s API** using a **Personal Access Token (PAT)**.
        
    * The API allows Claude to send commands just like a human would in the GitHub UI.
        
2. **Step-by-Step Example**
    
    * A **single prompt** like:
        
        > *"Create a new repo, add an index.html file, create a feature branch, add some CSS, push changes, and open a pull request."*
        
    * Claude will:
        
        1. **Create a repo** in your GitHub account.
            
        2. **Add an HTML file** (`index.html`).
            
        3. **Make a "feature" branch**.
            
        4. **Edit the HTML file & add CSS**.
            
        5. **Commit & push changes**.
            
        6. **Create a pull request** for review.
            
3. **Live GitHub Execution**
    
    * After approval, changes **appear on GitHub in real-time**.
        
    * You can manually verify **branches, commits, issues, and PRs**.
        

### 🔹 **GitHub API Token Setup**

1. **Create a GitHub PAT**:
    
    * Go to: [**GitHub → Developer Settings → Tokens (classic)**](https://github.com/settings/tokens)
        
    * Click **"Generate new token"**
        
    * Select:
        
        * ✅ `repo` → Read/write access
            
        * ✅ `write:packages` → Modify files
            
        * ✅ `user` → Access profile info
            
    * Click **Generate Token** & Copy it.
        
2. **Update Claude’s** `config.json`:
    
    * Open `CLA-desktop-config.json` in a text editor.
        
    * Add this:
        
        ```json
        {
          "mcp": {
            "github": {
              "api_key": "your_github_token_here"
            }
          }
        }
        ```
        
    * Save & restart Claude Desktop.
        

🛠 **Now Claude can manage repos, code, and issues automatically!**

---