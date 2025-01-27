---
title: "Tableau UI"
seoTitle: "Tableau UI"
seoDescription: "Tableau UI"
datePublished: Mon Jan 27 2025 00:08:47 GMT+0000 (Coordinated Universal Time)
cuid: cm6eajb6l000108l7h3fcbek0
slug: tableau-ui
tags: interface, tableau

---

## Main Workspace

```python
+--------------------------------------------------------------------------------+
|  File  Edit  Worksheet  Dashboard  Story  Analysis  Map  [⚙️]                    |
+--------------------------------------------------------------------------------+
|                                                                                 |
| Data Pane        Workspace                              Cards                   |
| ┌──────────┐    ┌─────────────────────────────────┐    ┌──────────────┐       |
| │ Tables   │    │                                 │    │ Filters      │       |
| │ ├── Sales│    │                                 │    │ ┌──────────┐ │       |
| │ └── Cust │    │                                 │    │ │Year      │ │       |
| │          │    │                                 │    │ │Region    │ │       |
| │ Fields   │    │                                 │    │ └──────────┘ │       |
| │ ├── Dim  │    │                                 │    │              │       |
| │ └── Meas │    │                                 │    │ Marks        │       |
| └──────────┘    └─────────────────────────────────┘    │ ○ Color     │       |
|                                                        │ □ Size      │       |
|                                                        │ ▲ Label     │       |
|                                                        └──────────────┘       |
+--------------------------------------------------------------------------------+
```

## Data Connection Flow

```python
       ┌───────────┐
       │  Connect  │
       └─────┬─────┘
             │
     ┌───────┴───────┐
     │  Data Source  │
     └───────┬───────┘
             │
    ┌────────┴────────┐
    │  Live or Extract│
    └────────┬────────┘
             │
     ┌───────┴───────┐
     │    Sheet      │
     └───────────────┘
```

## Dragging Fields to Build Viz

```python
Fields Pane                    Shelves
┌──────────┐                 ┌─────────────────┐
│ Sales    │    drag to     │ Columns         │
│ Date     │   ───────►     │                 │
│ Region   │                │ Rows            │
└──────────┘                └─────────────────┘

                    Results in
                       ▼
              ┌─────────────────┐
              │     Chart       │
              │   ▲             │
              │   │             │
              │   └─────►       │
              └─────────────────┘
```

## Show Me Panel (Chart Types)

```python
+-------------------------+
|     Show Me            |
|  ┌─────┐ ┌─────┐ ┌────┐|
|  │  ▊  │ │  ○  │ │ ▤  ||
|  │  ▊  │ │  ○  │ │ ▤  ||
|  └─────┘ └─────┘ └────┘|
|  Bar     Scatter  Heat |
|  ┌─────┐ ┌─────┐ ┌────┐|
|  │  ─  │ │  ◯  │ │ ╱  ||
|  │  ─  │ │  ◯  │ │ ╱  ||
|  └─────┘ └─────┘ └────┘|
|  Line    Pie     Area  |
+-------------------------+
```

## Dashboard Layout

```python
+----------------------------------------+
| Dashboard                              |
|┌──────────┐  ┌─────────────────────┐  |
|| Filter   |  |       Chart 1       |  |
|| Panel    |  |                     |  |
||          |  |                     |  |
|└──────────┘  └─────────────────────┘  |
|              ┌─────────────────────┐  |
|              |       Chart 2       |  |
|              |                     |  |
|              |                     |  |
|              └─────────────────────┘  |
+----------------------------------------+
```

## Common Workflows

1. Basic Visualization:
    

```python
Connect Data → Drag Dimension to Rows → Drag Measure to Columns → Select Chart Type
```

2. Creating Dashboard:
    

```python
New Dashboard → Drag Sheets → Add Filters → Add Actions → Format
```

3. Data Blending:
    

```python
Primary Data → Edit Blend Relationships → Add Secondary → Define Links
```

## Key UI Elements

### Shelves and Cards

```python
+------------------------------------------+
| Columns Shelf     [Field] [Field]        |
+------------------------------------------+
| Rows Shelf        [Field] [Field]        |
+------------------------------------------+
| Filters Card                             |
| ├── [Filter 1]                          |
| └── [Filter 2]                          |
+------------------------------------------+
| Marks Card                               |
| ├── Color        [Field]                |
| ├── Size         [Field]                |
| ├── Label        [Field]                |
| └── Detail       [Field]                |
+------------------------------------------+
```

### Calculation Editor

```python
+------------------------------------------+
| Create Calculated Field          [✓] [✗] |
+------------------------------------------+
| Name: [New Calculation]                  |
|                                          |
| Formula:                                 |
| ┌────────────────────────────────┐      |
| │SUM([Sales]) / SUM([Quantity])  │      |
| └────────────────────────────────┘      |
|                                          |
| Functions  Fields  Parameters            |
| ├── String  ├── Dim   └── [Param1]      |
| ├── Date    └── Meas                    |
| └── Number                              |
+------------------------------------------+
```

## Navigation Tips:

1. Top Menu Bar: Main functions and settings
    
    * File: New workbook, save, export
        
    * Worksheet: Create new views
        
    * Dashboard: Combine views
        
    * Analysis: Advanced analytics
        
2. Left Side: Data and fields
    
    * Connections
        
    * Fields list
        
    * Analytics objects
        
3. Right Side: Properties and formatting
    
    * Marks card
        
    * Filters
        
    * Field properties
        
4. Bottom: Sheets navigation
    
    * Worksheets
        
    * Dashboards
        
    * Stories
        

Remember:

* Drag and drop is your best friend
    
* Right-click menus have contextual options
    
* The Show Me button suggests chart types
    
* Double-click fields for quick analysis
    

This guide covers the core UI elements of Tableau Desktop. The actual software has more features, but mastering these elements will give you a strong foundation!