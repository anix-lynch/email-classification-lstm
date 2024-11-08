---
title: "20 Bokeh concepts with Before-and-After Examples"
seoTitle: "20 Bokeh concepts with Before-and-After Examples"
seoDescription: "20 Bokeh concepts with Before-and-After Examples"
datePublished: Thu Oct 03 2024 14:30:35 GMT+0000 (Coordinated Universal Time)
cuid: cm1te8rpv000p0amo5e3k36z7
slug: 20-bokeh-concepts-with-before-and-after-examples
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1728440991574/a32f1e5a-909f-4f53-9247-5435d1249879.png
tags: python, data-science, interactive, visualization, bokeh

---

### 1\. **Basic Line Plot (bokeh.plotting.figure.line)** 📈

**Boilerplate Code**:

```python
from bokeh.plotting import figure, show
```

**Use Case**: Create a **line plot** to display trends over time or continuous data. 📈

**Goal**: Visualize a series of data points connected by lines. 🎯

**Sample Code**:

```python
# Create a figure
p = figure(title="Line Plot Example", x_axis_label='X', y_axis_label='Y')

# Add a line renderer
p.line([1, 2, 3, 4], [1, 4, 2, 3])

# Show the plot
show(p)
```

**Before Example**: You have data points but no way to visualize their trend. 🤔

```python
Data: x = [1, 2, 3, 4], y = [1, 4, 2, 3]
```

**After Example**: With **line plot**, you clearly see the trend! 📈

```python
Output: A simple line plot.
```

**Challenge**: 🌟 Try adding a second line with a different color to compare two trends.

---

### 2\. **Scatter Plot (**[**bokeh.plotting.figure.circle**](http://bokeh.plotting.figure.circle)**)** 🔵

**Boilerplate Code**:

```python
from bokeh.plotting import figure, show
```

**Use Case**: Create a **scatter plot** to show the relationship between two variables. 🔵

**Goal**: Plot individual data points to visualize correlations or patterns. 🎯

**Sample Code**:

```python
# Create a figure
p = figure(title="Scatter Plot Example", x_axis_label='X', y_axis_label='Y')

# Add circle markers
p.circle([1, 2, 3, 4], [4, 3, 2, 1], size=10, color="blue")

# Show the plot
show(p)
```

**Before Example**: You have two sets of data but no way to <mark> visualize their relationship.</mark> 🤔

```python
Data: x = [1, 2, 3, 4], y = [4, 3, 2, 1]
```

**After Example**: With **scatter plot**, you can see the relationship between the variables! 🔵

```python
Output: A scatter plot showing individual data points.
```

**Challenge**: 🌟 Try using different marker shapes like `square` or `triangle`.

---

### 3\. **Bar Plot (**[**bokeh.charts.Bar**](http://bokeh.charts.Bar)**)** 📊

**Boilerplate Code**:

```python
from bokeh.plotting import figure, show
```

**Use Case**: Create a **bar plot** to compare categories. 📊

**Goal**: Visualize categorical data using bars to compare values. 🎯

**Sample Code**:

```python
# Create a figure
p = figure(x_range=["A", "B", "C"], title="Bar Plot Example", y_axis_label="Values")

# Add vertical bars
p.vbar(x=["A", "B", "C"], top=[10, 20, 30], width=0.5)

# Show the plot
show(p)
```

**Before Example**: You have categorical data but no way to compare them visually. 🤔

```python
Data: Categories = ["A", "B", "C"], Values = [10, 20, 30]
```

**After Example**: With **bar plot**, you can compare the categories visually! 📊

```python
Output: A bar plot showing comparisons across categories.
```

**Challenge**: 🌟 Try adding more categories and stacking bars with different colors.

---

### 4\. **Interactive Tools (bokeh.models)** 🛠️

**Boilerplate Code**:

```python
from bokeh.models import HoverTool
from bokeh.plotting import figure, show
```

**Use Case**: Add **interactive tools** like hover and zoom to your plots. 🛠️

**Goal**: Make the plot interactive to display more details on hover. 🎯

**Sample Code**:

```python
# Create a figure
p = figure(tools="pan,box_zoom,reset")

# Add hover tool
hover = HoverTool(tooltips=[("X", "@x"), ("Y", "@y")])
p.add_tools(hover)

# Add line plot
p.line([1, 2, 3, 4], [4, 3, 2, 1])

# Show the plot
show(p)
```

**Before Example**: You have a static plot without any interaction. 🤔

```python
Plot: A simple line plot.
```

**After Example**: With **interactive tools**, the plot becomes more engaging with hover and zoom! 🛠️

```python
Output: A line plot with zoom and hover capabilities.
```

**Challenge**: 🌟 Try adding more advanced interactive tools like tap or box selection.

---

### 5\. **Linked Plots (bokeh.plotting)** 🔗

**Boilerplate Code**:

```python
from bokeh.plotting import figure, show
from bokeh.layouts import column
```

**Use Case**: Create **linked plots** where interactions on one plot affect the other. 🔗

**Goal**: Link multiple plots together for synchronized zoom or pan. 🎯

**Sample Code**:

```python
# Create two figures with shared ranges
p1 = figure(x_range=(0, 10), y_range=(0, 10), title="Linked Plot 1")
p2 = figure(x_range=p1.x_range, y_range=p1.y_range, title="Linked Plot 2")

# Add scatter to both plots
p1.circle([1, 2, 3], [4, 5, 6], size=10)
p2.circle([2, 4, 6], [3, 6, 9], size=10)

# Show the linked plots
show(column(p1, p2))
```

**Before Example**: You have two separate plots but want them to interact. 🤔

```python
Two unlinked scatter plots.
```

**After Example**: With **linked plots**, both plots zoom and pan together! 🔗

```python
Output: Two scatter plots linked for interaction.
```

**Challenge**: 🌟 Try linking only one axis or using linked brushing to highlight points across both plots.

---

### 6\. **Categorical Heatmap (bokeh.plotting)** 🔥

**Boilerplate Code**:

```python
from bokeh.plotting import figure, show
from bokeh.transform import linear_cmap
from bokeh.models import ColorBar, ColumnDataSource
```

**Use Case**: Create a **heatmap** to display data values as color intensity. 🔥

**Goal**: Represent numerical data using color. 🎯

**Sample Code**:

```python
# Create data source
data = dict(x=["A", "B", "C"], y=["D", "E", "F"], values=[1, 2, 3])
source = ColumnDataSource(data)

# Create figure
p = figure(x_range=["A", "B", "C"], y_range=["D", "E", "F"], title="Heatmap")

# Add rectangles (heatmap cells)
p.rect(x="x", y="y", width=1, height=1, source=source, 
       fill_color=linear_cmap("values", "Viridis256", 1, 3), line_color=None)

# Add color bar
color_bar = ColorBar(color_mapper=linear_cmap("values", "Viridis256", 1, 3).color_mapper, location=(0, 0))
p.add_layout(color_bar, 'right')

# Show the plot
show(p)
```

**Before Example**: You have categorical data with values but no way to visualize their intensity. 🤔

```python
Data: x = ["A", "B", "C"], y = ["D", "E", "F"], values = [1, 2, 3]
```

**After Example**: With **heatmap**, you can visualize the intensity using colors! 🔥

```python
Output: A heatmap showing data intensity with colors.
```

**Challenge**: 🌟 Try using a different colormap or adding tooltips to display the exact value on hover.

---

### 7\. **Multi-Line Plot (bokeh.plotting.figure.multi\_line)** 🖍️

**Boilerplate Code**:

```python
from bokeh.plotting import figure, show
```

**Use Case**: Create a **multi-line plot** to visualize multiple data series on the same plot. 🖍️

**Goal**: Display several lines for comparison on one plot. 🎯

**Sample Code**:

```python
# Create a figure
p = figure(title="Multi-Line Plot Example", x_axis_label='X', y_axis_label='Y')

# Add multiple lines
p.multi_line(xs=[[1, 2, 3], [1, 2, 3]], ys=[[1, 4, 9], [2, 3, 4]], color=["blue",

 "green"])

# Show the plot
show(p)
```

**Before Example**: You have multiple data series but no way to plot them together. 🤔

```python
Data: xs = [[1, 2, 3], [1, 2, 3]], ys = [[1, 4, 9], [2, 3, 4]]
```

**After Example**: With **multi-line plot**, all the data series are visualized together! 🖍️

```python
Output: A plot with multiple lines representing different data series.
```

**Challenge**: 🌟 Try adding a legend to identify each line and use `line_dash` to differentiate them.

---

### 8\. **Patches Plot (bokeh.plotting.figure.patches)** 🖌️

**Boilerplate Code**:

```python
from bokeh.plotting import figure, show
```

**Use Case**: Create a **patches plot** to visualize areas or regions on the plot. 🖌️

**Goal**: Represent shapes or regions using connected points. 🎯

**Sample Code**:

```python
# Create a figure
p = figure(title="Patches Plot Example", x_axis_label='X', y_axis_label='Y')

# Add patches (areas)
p.patches(xs=[[1, 2, 3], [3, 4, 5]], ys=[[1, 4, 1], [2, 3, 2]], color=["blue", "green"], alpha=0.6)

# Show the plot
show(p)
```

**Before Example**: You have data representing regions but no way to visualize them. 🤔

```python
Data: xs = [[1, 2, 3], [3, 4, 5]], ys = [[1, 4, 1], [2, 3, 2]]
```

**After Example**: With **patches**, the regions are visualized as shaded areas! 🖌️

```python
Output: A plot showing colored patches representing different areas.
```

**Challenge**: 🌟 Try adding tooltips to each patch to display more information on hover.

---

### 9\. **Streaming Data (bokeh.models)** 📡

**Boilerplate Code**:

```python
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource
```

**Use Case**: Stream **live data** into your plots in real time. 📡

**Goal**: Continuously update plots with new data points in real time. 🎯

**Sample Code**:

```python
from random import randint

# Create data source
source = ColumnDataSource(data=dict(x=[0], y=[randint(0, 10)]))

# Create figure
p = figure(title="Streaming Data Example")
p.line('x', 'y', source=source)

# Update function
def update():
    new_data = dict(x=[source.data['x'][-1] + 1], y=[randint(0, 10)])
    source.stream(new_data)

# Add periodic callback
curdoc().add_periodic_callback(update, 1000)

# Show plot in Bokeh server
show(p)
```

**Before Example**: You want to visualize real-time data but the plot is static. 🤔

```python
Data: Randomly generated in real time.
```

**After Example**: With **streaming data**, the plot updates continuously! 📡

```python
Output: A plot that updates in real time with new data points.
```

**Challenge**: 🌟 Try plotting multiple streams of data simultaneously on the same plot.

---

### 10\. **Color Mappers (bokeh.transform)** 🎨

**Boilerplate Code**:

```python
from bokeh.transform import linear_cmap
```

**Use Case**: Use **color mappers** to map numerical data to colors dynamically. 🎨

**Goal**: Dynamically assign colors to data points based on a numerical range. 🎯

**Sample Code**:

```python
from bokeh.plotting import figure, show
from bokeh.transform import linear_cmap

# Create figure
p = figure(title="Color Mapper Example", x_axis_label='X', y_axis_label='Y')

# Add circle markers with dynamic color mapping
mapper = linear_cmap(field_name='y', palette="Viridis256", low=0, high=10)
p.circle([1, 2, 3, 4], [4, 3, 2, 1], color=mapper, size=10)

# Show the plot
show(p)
```

**Before Example**: You have numerical data but no way to dynamically color-code it. 🤔

```python
Data: y = [4, 3, 2, 1]
```

**After Example**: With **color mappers**, the data points are colored based on their value! 🎨

```python
Output: A scatter plot with dynamic color mapping based on the data.
```

**Challenge**: 🌟 Try using logarithmic or categorical color mappers for different types of data.

---

### 15\. **Box Plot (bokeh.models.BoxAnnotation)** 📦 (continued)

**Before Example**:  
You have numerical data but no way to summarize the range and distribution. 🤔

```python
Data: Various numerical values.
```

**After Example**:  
With **Box Plot**, you can summarize the data's range and outliers! 📦

```python
Output: A box annotation showing the distribution's middle 50% and outliers.
```

**Challenge**: 🌟 Try adding whiskers to show outliers or adding more boxes to compare multiple data ranges.

---

### 16\. **Time Series Plot (bokeh.plotting.figure.line)** 🕒

**Boilerplate Code**:

```python
from bokeh.plotting import figure, show
from datetime import datetime
```

**Use Case**: Create a **time series plot** to visualize trends over time. 🕒

**Goal**: Represent changes in data over time with a continuous line. 🎯

**Sample Code**:

```python
# Sample data
dates = [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]
values = [10, 20, 30]

# Create figure
p = figure(title="Time Series Example", x_axis_type="datetime")

# Add line plot
p.line(dates, values)

# Show the plot
show(p)
```

**Before Example**:  
You have time-based data but no way to display the changes over time. 🤔

```python
Data: Dates = [Jan 1, Jan 2, Jan 3], Values = [10, 20, 30]
```

**After Example**:  
With **time series plot**, the trend over time becomes clear! 🕒

```python
Output: A time series plot with dates on the x-axis.
```

**Challenge**: 🌟 Try adding `circle` markers to highlight data points along the time series.

---

### 17\. **Annotations (bokeh.models.Label)** 📝

**Boilerplate Code**:

```python
from bokeh.models import Label
from bokeh.plotting import figure, show
```

**Use Case**: Add **annotations** to your plot to explain or label data points. 📝

**Goal**: Provide additional information or clarify specific points on the plot. 🎯

**Sample Code**:

```python
# Create figure
p = figure(title="Annotation Example")

# Add a line
p.line([1, 2, 3], [4, 5, 6])

# Add a label annotation
label = Label(x=2, y=5, text="Important Point", text_font_size="10pt")
p.add_layout(label)

# Show the plot
show(p)
```

**Before Example**:  
You have key data points but no way to emphasize or explain them. 🤔

```python
Data: x = [1, 2, 3], y = [4, 5, 6]
```

**After Example**:  
With **annotations**, you can clarify important points! 📝

```python
Output: A plot with a label added to an important point.
```

**Challenge**: 🌟 Try using different annotation types, such as arrows or spans.

---

### 18\. **Geographical Plot (bokeh.tile\_providers)** 🗺️

**Boilerplate Code**:

```python
from bokeh.plotting import figure, show
from bokeh.tile_providers import get_provider, Vendors
```

**Use Case**: Create a **geographical plot** to display data on a map. 🗺️

**Goal**: Overlay data on a map using latitude and longitude coordinates. 🎯

**Sample Code**:

```python
# Create figure
p = figure(title="Geographical Plot Example", x_axis_type="mercator", y_axis_type="mercator")

# Add tile provider for map background
tile_provider = get_provider(Vendors.CARTODBPOSITRON)
p.add_tile(tile_provider)

# Add scatter plot for points on the map
p.circle(x=[0], y=[0], size=10)

# Show the plot
show(p)
```

**Before Example**:  
You have geographic data but no way to plot it on a map. 🤔

```python
Data: Latitude and longitude coordinates.
```

**After Example**:  
With **geographical plot**, your data is displayed on an interactive map! 🗺️

```python
Output: A map with markers at specific coordinates.
```

**Challenge**: 🌟 Try adding more markers or changing the tile provider to experiment with different map styles.

---

### 19\. **Network Graph (bokeh.models.from\_networkx)** 🕸️

**Boilerplate Code**:

```python
import networkx as nx
from bokeh.models import from_networkx
from bokeh.plotting import figure, show
```

**Use Case**: Create a **network graph** to visualize relationships between nodes. 🕸️

**Goal**: Represent data as a graph with nodes and edges. 🎯

**Sample Code**:

```python
# Create a NetworkX graph
G = nx.karate_club_graph()

# Create figure
p = figure(title="Network Graph Example")

# Convert NetworkX graph to Bokeh graph
graph = from_networkx(G, nx.spring_layout, scale=2, center=(0, 0))

# Add the graph to the plot
p.renderers.append(graph)

# Show the plot
show(p)
```

**Before Example**:  
You have relational data but no way to visualize connections. 🤔

```python
Data: Relationships between nodes.
```

**After Example**:  
With **network graph**, the relationships between nodes are clear! 🕸️

```python
Output: A network graph displaying nodes and edges.
```

**Challenge**: 🌟 Try adding attributes to the nodes, such as size or color, based on different properties.

---

### 20\. **Interactive Widgets (bokeh.models.widgets)** 🔧

**Boilerplate Code**:

```python
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import show
from bokeh.layouts import column
```

**Use Case**: Add **interactive widgets** like sliders or text inputs to make your plots interactive. 🔧

**Goal**: Provide controls to users for modifying plot data interactively. 🎯

**Sample Code**:

```python
# Create widgets
slider = Slider(start=0, end=10, value=1, step=0.1, title="Slider")
text_input = TextInput(value="Type here", title="Label:")

# Layout and show
layout = column(slider, text_input)
show(layout)
```

**Before Example**:  
You have a static plot but want users to interact with it. 🤔

```python
Data: A static plot with fixed data.
```

**After Example**:  
With **interactive widgets**, users can control the plot dynamically! 🔧

```python
Output: A layout with a slider and text input for interaction.
```

**Challenge**: 🌟 Try linking the slider to control plot parameters, such as zoom or data points.

### **Bonus Point: What Python libraries are best at interactive and animated plots?**

  
Yes, **Bokeh** is indeed **specialized in interactive visualizations**, but it’s not the only library that can do this. Several other Python libraries can also create **interactive** and even **animated** plots. Let me summarize what each of the popular libraries is good at:

**1\. Bokeh**

* **Specialty**:
    
    * **Interactive** visualizations (hover, zoom, pan, etc.).
        
    * Works well for dashboards and embedding visualizations in web applications.
        
    * Can create **animated visualizations** using **callbacks** and **streaming data**.
        
* **Use Cases**:
    
    * Interactive charts, real-time dashboards, and data streaming.
        

**2\. Plotly**

* **Specialty**:
    
    * **Highly interactive** visualizations with hover, zoom, pan, and click functionalities.
        
    * Supports **animated** visualizations with ease (especially for time series data).
        
    * Works in both web-based dashboards and Jupyter Notebooks.
        
* **Use Cases**:
    
    * Interactive 2D and 3D plots, animated charts, and dashboards.
        

**3\. Dash (built on Plotly)**

* **Specialty**:
    
    * **Web-based dashboards** with highly interactive and customizable elements.
        
    * Can integrate with Plotly’s interactive plots, making it great for dynamic applications.
        
* **Use Cases**:
    
    * Interactive web apps, real-time dashboards, and machine learning model monitoring.
        

**4\. Altair**

* **Specialty**:
    
    * Declarative syntax for creating **interactive** visualizations.
        
    * Simple to create **interactive plots** like zoomable charts and linked selections.
        
* **Use Cases**:
    
    * Interactive charts with minimal code, easy for beginners and fast prototyping.
        

**5\. Holoviews**

* **Specialty**:
    
    * **High-level** interface for creating complex interactive visualizations with minimal code.
        
    * Can work with other libraries like Bokeh, Plotly, and Matplotlib to create interactive plots.
        
* **Use Cases**:
    
    * Data exploration, interactive dashboards, and visual analysis.
        

**6\. Matplotlib (with** `FuncAnimation`)

* **Specialty**:
    
    * Known for static plots, but can create **animated** visualizations using the `FuncAnimation` module.
        
    * Limited interactivity compared to Plotly or Bokeh.
        
* **Use Cases**:
    
    * Static visualizations and basic animations (like animated line charts).
        

**7\. Seaborn**

* **Specialty**:
    
    * **Statistical visualizations**, mainly static.
        
    * While Seaborn focuses more on static plots, it integrates well with Matplotlib, which can add interactivity/animation.
        
* **Use Cases**:
    
    * Quick and beautiful statistical plots.
        

---

**Comparison for Interactive & Animated Visualizations**:

* **Best for Interactivity**:
    
    * **Plotly** (for easy and highly interactive visualizations) and **Bokeh** (for complex interactive plots and dashboards).
        
* **Best for Animation**:
    
    * **Plotly** (simpler to implement) and **Matplotlib** (for more control over the animation).
        
* **Best for Dashboards**:
    
    * **Dash** (if you need a complete dashboard), and **Bokeh** (for embedding interactive elements in apps).
        

---

**Summary**:

* **Bokeh** is great for **interactive and animated visualizations**, but **Plotly** and **Dash** also excel at creating highly interactive and animated visualizations, with **Plotly** being particularly easy to use for animations.
    
* **Altair** and **Holoviews** are also strong options for interactive data exploration.
    

Would you like a specific example of creating an animated or interactive plot in any of these libraries?

---