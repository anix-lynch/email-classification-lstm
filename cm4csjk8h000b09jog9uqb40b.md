---
title: "Python Automation #5: ✳️Excel w/pyxlsb, openpyxl, xlwings, pyexcel"
seoTitle: "Python Automation #5: ✳️Excel w/pyxlsb, openpyxl, xlwings, pyexcel"
seoDescription: "Python Automation #5: ✳️Excel w/pyxlsb, openpyxl, xlwings, pyexcel"
datePublished: Fri Dec 06 2024 13:37:55 GMT+0000 (Coordinated Universal Time)
cuid: cm4csjk8h000b09jog9uqb40b
slug: python-automation-5-excel-wpyxlsb-openpyxl-xlwings-pyexcel
tags: excel, openpyxl, xlwings, pyexcel, pyxlsb

---

### **1\. Read** `.xlsb` Files (pyxlsb)

```python
from pyxlsb import open_workbook

with open_workbook('example.xlsb') as wb:
    with wb.get_sheet(1) as sheet:
        for row in sheet.rows():
            print([item.v for item in row])
```

**Sample Output:**

```python
['Header1', 'Header2', 'Header3']
['Data1', 123, 45.67]
['Data2', 456, 89.01]
```

---

### **2\. Read and Write** `.xlsx` Files (openpyxl)

```python
from openpyxl import load_workbook, Workbook

# Reading
wb = load_workbook('example.xlsx')
sheet = wb.active
for row in sheet.iter_rows(values_only=True):
    print(row)

# Writing
new_wb = Workbook()
new_sheet = new_wb.active
new_sheet.append(['Name', 'Age', 'Score'])
new_sheet.append(['Alice', 30, 95])
new_wb.save('output.xlsx')
```

**Sample Output (Reading):**

```python
('Name', 'Age', 'Score')
('Alice', 30, 95)
('Bob', 25, 88)
```

---

### **3\. Directly Interact with Excel App (xlwings)**

```python
import xlwings as xw

app = xw.App(visible=True)
wb = xw.Book('example.xlsx')
sheet = wb.sheets[0]
sheet.range('A1').value = "Hello, Excel!"
wb.save()
wb.close()
app.quit()
```

**Effect:** Opens Excel, writes "Hello, Excel!" in cell A1, and saves the file.

---

### **4\. Batch Process Excel Files (pyexcel)**

```python
import pyexcel as p

# Convert Excel to CSV
p.save_as(file_name='example.xlsx', dest_file_name='output.csv')

# Batch process: read all files in a folder
from pathlib import Path
for file in Path('.').glob('*.xlsx'):
    data = p.get_array(file_name=file)
    print(f"Data from {file}: {data}")
```

**Sample Output:**

```python
Data from example1.xlsx: [['Header1', 'Header2'], [1, 2], [3, 4]]
Data from example2.xlsx: [['A', 'B'], [5, 6], [7, 8]]
```

---

### **5\. Modify Excel Cell Styles (openpyxl)**

```python
from openpyxl import Workbook
from openpyxl.styles import Font

wb = Workbook()
sheet = wb.active
sheet['A1'] = "Bold Text"
sheet['A1'].font = Font(bold=True)
wb.save('styled_output.xlsx')
```

**Effect:** Creates an Excel file where cell A1 has bold text.

---

### **6\. Automate Repetitive Excel Tasks (xlwings)**

```python
import xlwings as xw

app = xw.App(visible=False)
wb = xw.Book('example.xlsx')
sheet = wb.sheets[0]
for i in range(1, 11):
    sheet.range(f'A{i}').value = f"Row {i}"
wb.save()
wb.close()
app.quit()
```

**Effect:** Automatically fills rows 1-10 in column A with "Row 1", "Row 2", etc.

---

### **7\. Export Data to Multiple Formats (pyexcel)**

```python
import pyexcel as p

# Convert Excel to JSON
p.save_as(file_name='example.xlsx', dest_file_name='output.json')

# Convert Excel to CSV
p.save_as(file_name='example.xlsx', dest_file_name='output.csv')
```

**Effect:** Creates `output.json` and `output.csv` files from the Excel file.

---

### **8\. Manipulate Excel Formulas (openpyxl/xlwings)**

```python
from openpyxl import Workbook

wb = Workbook()
sheet = wb.active
sheet['A1'], sheet['A2'] = 10, 20
sheet['A3'] = "=SUM(A1:A2)"  # Adding formula
wb.save('formula_example.xlsx')
```

**Effect:** Adds a formula to calculate the sum of A1 and A2 in A3.

---

### **9\. Read Large Excel Files (pyxlsb)**

```python
from pyxlsb import open_workbook

with open_workbook('large_file.xlsb') as wb:
    with wb.get_sheet(1) as sheet:
        rows = [row for row in sheet.rows()]
        print(rows[:5])  # Print first 5 rows
```

**Sample Output:**

```python
[[Header1, Header2], [Data1, 123], [Data2, 456]]
```

---

### **10\. Edit Pivot Tables (xlwings)**

```python
import xlwings as xw

app = xw.App(visible=True)
wb = xw.Book('pivot_example.xlsx')
sheet = wb.sheets['Pivot']
sheet.range('B1').value = "Updated Value"
wb.save()
wb.close()
app.quit()
```

**Effect:** Updates a value in the pivot table Excel file.

---

### 1**1\. Generate Excel Reports (openpyxl/pyexcel)**

```python
from openpyxl import Workbook

wb = Workbook()
sheet = wb.active
sheet.title = "Report"
sheet.append(["Name", "Age", "Score"])
sheet.append(["Alice", 30, 95])
sheet.append(["Bob", 25, 88])
wb.save("report.xlsx")
```

**Effect:** Generates an `Excel` file `report.xlsx` with a basic table.

---

### **12\. Handle Old Excel Formats (.xls) (pyexcel)**

```python
import pyexcel

data = pyexcel.get_array(file_name="old_format.xls")
print(data)

pyexcel.save_as(array=data, dest_file_name="converted.xlsx")
```

**Effect:** Reads `.xls` files and converts them to `.xlsx`.

**Sample Output (Reading):**

```python
[['Name', 'Age', 'Score'], ['Alice', 30, 95], ['Bob', 25, 88]]
```

---

### **13\. Create and Save Charts (openpyxl)**

```python
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference

wb = Workbook()
sheet = wb.active
data = [["Name", "Score"], ["Alice", 90], ["Bob", 80], ["Charlie", 70]]
for row in data:
    sheet.append(row)

chart = BarChart()
chart_data = Reference(sheet, min_col=2, min_row=2, max_row=4, max_col=2)
chart.add_data(chart_data, titles_from_data=True)
sheet.add_chart(chart, "E5")
wb.save("chart_report.xlsx")
```

**Effect:** Adds a bar chart to the Excel file at cell `E5`.

---

### **14\. Integrate Python with VBA Macros (xlwings)**

```python
import xlwings as xw

app = xw.App(visible=True)
wb = xw.Book("macro_enabled.xlsm")
macro = wb.macro("Module1.MyMacro")
macro()
wb.save()
wb.close()
app.quit()
```

**Effect:** Executes the `MyMacro` VBA macro from the file `macro_enabled.xlsm`.

---

### **15\. Excel-to-DataFrame Conversion (openpyxl/pyexcel)**

```python
import pandas as pd
from openpyxl import load_workbook

wb = load_workbook("example.xlsx")
sheet = wb.active
data = sheet.values
df = pd.DataFrame(data, columns=next(data))  # Use first row as header
print(df)
```

**Sample Output:**

```python
    Name  Age  Score
0  Alice   30     95
1    Bob   25     88
```

---

### **16\. High-Speed Read/Write Operations (pyxlsb)**

```python
from pyxlsb import open_workbook

with open_workbook("large_file.xlsb") as wb:
    with wb.get_sheet(1) as sheet:
        rows = list(sheet.rows())
        print(rows[:5])  # Display first 5 rows
```

**Effect:** Quickly reads large `.xlsb` files.

---

### **17\. Merge or Split Excel Sheets (pyexcel)**

```python
import pyexcel

# Merging multiple Excel files
pyexcel.merge_books_to_a_book(["file1.xlsx", "file2.xlsx"], "merged.xlsx")

# Splitting sheets into separate files
sheets = pyexcel.get_book(file_name="merged.xlsx")
sheets.save_to_files(file_names=["sheet1.xlsx", "sheet2.xlsx"])
```

**Effect:** Merges sheets into one file or splits them into individual files.

---

### **18\. Work with Excel on Remote Servers (openpyxl/pyexcel)**

```python
from openpyxl import Workbook
import paramiko

# Save file locally
wb = Workbook()
wb.save("local_file.xlsx")

# Upload file to remote server
ssh = paramiko.SSHClient()
ssh.load_system_host_keys()
ssh.connect("your-server.com", username="user", password="password")
sftp = ssh.open_sftp()
sftp.put("local_file.xlsx", "/remote/path/remote_file.xlsx")
sftp.close()
ssh.close()
```

**Effect:** Transfers an Excel file to a remote server.

---

### **19\. Apply Conditional Formatting (openpyxl)**

```python
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from openpyxl.formatting.rule import CellIsRule

wb = Workbook()
sheet = wb.active
sheet.append(["Name", "Score"])
sheet.append(["Alice", 90])
sheet.append(["Bob", 75])

# Apply conditional formatting
red_fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
rule = CellIsRule(operator="lessThan", formula=["80"], stopIfTrue=True, fill=red_fill)
sheet.conditional_formatting.add("B2:B3", rule)
wb.save("conditional_format.xlsx")
```

**Effect:** Highlights scores below 80 in red.

---

### **20\. Automate Excel from Jupyter (xlwings)**

```python
import xlwings as xw

wb = xw.Book()  # Open a new workbook
sheet = wb.sheets[0]
sheet.range("A1").value = "Hello from Jupyter!"
wb.save("jupyter_automated.xlsx")
wb.close()
```

**Effect:** Automates Excel tasks directly from a Jupyter notebook.

---