# Pandas: Python Data Analysis Library

## What is Pandas?

Pandas is a powerful, flexible, and easy-to-use open-source data analysis and manipulation library built on top of Python. It provides high-performance, user-friendly data structures and tools for working with structured (tabular, multidimensional, potentially heterogeneous) and time series data.

## Core Data Structures

1. **DataFrame**

   - 2-dimensional labeled data structure
   - Similar to a spreadsheet or SQL table
   - Columns can be different types
   - Primary data structure for data analysis

2. **Series**
   - 1-dimensional labeled array
   - Can hold any data type
   - Like a column in a spreadsheet
   - Index-value pair structure

## Key Features

### 1. Data Manipulation

- **Merging and Joining**

  - Database-style operations
  - Multiple merge types (inner, outer, left, right)
  - Concatenation and combining

- **Reshaping**
  - Pivot tables
  - Melting and casting
  - Grouping and aggregation

### 2. Data Analysis

- **Statistical Functions**

  - Mean, median, mode
  - Standard deviation, variance
  - Correlation, covariance

- **Window Operations**
  - Rolling calculations
  - Expanding windows
  - Custom window functions

### 3. Data Cleaning

- **Missing Data Handling**

  - Multiple imputation methods
  - Dropping or filling NA values
  - Forward/backward fill

- **Duplicates Management**
  - Identification
  - Removal
  - Handling

## Advantages of Pandas

1. **Performance**

   - Optimized for speed
   - Efficient memory usage
   - Built on top of NumPy

2. **Data Handling**

   - Flexible data import/export
   - Handles missing data
   - Automatic data alignment

3. **Integration**

   - Works with many file formats
   - Integrates with other libraries
   - SQL database connectivity

4. **Functionality**
   - Rich data manipulation tools
   - Time series functionality
   - Text data processing

## Common Use Cases

1. **Data Science**

   - Data cleaning and preparation
   - Feature engineering
   - Exploratory data analysis

2. **Finance**

   - Time series analysis
   - Financial calculations
   - Risk analysis

3. **Business Analytics**
   - Sales reporting
   - Customer analytics
   - Performance metrics

## Code Examples

```python
import pandas as pd
import numpy as np

# Creating DataFrames
df = pd.DataFrame({
    'Name': ['John', 'Anna', 'Peter'],
    'Age': [28, 22, 35],
    'City': ['New York', 'Paris', 'London']
})

# Reading data
csv_df = pd.read_csv('data.csv')              # CSV files
excel_df = pd.read_excel('data.xlsx')         # Excel files
sql_df = pd.read_sql('query', connection)     # SQL databases

# Data manipulation
filtered = df[df['Age'] > 25]                 # Filtering
sorted_df = df.sort_values('Age')             # Sorting
grouped = df.groupby('City').mean()           # Grouping

# Data analysis
descriptive = df.describe()                   # Statistical summary
pivoted = df.pivot_table(                     # Pivot table
    values='Age',
    index='City',
    aggfunc='mean'
)

# Handling missing data
df.fillna(0)                                  # Fill NA with zero
df.dropna()                                   # Drop NA rows

# Merging data
merged = pd.merge(
    left_df,
    right_df,
    on='key_column',
    how='inner'
)
```

## Best Practices

1. **Data Import**

   - Use appropriate data types
   - Specify date columns
   - Handle missing values early

2. **Performance**

   - Use vectorized operations
   - Avoid loops when possible
   - Utilize method chaining

3. **Memory Management**
   - Use appropriate data types
   - Remove unnecessary columns
   - Use chunking for large files

## Common Operations Cheat Sheet

| Operation | Method             | Description          |
| --------- | ------------------ | -------------------- |
| Selection | `df[['col']]`      | Select columns       |
| Filtering | `df.loc[]`         | Label-based indexing |
| Sorting   | `df.sort_values()` | Sort by values       |
| Grouping  | `df.groupby()`     | Group by values      |
| Merging   | `pd.merge()`       | Combine DataFrames   |
| Stats     | `df.describe()`    | Statistical summary  |

## Installation

```bash
pip install pandas
```

## Further Resources

- [Official Pandas Documentation](https://pandas.pydata.org/docs/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Pandas API Reference](https://pandas.pydata.org/docs/reference/index.html)
- [Pandas Cookbook](https://pandas.pydata.org/docs/user_guide/cookbook.html)
