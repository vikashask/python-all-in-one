# 🚀 Pandas Crash Course — Quick Revision Guide

> **Time to revise: ~15 minutes** | Covers everything from the `pandas/` folder (including Matplotlib & Seaborn)

---

## 1. What is Pandas?
- **Data manipulation library** built on NumPy
- Two main structures: **Series** (1D) and **DataFrame** (2D table)
- Think of it as **Excel/SQL in Python**

---

## 2. Creating Data Structures

```python
import pandas as pd
import numpy as np

# Series
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
s = pd.Series({'x': 1, 'y': 2, 'z': 3})

# DataFrame — from dict
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})

# DataFrame — from list of dicts
df = pd.DataFrame([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])

# Quick info
df.head()       # First 5 rows
df.tail()       # Last 5 rows
df.info()       # Column types, non-null counts
df.describe()   # Stats: mean, std, min, max, quartiles
df.shape        # (rows, cols)
df.columns      # Column names
df.dtypes       # Data types
```

---

## 3. Selecting & Indexing Data

```python
# Single column (returns Series)
df['name']

# Multiple columns (returns DataFrame)
df[['name', 'age']]

# By label — .loc[row_label, col_label]
df.loc[0, 'name']           # Single value
df.loc[0:2, 'name':'age']   # Range (inclusive!)

# By position — .iloc[row_pos, col_pos]
df.iloc[0, 1]               # Row 0, Col 1
df.iloc[0:2, 0:2]           # Range (exclusive end)

# Boolean filtering
df[df['age'] > 25]
df[(df['age'] > 25) & (df['salary'] > 55000)]
df[df['name'].isin(['Alice', 'Bob'])]
```

---

## 4. Data Cleaning

### Missing Data
```python
df.isna().sum()              # Count NaN per column
df.dropna()                  # Drop rows with any NaN
df.dropna(how='all')         # Drop only if ALL values NaN
df.dropna(axis=1)            # Drop columns with NaN
df.fillna(0)                 # Fill NaN with 0
df['col'].fillna(df['col'].mean())  # Fill with mean
df.ffill()                   # Forward fill (previous value)
df.bfill()                   # Backward fill (next value)
```

### Duplicates
```python
df.duplicated().sum()        # Count duplicates
df.drop_duplicates()         # Remove duplicates
```

### String Cleaning
```python
df['name'].str.lower()       # lowercase
df['name'].str.upper()       # UPPERCASE
df['name'].str.strip()       # Remove whitespace
df['name'].str.title()       # Title Case
df['name'].str.contains('li') # Boolean: contains substring
df['name'].str.replace('old', 'new')
df['name'].str.split(' ', expand=True)  # Split into columns
df['name'].str.len()         # String lengths
```

---

## 5. Adding, Modifying & Deleting

```python
# Add new column
df['bonus'] = df['salary'] * 0.1
df['grade'] = np.where(df['age'] > 28, 'Senior', 'Junior')

# Modify
df['salary'] = df['salary'] + 5000

# Rename columns
df.rename(columns={'name': 'full_name'}, inplace=True)

# Delete
df.drop('bonus', axis=1, inplace=True)      # Drop column
df.drop([0, 1], inplace=True)                # Drop rows by index
```

---

## 6. Sorting & Ranking

```python
df.sort_values('age')                         # Ascending
df.sort_values('salary', ascending=False)     # Descending
df.sort_values(['age', 'salary'], ascending=[True, False])  # Multi
df.sort_index()                               # Sort by index
df['rank'] = df['salary'].rank(ascending=False)
```

---

## 7. Aggregation & GroupBy

```python
# Basic GroupBy
df.groupby('department')['salary'].mean()
df.groupby('department')['salary'].sum()
df.groupby('department').size()               # Count per group

# Multiple aggregations
df.groupby('department')['salary'].agg(['mean', 'min', 'max', 'count'])

# Named aggregation
df.groupby('department').agg(
    avg_salary=('salary', 'mean'),
    total=('salary', 'sum'),
    count=('salary', 'count')
)

# GroupBy multiple columns
df.groupby(['department', 'grade'])['salary'].mean()
```

---

## 8. Pivot Tables & Crosstab

```python
# Pivot Table (like Excel pivot)
pd.pivot_table(df, values='salary', index='department',
               columns='grade', aggfunc='mean', margins=True)

# Crosstab (frequency counts)
pd.crosstab(df['department'], df['grade'])
```

---

## 9. Merging & Joining

```python
# Merge (SQL-like joins)
pd.merge(df1, df2, on='id')                  # Inner join (default)
pd.merge(df1, df2, on='id', how='left')       # Left join
pd.merge(df1, df2, on='id', how='right')      # Right join
pd.merge(df1, df2, on='id', how='outer')      # Full outer join

# Concatenate
pd.concat([df1, df2])                         # Stack vertically ↓
pd.concat([df1, df2], axis=1)                 # Stack horizontally →
```

---

## 10. Reshaping Data

```python
# Wide → Long
pd.melt(df, id_vars=['name'], value_vars=['math', 'science'],
         var_name='subject', value_name='score')

# Long → Wide
df.pivot(index='name', columns='subject', values='score')

# Stack / Unstack (MultiIndex)
df.stack()        # Columns → rows
df.unstack()      # Rows → columns

# One-Hot Encoding
pd.get_dummies(df, columns=['category'])

# Transpose
df.T
```

---

## 11. Apply, Map & Transform

```python
# Apply function to column
df['salary'].apply(lambda x: x * 1.1)

# Apply function to each row
df.apply(lambda row: row['salary'] / row['age'], axis=1)

# Map values (dict replacement)
df['grade'].map({'A': 'Excellent', 'B': 'Good', 'C': 'Average'})

# Custom function
def categorize(age):
    return 'Senior' if age >= 30 else 'Junior'
df['category'] = df['age'].apply(categorize)
```

---

## 12. DateTime Operations

```python
df['date'] = pd.to_datetime(df['date'])
dates = pd.date_range('2024-01-01', periods=12, freq='M')

# Extract components
df['date'].dt.year
df['date'].dt.month
df['date'].dt.day
df['date'].dt.day_name()

# Rolling window
df['rolling_avg'] = df['sales'].rolling(window=3).mean()

# Resample (time-based groupby)
df.resample('M').sum()        # Monthly sum
```

---

## 13. File I/O

```python
# CSV
df.to_csv('output.csv', index=False)
df = pd.read_csv('data.csv', usecols=['col1', 'col2'], nrows=100)

# Excel
df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# JSON
df.to_json('output.json')
df = pd.read_json('data.json')
```

---

## 14. Window Functions

```python
df['rolling_mean'] = df['value'].rolling(3).mean()      # 3-period moving avg
df['expanding_mean'] = df['value'].expanding().mean()    # Expanding mean
df['cumsum'] = df['value'].cumsum()                      # Cumulative sum
df['ewm'] = df['value'].ewm(span=3).mean()              # Exponential weighted
```

---

## 15. MultiIndex

```python
# Create
idx = pd.MultiIndex.from_arrays([['A','A','B','B'], [1,2,1,2]])
df = pd.DataFrame({'val': [10,20,30,40]}, index=idx)

# Access
df.loc['A']                # All rows in group A
df.loc[('B', 2)]           # Specific multi-level
df.xs(1, level=1)          # Cross-section at level 1
df.unstack()               # Pivot inner level to columns
```

---

## 16. Performance Tips

| Tip | Impact |
|-----|--------|
| **Use vectorized ops** | 100x faster than `.iterrows()` |
| **Use `.astype('category')`** | For low-cardinality string columns |
| **Downcast numerics** | `pd.to_numeric(df['col'], downcast='integer')` |
| **Read only needed columns** | `pd.read_csv(usecols=[...])` |
| **Use chunking** | `pd.read_csv(chunksize=10000)` for huge files |
| **Use `.query()`** | `df.query('age > 25 and salary > 50000')` |

---

## 17. Advanced Tips

```python
df.query('age > 25 & salary > 50000')    # SQL-like filtering
df.nlargest(5, 'salary')                  # Top 5 by salary
df.nsmallest(3, 'age')                    # Bottom 3 by age
df['col'].value_counts()                  # Frequency counts
df.sample(10)                             # Random 10 rows
df.pipe(func1).pipe(func2)               # Method chaining
df.assign(new_col=lambda x: x['a'] + x['b'])  # Add col in chain
```

---

## 📊 Matplotlib Quick Reference

```python
import matplotlib.pyplot as plt

# Line Plot
plt.plot(x, y, label='Sales', color='blue', linestyle='--', marker='o')
plt.xlabel('Month'); plt.ylabel('Revenue'); plt.title('Monthly Sales')
plt.legend(); plt.grid(True); plt.show()

# Bar Chart
plt.bar(categories, values)               # Vertical
plt.barh(categories, values)              # Horizontal

# Histogram
plt.hist(data, bins=20, edgecolor='black', alpha=0.7)

# Scatter Plot
plt.scatter(x, y, c=colors, s=sizes, cmap='viridis', alpha=0.6)

# Pie Chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%', explode=[0.1, 0, 0, 0])

# Box Plot
plt.boxplot(data, labels=['A', 'B', 'C'])

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].plot(x, y)  # Top-left

# Heatmap (with colorbar)
plt.imshow(data, cmap='coolwarm'); plt.colorbar()

# Save
plt.savefig('chart.png', dpi=300, bbox_inches='tight')
```

**Key styles:** `plt.style.use('seaborn')` | `'ggplot'` | `'fivethirtyeight'`

---

## 📊 Seaborn Quick Reference

```python
import seaborn as sns

df = sns.load_dataset('tips')              # Built-in dataset

sns.scatterplot(x='total_bill', y='tip', hue='sex', data=df)
sns.lineplot(x='day', y='total_bill', hue='sex', data=df)
sns.boxplot(x='day', y='total_bill', hue='sex', data=df)
sns.barplot(x='day', y='total_bill', hue='sex', data=df)
sns.histplot(df['total_bill'], bins=20, kde=True)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
sns.pairplot(df, hue='sex')
sns.violinplot(x='day', y='total_bill', data=df)
```

> `hue` parameter = color by category group

---

## 🧾 Pandas Cheat Sheet

| Task | Code |
|------|------|
| Read CSV | `pd.read_csv('file.csv')` |
| First 5 rows | `df.head()` |
| Shape | `df.shape` |
| Column types | `df.dtypes` |
| Select column | `df['col']` |
| Filter rows | `df[df['col'] > 5]` |
| Sort | `df.sort_values('col')` |
| Group & mean | `df.groupby('col').mean()` |
| Merge | `pd.merge(df1, df2, on='key')` |
| Pivot | `pd.pivot_table(df, values, index, columns)` |
| Fill NaN | `df.fillna(0)` |
| Drop NaN | `df.dropna()` |
| Apply func | `df['col'].apply(func)` |
| Unique values | `df['col'].nunique()` |
| Value counts | `df['col'].value_counts()` |
| Save CSV | `df.to_csv('out.csv', index=False)` |

---

> **Source folders**: `pandas/useOfPandas.md`, `pandas/pandas-Zero-to-hero.ipynb`, `pandas/pandas_Example 1.ipynb`, `pandas/matplotlib.ipynb`, `pandas/seaborn.ipynb`
