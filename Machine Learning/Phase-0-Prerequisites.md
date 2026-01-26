# Phase 0: Prerequisites (Foundations)

## 🎯 Learning Objectives

By the end of this phase, you will:

- ✅ Write Python code for data manipulation
- ✅ Use NumPy for numerical operations
- ✅ Handle datasets with Pandas
- ✅ Visualize data with Matplotlib and Seaborn
- ✅ Load and prepare data for ML

**Time Required:** 1-2 weeks  
**Difficulty:** Beginner

---

## 0.1 Python Essentials for ML

### Variables and Data Types

```python
# Basic variables
name = "Machine Learning"
age = 25
height = 5.9
is_student = True

# Lists (ordered, mutable)
scores = [85, 90, 78, 92, 88]
print(scores[0])  # Output: 85

# Dictionaries (key-value pairs)
student = {
    "name": "John",
    "age": 22,
    "grades": [85, 90, 88]
}
print(student["name"])  # Output: John
```

### Loops and Conditionals

```python
# For loop
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    print(num * 2)  # Output: 2, 4, 6, 8, 10

# While loop
count = 0
while count < 5:
    print(f"Count is: {count}")
    count += 1

# If-else
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
else:
    grade = "C"
print(f"Grade: {grade}")  # Output: Grade: B
```

### Functions

```python
# Simple function
def calculate_average(numbers):
    """Calculate average of a list of numbers"""
    total = sum(numbers)
    count = len(numbers)
    return total / count

scores = [85, 90, 78, 92, 88]
avg = calculate_average(scores)
print(f"Average: {avg}")  # Output: Average: 86.6

# Function with multiple parameters
def greet(name, age=25):
    """Greet a person with their name and age"""
    return f"Hello {name}, you are {age} years old"

print(greet("Alice"))          # Output: Hello Alice, you are 25 years old
print(greet("Bob", 30))        # Output: Hello Bob, you are 30 years old
```

### File Handling

```python
# Writing to a file
data = ["Apple", "Banana", "Cherry"]
with open("fruits.txt", "w") as file:
    for fruit in data:
        file.write(fruit + "\n")

# Reading from a file
with open("fruits.txt", "r") as file:
    content = file.read()
    print(content)

# Reading line by line
with open("fruits.txt", "r") as file:
    for line in file:
        print(line.strip())  # Remove newline character
```

### List Comprehensions (ML Essential!)

```python
# Traditional way
squares = []
for x in range(10):
    squares.append(x ** 2)

# List comprehension way (preferred in ML)
squares = [x ** 2 for x in range(10)]
print(squares)  # Output: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition
even_squares = [x ** 2 for x in range(10) if x % 2 == 0]
print(even_squares)  # Output: [0, 4, 16, 36, 64]
```

---

## 0.2 Essential Libraries

### 📦 NumPy - Numerical Python

NumPy is the foundation of ML in Python. It provides fast array operations.

#### Why NumPy?

- ✅ 50x faster than Python lists
- ✅ Mathematical operations on entire arrays
- ✅ Foundation for Pandas and Scikit-learn

#### Basic NumPy Operations

```python
import numpy as np

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])  # 2D array

print(arr1)        # Output: [1 2 3 4 5]
print(arr1.shape)  # Output: (5,)
print(arr2.shape)  # Output: (2, 3)

# Array creation functions
zeros = np.zeros((3, 4))        # 3x4 array of zeros
ones = np.ones((2, 3))          # 2x3 array of ones
random = np.random.rand(3, 3)   # 3x3 array of random values (0-1)
range_arr = np.arange(0, 10, 2) # Array from 0 to 10, step 2
linspace = np.linspace(0, 1, 5) # 5 values from 0 to 1

print("Zeros:\n", zeros)
print("Range:", range_arr)      # Output: [0 2 4 6 8]
```

#### Array Operations (Vectorization)

```python
import numpy as np

# Element-wise operations (FAST!)
arr = np.array([1, 2, 3, 4, 5])

# Operations on entire array
print(arr + 10)      # Output: [11 12 13 14 15]
print(arr * 2)       # Output: [2 4 6 8 10]
print(arr ** 2)      # Output: [1 4 9 16 25]

# Mathematical functions
print(np.sqrt(arr))  # Square root
print(np.exp(arr))   # Exponential
print(np.log(arr))   # Natural log

# Array operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print(arr1 + arr2)   # Output: [5 7 9]
print(arr1 * arr2)   # Output: [4 10 18]

# Statistical operations
data = np.array([10, 20, 30, 40, 50])
print(f"Mean: {np.mean(data)}")      # Output: 30.0
print(f"Median: {np.median(data)}")  # Output: 30.0
print(f"Std: {np.std(data)}")        # Output: 14.14
print(f"Min: {np.min(data)}")        # Output: 10
print(f"Max: {np.max(data)}")        # Output: 50
```

#### Array Indexing and Slicing

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])

# Indexing
print(arr[0])     # Output: 10 (first element)
print(arr[-1])    # Output: 50 (last element)

# Slicing [start:end:step]
print(arr[1:4])   # Output: [20 30 40]
print(arr[:3])    # Output: [10 20 30]
print(arr[::2])   # Output: [10 30 50] (every 2nd element)

# Boolean indexing (ML ESSENTIAL!)
arr = np.array([1, 2, 3, 4, 5, 6])
mask = arr > 3
print(mask)       # Output: [False False False True True True]
print(arr[mask])  # Output: [4 5 6]

# One line
print(arr[arr > 3])  # Output: [4 5 6]
```

#### 2D Array Operations

```python
import numpy as np

# Creating 2D array
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(matrix)
print(f"Shape: {matrix.shape}")  # Output: (3, 3)

# Indexing 2D arrays
print(matrix[0, 0])    # Output: 1 (row 0, col 0)
print(matrix[1, 2])    # Output: 6 (row 1, col 2)
print(matrix[0])       # Output: [1 2 3] (entire first row)
print(matrix[:, 0])    # Output: [1 4 7] (entire first column)

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B:\n", A + B)      # Element-wise addition
print("A * B:\n", A * B)      # Element-wise multiplication
print("A @ B:\n", A @ B)      # Matrix multiplication
print("Transpose:\n", A.T)    # Transpose
```

---

### 📊 Pandas - Data Analysis

Pandas is the go-to library for working with tabular data (like Excel/CSV).

#### Creating DataFrames

```python
import pandas as pd
import numpy as np

# From dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'Salary': [50000, 60000, 75000, 55000],
    'Department': ['HR', 'IT', 'IT', 'Sales']
}
df = pd.DataFrame(data)
print(df)
```

Output:

```
      Name  Age  Salary Department
0    Alice   25   50000         HR
1      Bob   30   60000         IT
2  Charlie   35   75000         IT
3    David   28   55000      Sales
```

#### Basic DataFrame Operations

```python
import pandas as pd

# Load data
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'Salary': [50000, 60000, 75000, 55000, 65000],
    'Department': ['HR', 'IT', 'IT', 'Sales', 'HR']
})

# Basic info
print(df.head())         # First 5 rows
print(df.tail(3))        # Last 3 rows
print(df.shape)          # Output: (5, 4)
print(df.columns)        # Column names
print(df.dtypes)         # Data types
print(df.info())         # Detailed info
print(df.describe())     # Statistical summary

# Selecting columns
print(df['Name'])        # Single column (Series)
print(df[['Name', 'Age']])  # Multiple columns (DataFrame)

# Selecting rows
print(df.loc[0])         # Row by label
print(df.iloc[0])        # Row by position
print(df[df['Age'] > 30])  # Conditional selection

# Adding new column
df['Bonus'] = df['Salary'] * 0.1
print(df)
```

#### Data Filtering and Sorting

```python
import pandas as pd

df = pd.DataFrame({
    'Product': ['A', 'B', 'C', 'D', 'E'],
    'Price': [100, 150, 200, 120, 180],
    'Quantity': [10, 5, 8, 15, 7],
    'Category': ['Electronics', 'Clothing', 'Electronics', 'Clothing', 'Electronics']
})

# Filtering
high_price = df[df['Price'] > 150]
print(high_price)

# Multiple conditions
electronics_high_price = df[(df['Category'] == 'Electronics') & (df['Price'] > 150)]
print(electronics_high_price)

# Sorting
sorted_df = df.sort_values('Price', ascending=False)
print(sorted_df)

# Groupby (VERY IMPORTANT!)
category_avg = df.groupby('Category')['Price'].mean()
print(category_avg)
```

#### Handling Missing Data

```python
import pandas as pd
import numpy as np

# Data with missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [5, np.nan, np.nan, 8, 9],
    'C': [10, 11, 12, 13, 14]
})

print(df)

# Check for missing values
print(df.isnull())       # Boolean DataFrame
print(df.isnull().sum()) # Count per column

# Drop missing values
df_dropped = df.dropna()  # Drop rows with any NaN
print(df_dropped)

# Fill missing values
df_filled = df.fillna(0)  # Fill with 0
print(df_filled)

df_filled_mean = df.fillna(df.mean())  # Fill with column mean
print(df_filled_mean)
```

#### Reading and Writing Files

```python
import pandas as pd

# Read CSV
df = pd.read_csv('data.csv')

# Read Excel
df = pd.read_excel('data.xlsx')

# Read with specific parameters
df = pd.read_csv('data.csv',
                 sep=';',           # Delimiter
                 header=0,          # Row to use as header
                 na_values=['NA', 'N/A', ''])  # Missing value indicators

# Write to CSV
df.to_csv('output.csv', index=False)  # index=False to not save index column

# Write to Excel
df.to_excel('output.xlsx', index=False)
```

---

### 📈 Matplotlib - Data Visualization

#### Basic Plots

```python
import matplotlib.pyplot as plt
import numpy as np

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Sine Wave')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
sizes = 1000 * np.random.rand(50)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.colorbar()
plt.show()

# Bar plot
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

plt.figure(figsize=(8, 6))
plt.bar(categories, values, color='skyblue', edgecolor='black')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart')
plt.show()

# Histogram
data = np.random.randn(1000)

plt.figure(figsize=(8, 6))
plt.hist(data, bins=30, color='green', alpha=0.7, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()
```

#### Multiple Subplots

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Line
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sine Wave')

# Plot 2: Scatter
axes[0, 1].scatter(np.random.rand(50), np.random.rand(50))
axes[0, 1].set_title('Scatter Plot')

# Plot 3: Bar
axes[1, 0].bar(['A', 'B', 'C'], [1, 2, 3])
axes[1, 0].set_title('Bar Chart')

# Plot 4: Histogram
axes[1, 1].hist(np.random.randn(1000), bins=30)
axes[1, 1].set_title('Histogram')

plt.tight_layout()
plt.show()
```

---

### 🎨 Seaborn - Statistical Visualization

Seaborn is built on Matplotlib but provides better-looking plots and statistical functions.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set style
sns.set_style('whitegrid')

# Sample data
df = pd.DataFrame({
    'Age': np.random.randint(20, 60, 100),
    'Salary': np.random.randint(30000, 100000, 100),
    'Department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], 100)
})

# Distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, bins=20)
plt.title('Age Distribution')
plt.show()

# Scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='Age', y='Salary', data=df)
plt.title('Age vs Salary')
plt.show()

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Department', y='Salary', data=df)
plt.title('Salary by Department')
plt.show()

# Heatmap (correlation matrix)
numeric_df = df[['Age', 'Salary']]
correlation = numeric_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Pairplot (VERY USEFUL for ML!)
sns.pairplot(df, hue='Department')
plt.show()
```

---

### 🤖 Scikit-learn - ML Library (Preview)

Quick introduction - we'll dive deep in later phases.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Convert to DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y

print(df.head())
print(f"\nDataset shape: {X.shape}")
print(f"Number of classes: {len(set(y))}")

# Split data (we'll use this A LOT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Feature scaling (normalization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nOriginal data range: {X_train[:, 0].min():.2f} to {X_train[:, 0].max():.2f}")
print(f"Scaled data range: {X_train_scaled[:, 0].min():.2f} to {X_train_scaled[:, 0].max():.2f}")
```

---

## 🛠️ Setting Up Your Environment

### Option 1: Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv ml_env

# Activate
# On macOS/Linux:
source ml_env/bin/activate
# On Windows:
ml_env\Scripts\activate

# Install packages
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Verify installation
python -c "import numpy; print(numpy.__version__)"
```

### Option 2: Conda Environment

```bash
# Create environment
conda create -n ml_env python=3.10

# Activate
conda activate ml_env

# Install packages
conda install numpy pandas matplotlib seaborn scikit-learn jupyter

# Or use pip
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

---

## 📝 Practice Exercises

### Exercise 1: NumPy Array Operations

```python
# Create a 5x5 matrix with random values between 0 and 100
# Find: mean, median, standard deviation
# Extract all values greater than 50
# Normalize the matrix (subtract mean, divide by std)
```

### Exercise 2: Pandas Data Manipulation

```python
# Create a DataFrame with columns: Name, Age, City, Salary
# Add 5 rows of data
# Filter people older than 30
# Calculate average salary by city
# Add a new column: Salary_Category (Low, Medium, High based on salary)
```

### Exercise 3: Data Visualization

```python
# Create a dataset with 100 samples
# Plot: histogram, scatter plot, box plot
# Create a 2x2 subplot with different visualizations
# Use Seaborn for advanced plots
```

### Exercise 4: Real Dataset

```python
# Download a CSV from Kaggle or UCI ML Repository
# Load it with Pandas
# Handle missing values
# Create visualizations
# Calculate basic statistics
```

---

## ✅ Phase Completion Checklist

- [ ] Understand Python basics (loops, functions, lists, dicts)
- [ ] Create and manipulate NumPy arrays
- [ ] Perform vectorized operations
- [ ] Load and analyze data with Pandas
- [ ] Handle missing data
- [ ] Create visualizations with Matplotlib
- [ ] Use Seaborn for statistical plots
- [ ] Set up development environment
- [ ] Complete practice exercises
- [ ] Load and explore a real dataset

---

## 🎯 Key Takeaways

1. **NumPy**: Foundation for numerical computing - arrays are faster than lists
2. **Pandas**: Essential for data manipulation - DataFrames are like Excel on steroids
3. **Matplotlib**: Low-level plotting - complete control over visualizations
4. **Seaborn**: High-level plotting - beautiful statistical visualizations
5. **Vectorization**: Always prefer array operations over loops (50x faster!)

---

## 📚 Next Steps

Once you're comfortable with these tools, move to:
👉 [Phase 1: What is Machine Learning](./Phase-1-ML-Concepts.md)

---

**Remember:** These tools are your foundation. You'll use them in EVERY ML project. Practice until they become second nature!
