import json

# Create comprehensive notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Helper function to add cells
def add_markdown_cell(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [text.split("\n")]
    })

def add_code_cell(code):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split("\n")
    })

# Title and Introduction
add_markdown_cell("""# Australian Apparel Sales Analysis - Q4 2020

## Executive Summary
This report analyzes the sales data of the company for the fourth quarter (October-December 2020) in Australia, examining it on a state-by-state basis. The analysis provides insights to assist the company in making data-driven decisions for the upcoming year.

### Objectives:
1. Data Wrangling - Clean and prepare data
2. Data Analysis - Statistical analysis and insights
3. Data Visualization - Create comprehensive dashboard
4. Report Generation - Document findings and recommendations""")

# Import Libraries
add_markdown_cell("## Import Required Libraries")

add_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("Libraries imported successfully!")""")

# Section 1: Data Wrangling
add_markdown_cell("""---
# 1. DATA WRANGLING

## 1.1 Load and Inspect Data""")

add_code_cell("""# Load the dataset
df = pd.read_csv('AusApparalSales4thQrt2020.csv')

print("Dataset Shape:", df.shape)
print("\\nFirst 10 rows:")
df.head(10)""")

add_code_cell("""# Display basic information
print("Dataset Information:")
df.info()
print("\\n" + "="*80)
print("\\nColumn Names:")
print(df.columns.tolist())""")

# Missing values check
add_markdown_cell("## 1.2 Data Quality Check - Missing and Incorrect Values")

add_code_cell("""# Check for missing values using isna()
print("Missing Values Analysis:")
print("="*80)
missing_values = df.isna().sum()
missing_percentage = (df.isna().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing Count': missing_values.values,
    'Missing Percentage': missing_percentage.values
})

print(missing_df)
print("\\n" + "="*80)

# Check for non-null values using notna()
print("\\nNon-Null Values Count:")
print(df.notna().sum())""")

add_code_cell("""# Check for duplicate rows
print("Duplicate Rows Analysis:")
print("="*80)
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

if duplicate_count > 0:
    print("\\nDuplicate rows:")
    print(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)))""")

add_code_cell("""# Check for negative or zero values in numerical columns
print("Data Validity Check:")
print("="*80)
print(f"Negative values in 'Unit': {(df['Unit'] < 0).sum()}")
print(f"Zero values in 'Unit': {(df['Unit'] == 0).sum()}")
print(f"Negative values in 'Sales': {(df['Sales'] < 0).sum()}")
print(f"Zero values in 'Sales': {(df['Sales'] == 0).sum()}")

print("\\nUnique values in categorical columns:")
print(f"States: {df['State'].unique()}")
print(f"Groups: {df['Group'].unique()}")
print(f"Time: {df['Time'].unique()}")""")

# Recommendations
add_markdown_cell("""## 1.3 Recommendations for Handling Missing/Incorrect Data

**Analysis and Recommendations:**

1. **Missing Values Treatment:**
   - If missing values exist in numerical columns (Unit, Sales): Use median imputation for robust handling of outliers
   - If missing values exist in categorical columns (State, Group, Time): Use mode imputation or drop if < 5% of data
   - For Date/Time missing values: Drop rows as temporal data is critical for trend analysis

2. **Incorrect/Invalid Data Treatment:**
   - Remove rows with negative values in Unit or Sales (business rule violation)
   - For zero values: Investigate if they represent valid transactions or data entry errors
   - Standardize text fields (remove leading/trailing spaces in State, Group, Time)

3. **Duplicate Handling:**
   - Remove exact duplicates
   - Investigate near-duplicates for potential aggregation""")

add_code_cell("""# Clean the data based on findings
df_clean = df.copy()

# Strip whitespaces from string columns
string_columns = ['Date', 'Time', 'State', 'Group']
for col in string_columns:
    df_clean[col] = df_clean[col].str.strip()

# Remove duplicates if any
initial_rows = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"Removed {initial_rows - len(df_clean)} duplicate rows")

# Handle missing values (if any)
if df_clean.isna().sum().sum() > 0:
    print("\\nHandling missing values...")
    # Fill numerical columns with median
    df_clean['Unit'].fillna(df_clean['Unit'].median(), inplace=True)
    df_clean['Sales'].fillna(df_clean['Sales'].median(), inplace=True)
    # Fill categorical columns with mode
    for col in ['State', 'Group', 'Time']:
        if df_clean[col].isna().sum() > 0:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    # Drop rows with missing dates
    df_clean.dropna(subset=['Date'], inplace=True)

print(f"\\nCleaned dataset shape: {df_clean.shape}")
print(f"Missing values after cleaning: {df_clean.isna().sum().sum()}")""")

# Feature Engineering
add_markdown_cell("## 1.4 Data Transformation and Feature Engineering")

add_code_cell("""# Convert Date column to datetime format
df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%d-%b-%Y')

# Extract additional time features
df_clean['Year'] = df_clean['Date'].dt.year
df_clean['Month'] = df_clean['Date'].dt.month
df_clean['Month_Name'] = df_clean['Date'].dt.strftime('%B')
df_clean['Week'] = df_clean['Date'].dt.isocalendar().week
df_clean['Day'] = df_clean['Date'].dt.day
df_clean['DayOfWeek'] = df_clean['Date'].dt.day_name()
df_clean['Quarter'] = df_clean['Date'].dt.quarter

print("Date parsing completed!")
print("\\nNew columns added:")
print(df_clean[['Date', 'Month_Name', 'Week', 'DayOfWeek', 'Quarter']].head())""")

# Normalization
add_markdown_cell("""## 1.5 Data Normalization

**Why Normalization?**
- Normalization scales data to a fixed range (0-1), which is preferred for this analysis
- It preserves the shape of the distribution and handles outliers better than standardization
- Useful for comparing sales across different states and groups on the same scale

**Method:** Min-Max Normalization using the formula: `(X - X_min) / (X_max - X_min)`""")

add_code_cell("""# Apply Min-Max Normalization
scaler = MinMaxScaler()

# Normalize Unit and Sales columns
df_clean['Unit_Normalized'] = scaler.fit_transform(df_clean[['Unit']])
df_clean['Sales_Normalized'] = scaler.fit_transform(df_clean[['Sales']])

print("Normalization completed!")
print("\\nComparison of original vs normalized values:")
comparison_df = df_clean[['Unit', 'Unit_Normalized', 'Sales', 'Sales_Normalized']].describe()
print(comparison_df)""")

# GroupBy Analysis
add_markdown_cell("""## 1.6 GroupBy Function Analysis and Recommendations

**Application of GroupBy():**

The `GroupBy()` function is essential for:
1. **Data Chunking:** Splitting data into meaningful segments (State, Group, Time)
2. **Data Aggregation:** Computing summary statistics for each segment
3. **Comparative Analysis:** Comparing performance across different dimensions

**Recommendation:**
- Use GroupBy for aggregating sales data by State, Group, and Time period
- Create multi-level groupings for comprehensive analysis
- Apply aggregate functions (sum, mean, count) to derive insights""")

add_code_cell("""# Demonstrate GroupBy applications

# 1. Group by State
print("State-wise Sales Summary:")
print("="*80)
state_summary = df_clean.groupby('State').agg({
    'Sales': ['sum', 'mean', 'count'],
    'Unit': ['sum', 'mean']
}).round(2)
print(state_summary)

print("\\n" + "="*80)

# 2. Group by Demographic Group
print("\\nGroup-wise Sales Summary:")
print("="*80)
group_summary = df_clean.groupby('Group').agg({
    'Sales': ['sum', 'mean', 'count'],
    'Unit': ['sum', 'mean']
}).round(2)
print(group_summary)

print("\\n" + "="*80)

# 3. Group by Time of Day
print("\\nTime-of-Day Sales Summary:")
print("="*80)
time_summary = df_clean.groupby('Time').agg({
    'Sales': ['sum', 'mean', 'count'],
    'Unit': ['sum', 'mean']
}).round(2)
print(time_summary)""")

add_code_cell("""# Multi-level grouping for deeper insights
print("Multi-dimensional Analysis (State x Group):")
print("="*80)
state_group_summary = df_clean.groupby(['State', 'Group'])['Sales'].agg(['sum', 'mean', 'count']).round(2)
print(state_group_summary.head(20))

print("\\n" + "="*80)
print("\\nMulti-dimensional Analysis (State x Time):")
print("="*80)
state_time_summary = df_clean.groupby(['State', 'Time'])['Sales'].sum().unstack(fill_value=0)
print(state_time_summary)""")

# Save first part
with open('AusApparalSales_Analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook Part 1 created successfully!")
