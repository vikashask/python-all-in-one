# Phase 4: Feature Engineering & Data Processing

**Navigation:** [Previous (Phase 3)](./Phase-3-Unsupervised-Learning.md) | [Practice Notebook](./Phase-4-Feature-Engineering.ipynb) | [Next (Phases 5-9)](./Phase-5-9-Advanced-ML.md)

## 🎯 Learning Objectives

By the end of this phase, you will:

- ✅ Handle missing data effectively
- ✅ Deal with outliers
- ✅ Encode categorical variables
- ✅ Scale and normalize features
- ✅ Create new features
- ✅ Select important features
- ✅ Build complete data pipelines

**Time Required:** 2 weeks ⭐ CRITICAL SKILL
**Difficulty:** Intermediate
**Note:** This is 70% of real ML work!

---

## Why Feature Engineering Matters

> **"Garbage In = Garbage Out"**

Even the best algorithm fails with poor data. Feature engineering can improve model performance more than any algorithm tuning!

**Real Impact:**

- ❌ Bad features + Good algorithm = Poor results
- ✅ Good features + Simple algorithm = Excellent results

---

## 4.1 Handling Missing Data

### Types of Missing Data

1. **MCAR** (Missing Completely At Random): Random, no pattern
2. **MAR** (Missing At Random): Related to other variables
3. **MNAR** (Missing Not At Random): Related to the missing value itself

### Detecting Missing Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create dataset with missing values
data = {
    'Age': [25, 30, np.nan, 45, np.nan, 35, 50, np.nan, 40, 28],
    'Salary': [50000, np.nan, 70000, np.nan, 45000, 60000, np.nan, 90000, 65000, 55000],
    'Experience': [2, 5, np.nan, 10, 1, 6, 15, np.nan, 8, 3],
    'City': ['NY', 'LA', 'NY', np.nan, 'Chicago', 'LA', np.nan, 'NY', 'Chicago', 'LA']
}
df = pd.DataFrame(data)

print("Dataset with Missing Values:")
print(df)

# Check missing values
print("\n=== Missing Values Summary ===")
print(df.isnull().sum())
print("\nPercentage missing:")
print(df.isnull().sum() / len(df) * 100)

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap (Yellow = Missing)')
plt.show()
```

### Strategy 1: Remove Missing Data

```python
# Remove rows with ANY missing value
df_dropna_rows = df.dropna()
print(f"Original rows: {len(df)}, After dropping: {len(df_dropna_rows)}")

# Remove columns with ANY missing value
df_dropna_cols = df.dropna(axis=1)
print(f"Original columns: {df.shape[1]}, After dropping: {df_dropna_cols.shape[1]}")

# Remove rows with more than 2 missing values
df_thresh = df.dropna(thresh=2)
print(f"Rows with at least 2 non-null values: {len(df_thresh)}")

# ⚠️ Use carefully! Only when:
# - Small percentage of missing data (<5%)
# - Lots of data available
# - MCAR (missing completely at random)
```

### Strategy 2: Imputation (Fill Missing Values)

```python
from sklearn.impute import SimpleImputer

# Method 1: Fill with mean
df_mean = df.copy()
df_mean['Age'].fillna(df_mean['Age'].mean(), inplace=True)
df_mean['Salary'].fillna(df_mean['Salary'].mean(), inplace=True)

print("\n=== Mean Imputation ===")
print(df_mean[['Age', 'Salary']])

# Method 2: Fill with median (better for outliers)
df_median = df.copy()
df_median['Age'].fillna(df_median['Age'].median(), inplace=True)

# Method 3: Fill with mode (for categorical)
df_mode = df.copy()
df_mode['City'].fillna(df_mode['City'].mode()[0], inplace=True)

# Method 4: Forward fill (for time series)
df_ffill = df.copy()
df_ffill.fillna(method='ffill', inplace=True)

# Method 5: Backward fill
df_bfill = df.copy()
df_bfill.fillna(method='bfill', inplace=True)

# Method 6: Fill with constant
df_const = df.copy()
df_const.fillna({'Age': 0, 'Salary': 0, 'City': 'Unknown'}, inplace=True)
```

### Strategy 3: Advanced Imputation

```python
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# KNN Imputer (uses similar rows)
knn_imputer = KNNImputer(n_neighbors=3)
df_knn = pd.DataFrame(
    knn_imputer.fit_transform(df[['Age', 'Salary', 'Experience']]),
    columns=['Age', 'Salary', 'Experience']
)

print("\n=== KNN Imputation ===")
print(df_knn)

# Iterative Imputer (predicts missing values)
iter_imputer = IterativeImputer(random_state=42)
df_iter = pd.DataFrame(
    iter_imputer.fit_transform(df[['Age', 'Salary', 'Experience']]),
    columns=['Age', 'Salary', 'Experience']
)

print("\n=== Iterative Imputation ===")
print(df_iter)
```

### Comparison of Methods

```python
# Generate data with missing values
np.random.seed(42)
true_values = np.random.normal(50, 10, 100)
observed = true_values.copy()
missing_indices = np.random.choice(100, 20, replace=False)
observed[missing_indices] = np.nan

# Compare imputation methods
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error

methods = {
    'Mean': SimpleImputer(strategy='mean'),
    'Median': SimpleImputer(strategy='median'),
    'KNN (k=5)': KNNImputer(n_neighbors=5)
}

results = {}

for name, imputer in methods.items():
    imputed = imputer.fit_transform(observed.reshape(-1, 1)).ravel()
    mse = mean_squared_error(true_values[missing_indices], imputed[missing_indices])
    results[name] = mse
    print(f"{name} MSE: {mse:.2f}")

# Best method has lowest MSE
```

---

## 4.2 Handling Outliers

### Detecting Outliers

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate data with outliers
np.random.seed(42)
data = np.random.normal(100, 15, 200)
outliers = np.array([200, 220, 50, 30])  # Add outliers
data_with_outliers = np.concatenate([data, outliers])

# Method 1: Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram
axes[0].hist(data_with_outliers, bins=30, edgecolor='black')
axes[0].set_title('Histogram')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Boxplot
axes[1].boxplot(data_with_outliers)
axes[1].set_title('Box Plot')
axes[1].set_ylabel('Value')

# Scatter with index
axes[2].scatter(range(len(data_with_outliers)), data_with_outliers)
axes[2].set_title('Scatter Plot')
axes[2].set_xlabel('Index')
axes[2].set_ylabel('Value')

plt.tight_layout()
plt.show()

# Method 2: Z-Score
from scipy import stats

z_scores = np.abs(stats.zscore(data_with_outliers))
threshold = 3
outliers_z = np.where(z_scores > threshold)[0]
print(f"Z-Score method: {len(outliers_z)} outliers detected")
print(f"Outlier indices: {outliers_z}")
print(f"Outlier values: {data_with_outliers[outliers_z]}")

# Method 3: IQR (Interquartile Range)
Q1 = np.percentile(data_with_outliers, 25)
Q3 = np.percentile(data_with_outliers, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = np.where((data_with_outliers < lower_bound) |
                        (data_with_outliers > upper_bound))[0]
print(f"\nIQR method: {len(outliers_iqr)} outliers detected")
print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
```

### Handling Outliers

```python
# Strategy 1: Remove outliers
data_no_outliers = data_with_outliers[(data_with_outliers >= lower_bound) &
                                       (data_with_outliers <= upper_bound)]
print(f"Original size: {len(data_with_outliers)}, After removal: {len(data_no_outliers)}")

# Strategy 2: Cap (winsorize)
data_capped = data_with_outliers.copy()
data_capped[data_capped < lower_bound] = lower_bound
data_capped[data_capped > upper_bound] = upper_bound

# Strategy 3: Transform (log)
data_log = np.log1p(data_with_outliers)  # log1p = log(1 + x)

# Compare distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].boxplot(data_with_outliers)
axes[0].set_title('Original Data')
axes[0].set_ylabel('Value')

axes[1].boxplot(data_capped)
axes[1].set_title('Capped Data')
axes[1].set_ylabel('Value')

axes[2].boxplot(data_log)
axes[2].set_title('Log Transformed')
axes[2].set_ylabel('Log Value')

plt.tight_layout()
plt.show()
```

---

## 4.3 Encoding Categorical Variables

### Types of Encoding

#### 1. Label Encoding (Ordinal)

```python
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Data with ordinal categories (order matters)
df = pd.DataFrame({
    'Size': ['Small', 'Medium', 'Large', 'Small', 'Large', 'Medium']
})

# Manual mapping (when you know the order)
size_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
df['Size_Encoded'] = df['Size'].map(size_mapping)

print(df)

# LabelEncoder (automatic)
le = LabelEncoder()
df['Size_LE'] = le.fit_transform(df['Size'])

print("\nLabel Encoding:")
print(df)

# ⚠️ Use only for ordinal data (Low < Medium < High)
```

#### 2. One-Hot Encoding (Nominal)

```python
# Data with nominal categories (no order)
df = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Green', 'Blue'],
    'Price': [10, 20, 15, 12, 18, 22]
})

# Method 1: pandas get_dummies
df_dummies = pd.get_dummies(df, columns=['Color'], prefix='Color')
print("One-Hot Encoding (pandas):")
print(df_dummies)

# Method 2: sklearn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, drop='first')  # drop='first' avoids dummy trap
color_encoded = ohe.fit_transform(df[['Color']])

print("\nOne-Hot Encoding (sklearn):")
print(color_encoded)
print(f"Categories: {ohe.categories_}")

# ⚠️ Creates many columns if categories are many!
```

#### 3. Target Encoding (Advanced)

```python
import pandas as pd
import numpy as np

# Target encoding uses target variable
df = pd.DataFrame({
    'City': ['NY', 'LA', 'NY', 'Chicago', 'LA', 'NY', 'Chicago', 'LA'],
    'Salary': [70000, 60000, 75000, 55000, 62000, 72000, 58000, 64000]
})

# Calculate mean salary per city
target_encoding = df.groupby('City')['Salary'].mean()
df['City_Encoded'] = df['City'].map(target_encoding)

print("Target Encoding:")
print(df)
print(f"\nEncoding map:")
print(target_encoding)

# ⚠️ Risk of data leakage! Use with cross-validation
```

#### 4. Frequency Encoding

```python
# Encode by frequency of occurrence
df = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'C', 'A', 'B', 'A', 'D', 'A']
})

freq_encoding = df['Category'].value_counts() / len(df)
df['Category_Freq'] = df['Category'].map(freq_encoding)

print("Frequency Encoding:")
print(df)
```

### Choosing Encoding Method

| Data Type              | Encoding Method  | Example                                   |
| ---------------------- | ---------------- | ----------------------------------------- |
| **Ordinal** (ordered)  | Label Encoding   | Education: HighSchool < Bachelor < Master |
| **Nominal** (no order) | One-Hot Encoding | Color: Red, Blue, Green                   |
| **High Cardinality**   | Target/Frequency | City names (1000s of cities)              |
| **Binary**             | Label (0/1)      | Yes/No                                    |

---

## 4.4 Feature Scaling

### Why Scale?

Different features have different ranges. Algorithms like KNN, SVM, Neural Networks are sensitive to scale.

```python
# Example: Impact of scaling
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate data with different scales
np.random.seed(42)
X1 = np.random.randn(100, 1) * 1      # Range: ~[-3, 3]
X2 = np.random.randn(100, 1) * 1000   # Range: ~[-3000, 3000]
X = np.hstack([X1, X2])
y = (X1 + X2 > 0).astype(int).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Without scaling
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
acc_no_scale = accuracy_score(y_test, knn.predict(X_test))

# With scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_scaled = KNeighborsClassifier()
knn_scaled.fit(X_train_scaled, y_train)
acc_scaled = accuracy_score(y_test, knn_scaled.predict(X_test_scaled))

print(f"Accuracy without scaling: {acc_no_scale:.2%}")
print(f"Accuracy with scaling: {acc_scaled:.2%}")
```

### Scaling Methods

#### 1. Standardization (Z-Score Normalization)

```python
from sklearn.preprocessing import StandardScaler

data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

print("Original data:")
print(data)
print("\nStandardized (mean=0, std=1):")
print(data_standardized)
print(f"\nMean: {data_standardized.mean(axis=0)}")
print(f"Std: {data_standardized.std(axis=0)}")

# Formula: (x - mean) / std
# Result: Mean=0, Std=1
# Use when: Data is normally distributed
```

#### 2. Min-Max Scaling (Normalization)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_minmax = scaler.fit_transform(data)

print("Min-Max Scaled (0 to 1):")
print(data_minmax)
print(f"\nMin: {data_minmax.min(axis=0)}")
print(f"\nMax: {data_minmax.max(axis=0)}")

# Formula: (x - min) / (max - min)
# Result: Range [0, 1]
# Use when: Need specific range, no extreme outliers

# Custom range (e.g., -1 to 1)
scaler_custom = MinMaxScaler(feature_range=(-1, 1))
data_custom = scaler_custom.fit_transform(data)
print("\nCustom range (-1 to 1):")
print(data_custom)
```

#### 3. Robust Scaler (for outliers)

```python
from sklearn.preprocessing import RobustScaler

# Data with outliers
data_outliers = np.array([[1], [2], [3], [4], [100]])  # 100 is outlier

robust_scaler = RobustScaler()
data_robust = robust_scaler.fit_transform(data_outliers)

standard_scaler = StandardScaler()
data_standard = standard_scaler.fit_transform(data_outliers)

print("Original:", data_outliers.T)
print("Robust Scaled:", data_robust.T)
print("Standard Scaled:", data_standard.T)

# RobustScaler uses median and IQR, less affected by outliers
```

### When to Use Which Scaler?

| Scaler             | Use When                             | Pros               | Cons                       |
| ------------------ | ------------------------------------ | ------------------ | -------------------------- |
| **StandardScaler** | Normal distribution, ML algorithms   | Most common        | Sensitive to outliers      |
| **MinMaxScaler**   | Need specific range, Neural Networks | Bounded range      | Very sensitive to outliers |
| **RobustScaler**   | Data has outliers                    | Robust to outliers | Less common                |
| **Normalizer**     | Text classification (rare)           | Unit norm          | Special use case           |

---

## 4.5 Feature Creation

### Domain-Based Features

```python
import pandas as pd
import numpy as np

# Example: E-commerce data
df = pd.DataFrame({
    'PurchaseDate': pd.date_range('2023-01-01', periods=10),
    'ItemPrice': [100, 200, 150, 300, 250, 180, 220, 190, 280, 160],
    'Quantity': [1, 2, 1, 3, 2, 1, 2, 1, 3, 1],
    'ShippingCost': [10, 15, 10, 20, 15, 10, 15, 10, 20, 10]
})

# Create new features
df['TotalPrice'] = df['ItemPrice'] * df['Quantity']
df['TotalCost'] = df['TotalPrice'] + df['ShippingCost']
df['PricePerItem'] = df['TotalCost'] / df['Quantity']

# Date features
df['DayOfWeek'] = df['PurchaseDate'].dt.dayofweek
df['Month'] = df['PurchaseDate'].dt.month
df['Quarter'] = df['PurchaseDate'].dt.quarter
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# Binning
df['PriceCategory'] = pd.cut(df['ItemPrice'],
                              bins=[0, 150, 250, 1000],
                              labels=['Low', 'Medium', 'High'])

print(df)
```

### Mathematical Transformations

```python
# Generate skewed data
data = pd.DataFrame({
    'Income': np.random.exponential(50000, 100),
    'Age': np.random.normal(35, 10, 100)
})

# Log transformation (for right-skewed data)
data['Income_Log'] = np.log1p(data['Income'])  # log(1 + x)

# Square root
data['Income_Sqrt'] = np.sqrt(data['Income'])

# Box-Cox transformation
from scipy.stats import boxcox
data['Income_BoxCox'], _ = boxcox(data['Income'] + 1)

# Compare distributions
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(20, 4))

data['Income'].hist(bins=30, ax=axes[0])
axes[0].set_title('Original')

data['Income_Log'].hist(bins=30, ax=axes[1])
axes[1].set_title('Log Transform')

data['Income_Sqrt'].hist(bins=30, ax=axes[2])
axes[2].set_title('Sqrt Transform')

data['Income_BoxCox'].hist(bins=30, ax=axes[3])
axes[3].set_title('Box-Cox Transform')

plt.tight_layout()
plt.show()
```

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

# Original features
X = np.array([[2, 3]])

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print("Original:", X)
print("Polynomial:", X_poly)
print("Features:", poly.get_feature_names_out(['x1', 'x2']))
# Output: ['x1', 'x2', 'x1^2', 'x1*x2', 'x2^2']
```

### Interaction Features

```python
# Example: House price prediction
df = pd.DataFrame({
    'Size': [1500, 2000, 2500],
    'Bedrooms': [3, 4, 4],
    'Bathrooms': [2, 2, 3]
})

# Create interactions
df['Size_per_Bedroom'] = df['Size'] / df['Bedrooms']
df['Size_per_Bathroom'] = df['Size'] / df['Bathrooms']
df['Bedroom_Bath_Ratio'] = df['Bedrooms'] / df['Bathrooms']

print(df)
```

---

## 4.6 Feature Selection

### Why Feature Selection?

- ✅ Reduces overfitting
- ✅ Improves accuracy
- ✅ Reduces training time
- ✅ Improves interpretability

### Method 1: Correlation Analysis

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
df = pd.DataFrame({
    'Feature1': np.random.randn(100),
    'Feature2': np.random.randn(100),
    'Feature3': np.random.randn(100),
    'Feature4': np.random.randn(100),
    'Target': np.random.randn(100)
})

# Add correlation
df['Feature2'] = df['Feature1'] + np.random.randn(100) * 0.1
df['Target'] = df['Feature1'] * 2 + df['Feature3'] + np.random.randn(100) * 0.5

# Correlation matrix
corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Select features highly correlated with target
target_corr = corr_matrix['Target'].abs().sort_values(ascending=False)
print("\nCorrelation with Target:")
print(target_corr)

# Remove highly correlated features (multicollinearity)
threshold = 0.9
to_drop = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            to_drop.add(corr_matrix.columns[j])

print(f"\nFeatures to drop (correlation > {threshold}): {to_drop}")
```

### Method 2: Feature Importance (Tree-based)

```python
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Generate data
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=10, n_informative=5, random_state=42)

feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Feature importance
importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(importances)

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(importances['Feature'], importances['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.show()

# Select top K features
k = 5
selected_features = importances.head(k)['Feature'].tolist()
print(f"\nTop {k} features: {selected_features}")
```

### Method 3: Recursive Feature Elimination (RFE)

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Create model
model = LinearRegression()

# RFE selector
rfe = RFE(estimator=model, n_features_to_select=5)
rfe.fit(X, y)

# Selected features
selected_mask = rfe.support_
selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]

print("\nRFE Selected Features:")
print(selected_features)

# Feature ranking
ranking = pd.DataFrame({
    'Feature': feature_names,
    'Ranking': rfe.ranking_
}).sort_values('Ranking')

print("\nFeature Ranking:")
print(ranking)
```

### Method 4: Statistical Tests

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.datasets import make_classification

# Classification data
X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)

# ANOVA F-test
selector_f = SelectKBest(f_classif, k=5)
X_selected_f = selector_f.fit_transform(X, y)

print("F-test scores:")
print(selector_f.scores_)

# Mutual Information
selector_mi = SelectKBest(mutual_info_classif, k=5)
X_selected_mi = selector_mi.fit_transform(X, y)

print("\nMutual Information scores:")
print(selector_mi.scores_)
```

---

## 4.7 Complete Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
data = {
    'Age': [25, 30, np.nan, 45, 35, np.nan, 50, 40, 28, 55],
    'Salary': [50000, 60000, 70000, np.nan, 65000, 75000, np.nan, 85000, 55000, 90000],
    'City': ['NY', 'LA', 'NY', 'Chicago', 'LA', 'NY', 'Chicago', 'LA', 'NY', 'Chicago'],
    'Purchased': [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

# Separate features and target
X = df.drop('Purchased', axis=1)
y = df['Purchased']

# Define numeric and categorical columns
numeric_features = ['Age', 'Salary']
categorical_features = ['City']

# Create transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(drop='first', sparse=False))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit pipeline
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.1f}%")

# Pipeline handles everything automatically!
print("\n✓ Pipeline handles:")
print("  - Missing value imputation")
print("  - Feature scaling")
print("  - Categorical encoding")
print("  - Model training")
```

---

## ✅ Phase Completion Checklist

- [ ] Handle missing data with appropriate strategies
- [ ] Detect and handle outliers
- [ ] Encode categorical variables correctly
- [ ] Scale features appropriately
- [ ] Create domain-specific features
- [ ] Apply mathematical transformations
- [ ] Select important features
- [ ] Build complete preprocessing pipelines
- [ ] Understand when to use each technique

---

## 🎯 Key Takeaways

1. **Feature engineering** is 70% of ML work
2. **Missing data**: Impute carefully, consider patterns
3. **Outliers**: Detect (IQR, Z-score), handle (remove/cap/transform)
4. **Encoding**: One-hot for nominal, label for ordinal
5. **Scaling**: Essential for distance-based algorithms
6. **Feature creation**: Domain knowledge is key
7. **Feature selection**: More features ≠ better model
8. **Pipelines**: Automate preprocessing to avoid data leakage

---

## 📚 Next Phase

Ready to evaluate and tune your models?
👉 [Phases 5-9: Advanced Machine Learning](./Phase-5-9-Advanced-ML.md)
👉 Practice notebook: [Phase-4-Feature-Engineering.ipynb](./Phase-4-Feature-Engineering.ipynb)
