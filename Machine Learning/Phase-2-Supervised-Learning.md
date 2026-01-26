# Phase 2: Supervised Learning

## 🎯 Learning Objectives

By the end of this phase, you will:

- ✅ Master regression algorithms for predicting numbers
- ✅ Master classification algorithms for predicting categories
- ✅ Understand evaluation metrics for both types
- ✅ Build end-to-end prediction models
- ✅ Choose the right algorithm for your problem

**Time Required:** 3-4 weeks ⭐ MOST IMPORTANT PHASE  
**Difficulty:** Intermediate  
**Prerequisites:** Phases 0-1 completed

---

# Part A: Regression (Predicting Numbers)

## 📊 What is Regression?

**Goal:** Predict a continuous numerical value.

**Examples:**

- 🏠 House prices ($200,000)
- 🌡️ Temperature (25.5°C)
- 📈 Stock prices ($150.75)
- 🚗 Car fuel efficiency (32.5 mpg)
- 💰 Salary ($75,000)

---

## 2.1 Linear Regression

### Concept

Find the best-fit straight line through data points.

**Formula:** $y = mx + b$

- $m$ = slope
- $b$ = y-intercept
- $x$ = input feature
- $y$ = prediction

### Simple Example: Salary Prediction

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Data: Years of experience vs Salary
experience = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
salary = np.array([30000, 35000, 42000, 48000, 55000, 60000, 67000, 75000, 82000, 90000])

# Create and train model
model = LinearRegression()
model.fit(experience, salary)

# Make predictions
predictions = model.predict(experience)

# Model parameters
print(f"Slope (m): ${model.coef_[0]:.2f} per year")
print(f"Intercept (b): ${model.intercept_:.2f}")
print(f"R² Score: {r2_score(salary, predictions):.3f}")

# Predict for new value
new_experience = np.array([[12]])
predicted_salary = model.predict(new_experience)
print(f"\nPredicted salary for 12 years: ${predicted_salary[0]:.2f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(experience, salary, color='blue', s=100, label='Actual data')
plt.plot(experience, predictions, color='red', linewidth=2, label='Linear regression line')
plt.scatter(new_experience, predicted_salary, color='green', s=200, marker='*',
            label='Prediction for 12 years')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.title('Salary Prediction using Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Understanding the Math (Simplified)

```python
# What the model does internally:
# 1. Try random line: y = m*x + b
# 2. Calculate error (how far off predictions are)
# 3. Adjust m and b to reduce error
# 4. Repeat until error is minimized

# Cost Function (what we minimize)
def cost_function(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

# Example
y_true = np.array([100, 200, 300])
y_pred_bad = np.array([150, 150, 150])
y_pred_good = np.array([110, 190, 310])

print(f"Bad predictions cost: {cost_function(y_true, y_pred_bad):.2f}")
print(f"Good predictions cost: {cost_function(y_true, y_pred_good):.2f}")
```

---

## 2.2 Multiple Linear Regression

### Concept

Use **multiple features** to make predictions.

**Formula:** $y = b_0 + b_1x_1 + b_2x_2 + b_3x_3 + ...$

### Example: House Price Prediction

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create dataset
data = {
    'Size': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    'Bedrooms': [3, 3, 2, 4, 2, 3, 4, 4, 3, 3],
    'Age': [0, 10, 15, 2, 20, 8, 5, 3, 12, 7],
    'Price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000,
              319000, 255000]
}
df = pd.DataFrame(data)

print(df.head())
print(f"\nDataset shape: {df.shape}")

# Features and target
X = df[['Size', 'Bedrooms', 'Age']]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n=== Model Performance ===")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"Mean Absolute Error: ${mean_absolute_error(y_test, y_pred):.2f}")
print(f"Root Mean Squared Error: ${np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Coefficients
print("\n=== Model Coefficients ===")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: ${coef:.2f}")
print(f"Intercept: ${model.intercept_:.2f}")

# Predict new house
new_house = pd.DataFrame({
    'Size': [2000],
    'Bedrooms': [3],
    'Age': [5]
})
predicted_price = model.predict(new_house)
print(f"\n=== New Prediction ===")
print(f"House: {new_house.values[0]}")
print(f"Predicted Price: ${predicted_price[0]:.2f}")

# Compare actual vs predicted
comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Difference': y_test.values - y_pred
})
print("\n=== Test Set Predictions ===")
print(comparison)
```

### Feature Importance

```python
import matplotlib.pyplot as plt

# Visualize feature importance
features = X.columns
coefficients = model.coef_

plt.figure(figsize=(10, 6))
plt.barh(features, coefficients)
plt.xlabel('Coefficient Value')
plt.title('Feature Importance')
plt.grid(True, alpha=0.3)
plt.show()

# Interpretation:
# Positive coefficient = feature increases price
# Negative coefficient = feature decreases price
# Larger magnitude = more important
```

---

## 2.3 Polynomial Regression

### Concept

Fit a curve instead of a straight line for non-linear data.

**Formula:** $y = b_0 + b_1x + b_2x^2 + b_3x^3 + ...$

### Example: Product Sales over Time

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Non-linear data (sales curve)
months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
sales = np.array([20, 25, 35, 50, 70, 95, 115, 130, 140, 145, 148, 150])

# Try different polynomial degrees
degrees = [1, 2, 3, 5]
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, degree in enumerate(degrees):
    # Create polynomial model
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(months, sales)

    # Predictions
    months_plot = np.linspace(1, 12, 100).reshape(-1, 1)
    sales_pred = model.predict(months_plot)

    # Calculate R²
    r2 = model.score(months, sales)

    # Plot
    axes[idx].scatter(months, sales, color='blue', s=100, label='Actual data')
    axes[idx].plot(months_plot, sales_pred, color='red', linewidth=2,
                   label=f'Degree {degree} (R²={r2:.3f})')
    axes[idx].set_xlabel('Month')
    axes[idx].set_ylabel('Sales (1000s)')
    axes[idx].set_title(f'Polynomial Regression (Degree {degree})')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Best model (degree 2 or 3)
best_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
best_model.fit(months, sales)

# Predict future months
future_months = np.array([[13], [14], [15]])
future_sales = best_model.predict(future_months)

print("=== Future Predictions ===")
for month, sale in zip(future_months.ravel(), future_sales):
    print(f"Month {month}: {sale:.2f}k sales")
```

### Choosing Polynomial Degree

```python
from sklearn.metrics import mean_squared_error

# Test different degrees
degrees_range = range(1, 10)
train_errors = []
test_errors = []

# Split data
X_train, X_test = months[:9], months[9:]
y_train, y_test = sales[:9], sales[9:]

for degree in degrees_range:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_errors.append(mean_squared_error(y_train, train_pred))
    test_errors.append(mean_squared_error(y_test, test_pred))

# Plot errors
plt.figure(figsize=(10, 6))
plt.plot(degrees_range, train_errors, 'o-', label='Training Error')
plt.plot(degrees_range, test_errors, 's-', label='Test Error')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Model Complexity vs Error')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Best degree is where test error is minimum
best_degree = degrees_range[np.argmin(test_errors)]
print(f"Best polynomial degree: {best_degree}")
```

---

## 2.4 Ridge Regression (L2 Regularization)

### Concept

Prevents overfitting by penalizing large coefficients.

**Use When:**

- Many features
- Features are correlated
- Want to prevent overfitting

### Example

```python
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate data with many features
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 20)  # 20 features
y = X[:, 0] + 2 * X[:, 1] + np.random.randn(n_samples) * 0.1  # Only 2 features matter

# Split data
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Scale features (important for regularization!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=0.1)': Ridge(alpha=0.1),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0),
    'Ridge (alpha=10)': Ridge(alpha=10)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"{name}:")
    print(f"  Train R²: {train_score:.3f}")
    print(f"  Test R²: {test_score:.3f}")
    print(f"  Coefficient sum: {np.sum(np.abs(model.coef_)):.2f}")
    print()
```

### Visualizing Regularization

```python
import matplotlib.pyplot as plt

# Train models with different alpha values
alphas = np.logspace(-2, 3, 50)
coefficients = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    coefficients.append(ridge.coef_)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(alphas, coefficients)
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regression: Coefficient vs Alpha')
plt.grid(True, alpha=0.3)
plt.show()

# As alpha increases, coefficients shrink towards zero
```

---

## 2.5 Lasso Regression (L1 Regularization)

### Concept

Like Ridge, but can reduce coefficients to exactly zero (feature selection).

**Use When:**

- Want automatic feature selection
- Many features, some irrelevant
- Need interpretable model

### Example

```python
from sklearn.linear_model import Lasso

# Same data as before
# Lasso will automatically identify important features

alphas = [0.01, 0.1, 1.0, 10.0]

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_scaled, y_train)

    # Count non-zero coefficients
    non_zero = np.sum(lasso.coef_ != 0)

    print(f"Alpha = {alpha}")
    print(f"  Non-zero features: {non_zero}/20")
    print(f"  Train R²: {lasso.score(X_train_scaled, y_train):.3f}")
    print(f"  Test R²: {lasso.score(X_test_scaled, y_test):.3f}")
    print(f"  Top 5 coefficients: {sorted(np.abs(lasso.coef_), reverse=True)[:5]}")
    print()
```

### Ridge vs Lasso Comparison

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Ridge coefficients
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ax1.bar(range(len(ridge.coef_)), ridge.coef_)
ax1.set_title('Ridge Regression Coefficients')
ax1.set_xlabel('Feature Index')
ax1.set_ylabel('Coefficient Value')
ax1.grid(True, alpha=0.3)

# Lasso coefficients
lasso = Lasso(alpha=1.0)
lasso.fit(X_train_scaled, y_train)
ax2.bar(range(len(lasso.coef_)), lasso.coef_)
ax2.set_title('Lasso Regression Coefficients')
ax2.set_xlabel('Feature Index')
ax2.set_ylabel('Coefficient Value')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Notice: Lasso sets many coefficients to exactly 0
```

---

## 2.6 ElasticNet (L1 + L2)

### Concept

Combination of Ridge and Lasso.

```python
from sklearn.linear_model import ElasticNet

# ElasticNet has two parameters:
# - alpha: overall regularization strength
# - l1_ratio: balance between L1 and L2 (0=Ridge, 1=Lasso)

elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)

print(f"ElasticNet Performance:")
print(f"  Train R²: {elastic.score(X_train_scaled, y_train):.3f}")
print(f"  Test R²: {elastic.score(X_test_scaled, y_test):.3f}")
print(f"  Non-zero features: {np.sum(elastic.coef_ != 0)}/20")
```

---

## 📊 Regression Metrics

### 1. Mean Absolute Error (MAE)

Average absolute difference between predictions and actual values.

```python
from sklearn.metrics import mean_absolute_error

y_true = [100, 200, 300, 400]
y_pred = [110, 190, 310, 390]

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae}")  # Output: 10.0

# Interpretation: On average, predictions are off by 10 units
```

**Pros:** Easy to interpret, same units as target  
**Cons:** Doesn't penalize large errors heavily

### 2. Mean Squared Error (MSE)

Average of squared differences.

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse}")  # Output: 125.0

# Root MSE (back to original units)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse}")  # Output: 11.18
```

**Pros:** Penalizes large errors  
**Cons:** Sensitive to outliers, different units (squared)

### 3. R² Score (Coefficient of Determination)

Proportion of variance explained by the model.

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R²: {r2}")  # Output: 0.975

# Interpretation:
# 1.0 = perfect predictions
# 0.0 = model as good as predicting mean
# negative = model worse than predicting mean
```

**Pros:** Standardized (0-1), easy to interpret  
**Cons:** Can be misleading with non-linear data

### Complete Example

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_regression(y_true, y_pred, model_name="Model"):
    """Comprehensive regression evaluation"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"=== {model_name} Evaluation ===")
    print(f"MAE:  {mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.3f}")

    # Visual comparison
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', linewidth=2, label='Perfect predictions')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Example usage
y_true = np.array([100, 150, 200, 250, 300])
y_pred = np.array([110, 145, 210, 240, 295])

evaluate_regression(y_true, y_pred, "Linear Regression")
```

---

## 🎯 Regression Algorithm Selection Guide

| Scenario                              | Recommended Algorithm     |
| ------------------------------------- | ------------------------- |
| Linear relationship, few features     | **Linear Regression**     |
| Non-linear relationship               | **Polynomial Regression** |
| Many features, some correlated        | **Ridge Regression**      |
| Many features, want feature selection | **Lasso Regression**      |
| Complex scenario, need balance        | **ElasticNet**            |

---

## 📝 Mini-Project: House Price Prediction

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Create comprehensive dataset
np.random.seed(42)
n_samples = 200

data = {
    'Size': np.random.randint(1000, 4000, n_samples),
    'Bedrooms': np.random.randint(1, 6, n_samples),
    'Bathrooms': np.random.randint(1, 4, n_samples),
    'Age': np.random.randint(0, 50, n_samples),
    'Location_Score': np.random.randint(1, 11, n_samples),
    'HasGarage': np.random.choice([0, 1], n_samples),
    'HasPool': np.random.choice([0, 1], n_samples)
}

# Calculate price based on features
data['Price'] = (
    data['Size'] * 100 +
    data['Bedrooms'] * 10000 +
    data['Bathrooms'] * 15000 -
    data['Age'] * 1000 +
    data['Location_Score'] * 20000 +
    data['HasGarage'] * 25000 +
    data['HasPool'] * 30000 +
    np.random.randn(n_samples) * 20000  # Add noise
)

df = pd.DataFrame(data)

print("Dataset Overview:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nStatistics:")
print(df.describe())

# Prepare data
X = df.drop('Price', axis=1)
y = df['Price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=10),
    'Lasso': Lasso(alpha=10)
}

results = {}

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {'MAE': mae, 'R2': r2, 'predictions': y_pred}

    print(f"\n=== {name} ===")
    print(f"MAE: ${mae:,.2f}")
    print(f"R²: {r2:.3f}")

# Compare models visually
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    y_pred = results[name]['predictions']

    axes[idx].scatter(y_test, y_pred, alpha=0.6)
    axes[idx].plot([y_test.min(), y_test.max()],
                   [y_test.min(), y_test.max()],
                   'r--', linewidth=2)
    axes[idx].set_xlabel('Actual Price')
    axes[idx].set_ylabel('Predicted Price')
    axes[idx].set_title(f'{name}\nR²={results[name]["R2"]:.3f}')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Feature importance
print("\n=== Feature Importance (Linear Regression) ===")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': models['Linear Regression'].coef_
}).sort_values('Coefficient', ascending=False)

print(feature_importance)

# Predict new house
new_house = pd.DataFrame({
    'Size': [2500],
    'Bedrooms': [4],
    'Bathrooms': [3],
    'Age': [5],
    'Location_Score': [8],
    'HasGarage': [1],
    'HasPool': [1]
})

new_house_scaled = scaler.transform(new_house)
prediction = models['Linear Regression'].predict(new_house_scaled)

print(f"\n=== New House Prediction ===")
print(new_house.T)
print(f"\nPredicted Price: ${prediction[0]:,.2f}")
```

---

**Continued in next part: [Classification Algorithms](./Phase-2-Supervised-Learning-Classification.md)**

---

## ✅ Regression Checklist

- [ ] Understand when to use regression
- [ ] Implement Linear Regression
- [ ] Use Multiple Linear Regression with multiple features
- [ ] Apply Polynomial Regression for non-linear data
- [ ] Use Ridge for regularization
- [ ] Use Lasso for feature selection
- [ ] Calculate MAE, MSE, RMSE, R²
- [ ] Interpret regression coefficients
- [ ] Complete mini-project

---

## 📚 Next Section

👉 Continue to [Classification Algorithms](./Phase-2-Supervised-Learning-Classification.md) to learn how to predict categories!
