# 🚀 Machine Learning Crash Course — Quick Revision Guide

> **Time to revise: ~20 minutes** | Covers everything from the `Machine Learning/` folder (Phase 0-9)

---

## 1. What is Machine Learning?

| Traditional Programming    | Machine Learning                   |
| -------------------------- | ---------------------------------- |
| Input + **Rules** → Output | Input + **Output** → Rules (Model) |

**Key Terms:**

- **Features (X)** — input columns (age, salary, etc.)
- **Labels (y)** — what we want to predict
- **Model** — algorithm that learns patterns
- **Training** — model learns from data
- **Testing** — evaluate on unseen data

---

## 2. Types of ML

| Type                | What it does               | Algorithms                  |
| ------------------- | -------------------------- | --------------------------- |
| **Supervised**      | Has labeled data → predict | Regression, Classification  |
| **Unsupervised**    | No labels → find patterns  | Clustering, PCA             |
| **Semi-supervised** | Mix of labeled + unlabeled | —                           |
| **Reinforcement**   | Agent learns via rewards   | Q-learning, Policy Gradient |

---

## 3. ML Workflow (7 Steps)

```
1. Collect Data → 2. Clean/Prepare → 3. Train/Test Split
  → 4. Choose Algorithm → 5. Train Model
  → 6. Evaluate → 7. Predict
```

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 4. Overfitting vs Underfitting

| Problem          | What happens                         | Solution                                                   |
| ---------------- | ------------------------------------ | ---------------------------------------------------------- |
| **Overfitting**  | Works great on training, bad on test | More data, regularization, simpler model, cross-validation |
| **Underfitting** | Bad on both training and test        | More features, complex model, less regularization          |

---

## 5. Supervised Learning — REGRESSION

> Predict a **continuous number** (price, salary, temperature)

### Algorithms:

| Algorithm                 | When to use                            | Code                                                  |
| ------------------------- | -------------------------------------- | ----------------------------------------------------- |
| **Linear Regression**     | Simple linear relationship             | `LinearRegression()`                                  |
| **Polynomial Regression** | Curved/non-linear data                 | `PolynomialFeatures(degree=n)` + `LinearRegression()` |
| **Ridge (L2)**            | Prevent overfitting, keep all features | `Ridge(alpha=1.0)`                                    |
| **Lasso (L1)**            | Feature selection (sets some to 0)     | `Lasso(alpha=0.1)`                                    |
| **ElasticNet**            | Best of Ridge + Lasso                  | `ElasticNet(alpha=0.1, l1_ratio=0.5)`                 |

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Regression Metrics:

| Metric   | What it means                            | Good value           |
| -------- | ---------------------------------------- | -------------------- |
| **MAE**  | Avg absolute error                       | Lower = better       |
| **MSE**  | Avg squared error (penalizes big errors) | Lower = better       |
| **RMSE** | √MSE (same unit as target)               | Lower = better       |
| **R²**   | % of variance explained                  | Closer to 1 = better |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
```

---

## 6. Supervised Learning — CLASSIFICATION

> Predict a **category/class** (spam/not spam, yes/no, cat/dog)

### Algorithms:

| Algorithm               | When to use                       | Key param                          |
| ----------------------- | --------------------------------- | ---------------------------------- |
| **Logistic Regression** | Binary/multi-class, fast baseline | `C` (regularization)               |
| **KNN**                 | Simple, small datasets            | `n_neighbors=5`                    |
| **Naive Bayes**         | Text classification, fast         | `MultinomialNB()` / `GaussianNB()` |
| **Decision Tree**       | Interpretable, visual             | `max_depth`                        |
| **Random Forest**       | Best all-rounder                  | `n_estimators`, `max_depth`        |
| **SVM**                 | High-dimensional data             | `kernel='rbf'`, `C`                |

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Classification Metrics:

| Metric               | What it means                                      |
| -------------------- | -------------------------------------------------- |
| **Accuracy**         | % correct predictions (misleading if imbalanced)   |
| **Precision**        | Of predicted positives, how many are correct?      |
| **Recall**           | Of actual positives, how many did we catch?        |
| **F1 Score**         | Harmonic mean of Precision & Recall                |
| **ROC-AUC**          | Area under ROC curve (0.5 = random, 1.0 = perfect) |
| **Confusion Matrix** | Table of TP, TN, FP, FN                            |

```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
```

---

## 7. Unsupervised Learning

### Clustering (Group similar data)

| Algorithm        | How it works                  | Key param                      |
| ---------------- | ----------------------------- | ------------------------------ |
| **K-Means**      | Assign to K centroids         | `n_clusters`, use Elbow method |
| **Hierarchical** | Build tree of clusters        | `linkage='ward'`               |
| **DBSCAN**       | Density-based, finds outliers | `eps`, `min_samples`           |

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
```

**Elbow Method:** Plot inertia vs K → pick the "elbow" point

### Dimensionality Reduction

| Algorithm | Purpose                                   |
| --------- | ----------------------------------------- |
| **PCA**   | Reduce features, keep variance            |
| **t-SNE** | 2D visualization of high-dimensional data |

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)  # or n_components=0.95 for 95% variance
X_reduced = pca.fit_transform(X)
```

---

## 8. Feature Engineering (70% of real ML Work!)

### Missing Data

```python
from sklearn.impute import SimpleImputer, KNNImputer
imputer = SimpleImputer(strategy='mean')      # or 'median', 'most_frequent'
X_filled = imputer.fit_transform(X)
```

### Outliers

- **Detect:** Box plot, Z-score (`|z| > 3`), IQR method
- **Handle:** Remove, cap/winsorize, log transform

### Encoding Categorical Variables

| Method               | When to use                    |
| -------------------- | ------------------------------ |
| **Label Encoding**   | Ordinal data (low/medium/high) |
| **One-Hot Encoding** | Nominal data (red/blue/green)  |
| **Target Encoding**  | High cardinality features      |

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
pd.get_dummies(df, columns=['color'])     # Easiest one-hot
```

### Feature Scaling

| Scaler             | Formula                 | When               |
| ------------------ | ----------------------- | ------------------ |
| **StandardScaler** | (x - mean) / std        | Most algorithms    |
| **MinMaxScaler**   | (x - min) / (max - min) | Neural networks    |
| **RobustScaler**   | (x - median) / IQR      | Data with outliers |

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

### Feature Selection

```python
# Correlation
df.corr()['target'].sort_values()

# Feature importance (Random Forest)
model.feature_importances_

# RFE (Recursive Feature Elimination)
from sklearn.feature_selection import RFE
selector = RFE(model, n_features_to_select=5)
X_selected = selector.fit_transform(X, y)
```

### Sklearn Pipeline (Automate everything!)

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
pipe.fit(X_train, y_train)
```

---

## 9. Model Evaluation & Tuning

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Hyperparameter Tuning

| Method                 | Speed                   | Quality |
| ---------------------- | ----------------------- | ------- |
| **GridSearchCV**       | Slow (tries all combos) | Best    |
| **RandomizedSearchCV** | Fast (random sample)    | Good    |

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(grid.best_params_)
```

---

## 10. Ensemble Methods

| Method                | How it works                                   |
| --------------------- | ---------------------------------------------- |
| **Bagging**           | Train multiple models on random subsets → vote |
| **Random Forest**     | Bagging + random feature subsets               |
| **AdaBoost**          | Focus on mistakes of previous model            |
| **Gradient Boosting** | Each model corrects previous errors            |
| **XGBoost**           | Optimized gradient boosting (fast!)            |
| **LightGBM**          | Even faster, leaf-wise growth                  |
| **Stacking**          | Use predictions of models as features          |

```python
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)
```

---

## 11. ML in Production

```python
# Save model
import pickle
pickle.dump(model, open('model.pkl', 'wb'))

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# FastAPI deployment
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    prediction = model.predict([data['features']])
    return {"prediction": prediction.tolist()}
```

---

## 12. Algorithm Selection Quick Guide

| Problem                      | Start With          | Then Try                      |
| ---------------------------- | ------------------- | ----------------------------- |
| **Regression**               | Linear Regression   | Ridge, Random Forest, XGBoost |
| **Binary Classification**    | Logistic Regression | Random Forest, SVM, XGBoost   |
| **Multi-class**              | Random Forest       | SVM, XGBoost                  |
| **Clustering**               | K-Means             | DBSCAN, Hierarchical          |
| **Dimensionality Reduction** | PCA                 | t-SNE (visualization only)    |
| **Text Classification**      | Naive Bayes         | SVM, Random Forest            |

---

## 🧾 Complete ML Flow Cheat Sheet

```
Data → Clean (missing, outliers) → Encode (categorical) → Scale (features)
     → Split (train/test) → Train (algorithm) → Evaluate (metrics)
     → Tune (GridSearch/CV) → Final Model → Deploy (pickle + API)
```

---

## 📋 Key Imports

```python
# Data
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

---

> **Source folder**: `Machine Learning/` (Phase-0 through Phase-9, all .md and .ipynb files)
