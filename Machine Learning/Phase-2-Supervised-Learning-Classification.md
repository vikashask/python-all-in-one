# Phase 2B: Classification Algorithms

## 🎯 What is Classification?

**Goal:** Predict which category (class) something belongs to.

**Examples:**

- 📧 Email: Spam or Not Spam
- 🏥 Patient: Disease or Healthy
- 💳 Transaction: Fraud or Legitimate
- 🌦️ Weather: Sunny, Rainy, or Cloudy
- 🐱 Image: Cat, Dog, or Bird

---

## 2.7 Logistic Regression

### Concept

Despite the name, it's for **classification**, not regression! Predicts probability of belonging to a class.

**Formula:** $P(y=1) = \frac{1}{1 + e^{-(mx + b)}}$  
Output: Probability between 0 and 1

### Binary Classification Example: Disease Prediction

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate data: Age vs Disease (0=Healthy, 1=Diseased)
np.random.seed(42)

# Healthy people (younger, lower risk)
healthy_age = np.random.normal(35, 10, 100)
healthy_labels = np.zeros(100)

# Diseased people (older, higher risk)
diseased_age = np.random.normal(55, 10, 100)
diseased_labels = np.ones(100)

# Combine
X = np.concatenate([healthy_age, diseased_age]).reshape(-1, 1)
y = np.concatenate([healthy_labels, diseased_labels])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)  # Probabilities

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.1f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Diseased']))

# Visualize
plt.figure(figsize=(12, 5))

# Plot 1: Data points
plt.subplot(1, 2, 1)
plt.scatter(X_train[y_train==0], y_train[y_train==0], color='blue', label='Healthy', alpha=0.6)
plt.scatter(X_train[y_train==1], y_train[y_train==1], color='red', label='Diseased', alpha=0.6)

# Decision boundary
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_plot_proba = model.predict_proba(X_plot)[:, 1]
plt.plot(X_plot, y_plot_proba, color='green', linewidth=2, label='Probability curve')
plt.axhline(y=0.5, color='black', linestyle='--', label='Decision boundary')
plt.xlabel('Age')
plt.ylabel('Probability of Disease')
plt.title('Logistic Regression: Disease Prediction')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Probability distribution
plt.subplot(1, 2, 2)
for i in range(len(X_test)):
    color = 'red' if y_test.iloc[i] == 1 else 'blue'
    plt.scatter(X_test.iloc[i], y_pred_proba[i, 1], color=color, alpha=0.6)
plt.axhline(y=0.5, color='black', linestyle='--', label='Threshold')
plt.xlabel('Age')
plt.ylabel('Predicted Probability')
plt.title('Test Set Predictions')
plt.legend(['Threshold=0.5', 'Diseased', 'Healthy'])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Predict for new patients
new_patients = np.array([[30], [45], [60], [70]])
predictions = model.predict(new_patients)
probabilities = model.predict_proba(new_patients)

print("\n=== New Patient Predictions ===")
for age, pred, prob in zip(new_patients, predictions, probabilities):
    status = "Diseased" if pred == 1 else "Healthy"
    confidence = prob[int(pred)] * 100
    print(f"Age {age[0]}: {status} (Confidence: {confidence:.1f}%)")
```

### Multi-class Classification Example: Iris Dataset

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.1f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Iris Classification')
plt.show()

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## 2.8 K-Nearest Neighbors (KNN)

### Concept

Classify based on the K nearest training examples.

**How it works:**

1. Find K nearest neighbors to new point
2. Take majority vote
3. Assign that class

**Analogy:** "You are the average of your 5 closest friends"

### Example: Simple Classification

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1,
                          random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with different K values
k_values = [1, 3, 5, 10, 20]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, k in enumerate(k_values):
    # Train
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    axes[idx].scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                     cmap='RdYlBu', edgecolors='black', s=50, alpha=0.7)
    axes[idx].set_title(f'K={k}, Accuracy={accuracy:.2f}')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

# Remove extra subplot
fig.delaxes(axes[5])
plt.tight_layout()
plt.show()
```

### Choosing K

```python
from sklearn.model_selection import cross_val_score

# Test different K values
k_range = range(1, 31)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Use cross-validation for better estimate
    cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    scores.append(cv_scores.mean())

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_range, scores, marker='o')
plt.xlabel('K Value')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Finding Optimal K')
plt.grid(True, alpha=0.3)
plt.show()

# Best K
best_k = k_range[np.argmax(scores)]
print(f"Best K: {best_k}")
print(f"Best Accuracy: {max(scores):.3f}")
```

**Pros of KNN:**

- ✅ Simple to understand
- ✅ No training phase
- ✅ Works well with non-linear data

**Cons of KNN:**

- ❌ Slow for large datasets
- ❌ Sensitive to feature scaling
- ❌ Struggles with high dimensions

---

## 2.9 Naive Bayes

### Concept

Based on Bayes' theorem: Calculate probability of each class and pick the highest.

**Use Cases:**

- 📧 Spam detection (text classification)
- 📄 Document categorization
- 🩺 Medical diagnosis

### Example: Email Spam Detection

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample emails
emails = [
    "win free money now",
    "claim your prize today",
    "meeting scheduled for tomorrow",
    "project update required",
    "congratulations you won lottery",
    "team lunch at noon",
    "click here for free gift",
    "quarterly report attached",
    "limited time offer act now",
    "please review the document",
    "get rich quick scheme",
    "conference call at 3pm",
    "buy now save 50 percent",
    "status update needed",
    "earn money from home",
    "budget planning meeting"
]

# Labels: 1=Spam, 0=Not Spam
labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Train
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.1f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))

# Test new emails
new_emails = [
    "urgent meeting tomorrow",
    "free money click now",
    "project deadline approaching",
    "win prize claim now"
]

X_new = vectorizer.transform(new_emails)
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)

print("\n=== New Email Predictions ===")
for email, pred, prob in zip(new_emails, predictions, probabilities):
    label = "SPAM" if pred == 1 else "NOT SPAM"
    confidence = max(prob) * 100
    print(f"'{email}'")
    print(f"  → {label} (Confidence: {confidence:.1f}%)\n")
```

### Gaussian Naive Bayes (for numerical features)

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict
y_pred = gnb.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Gaussian Naive Bayes Accuracy: {accuracy * 100:.1f}%")
```

---

## 2.10 Decision Tree

### Concept

Creates a tree of decisions based on feature values.

**How it works:**

1. Split data based on feature that best separates classes
2. Repeat for each subset
3. Stop when classes are pure or max depth reached

### Example with Visualization

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X = iris.data[:, :2]  # Use only 2 features for visualization
y = iris.target

# Train
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=iris.feature_names[:2],
          class_names=iris.target_names,
          filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree: Iris Classification')
plt.show()

# Predict
y_pred = dt.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy * 100:.1f}%")
```

### Decision Boundary Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                          n_informative=2, random_state=42)

# Train with different max_depth
depths = [2, 5, 10, 20]
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, depth in enumerate(depths):
    # Train
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X, y)

    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                     edgecolors='black', s=50)
    axes[idx].set_title(f'Max Depth = {depth}')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Notice: Higher depth = more complex boundaries = risk of overfitting
```

### Feature Importance

```python
# Train on full iris dataset
iris = load_iris()
X = iris.data
y = iris.target

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X, y)

# Feature importance
importances = dt.feature_importances_
features = iris.feature_names

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel('Importance')
plt.title('Feature Importance in Decision Tree')
plt.grid(True, alpha=0.3)
plt.show()

for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.3f}")
```

**Pros:**

- ✅ Easy to interpret
- ✅ Handles non-linear data
- ✅ No feature scaling needed
- ✅ Shows feature importance

**Cons:**

- ❌ Prone to overfitting
- ❌ Unstable (small data changes = different tree)
- ❌ Not great with small datasets

---

## 2.11 Random Forest

### Concept

Ensemble of many decision trees voting together. **"Wisdom of the crowd"**

**How it works:**

1. Create multiple decision trees
2. Each tree trained on random subset of data
3. Each tree votes
4. Final prediction = majority vote

### Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Compare Decision Tree vs Random Forest
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train both
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predict
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)

# Compare
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

print("\n=== Random Forest Classification Report ===")
print(classification_report(y_test, rf_pred, target_names=iris.target_names))
```

### Feature Importance in Random Forest

```python
# Feature importance
importances = rf.feature_importances_
features = iris.feature_names
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance: Random Forest')
plt.tight_layout()
plt.show()
```

### Tuning Random Forest

```python
from sklearn.model_selection import GridSearchCV

# Parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Grid search
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Use best model
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
```

**Pros:**

- ✅ More accurate than single decision tree
- ✅ Less prone to overfitting
- ✅ Handles large datasets well
- ✅ Feature importance

**Cons:**

- ❌ Slower than decision tree
- ❌ Less interpretable
- ❌ Requires more memory

---

## 2.12 Support Vector Machine (SVM)

### Concept

Find the best boundary (hyperplane) that separates classes with maximum margin.

**Intuition:** Draw a line that's as far as possible from both classes.

### Linear SVM Example

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

# Generate linearly separable data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1,
                          class_sep=2, random_state=42)

# Train
svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

# Plot
plt.figure(figsize=(10, 6))

# Decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
           alpha=0.5, linestyles=['--', '-', '--'])

# Plot support vectors
ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
           s=200, facecolors='none', edgecolors='k', linewidths=2)

# Plot data points
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=50, edgecolors='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Support Vector Machine: Linear Kernel')
plt.colorbar(scatter)
plt.show()

print(f"Number of support vectors: {len(svm.support_vectors_)}")
```

### Non-Linear SVM (RBF Kernel)

```python
from sklearn.datasets import make_moons

# Generate non-linear data
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

# Compare different kernels
kernels = ['linear', 'rbf', 'poly']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, kernel in enumerate(kernels):
    # Train
    svm = SVC(kernel=kernel, gamma='auto')
    svm.fit(X, y)

    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                     edgecolors='black', s=50)
    axes[idx].set_title(f'SVM: {kernel.upper()} Kernel')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

**Pros:**

- ✅ Effective in high dimensions
- ✅ Works well with clear margin
- ✅ Memory efficient (uses support vectors)

**Cons:**

- ❌ Slow on large datasets
- ❌ Sensitive to feature scaling
- ❌ Hard to interpret

---

## 📊 Classification Metrics

### 1. Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Example predictions
y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 1, 0, 0, 1, 0, 1, 1]

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Components:
# True Positive (TP): Predicted 1, Actual 1
# True Negative (TN): Predicted 0, Actual 0
# False Positive (FP): Predicted 1, Actual 0 (Type I Error)
# False Negative (FN): Predicted 0, Actual 1 (Type II Error)
```

### 2. Accuracy, Precision, Recall, F1-Score

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Accuracy: Overall correctness
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy * 100:.1f}%")
# Formula: (TP + TN) / (TP + TN + FP + FN)

# Precision: Of all predicted positives, how many were correct?
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision * 100:.1f}%")
# Formula: TP / (TP + FP)
# Use when: False Positives are costly (e.g., spam detection)

# Recall: Of all actual positives, how many did we catch?
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall * 100:.1f}%")
# Formula: TP / (TP + FN)
# Use when: False Negatives are costly (e.g., disease detection)

# F1-Score: Harmonic mean of precision and recall
f1 = f1_score(y_true, y_pred)
print(f"F1-Score: {f1 * 100:.1f}%")
# Formula: 2 * (Precision * Recall) / (Precision + Recall)
# Use when: Need balance between precision and recall
```

### Real-World Example: Medical Diagnosis

```python
# Scenario: Detecting rare disease (1% of population)

# Bad Model 1: Always predict "No disease"
y_true = [0]*99 + [1]*1  # 99 healthy, 1 diseased
y_pred_bad = [0]*100      # Always predict healthy

print("Model 1 (Always predict healthy):")
print(f"  Accuracy: {accuracy_score(y_true, y_pred_bad) * 100:.1f}%")  # 99%!
print(f"  Recall: {recall_score(y_true, y_pred_bad) * 100:.1f}%")      # 0%!
print("  → Misses all diseased patients!\n")

# Good Model: Catches disease
y_pred_good = [0]*90 + [1]*9 + [0]*1  # Some false positives, but catches disease
print("Model 2 (Balanced):")
print(f"  Accuracy: {accuracy_score(y_true, y_pred_good) * 100:.1f}%")
print(f"  Recall: {recall_score(y_true, y_pred_good) * 100:.1f}%")
print("  → Catches diseased patients!")
```

**Lesson:** Accuracy alone can be misleading with imbalanced data!

### 3. ROC Curve and AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Get probability predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# AUC interpretation:
# 1.0 = Perfect classifier
# 0.5 = Random classifier
# < 0.5 = Worse than random
```

---

## 🎯 Algorithm Selection Guide

| Scenario                            | Recommended Algorithm   |
| ----------------------------------- | ----------------------- |
| Linear separation, interpretability | **Logistic Regression** |
| Small dataset, fast prediction      | **KNN**                 |
| Text classification                 | **Naive Bayes**         |
| Interpretable rules                 | **Decision Tree**       |
| High accuracy, less overfitting     | **Random Forest**       |
| High-dimensional data               | **SVM (RBF)**           |

---

## 📝 Complete Classification Project: Customer Churn Prediction

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create dataset
np.random.seed(42)
n_samples = 500

data = {
    'Age': np.random.randint(18, 70, n_samples),
    'Tenure_Months': np.random.randint(1, 72, n_samples),
    'Monthly_Charges': np.random.uniform(20, 100, n_samples),
    'Total_Charges': np.random.uniform(100, 7000, n_samples),
    'Num_Products': np.random.randint(1, 5, n_samples),
    'Support_Calls': np.random.randint(0, 10, n_samples),
    'Contract_Type': np.random.choice(['Monthly', 'Annual', '2-Year'], n_samples)
}

# Calculate churn (synthetic logic)
churn_probability = (
    (70 - data['Age']) / 100 +
    (10 - data['Tenure_Months']) / 100 +
    data['Support_Calls'] / 20 +
    (data['Contract_Type'] == 'Monthly') * 0.3
)
data['Churn'] = (churn_probability + np.random.randn(n_samples) * 0.1 > 0.5).astype(int)

df = pd.DataFrame(data)

print("Dataset Overview:")
print(df.head())
print(f"\nChurn Distribution:")
print(df['Churn'].value_counts())
print(f"\nChurn Rate: {df['Churn'].mean() * 100:.1f}%")

# Prepare data
# Encode categorical
le = LabelEncoder()
df['Contract_Type_Encoded'] = le.fit_transform(df['Contract_Type'])

# Features and target
X = df.drop(['Churn', 'Contract_Type'], axis=1)
y = df['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

results = {}

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = {
        'accuracy': accuracy,
        'predictions': y_pred
    }

    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy * 100:.1f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stay', 'Churn']))

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, (name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['predictions'])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Feature importance (Random Forest)
rf = models['Random Forest']
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== Feature Importance (Random Forest) ===")
print(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance: Customer Churn')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Predict new customers
new_customers = pd.DataFrame({
    'Age': [25, 55],
    'Tenure_Months': [3, 48],
    'Monthly_Charges': [80, 45],
    'Total_Charges': [240, 2160],
    'Num_Products': [1, 3],
    'Support_Calls': [5, 1],
    'Contract_Type_Encoded': [0, 2]  # Monthly, 2-Year
})

new_customers_scaled = scaler.transform(new_customers)
best_model = models['Random Forest']
predictions = best_model.predict(new_customers_scaled)

print("\n=== New Customer Predictions ===")
for idx, pred in enumerate(predictions):
    status = "WILL CHURN" if pred == 1 else "WILL STAY"
    print(f"Customer {idx+1}: {status}")
    print(new_customers.iloc[idx])
    print()
```

---

## ✅ Classification Checklist

- [ ] Understand binary vs multi-class classification
- [ ] Implement Logistic Regression
- [ ] Use KNN with optimal K
- [ ] Apply Naive Bayes for text/categorical data
- [ ] Build Decision Trees
- [ ] Use Random Forest for better accuracy
- [ ] Apply SVM with different kernels
- [ ] Calculate accuracy, precision, recall, F1
- [ ] Interpret confusion matrix
- [ ] Understand ROC-AUC
- [ ] Complete churn prediction project

---

## 🎯 Key Takeaways

1. **Classification** predicts categories, not numbers
2. **Logistic Regression** for interpretable linear models
3. **KNN** for simple, instance-based learning
4. **Naive Bayes** for text and fast predictions
5. **Decision Tree** for interpretability
6. **Random Forest** for accuracy
7. **SVM** for high-dimensional data
8. **Metrics matter**: Choose based on problem (accuracy vs precision vs recall)

---

## 📚 Next Phase

Congratulations! You've mastered supervised learning. Move to:
👉 [Phase 3: Unsupervised Learning](./Phase-3-Unsupervised-Learning.md)

---

**Remember:** Practice is key! Try these algorithms on different datasets from Kaggle or UCI ML Repository.
