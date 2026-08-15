# Phase 1: What is Machine Learning

**Navigation:** [Previous (Phase 0 Notebook)](./Phase-0-Prerequisites.ipynb) | [Practice Notebook](./Phase-1-ML-Concepts.ipynb) | [Next (Phase 2A)](./Phase-2-Supervised-Learning.md)

## 🎯 Learning Objectives

By the end of this phase, you will:

- ✅ Understand what machine learning is and isn't
- ✅ Differentiate ML from traditional programming
- ✅ Identify the main types of ML
- ✅ Understand core concepts: features, labels, training, testing
- ✅ Recognize overfitting and underfitting

**Time Required:** 1 week
**Difficulty:** Beginner
**Prerequisites:** Phase 0 completed

---

## 1.1 What ML Actually Is

### Traditional Programming vs Machine Learning

#### Traditional Programming

```
INPUT + RULES → OUTPUT

Example:
def is_spam(email):
    if "free money" in email or "click here" in email:
        return "SPAM"
    else:
        return "NOT SPAM"
```

**Problem:** Rules are hard-coded. What about "FR33 M0NEY"? You need infinite rules!

#### Machine Learning

```
INPUT + OUTPUT → RULES (learned by the model)

Example:
# Give the model thousands of emails labeled as spam/not spam
# Model learns patterns automatically
model.fit(emails, labels)  # Training
prediction = model.predict(new_email)  # Testing
```

**Advantage:** Model learns patterns we might not see. Handles variations automatically!

---

### Real-World Example: Email Spam Detection

#### Traditional Approach (Rules-Based)

```python
def is_spam_traditional(email):
    spam_words = ['free', 'win', 'click here', 'viagra', 'lottery']
    spam_score = 0

    for word in spam_words:
        if word in email.lower():
            spam_score += 1

    return "SPAM" if spam_score >= 2 else "NOT SPAM"

# Test
email1 = "Click here to win free money!"
email2 = "Hey, let's meet for coffee tomorrow"

print(is_spam_traditional(email1))  # SPAM ✓
print(is_spam_traditional(email2))  # NOT SPAM ✓

# But what about...
email3 = "Cl1ck h3r3 to w1n fr33 m0n3y!"  # Bypasses rules! ✗
```

#### Machine Learning Approach

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
emails = [
    "win free money now",
    "click here for prizes",
    "meeting tomorrow at 3pm",
    "project deadline next week",
    "get rich quick scheme",
    "lunch plans for friday"
]
labels = [1, 1, 0, 0, 1, 0]  # 1=SPAM, 0=NOT SPAM

# Convert text to numbers (vectorization)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Predict
new_emails = [
    "Cl1ck h3r3 to w1n fr33 m0n3y",  # Tricky spam
    "quarterly report meeting"
]
X_new = vectorizer.transform(new_emails)
predictions = model.predict(X_new)

print(predictions)  # Model learns patterns, not exact words!
```

---

### Key ML Concepts

#### 1. **Features** (Input)

The measurable properties used to make predictions.

```python
# Example: House Price Prediction
house = {
    'size': 2000,           # Feature 1: Square feet
    'bedrooms': 3,          # Feature 2: Number of bedrooms
    'age': 10,              # Feature 3: Age of house
    'location_score': 8     # Feature 4: Location rating
}
# Model uses these to predict price
```

#### 2. **Labels** (Output)

The thing we want to predict.

```python
# In supervised learning, we have both features and labels
data = [
    {'size': 1500, 'bedrooms': 2, 'price': 200000},  # price is label
    {'size': 2000, 'bedrooms': 3, 'price': 300000},
    {'size': 2500, 'bedrooms': 4, 'price': 400000}
]

# Goal: Given new house features, predict the price (label)
```

#### 3. **Model**

The mathematical function that maps features to labels.

```python
# Simplified view
def model(features):
    # Learns patterns during training
    # Returns prediction
    return prediction

# Example
features = [2000, 3, 10, 8]  # size, bedrooms, age, location
predicted_price = model(features)  # Returns: 320000
```

#### 4. **Training**

The process of teaching the model by showing it examples.

```python
from sklearn.linear_model import LinearRegression

# Training data
X_train = [[1500, 2], [2000, 3], [2500, 4]]  # Features
y_train = [200000, 300000, 400000]            # Labels

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)  # Learning happens here!

print("Training complete!")
```

#### 5. **Testing**

Evaluating how well the model performs on unseen data.

```python
# Test data (model hasn't seen this before)
X_test = [[1800, 2], [2200, 3]]
y_test = [250000, 350000]  # Actual prices

# Predict
y_pred = model.predict(X_test)

print(f"Predicted: {y_pred}")
print(f"Actual: {y_test}")

# Calculate error
from sklearn.metrics import mean_absolute_error
error = mean_absolute_error(y_test, y_pred)
print(f"Average error: ${error:.2f}")
```

---

### Simple Complete Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Problem: Predict salary based on years of experience

# Training data
experience = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)  # Features
salary = np.array([30000, 35000, 42000, 48000, 55000, 60000, 67000, 75000])  # Labels

# Create model
model = LinearRegression()

# Train
model.fit(experience, salary)
print("✓ Model trained!")

# Test: Predict salary for 10 years experience
new_experience = np.array([[10]])
predicted_salary = model.predict(new_experience)
print(f"Predicted salary for 10 years: ${predicted_salary[0]:.2f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(experience, salary, color='blue', label='Actual data')
plt.plot(experience, model.predict(experience), color='red', label='Model prediction')
plt.scatter(new_experience, predicted_salary, color='green', s=200, label='New prediction')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.title('Salary Prediction Model')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 1.2 Types of Machine Learning

### 1. Supervised Learning 🏆 (Most Common - 80% of ML)

You have both **features** and **labels** (correct answers).

#### Example 1: Classification

```python
# Email Spam Detection
X = [
    "buy this product now",     # Features
    "meeting at 3pm",
    "win free money"
]
y = ["spam", "not spam", "spam"]  # Labels

# Model learns: "What patterns indicate spam?"
```

#### Example 2: Regression

```python
# House Price Prediction
X = [[1500, 2], [2000, 3], [2500, 4]]  # [size, bedrooms]
y = [200000, 300000, 400000]            # prices

# Model learns: "How does size and bedrooms affect price?"
```

**Real-World Uses:**

- 📧 Email spam filtering
- 🏥 Disease diagnosis
- 📈 Stock price prediction
- 🎬 Movie recommendation
- 🚗 Self-driving cars (object detection)

---

### 2. Unsupervised Learning 🔍

You have **only features**, no labels. Model finds hidden patterns.

#### Example: Customer Segmentation

```python
from sklearn.cluster import KMeans

# Customer data (no labels!)
customers = [
    [25, 30000],   # [age, income]
    [30, 35000],
    [45, 80000],
    [50, 90000],
    [28, 32000],
    [48, 85000]
]

# Find groups
kmeans = KMeans(n_clusters=2)
groups = kmeans.fit_predict(customers)

print(groups)  # Output: [0, 0, 1, 1, 0, 1]
# Group 0: Younger, lower income
# Group 1: Older, higher income
```

**Real-World Uses:**

- 👥 Customer segmentation
- 🔍 Anomaly detection (fraud)
- 📊 Data exploration
- 🧬 Gene sequence analysis
- 🎵 Music genre discovery

---

### 3. Semi-Supervised Learning 🔄

Mix of labeled and unlabeled data (common in real world).

```python
# Scenario: You have 1000 images, but only 100 are labeled
labeled_images = 100    # Expensive to label!
unlabeled_images = 900  # Free but no labels

# Semi-supervised learning uses both
# Common in: image recognition, speech recognition
```

**Why Important:**

- Labeling data is expensive (humans need to do it)
- Often we have lots of data but few labels
- Common in: medical imaging, speech recognition

---

### 4. Reinforcement Learning 🎮

Agent learns by trial and error (reward/punishment).

```python
# Simple example (conceptual)
class GameAgent:
    def play_action(self, state):
        action = self.choose_action(state)
        reward = environment.execute(action)

        if reward > 0:
            self.learn("This action was good!")
        else:
            self.learn("This action was bad!")

        return action

# Agent learns optimal strategy through practice
```

**Real-World Uses:**

- 🎮 Game AI (AlphaGo, Chess)
- 🚗 Self-driving cars
- 🤖 Robotics
- 💰 Trading algorithms
- 🎯 Ad placement optimization

---

### Comparison Table

| Type                | Has Labels? | Goal                         | Example                                  |
| ------------------- | ----------- | ---------------------------- | ---------------------------------------- |
| **Supervised**      | ✅ Yes      | Predict output from input    | Spam detection, Price prediction         |
| **Unsupervised**    | ❌ No       | Find patterns/groups         | Customer segmentation, Anomaly detection |
| **Semi-Supervised** | ⚠️ Some     | Use both labeled & unlabeled | Image classification with few labels     |
| **Reinforcement**   | 🎯 Rewards  | Learn optimal strategy       | Game playing, Robotics                   |

---

## 1.3 Core ML Workflow

```python
# Step 1: Collect Data
import pandas as pd
data = pd.read_csv('house_prices.csv')

# Step 2: Prepare Data
X = data[['size', 'bedrooms', 'age']]  # Features
y = data['price']                       # Label

# Step 3: Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Choose Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Step 5: Train Model
model.fit(X_train, y_train)

# Step 6: Evaluate Model
score = model.score(X_test, y_test)
print(f"Model accuracy: {score}")

# Step 7: Make Predictions
new_house = [[2000, 3, 10]]
predicted_price = model.predict(new_house)
print(f"Predicted price: ${predicted_price[0]:.2f}")
```

---

## 1.4 Overfitting vs Underfitting ⚠️

### The Goldilocks Problem

#### Underfitting (Too Simple)

Model is too simple to capture patterns.

```python
# Example: Linear model for complex data
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Complex, non-linear data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, 100)

# Too simple model (linear)
model = LinearRegression()
model.fit(X, y)

plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, model.predict(X), color='red', label='Underfit model')
plt.legend()
plt.title('Underfitting: Model too simple')
plt.show()

# Result: Poor performance on both training and test data
```

#### Overfitting (Too Complex)

Model memorizes training data instead of learning patterns.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Too complex model (high-degree polynomial)
model = make_pipeline(PolynomialFeatures(20), LinearRegression())
model.fit(X, y)

plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, model.predict(X), color='red', label='Overfit model')
plt.legend()
plt.title('Overfitting: Model too complex')
plt.show()

# Result: Perfect on training data, poor on test data
```

#### Just Right (Good Fit)

Model captures patterns without memorizing.

```python
# Appropriate complexity
model = make_pipeline(PolynomialFeatures(3), LinearRegression())
model.fit(X, y)

plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, model.predict(X), color='green', label='Good fit')
plt.legend()
plt.title('Good Fit: Balanced complexity')
plt.show()

# Result: Good on both training and test data
```

### Visual Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generate data
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, 50)

# Split
X_train, X_test = X[:40], X[40:]
y_train, y_test = y[:40], y[40:]

# Three models
model_underfit = LinearRegression()  # Degree 1
model_good = make_pipeline(PolynomialFeatures(3), LinearRegression())
model_overfit = make_pipeline(PolynomialFeatures(20), LinearRegression())

# Train all
model_underfit.fit(X_train, y_train)
model_good.fit(X_train, y_train)
model_overfit.fit(X_train, y_train)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, model, title in zip(axes,
                            [model_underfit, model_good, model_overfit],
                            ['Underfitting', 'Good Fit', 'Overfitting']):
    ax.scatter(X_train, y_train, label='Train', alpha=0.6)
    ax.scatter(X_test, y_test, label='Test', alpha=0.6)

    X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
    ax.plot(X_plot, model.predict(X_plot), 'r-', linewidth=2)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    ax.set_title(f'{title}\nTrain: {train_score:.2f} | Test: {test_score:.2f}')
    ax.legend()

plt.tight_layout()
plt.show()
```

### How to Detect?

| Condition    | Training Accuracy | Test Accuracy | Problem          |
| ------------ | ----------------- | ------------- | ---------------- |
| Underfitting | Low (60%)         | Low (58%)     | Model too simple |
| Good Fit     | High (90%)        | High (88%)    | ✓ Balanced       |
| Overfitting  | Very High (99%)   | Low (70%)     | Model memorizes  |

### Solutions

```python
# For Underfitting:
# 1. Use more complex model
# 2. Add more features
# 3. Train longer

# For Overfitting:
# 1. Get more training data
# 2. Use simpler model
# 3. Apply regularization
# 4. Use dropout (neural networks)

# Example: Early stopping
from sklearn.model_selection import learning_curve

# Plot learning curve to detect over/underfitting
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, test_scores.mean(axis=1), label='Test score')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.legend()
plt.title('Learning Curve')
plt.show()
```

---

## 1.5 Model Performance Metrics (Introduction)

### For Regression (Predicting Numbers)

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Actual vs Predicted
y_true = np.array([100, 150, 200, 250, 300])
y_pred = np.array([110, 145, 210, 240, 295])

# Mean Absolute Error (average error in same units)
mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: ${mae:.2f}")  # Average error in dollars

# Mean Squared Error (penalizes large errors more)
mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.2f}")

# R² Score (0-1, higher is better)
r2 = r2_score(y_true, y_pred)
print(f"R² Score: {r2:.3f}")  # 1.0 = perfect, 0.0 = useless
```

### For Classification (Predicting Categories)

```python
from sklearn.metrics import accuracy_score, confusion_matrix

# Actual vs Predicted
y_true = ['spam', 'spam', 'not spam', 'spam', 'not spam']
y_pred = ['spam', 'not spam', 'not spam', 'spam', 'not spam']

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy * 100:.1f}%")  # 80%

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)
```

---

## 📝 Practice Exercises

### Exercise 1: Identify ML Type

```python
# For each scenario, identify if it's:
# - Supervised (Regression or Classification)
# - Unsupervised
# - Reinforcement

scenarios = [
    "Predicting tomorrow's temperature",        # ?
    "Grouping similar customers",               # ?
    "Teaching robot to walk",                   # ?
    "Detecting fraudulent transactions",        # ?
    "Finding topics in documents",              # ?
]
```

### Exercise 2: Build Simple Model

```python
# Create a simple model to predict exam scores based on study hours
study_hours = [1, 2, 3, 4, 5, 6, 7, 8]
exam_scores = [50, 55, 60, 65, 70, 75, 80, 85]

# TODO:
# 1. Split into train/test
# 2. Train a model
# 3. Predict score for 10 hours of study
# 4. Calculate accuracy
```

### Exercise 3: Detect Overfitting

```python
# Given:
model_a_train_acc = 0.95
model_a_test_acc = 0.93

model_b_train_acc = 0.99
model_b_test_acc = 0.70

model_c_train_acc = 0.65
model_c_test_acc = 0.64

# Which model is:
# - Overfitting?
# - Underfitting?
# - Good fit?
```

---

## ✅ Phase Completion Checklist

- [ ] Understand difference between traditional programming and ML
- [ ] Explain what features, labels, and models are
- [ ] Identify the 4 types of ML with examples
- [ ] Describe the ML workflow (collect, prepare, train, test, predict)
- [ ] Recognize overfitting and underfitting
- [ ] Know basic evaluation metrics (accuracy, MAE, R²)
- [ ] Complete practice exercises
- [ ] Explain ML concepts to someone else (best test!)

---

## 🎯 Key Takeaways

1. **ML learns patterns from data** - no hard-coded rules
2. **Supervised learning** is most common (80% of ML)
3. **Features** = inputs, **Labels** = outputs we want to predict
4. **Training** = learning, **Testing** = evaluation
5. **Overfitting** = memorizing, **Underfitting** = too simple
6. **Good model** balances complexity with generalization

---

## 💡 Common Misconceptions

❌ **"ML is magic"**
✅ ML is statistics + optimization. It finds patterns in data.

❌ **"More data always helps"**
✅ Quality > Quantity. 100 good examples beat 10,000 bad ones.

❌ **"Complex models are better"**
✅ Simple models often work best. Start simple, add complexity if needed.

❌ **"100% accuracy is the goal"**
✅ Perfect accuracy on training data = overfitting!

---

## 📚 Next Steps

Ready to build real models? Move to:
👉 [Phase 2: Supervised Learning](./Phase-2-Supervised-Learning.md)
👉 Practice notebook first: [Phase-1-ML-Concepts.ipynb](./Phase-1-ML-Concepts.ipynb)

This is where the fun begins! 🚀

---

**Pro Tip:** The best way to understand ML is to build models. Don't just read - code along with every example!
