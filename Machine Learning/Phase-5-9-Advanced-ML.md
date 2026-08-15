# Phases 5-9: Advanced Machine Learning

**Navigation:** [Previous (Phase 4)](./Phase-4-Feature-Engineering.md) | [Practice Notebook (Phases 5-6)](./Phase-5-6-Advanced-ML.ipynb) | [Back to Start](./00-Start-Here.ipynb)

This document covers the advanced phases of machine learning. Each section can be expanded into separate files as needed.

---

# Phase 5: Model Evaluation & Tuning

## 🎯 Learning Objectives

- ✅ Master cross-validation techniques
- ✅ Perform hyperparameter tuning
- ✅ Understand model selection strategies
- ✅ Avoid common pitfalls (data leakage, overfitting)

**Time Required:** 1-2 weeks

---

## 5.1 Cross-Validation

### K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Create model
model = LogisticRegression()

# K-Fold Cross-Validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print(f"Cross-Validation Scores: {scores}")
print(f"Mean Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

### Stratified K-Fold (for imbalanced data)

```python
from sklearn.model_selection import StratifiedKFold

# Stratified keeps class distribution in each fold
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=stratified_cv, scoring='accuracy')

print(f"Stratified CV Scores: {scores}")
print(f"Mean Accuracy: {scores.mean():.3f}")
```

### Learning Curves

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

# Generate learning curves
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Interpretation:
# - Both low: underfitting
# - Train high, val low: overfitting
# - Both high: good fit
```

---

## 5.2 Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create grid search
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf, param_grid, cv=5,
    scoring='accuracy', n_jobs=-1, verbose=1
)

# Fit
grid_search.fit(X, y)

# Best parameters
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.3f}")

# Use best model
best_model = grid_search.best_estimator_
```

### Random Search (faster for large parameter spaces)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

# Random search
random_search = RandomizedSearchCV(
    rf, param_dist, n_iter=50, cv=5,
    scoring='accuracy', n_jobs=-1, random_state=42
)

random_search.fit(X, y)

print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Score: {random_search.best_score_:.3f}")
```

### Bayesian Optimization (most efficient)

```python
# Install: pip install scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Define search space
search_spaces = {
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(5, 50),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.1, 0.9)
}

# Bayesian optimization
bayes_search = BayesSearchCV(
    rf, search_spaces, n_iter=50, cv=5,
    scoring='accuracy', n_jobs=-1, random_state=42
)

bayes_search.fit(X, y)

print(f"Best Parameters: {bayes_search.best_params_}")
print(f"Best Score: {bayes_search.best_score_:.3f}")
```

---

## 5.3 Model Selection

### Comparing Multiple Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

# Compare models
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std()
    }
    print(f"{name}:")
    print(f"  Mean Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Visualize comparison
import matplotlib.pyplot as plt

names = list(results.keys())
means = [results[name]['mean'] for name in names]
stds = [results[name]['std'] for name in names]

plt.figure(figsize=(12, 6))
plt.bar(names, means, yerr=stds, capsize=5)
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## ✅ Phase 5 Key Takeaways

1. **Always use cross-validation** (not just train-test split)
2. **Grid Search** for small parameter spaces
3. **Random/Bayesian Search** for large spaces
4. **Compare multiple models** before choosing
5. **Learning curves** diagnose over/underfitting

---

# Phase 6: Ensemble Learning

## 🎯 Learning Objectives

- ✅ Understand ensemble methods
- ✅ Master bagging and boosting
- ✅ Use XGBoost and LightGBM
- ✅ Build stacking ensembles

**Time Required:** 2 weeks ⭐ ADVANCED

---

## 6.1 Bagging (Bootstrap Aggregating)

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Base model
base_model = DecisionTreeClassifier(random_state=42)

# Bagging ensemble
bagging = BaggingClassifier(
    base_estimator=base_model,
    n_estimators=50,
    random_state=42
)

bagging.fit(X, y)

# Compare with single tree
single_scores = cross_val_score(base_model, X, y, cv=5)
bagging_scores = cross_val_score(bagging, X, y, cv=5)

print(f"Single Tree: {single_scores.mean():.3f}")
print(f"Bagging: {bagging_scores.mean():.3f}")
```

## 6.2 Random Forest (Bagging + Feature Randomness)

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

rf.fit(X, y)

# Feature importance
importances = rf.feature_importances_
```

## 6.3 AdaBoost (Adaptive Boosting)

```python
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

adaboost.fit(X, y)
scores = cross_val_score(adaboost, X, y, cv=5)
print(f"AdaBoost Accuracy: {scores.mean():.3f}")
```

## 6.4 Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb.fit(X, y)
scores = cross_val_score(gb, X, y, cv=5)
print(f"Gradient Boosting Accuracy: {scores.mean():.3f}")
```

## 6.5 XGBoost (eXtreme Gradient Boosting)

```python
# Install: pip install xgboost
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

xgb_model.fit(X, y)
scores = cross_val_score(xgb_model, X, y, cv=5)
print(f"XGBoost Accuracy: {scores.mean():.3f}")

# Feature importance
xgb.plot_importance(xgb_model)
plt.show()
```

## 6.6 LightGBM (Fast Gradient Boosting)

```python
# Install: pip install lightgbm
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

lgb_model.fit(X, y)
scores = cross_val_score(lgb_model, X, y, cv=5)
print(f"LightGBM Accuracy: {scores.mean():.3f}")
```

## 6.7 Stacking

```python
from sklearn.ensemble import StackingClassifier

# Base models
estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=10, random_state=42)),
    ('lr', LogisticRegression())
]

# Meta model
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

stacking.fit(X, y)
scores = cross_val_score(stacking, X, y, cv=5)
print(f"Stacking Accuracy: {scores.mean():.3f}")
```

---

## ✅ Phase 6 Key Takeaways

1. **Bagging**: Reduces variance (Random Forest)
2. **Boosting**: Reduces bias (XGBoost, LightGBM)
3. **XGBoost/LightGBM**: Industry standard, wins competitions
4. **Stacking**: Combine different model types
5. **Ensemble > Single model** (almost always)

---

# Phase 7: Introduction to Deep Learning

## 🎯 Learning Objectives

- ✅ Understand neural networks basics
- ✅ Build simple neural networks
- ✅ Use TensorFlow/Keras or PyTorch
- ✅ Apply to basic problems

**Time Required:** 2 weeks

---

## 7.1 Neural Network Basics

```python
from tensorflow import keras
from tensorflow.keras import layers

# Simple neural network
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate
loss, accuracy = model.evaluate(X, y)
print(f"Accuracy: {accuracy:.3f}")

# Plot training history
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

## 7.2 Common Architectures

### For Tabular Data

- Input Layer → Hidden Layers → Output Layer
- Use ReLU activation for hidden layers
- Use sigmoid (binary) or softmax (multi-class) for output

### For Images

- Convolutional Neural Networks (CNNs)

### For Sequences

- Recurrent Neural Networks (RNNs)
- LSTMs, GRUs

## 7.3 Activation Functions

- **ReLU**: f(x) = max(0, x) - Most common
- **Sigmoid**: f(x) = 1/(1+e^-x) - Binary classification output
- **Softmax**: Multi-class classification output
- **Tanh**: f(x) = tanh(x) - Range [-1, 1]

---

## ✅ Phase 7 Key Takeaways

1. **Neural networks** = stacked layers of neurons
2. **Backpropagation** updates weights to minimize loss
3. **TensorFlow/Keras** for easy implementation
4. **Start simple**, add complexity only if needed
5. **Deep learning** powerful but needs lots of data

---

# Phase 8: ML in Production

## 🎯 Learning Objectives

- ✅ Build ML pipelines
- ✅ Deploy models as APIs
- ✅ Monitor model performance
- ✅ Understand MLOps basics

**Time Required:** 2 weeks ⭐ INDUSTRY LEVEL

---

## 8.1 Model Serialization

```python
import pickle
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Use loaded model
predictions = loaded_model.predict(X)
```

## 8.2 Building REST API with FastAPI

```python
# Install: pip install fastapi uvicorn
from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(features: list):
    """Make prediction"""
    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}

@app.get("/health")
def health():
    """Health check"""
    return {"status": "ok"}

# Run: uvicorn main:app --reload
```

## 8.3 Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

## 8.4 Model Monitoring

```python
# Track predictions and actuals
import pandas as pd

predictions_log = []

def log_prediction(features, prediction, actual=None):
    predictions_log.append({
        'timestamp': pd.Timestamp.now(),
        'features': features,
        'prediction': prediction,
        'actual': actual
    })

# Analyze model drift
df_log = pd.DataFrame(predictions_log)

# Check accuracy over time
df_log['correct'] = df_log['prediction'] == df_log['actual']
accuracy_over_time = df_log.groupby(df_log['timestamp'].dt.date)['correct'].mean()

# Alert if accuracy drops
if accuracy_over_time.iloc[-1] < 0.8:
    print("⚠️ Model performance degraded! Retrain needed.")
```

---

## ✅ Phase 8 Key Takeaways

1. **Serialize models** for deployment
2. **FastAPI** for quick REST APIs
3. **Docker** for containerization
4. **Monitor** model performance continuously
5. **Retrain** when performance degrades

---

# Phase 9: End-to-End Projects

## 🎯 Learning Objectives

- ✅ Build complete ML projects
- ✅ Apply all learned concepts
- ✅ Create portfolio projects

**Time Required:** Ongoing ⭐ MOST IMPORTANT

---

## 9.1 Project Structure

```
project/
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_modeling.ipynb
├── src/
│   ├── data_processing.py
│   ├── model.py
│   ├── evaluation.py
├── models/
│   ├── model.pkl
├── api/
│   ├── main.py
├── tests/
├── requirements.txt
├── README.md
└── Dockerfile
```

## 9.2 Beginner Projects

### 1. Iris Flower Classification

- **Dataset**: Iris dataset
- **Goal**: Classify flower species
- **Skills**: Basic classification, evaluation

### 2. House Price Prediction

- **Dataset**: California housing
- **Goal**: Predict house prices
- **Skills**: Regression, feature engineering

### 3. Titanic Survival Prediction

- **Dataset**: Kaggle Titanic
- **Goal**: Predict survival
- **Skills**: Feature engineering, classification

## 9.3 Intermediate Projects

### 1. Credit Card Fraud Detection

- **Challenge**: Highly imbalanced data
- **Skills**: Handling imbalance, anomaly detection
- **Algorithms**: Random Forest, XGBoost

### 2. Customer Churn Prediction

- **Goal**: Predict customer churn
- **Skills**: Feature engineering, business insights
- **Metrics**: Precision, recall, F1-score

### 3. Sentiment Analysis

- **Goal**: Classify text sentiment
- **Skills**: NLP, text preprocessing
- **Algorithms**: Naive Bayes, LSTM

## 9.4 Advanced Projects

### 1. Recommendation System

- **Goal**: Recommend products/movies
- **Approaches**: Collaborative filtering, content-based
- **Skills**: Matrix factorization, embeddings

### 2. Time Series Forecasting

- **Goal**: Predict future values
- **Skills**: ARIMA, LSTM
- **Example**: Stock prices, sales forecasting

### 3. Image Classification

- **Goal**: Classify images
- **Skills**: CNNs, transfer learning
- **Framework**: TensorFlow/PyTorch

---

## 9.5 Project Template

```python
"""
Complete ML Project Template
"""

# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 2. Load Data
df = pd.read_csv('data.csv')
print(df.head())

# 3. Exploratory Data Analysis
print(df.info())
print(df.describe())
print(df.isnull().sum())

# 4. Data Preprocessing
# Handle missing values
df = df.dropna()

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 8. Evaluate Model
y_pred = model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Save Model
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n✓ Model saved successfully!")
```

---

## ✅ Phase 9 Key Takeaways

1. **Start with simple projects**, progress to complex
2. **Document everything** - README, comments
3. **Version control** - Use Git/GitHub
4. **Deploy projects** - Make them accessible
5. **Build portfolio** - Showcase your work

---

## 🎓 Complete ML Roadmap Summary

| Phase | Focus                 | Duration  | Importance           |
| ----- | --------------------- | --------- | -------------------- |
| 0     | Prerequisites         | 1-2 weeks | Foundation           |
| 1     | ML Concepts           | 1 week    | Core Understanding   |
| 2     | Supervised Learning   | 3-4 weeks | ⭐ Most Important    |
| 3     | Unsupervised Learning | 1-2 weeks | Pattern Discovery    |
| 4     | Feature Engineering   | 2 weeks   | ⭐ 70% of work       |
| 5     | Model Evaluation      | 1-2 weeks | Optimization         |
| 6     | Ensemble Learning     | 2 weeks   | ⭐ Competition Level |
| 7     | Deep Learning         | 2 weeks   | Modern AI            |
| 8     | Production            | 2 weeks   | ⭐ Industry Ready    |
| 9     | Projects              | Ongoing   | ⭐ Portfolio         |

**Total Time: 3-4 months of dedicated study**

---

## 📚 Additional Resources

### Books

- "Hands-On Machine Learning" - Aurélien Géron
- "Pattern Recognition and Machine Learning" - Christopher Bishop
- "Deep Learning" - Goodfellow, Bengio, Courville

### Online Courses

- Andrew Ng's ML Course (Coursera)
- Fast.ai
- Google ML Crash Course

### Practice Platforms

- Kaggle
- DataCamp
- LeetCode (for ML coding interviews)

### Communities

- Reddit: r/MachineLearning, r/learnmachinelearning
- Discord: ML communities
- Twitter: Follow ML researchers

---

## 🎯 Final Tips for Success

1. **Code Along**: Don't just read, implement everything
2. **Build Projects**: Apply concepts immediately
3. **Join Competitions**: Kaggle is excellent
4. **Read Papers**: Stay updated with research
5. **Network**: Join ML communities
6. **Be Patient**: Learning takes time
7. **Focus on Fundamentals**: Don't jump to deep learning
8. **Explain to Others**: Teaching solidifies learning
9. **Iterate**: Revisit phases as you learn
10. **Never Stop Learning**: ML is constantly evolving

---

## 🌟 Congratulations!

You now have a complete roadmap to becoming an ML practitioner. Remember:

> "The best time to start was yesterday. The second best time is now."

**Your ML Journey Starts Today! 🚀**

---

**Last Updated:** January 2026
**Author:** ML Learning Path
**Version:** 1.0
