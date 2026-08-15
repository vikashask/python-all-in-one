# Phase 3: Unsupervised Learning

**Navigation:** [Previous (Phase 2B)](./Phase-2-Supervised-Learning-Classification.md) | [Practice Notebook](./Phase-3-Unsupervised-Learning.ipynb) | [Next (Phase 4)](./Phase-4-Feature-Engineering.md)

## 🎯 Learning Objectives

By the end of this phase, you will:

- ✅ Understand clustering algorithms
- ✅ Apply K-Means, Hierarchical, and DBSCAN clustering
- ✅ Use dimensionality reduction (PCA, t-SNE)
- ✅ Identify patterns in unlabeled data
- ✅ Visualize high-dimensional data

**Time Required:** 1-2 weeks
**Difficulty:** Intermediate
**Prerequisites:** Phases 0-2 completed

---

## What is Unsupervised Learning?

**Key Difference from Supervised:** NO LABELS!

**Goal:** Find hidden patterns, structures, or groups in data.

**Use Cases:**

- 👥 Customer segmentation
- 🔍 Anomaly detection (fraud)
- 📄 Topic modeling in documents
- 🧬 Gene clustering
- 🎵 Music recommendation

---

# Part A: Clustering

## 3.1 K-Means Clustering

### Concept

Group similar data points into K clusters.

**Algorithm:**

1. Choose K (number of clusters)
2. Randomly place K centroids
3. Assign each point to nearest centroid
4. Move centroids to center of assigned points
5. Repeat steps 3-4 until convergence

### Simple Example: Customer Segmentation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, n_features=2,
                  cluster_std=0.6, random_state=42)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X',
            edgecolors='black', linewidths=2, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering (K=4)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Cluster centers:\n{centers}")
print(f"Inertia (sum of squared distances): {kmeans.inertia_:.2f}")
```

### Real Example: Customer Segmentation by Spending

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate customer data
np.random.seed(42)
n_customers = 200

data = {
    'Annual_Income': np.random.normal(50000, 20000, n_customers),
    'Spending_Score': np.random.randint(1, 100, n_customers),
    'Age': np.random.randint(18, 70, n_customers)
}
df = pd.DataFrame(data)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize (2D projection)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(df['Annual_Income'], df['Spending_Score'],
                     c=df['Cluster'], cmap='viridis', s=100, alpha=0.6)
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score')
plt.title('Customer Segments')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

# Cluster analysis
plt.subplot(1, 2, 2)
cluster_summary = df.groupby('Cluster').mean()
cluster_summary.plot(kind='bar', ax=plt.gca())
plt.xlabel('Cluster')
plt.ylabel('Average Value')
plt.title('Cluster Characteristics')
plt.xticks(rotation=0)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== Cluster Summary ===")
print(cluster_summary)

# Interpret clusters
print("\n=== Cluster Interpretation ===")
for cluster in range(4):
    cluster_data = df[df['Cluster'] == cluster]
    print(f"\nCluster {cluster} ({len(cluster_data)} customers):")
    print(f"  Avg Income: ${cluster_data['Annual_Income'].mean():.2f}")
    print(f"  Avg Spending: {cluster_data['Spending_Score'].mean():.1f}")
    print(f"  Avg Age: {cluster_data['Age'].mean():.1f}")
```

### Choosing Optimal K: Elbow Method

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate data
X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

# Try different K values
inertias = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal K')
plt.grid(True, alpha=0.3)
plt.show()

# The "elbow" point is optimal K (where curve bends)
print("Look for the 'elbow' point where the curve bends sharply")
```

### Silhouette Score (Another Method)

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score = {score:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.grid(True, alpha=0.3)
plt.show()

# Higher silhouette score = better clustering
best_k = range(2, 11)[np.argmax(silhouette_scores)]
print(f"\nBest K: {best_k} (Silhouette Score: {max(silhouette_scores):.3f})")
```

---

## 3.2 Hierarchical Clustering

### Concept

Build a tree (dendrogram) of clusters, either bottom-up (agglomerative) or top-down (divisive).

**Agglomerative (Bottom-Up):**

1. Start: Each point is its own cluster
2. Merge closest clusters
3. Repeat until one cluster

### Example with Dendrogram

```python
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np

# Generate data
np.random.seed(42)
X = np.random.randn(50, 2)

# Perform hierarchical clustering
linkage_matrix = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.axhline(y=6, color='r', linestyle='--', label='Cut line (3 clusters)')
plt.legend()
plt.show()

# Cut dendrogram at specific height to get clusters
agg_clustering = AgglomerativeClustering(n_clusters=3)
labels = agg_clustering.fit_predict(X)

# Visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=100, alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Hierarchical Clustering Result (3 clusters)')
plt.colorbar(label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Number of clusters: {len(set(labels))}")
```

### Linkage Methods

```python
# Compare different linkage methods
methods = ['ward', 'complete', 'average', 'single']
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, method in enumerate(methods):
    agg = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = agg.fit_predict(X)

    axes[idx].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=100, alpha=0.6)
    axes[idx].set_title(f'Linkage: {method.upper()}')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Pros:**

- ✅ No need to specify K upfront
- ✅ Dendrogram shows hierarchy
- ✅ Works with any distance metric

**Cons:**

- ❌ Slow for large datasets (O(n²))
- ❌ Can't undo merges

---

## 3.3 DBSCAN (Density-Based)

### Concept

Find clusters based on density. Points in dense regions = same cluster.

**Parameters:**

- `eps`: Maximum distance between points
- `min_samples`: Minimum points to form cluster

**Advantages:**

- ✅ Automatically finds number of clusters
- ✅ Can find arbitrary shaped clusters
- ✅ Identifies outliers/noise

### Example

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Generate non-linear data
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Compare K-Means vs DBSCAN
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# K-Means (struggles with non-linear shapes)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

ax1.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50)
ax1.set_title('K-Means (Linear boundaries)')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.grid(True, alpha=0.3)

# DBSCAN (handles non-linear shapes)
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

ax2.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', s=50)
ax2.set_title('DBSCAN (Flexible boundaries)')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Count clusters (excluding noise labeled as -1)
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
```

### Finding Optimal eps

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Calculate distances to k-nearest neighbors
k = 5
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X)
distances, indices = neighbors.kneighbors(X)

# Sort and plot distances
distances = np.sort(distances[:, k-1], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('Points sorted by distance')
plt.ylabel(f'Distance to {k}th nearest neighbor')
plt.title('K-Distance Graph (for choosing eps)')
plt.grid(True, alpha=0.3)
plt.show()

# The "elbow" point suggests optimal eps
print("Look for the 'elbow' point - that's your optimal eps value")
```

### Outlier Detection with DBSCAN

```python
# Generate data with outliers
np.random.seed(42)
normal_data = np.random.randn(200, 2)
outliers = np.random.uniform(-5, 5, (20, 2))
X_with_outliers = np.vstack([normal_data, outliers])

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=10)
labels = dbscan.fit_predict(X_with_outliers)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_with_outliers[labels != -1, 0], X_with_outliers[labels != -1, 1],
           c=labels[labels != -1], cmap='viridis', s=50, alpha=0.6, label='Clusters')
plt.scatter(X_with_outliers[labels == -1, 0], X_with_outliers[labels == -1, 1],
           c='red', s=100, marker='x', label='Outliers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Outlier Detection with DBSCAN')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Number of outliers detected: {(labels == -1).sum()}")
```

---

## 🎯 Clustering Algorithm Comparison

| Algorithm        | Pros                                                    | Cons                                          | Use When                               |
| ---------------- | ------------------------------------------------------- | --------------------------------------------- | -------------------------------------- |
| **K-Means**      | Fast, simple, scalable                                  | Need to specify K, assumes spherical clusters | Large datasets, clear clusters         |
| **Hierarchical** | No K needed, dendrogram                                 | Slow (O(n²)), can't undo                      | Small datasets, hierarchy important    |
| **DBSCAN**       | Finds K automatically, handles shapes, detects outliers | Sensitive to parameters                       | Non-linear clusters, outlier detection |

---

# Part B: Dimensionality Reduction

## 3.4 PCA (Principal Component Analysis)

### Concept

Reduce dimensions while keeping most important information (variance).

**Use Cases:**

- 📉 Reduce features for visualization
- ⚡ Speed up training
- 🧹 Remove noise
- 📊 Exploratory analysis

### Simple 2D Example

```python
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Generate correlated data
np.random.seed(42)
X = np.random.randn(200, 2)
X[:, 1] = X[:, 0] + np.random.randn(200) * 0.3  # Create correlation

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Original data
ax1.scatter(X[:, 0], X[:, 1], alpha=0.6)
ax1.set_xlabel('Original Feature 1')
ax1.set_ylabel('Original Feature 2')
ax1.set_title('Original Data')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# PCA components
origin = [0, 0]
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    ax1.arrow(*origin, *(comp * var * 3), head_width=0.1, head_length=0.1,
             fc=f'C{i}', ec=f'C{i}', label=f'PC{i+1}')
ax1.legend()

# Transformed data
ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')
ax2.set_title('PCA Transformed Data')
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

plt.tight_layout()
plt.show()

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")
```

### Real Example: Iris Dataset Dimensionality Reduction

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, 5), pca.explained_variance_ratio_, alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained Ratio')
plt.title('Variance Explained by Each Component')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, 5), cumsum, marker='o', linewidth=2, markersize=8)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Explained Variance by Component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.2%}")
print(f"\nTotal: {pca.explained_variance_ratio_.sum():.2%}")

# Visualize in 2D
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis', s=50)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
plt.title('Iris Dataset: PCA Visualization')
plt.colorbar(scatter, label='Species')
plt.grid(True, alpha=0.3)
plt.show()

# 2 components explain:
print(f"\n2 PCs explain {pca_2d.explained_variance_ratio_.sum():.2%} of variance")
```

### PCA for Feature Reduction

```python
# When to keep how many components?

# Method 1: Keep 95% variance
pca_95 = PCA(n_components=0.95)  # Keep components explaining 95% variance
X_reduced = pca_95.fit_transform(X_scaled)
print(f"Original features: {X_scaled.shape[1]}")
print(f"Reduced features: {X_reduced.shape[1]}")
print(f"Variance preserved: {pca_95.explained_variance_ratio_.sum():.2%}")

# Method 2: Scree plot (elbow method)
pca_full = PCA()
pca_full.fit(X_scaled)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         pca_full.explained_variance_ratio_, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Scree Plot')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 3.5 t-SNE (t-Distributed Stochastic Neighbor Embedding)

### Concept

Powerful for visualizing high-dimensional data in 2D/3D.

**Key Difference from PCA:**

- PCA: Linear, preserves global structure
- t-SNE: Non-linear, preserves local structure (neighborhood)

**Use Case:** Visualization ONLY (not for feature reduction)

### Example: MNIST Digits Visualization

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load data
digits = load_digits()
X = digits.data
y = digits.target

# Take subset for speed (t-SNE is slow)
n_samples = 500
X_subset = X[:n_samples]
y_subset = y[:n_samples]

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_subset)

# Visualize
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset,
                     cmap='tab10', s=50, alpha=0.7)
plt.colorbar(scatter, label='Digit')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of Handwritten Digits')
plt.grid(True, alpha=0.3)
plt.show()

# Similar digits cluster together!
```

### PCA vs t-SNE Comparison

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load data
digits = load_digits()
X = digits.data[:1000]
y = digits.target[:1000]

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Compare
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# PCA
scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=30, alpha=0.7)
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_title('PCA: Global Structure')
ax1.grid(True, alpha=0.3)

# t-SNE
scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=30, alpha=0.7)
ax2.set_xlabel('t-SNE Dim 1')
ax2.set_ylabel('t-SNE Dim 2')
ax2.set_title('t-SNE: Local Structure')
ax2.grid(True, alpha=0.3)

plt.colorbar(scatter2, ax=ax2, label='Digit')
plt.tight_layout()
plt.show()

# Notice: t-SNE creates clearer clusters
```

**When to Use:**

- **PCA**: Feature reduction, speeding up training, understanding variance
- **t-SNE**: Visualization only, exploring high-dimensional data

---

## 📝 Complete Project: Customer Segmentation

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Generate customer data
np.random.seed(42)
n_customers = 500

data = {
    'Age': np.random.randint(18, 70, n_customers),
    'Income': np.random.normal(60000, 25000, n_customers),
    'Spending_Score': np.random.randint(1, 100, n_customers),
    'Purchase_Frequency': np.random.randint(1, 50, n_customers),
    'Avg_Purchase_Value': np.random.uniform(10, 500, n_customers),
    'Years_Customer': np.random.randint(0, 20, n_customers),
    'Website_Visits': np.random.randint(0, 100, n_customers),
    'Email_Opens': np.random.randint(0, 50, n_customers)
}

df = pd.DataFrame(data)

print("Dataset Overview:")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nStatistics:")
print(df.describe())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Find optimal K
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

    from sklearn.metrics import silhouette_score
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Plot metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')
ax1.grid(True, alpha=0.3)

ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Choose optimal K (let's say 4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\n=== Cluster Distribution ===")
print(df['Cluster'].value_counts().sort_index())

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'],
                     cmap='viridis', s=50, alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Customer Segments (PCA Visualization)')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

# Analyze clusters
print("\n=== Cluster Analysis ===")
cluster_analysis = df.groupby('Cluster').mean()
print(cluster_analysis)

# Visualize cluster characteristics
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for idx, col in enumerate(df.columns[:-1]):  # Exclude 'Cluster' column
    df.boxplot(column=col, by='Cluster', ax=axes[idx])
    axes[idx].set_title(col)
    axes[idx].set_xlabel('Cluster')

plt.suptitle('Cluster Characteristics', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

# Name clusters based on characteristics
cluster_names = {
    0: "Budget Shoppers",
    1: "Premium Customers",
    2: "Occasional Buyers",
    3: "Loyal Regulars"
}

df['Segment_Name'] = df['Cluster'].map(cluster_names)

print("\n=== Segment Summary ===")
for cluster, name in cluster_names.items():
    cluster_data = df[df['Cluster'] == cluster]
    print(f"\n{name} (Cluster {cluster}):")
    print(f"  Count: {len(cluster_data)}")
    print(f"  Avg Age: {cluster_data['Age'].mean():.1f}")
    print(f"  Avg Income: ${cluster_data['Income'].mean():.2f}")
    print(f"  Avg Spending: {cluster_data['Spending_Score'].mean():.1f}")
    print(f"  Avg Purchase Frequency: {cluster_data['Purchase_Frequency'].mean():.1f}")

# Save results
df.to_csv('customer_segments.csv', index=False)
print("\n✓ Results saved to customer_segments.csv")
```

---

## ✅ Phase Completion Checklist

- [ ] Understand difference between supervised and unsupervised learning
- [ ] Implement K-Means clustering
- [ ] Use Elbow and Silhouette methods
- [ ] Apply Hierarchical clustering with dendrograms
- [ ] Use DBSCAN for density-based clustering
- [ ] Apply PCA for dimensionality reduction
- [ ] Visualize high-dimensional data with t-SNE
- [ ] Complete customer segmentation project
- [ ] Interpret cluster characteristics

---

## 🎯 Key Takeaways

1. **Unsupervised learning** finds patterns without labels
2. **K-Means** for fast, simple clustering (need to specify K)
3. **Hierarchical** for dendrogram and no K specification
4. **DBSCAN** for arbitrary shapes and outlier detection
5. **PCA** for feature reduction and understanding variance
6. **t-SNE** for visualization only (preserves local structure)
7. Always **standardize** features before clustering

---

## 📚 Next Phase

Excellent work! Now master the practical skills:
👉 [Phase 4: Feature Engineering & Data Processing](./Phase-4-Feature-Engineering.md)
👉 Practice notebook: [Phase-3-Unsupervised-Learning.ipynb](./Phase-3-Unsupervised-Learning.ipynb)

**Remember:** 70% of ML work is data preparation!
