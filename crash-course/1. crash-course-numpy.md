# 🚀 NumPy Crash Course — Quick Revision Guide

> **Time to revise: ~10 minutes** | Covers everything from the `numpy/` folder

---

## 1. What is NumPy?

- **Numerical Python** — foundation for scientific computing
- Provides **N-dimensional arrays (ndarray)** — faster than Python lists (50-100x)
- Built on C → vectorized operations, no explicit loops needed

---

## 2. Array Creation

```python
import numpy as np

# From list
a = np.array([1, 2, 3])              # 1D
b = np.array([[1,2],[3,4]])           # 2D

# Built-in generators
np.zeros((3, 3))                      # All zeros
np.ones((2, 4))                       # All ones
np.full((2, 2), 7)                    # All 7s
np.eye(3)                             # Identity matrix
np.empty((2, 2))                      # Uninitialized

# Ranges
np.arange(0, 10, 2)                   # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)                  # 5 evenly spaced: 0 to 1
np.logspace(0, 3, 4)                  # [1, 10, 100, 1000]
```

---

## 3. Array Attributes

```python
arr = np.arange(24).reshape(2, 3, 4)

arr.shape      # (2, 3, 4)
arr.ndim       # 3 (number of dimensions)
arr.size       # 24 (total elements)
arr.dtype      # int64
arr.itemsize   # 8 bytes per element
arr.nbytes     # 192 total bytes
```

---

## 4. Indexing & Slicing

```python
arr = np.array([10, 20, 30, 40, 50])

arr[0]         # 10 (first)
arr[-1]        # 50 (last)
arr[1:4]       # [20, 30, 40]
arr[::2]       # [10, 30, 50] (every 2nd)
arr[::-1]      # [50, 40, 30, 20, 10] (reverse)

# 2D
m = np.array([[1,2,3],[4,5,6],[7,8,9]])
m[1, 2]        # 6 (row 1, col 2)
m[1, :]        # [4, 5, 6] (full row)
m[:, 2]        # [3, 6, 9] (full column)

# Fancy indexing
arr[[0, 2, 4]] # Pick specific indices
```

> ⚠️ **Views vs Copies**: Slicing returns a **view** (shared memory). Use `.copy()` for independent copy.

---

## 5. Vectorized Operations (No Loops!)

```python
arr = np.array([1, 2, 3, 4])

arr + 10       # [11, 12, 13, 14]
arr * 3        # [3, 6, 9, 12]
arr ** 2       # [1, 4, 9, 16]

# Element-wise between arrays
a + b          # Add corresponding elements
a * b          # Multiply corresponding elements
```

---

## 6. Broadcasting Rules

```python
# Scalar → broadcasts to all elements
arr + 10

# Row vector → broadcasts across all rows
matrix + row_vector

# Column vector → broadcasts across all columns
matrix + col_vector
```

**Rules:**

1. Pad dimensions with 1 on the left
2. Size-1 dimensions stretch to match
3. Incompatible sizes → Error

---

## 7. Math Operations & Universal Functions (ufuncs)

```python
np.sqrt(arr)              # Square root
np.square(arr)            # Square
np.exp(arr)               # e^x
np.log(arr)               # Natural log
np.log10(arr)             # Log base 10
np.sin(arr)               # Sine
np.cos(arr)               # Cosine
np.abs(arr)               # Absolute value
np.floor(arr)             # Round down
np.ceil(arr)              # Round up
np.round(arr, 2)          # Round to 2 decimals
np.maximum(a, b)          # Element-wise max
np.minimum(a, b)          # Element-wise min
```

---

## 8. Aggregation / Statistics

```python
arr.sum()                 # Total sum
arr.mean()                # Average
arr.std()                 # Standard deviation
arr.min()                 # Minimum
arr.max()                 # Maximum
arr.prod()                # Product of all
arr.argmax()              # Index of max
arr.argmin()              # Index of min
arr.cumsum()              # Cumulative sum

# With axis (2D)
arr.sum(axis=0)           # Sum per COLUMN (↓)
arr.sum(axis=1)           # Sum per ROW (→)

np.median(arr)            # Median
np.percentile(arr, 25)    # 25th percentile
```

---

## 9. Boolean Masking & Filtering

```python
arr = np.array([1, 5, 8, 3, 9, 2])

arr[arr > 5]                      # [8, 9]
arr[(arr > 3) & (arr < 8)]        # [5] — AND
arr[(arr < 3) | (arr > 7)]        # [1, 8, 9, 2] — OR
arr[~(arr > 5)]                   # [1, 5, 3, 2] — NOT

# Conditional replace
np.where(arr > 5, 'big', 'small')
np.where(arr > 5, arr * 2, arr)   # Double if > 5, else keep

# Counting
np.sum(arr > 5)                   # How many > 5
np.any(arr > 5)                   # Any > 5? True
np.all(arr > 0)                   # All > 0? True
```

---

## 10. Reshaping & Stacking

```python
arr = np.arange(12)

arr.reshape(3, 4)         # 3 rows × 4 cols
arr.reshape(4, -1)        # -1 = auto-calculate
matrix.flatten()          # 2D → 1D (copy)
matrix.ravel()            # 2D → 1D (view)
matrix.T                  # Transpose

# Combining
np.vstack((a, b))         # Stack vertically ↓
np.hstack((a, b))         # Stack horizontally →
np.concatenate((a, b), axis=0)
```

---

## 11. Random Numbers (Modern API)

```python
rng = np.random.default_rng(seed=42)

rng.random((3, 3))                        # Uniform [0, 1)
rng.integers(0, 100, size=5)              # Random ints
rng.normal(loc=100, scale=15, size=5)     # Normal distribution
rng.uniform(low=0, high=10, size=5)       # Uniform distribution
rng.choice(['a', 'b', 'c'], size=5)       # Random pick
rng.shuffle(arr)                          # In-place shuffle
rng.permutation(arr)                      # Returns shuffled copy
```

---

## 12. Linear Algebra

```python
A @ B                     # Matrix multiplication
np.dot(v1, v2)            # Dot product
np.linalg.det(A)          # Determinant
np.linalg.inv(A)          # Inverse
np.linalg.solve(A, b)     # Solve Ax = b
np.linalg.norm(A)         # Matrix norm
np.linalg.eig(A)          # Eigenvalues & eigenvectors
```

---

## 13. Save & Load

```python
np.save("data.npy", arr)              # Binary format
arr = np.load("data.npy")

np.savetxt("data.csv", arr, delimiter=",", fmt="%d")  # CSV
arr = np.loadtxt("data.csv", delimiter=",")
```

---

## 14. Einsum (Einstein Summation)

```python
np.einsum('ii->', A)               # Trace (diagonal sum)
np.einsum('ij->', A)               # Sum all elements
np.einsum('ij,jk->ik', A, B)      # Matrix multiply
np.einsum('i,j->ij', a, b)        # Outer product
```

---

## 15. Set Operations

```python
np.unique(a)              # Unique values
np.union1d(a, b)          # Union
np.intersect1d(a, b)      # Intersection
np.setdiff1d(a, b)        # In a but not b
np.in1d([1, 4], a)        # Membership check
```

---

## 16. Structured Arrays (Mini Database)

```python
dt = np.dtype([('name', 'U10'), ('age', int), ('weight', float)])
people = np.array([('Alice', 25, 55.5), ('Bob', 30, 80.0)], dtype=dt)
people['age']             # Access column → [25, 30]
```

---

## ⚡ Performance Tips

| Tip                                | Why                                     |
| ---------------------------------- | --------------------------------------- |
| **Vectorize** — avoid Python loops | 100x+ faster                            |
| **Use correct dtype**              | `float32` saves 50% memory vs `float64` |
| **Use views, not copies**          | Slicing = view, fancy indexing = copy   |
| **Pre-allocate arrays**            | Avoid growing arrays dynamically        |
| **Leverage broadcasting**          | No need to manually expand dimensions   |

---

## 🧾 Cheat Sheet

| Task         | Code                                      |
| ------------ | ----------------------------------------- |
| Create array | `np.array([1,2,3])`                       |
| Zeros/Ones   | `np.zeros((3,3))` / `np.ones((2,2))`      |
| Range        | `np.arange(0,10,2)`                       |
| Shape        | `arr.shape`                               |
| Reshape      | `arr.reshape(3,4)`                        |
| Transpose    | `arr.T`                                   |
| Sum/Mean     | `arr.sum()` / `arr.mean()`                |
| Filter       | `arr[arr > 5]`                            |
| Conditional  | `np.where(arr > 5, 'Y', 'N')`             |
| Stack        | `np.vstack((a,b))` / `np.hstack((a,b))`   |
| Random       | `np.random.default_rng(42).random((3,3))` |
| Matrix mult  | `A @ B`                                   |
| Save/Load    | `np.save()` / `np.load()`                 |

---

> **Source folders**: `numpy/useOfNumpy.md`, `numpy/numpy-Zero-to-hero.ipynb`, `numpy/numpy_Example 1.ipynb`
