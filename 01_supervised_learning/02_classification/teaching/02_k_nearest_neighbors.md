# K-Nearest Neighbors (K-NN) - Complete Guide

## Table of Contents
1. [What is K-Nearest Neighbors?](#what-is-k-nearest-neighbors)
2. [How Does K-NN Work?](#how-does-k-nn-work)
3. [Distance Metrics](#distance-metrics)
4. [Choosing the Right K Value](#choosing-the-right-k-value)
5. [Advantages and Disadvantages](#advantages-and-disadvantages)
6. [When to Use K-NN](#when-to-use-k-nn)
7. [Real-World Applications](#real-world-applications)
8. [Implementation Steps](#implementation-steps)
9. [Optimization Techniques](#optimization-techniques)
10. [Common Pitfalls](#common-pitfalls)

---

## What is K-Nearest Neighbors?

**K-Nearest Neighbors (K-NN)** is a simple, intuitive classification algorithm that makes predictions based on the **similarity** of data points. It's called a "lazy learning" algorithm because it doesn't build a model during training - instead, it stores all training data and makes decisions at prediction time.

### Core Concept
**"Tell me who your neighbors are, and I'll tell you who you are"**

For any new data point:
1. Find the K closest training examples
2. Look at their classes/labels
3. Predict the most common class among these neighbors

**Example**: To classify a new email as spam or not spam:
- Find the 5 most similar emails in your training data
- If 3 are spam and 2 are not spam → Predict "spam"

---

## How Does K-NN Work?

### The Algorithm (Step by Step)

#### For Classification:
1. **Store Training Data**: Keep all training examples in memory
2. **Calculate Distances**: For a new point, calculate distance to all training points
3. **Find K Neighbors**: Select the K closest training points
4. **Vote**: Count the classes of these K neighbors
5. **Predict**: Choose the most frequent class (majority vote)

#### For Regression:
- Same steps, but instead of voting, **average** the target values of K neighbors

### Visual Example
```
Training Data:
• Red points: Class A
■ Blue squares: Class B

New point: ? (to classify)

K=3: Find 3 nearest neighbors
Result: 2 Class A, 1 Class B → Predict Class A

K=5: Find 5 nearest neighbors  
Result: 2 Class A, 3 Class B → Predict Class B
```

**Key Insight**: The choice of K dramatically affects the prediction!

---

## Distance Metrics

### 1. **Euclidean Distance** (Most Common)
```
d = √[(x₁-y₁)² + (x₂-y₂)² + ... + (xₙ-yₙ)²]
```
- **Best for**: Continuous features, when all features are equally important
- **Think**: Straight-line distance between two points

### 2. **Manhattan Distance** (City Block)
```
d = |x₁-y₁| + |x₂-y₂| + ... + |xₙ-yₙ|
```
- **Best for**: When you can only move along grid lines (like city blocks)
- **More robust** to outliers than Euclidean

### 3. **Minkowski Distance** (Generalized)
```
d = (|x₁-y₁|ᵖ + |x₂-y₂|ᵖ + ... + |xₙ-yₙ|ᵖ)^(1/p)
```
- **p=1**: Manhattan distance
- **p=2**: Euclidean distance
- **p=∞**: Chebyshev distance (maximum difference in any dimension)

### 4. **Cosine Distance**
```
d = 1 - (A·B) / (||A|| × ||B||)
```
- **Best for**: High-dimensional data, text analysis
- **Measures angle** between vectors, not magnitude

### 5. **Hamming Distance**
```
d = number of positions where bits differ
```
- **Best for**: Categorical data, binary features

### Choosing Distance Metric
| Data Type | Recommended Distance |
|-----------|---------------------|
| Continuous, scaled | Euclidean |
| Mixed types | Manhattan |
| High-dimensional | Cosine |
| Categorical | Hamming |
| With outliers | Manhattan |

---

## Choosing the Right K Value

### The K Dilemma

#### Small K (K=1, K=3):
**Pros**:
- More sensitive to local patterns
- Can capture fine details
- Better for complex boundaries

**Cons**:
- More sensitive to noise
- Higher variance
- Overfitting risk

#### Large K (K=50, K=100):
**Pros**:
- More stable predictions
- Less sensitive to noise
- Smoother decision boundaries

**Cons**:
- May miss local patterns
- Higher bias
- Underfitting risk

### Finding Optimal K

#### 1. **Cross-Validation Method**
```python
from sklearn.model_selection import cross_val_score

k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
```

#### 2. **Rule of Thumb**
- **K = √N** (where N is number of training samples)
- **Always use odd K** for binary classification (avoids ties)
- **Start with K=5** and tune from there

#### 3. **Consider Your Data**
- **Few samples**: Use smaller K (3-7)
- **Many samples**: Can use larger K (10-50)
- **Noisy data**: Use larger K
- **Clean data**: Can use smaller K

---

## Advantages and Disadvantages

### ✅ Advantages

1. **Simple and Intuitive**
   - Easy to understand and explain
   - No complex math or assumptions
   - Great for beginners

2. **No Training Required**
   - No model to build or parameters to learn
   - New data can be added easily
   - Adapts automatically to new patterns

3. **Naturally Handles Multi-Class**
   - Works with any number of classes
   - No need to modify algorithm

4. **Non-Parametric**
   - No assumptions about data distribution
   - Can learn any decision boundary shape
   - Flexible and adaptive

5. **Can Do Both Classification and Regression**
   - Same algorithm works for both problems
   - Just change the final step (vote vs average)

6. **Provides Confidence Measure**
   - Can see how "sure" prediction is based on neighbor agreement
   - Useful for uncertain cases

### ❌ Disadvantages

1. **Computationally Expensive**
   - Must calculate distance to ALL training points
   - Slow prediction time (O(N) per prediction)
   - Memory intensive (stores all data)

2. **Curse of Dimensionality**
   - Performance degrades with many features
   - All points become equally "far" in high dimensions
   - Need dimensionality reduction

3. **Sensitive to Feature Scaling**
   - Features with larger scales dominate distance calculation
   - Must normalize/standardize features

4. **Sensitive to Irrelevant Features**
   - Noise features affect distance calculation
   - Need feature selection

5. **Imbalanced Data Issues**
   - Majority class can dominate neighborhoods
   - May need special handling

6. **No Model Interpretability**
   - Can't explain WHY a prediction was made
   - Just "because neighbors were this class"

---

## When to Use K-NN

### ✅ Good Choice When:
- **Small to medium datasets** (< 100K samples)
- **Low to medium dimensions** (< 20 features)
- **Non-linear decision boundaries** expected
- **Irregular data distributions**
- **Need simple, explainable method**
- **Local patterns** are important
- **Quick prototyping** or baseline model

### ❌ Avoid When:
- **Large datasets** (millions of samples)
- **High-dimensional data** (hundreds of features)
- **Real-time prediction** required
- **Memory is limited**
- **Many irrelevant features**
- **Need feature importance** insights

---

## Real-World Applications

### 1. **Recommendation Systems**
- **Product Recommendations**: Find users with similar purchase history
- **Movie/Music**: Recommend based on similar user preferences
- **Content Filtering**: Suggest articles based on reading patterns

### 2. **Image Recognition**
- **Handwriting Recognition**: Compare new character to known examples
- **Face Recognition**: Find similar faces in database
- **Medical Imaging**: Classify based on similar cases

### 3. **Text Mining & NLP**
- **Document Classification**: Group similar documents
- **Sentiment Analysis**: Classify based on similar text patterns
- **Language Detection**: Identify language based on character patterns

### 4. **Finance**
- **Credit Risk**: Find customers with similar profiles
- **Fraud Detection**: Identify unusual patterns compared to normal behavior
- **Portfolio Management**: Group similar stocks or investments

### 5. **Healthcare**
- **Diagnosis**: Compare patient symptoms to similar cases
- **Drug Discovery**: Find compounds with similar properties
- **Treatment Recommendation**: Based on similar patient responses

### 6. **Market Research**
- **Customer Segmentation**: Group customers with similar behavior
- **Price Optimization**: Set prices based on similar products
- **A/B Testing**: Compare performance to similar campaigns

---

## Implementation Steps

### Step 1: Data Preparation
```python
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = pd.read_csv('your_data.csv')
X = data.drop('target', axis=1)
y = data['target']
```

### Step 2: Feature Scaling (CRITICAL!)
```python
# K-NN is very sensitive to feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

### Step 3: Find Optimal K
```python
# Test different K values
k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    k_scores.append(accuracy_score(y_test, y_pred))

optimal_k = k_range[np.argmax(k_scores)]
print(f"Optimal K: {optimal_k}")
```

### Step 4: Train Final Model
```python
# Train with optimal K
knn_final = KNeighborsClassifier(n_neighbors=optimal_k)
knn_final.fit(X_train, y_train)
```

### Step 5: Make Predictions
```python
# Predictions
y_pred = knn_final.predict(X_test)
y_pred_proba = knn_final.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Step 6: Visualize Results (2D case)
```python
import matplotlib.pyplot as plt

# For 2D data, visualize decision boundary
def plot_decision_boundary(X, y, model, title):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
    plt.title(title)
    plt.show()
```

---

## Optimization Techniques

### 1. **Approximate Nearest Neighbors**
```python
# Use libraries like FAISS, Annoy, or NMSLIB for large datasets
from sklearn.neighbors import NearestNeighbors
import faiss

# FAISS for GPU-accelerated search
index = faiss.IndexFlatL2(d)  # d = number of dimensions
index.add(X_train)
distances, indices = index.search(X_test, k)
```

### 2. **Dimensionality Reduction**
```python
from sklearn.decomposition import PCA

# Reduce dimensions before K-NN
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)
```

### 3. **Feature Selection**
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select most relevant features
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

### 4. **Distance-Weighted Voting**
```python
# Closer neighbors have more influence
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
```

### 5. **Data Structures for Speed**
```python
# Use efficient data structures
knn = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='ball_tree',  # or 'kd_tree' for low dimensions
    leaf_size=30
)
```

---

## Common Pitfalls

### 1. **Forgetting Feature Scaling**
```python
# ❌ Wrong - features not scaled
knn.fit(X_train, y_train)

# ✅ Correct - always scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn.fit(X_train_scaled, y_train)
```

### 2. **Using Even K for Binary Classification**
```python
# ❌ Can cause ties
knn = KNeighborsClassifier(n_neighbors=4)

# ✅ Use odd K to avoid ties
knn = KNeighborsClassifier(n_neighbors=5)
```

### 3. **Not Handling Imbalanced Data**
```python
# ✅ Use stratified sampling and appropriate metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)
```

### 4. **Using K-NN with Too Many Features**
```python
# ✅ Reduce dimensions first
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_reduced = pca.fit_transform(X)
```

### 5. **Not Cross-Validating K**
```python
# ✅ Always validate your K choice
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': range(1, 31)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_k = grid_search.best_params_['n_neighbors']
```

---

## Summary

**K-Nearest Neighbors** is a simple but powerful algorithm that:
- Makes predictions based on similarity to training examples
- Requires careful choice of K value and distance metric
- Needs feature scaling and dimensionality consideration
- Works well for non-linear patterns and local clusters
- Can be slow but is highly intuitive

**Best for**: Non-linear classification problems with moderate-sized, well-preprocessed datasets.

**Remember**: K-NN's success depends heavily on data preprocessing and parameter tuning!

---

*Next: Explore Support Vector Machines for maximum margin classification*