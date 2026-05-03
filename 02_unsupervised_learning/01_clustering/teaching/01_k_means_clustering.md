# K-Means Clustering — Complete Guide

## Table of Contents
1. [What is K-Means?](#what-is-k-means)
2. [How the Algorithm Works](#how-the-algorithm-works)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Choosing K: The Elbow Method](#choosing-k-the-elbow-method)
5. [Advantages and Disadvantages](#advantages-and-disadvantages)
6. [When to Use K-Means](#when-to-use-k-means)
7. [Real-World Applications](#real-world-applications)
8. [Implementation Steps](#implementation-steps)
9. [Common Pitfalls](#common-pitfalls)

---

## What is K-Means?

**K-Means** is an unsupervised learning algorithm that partitions a dataset into **K distinct, non-overlapping clusters**. Unlike supervised learning, there are no labels — the algorithm discovers structure in the data on its own.

**The core idea:** Assign every data point to one of K clusters such that points within the same cluster are as similar as possible (small within-cluster variance) and points in different clusters are as different as possible (large between-cluster variance).

**The key constraint you must decide upfront:** You must specify K — the number of clusters — before running the algorithm. Choosing K poorly is the most common source of bad results.

---

## How the Algorithm Works

K-Means is an iterative algorithm with two alternating steps:

### Step 1: Initialisation
Choose K initial centroids. The default (k-means++) picks the first centroid randomly, then picks each subsequent centroid with probability proportional to distance from the nearest existing centroid. This avoids bad initialisation that traps the algorithm in poor local optima.

### Step 2: Assignment
Assign each data point to the nearest centroid (measured by Euclidean distance):

```
cluster(xᵢ) = argmin_k ||xᵢ - μₖ||²
```

Every point belongs to exactly one cluster.

### Step 3: Update
Recompute each centroid as the mean of all points assigned to it:

```
μₖ = (1/|Cₖ|) × Σ xᵢ  for all xᵢ in cluster k
```

### Step 4: Repeat
Repeat assignment and update until centroids stop moving (convergence) or a maximum number of iterations is reached.

**Convergence is guaranteed** — WCSS (within-cluster sum of squares) decreases or stays the same at each iteration. But convergence to the *global* optimum is not guaranteed; the algorithm may find a local minimum.

---

## Mathematical Foundation

### Objective Function (WCSS)
K-Means minimises the **Within-Cluster Sum of Squares**:

```
WCSS = Σₖ Σ_{xᵢ ∈ Cₖ} ||xᵢ - μₖ||²
```

Where:
- K = number of clusters
- Cₖ = set of points in cluster k
- μₖ = centroid of cluster k
- `||xᵢ - μₖ||²` = squared Euclidean distance

Lower WCSS means tighter, more compact clusters.

### Why Squared Distance?
Squaring the distance:
1. Makes the objective differentiable (needed for mathematical analysis)
2. Penalises large deviations more heavily than small ones
3. Ensures the mean is the optimal centroid (the value that minimises squared distances from all cluster members)

### Distance Metric
K-Means uses **Euclidean distance** by default. This is why **feature scaling is required** — if salary ranges from 0–100,000 and age from 0–100, the salary dimension dominates the distance calculation entirely. StandardScaler equalises the influence of each feature.

---

## Choosing K: The Elbow Method

Since you must choose K in advance, the **Elbow Method** helps you pick it empirically:

1. Run K-Means for K = 1, 2, 3, ..., 10 (or more)
2. Record the WCSS for each K
3. Plot WCSS vs K
4. Look for an "elbow" — the point where adding more clusters gives diminishing WCSS reduction

**Why WCSS always decreases with more K:** With K = N (one cluster per point), WCSS = 0. The elbow identifies the K where you're getting the most "bang for your buck" — each additional cluster still meaningfully reduces WCSS.

**The elbow is not always obvious.** If the data has no clear cluster structure, the curve decreases smoothly with no elbow. This is a signal that K-Means may not be the right approach for this data.

**Alternative methods:**
- **Silhouette score** — measures how well-separated clusters are (−1 to +1, higher is better)
- **Gap statistic** — compares WCSS to expected WCSS under random data
- **Domain knowledge** — often the most reliable: "we have 5 product lines, so K=5"

---

## Advantages and Disadvantages

### Advantages

| Advantage | Detail |
|-----------|--------|
| Simple and fast | O(n × K × iterations) — scales to large datasets |
| Guaranteed convergence | Always terminates in finite steps |
| Interpretable output | Centroids represent the "typical" member of each cluster |
| Works well when clusters are roughly spherical and equal-sized | The most common real-world case |

### Disadvantages

| Disadvantage | Detail |
|-------------|--------|
| Must specify K | Poor choice of K gives meaningless clusters |
| Sensitive to initialisation | Different random starts can produce different results (use `n_init=10`) |
| Assumes spherical clusters | Fails on elongated, concave, or irregularly shaped clusters |
| Sensitive to outliers | One outlier pulls the centroid towards it |
| Hard assignment | Every point belongs to exactly one cluster; no notion of ambiguity |
| Scale-sensitive | Must normalise features first |

---

## When to Use K-Means

**Use K-Means when:**
- You have a rough idea of how many clusters to expect
- Clusters are roughly equal in size and roughly spherical
- You need results quickly on a large dataset
- Interpretability matters (centroids are human-readable)

**Avoid K-Means when:**
- Clusters have very different sizes or densities
- Clusters are non-convex (e.g., two concentric rings)
- The data has many outliers
- You have no idea how many clusters exist (try hierarchical first — the dendrogram shows natural groupings)

---

## Real-World Applications

**Customer Segmentation:**
Segment mall customers by income and spending score into groups like "high income, low spending" (savers), "low income, high spending" (impulsive buyers), "moderate income, moderate spending" (mainstream). Each segment gets a different marketing strategy.

**Image Compression:**
Represent each pixel by its cluster's centroid colour. With K=16, a 24-bit colour image becomes a 4-bit image — 6x compression — with visually acceptable quality.

**Document Clustering:**
Group news articles by topic for content recommendation. Feed TF-IDF vectors to K-Means; each cluster represents a topic.

**Anomaly Detection:**
Points far from any centroid are potential anomalies. After clustering normal data, flag new points whose distance to the nearest centroid exceeds a threshold.

---

## Implementation Steps

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Scale features (required for Euclidean-distance-based methods)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Find optimal K with Elbow Method
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel('Number of Clusters K')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# 3. Train with chosen K
km = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
y_km = km.fit_predict(X_scaled)

# 4. Access results
labels = km.labels_          # cluster assignment for each point
centroids = km.cluster_centers_  # centroid coordinates (in scaled space)
inertia = km.inertia_        # WCSS
```

**Key parameters:**
- `init='k-means++'` — smart initialisation (default, always use this)
- `n_init=10` — run the algorithm 10 times with different initialisations, keep the best result
- `random_state=42` — for reproducibility

---

## Common Pitfalls

**1. Forgetting to scale features**
Without scaling, high-magnitude features dominate distance calculations. Salary will completely overshadow age. Always `StandardScaler` before K-Means.

**2. Ignoring local optima**
K-Means can converge to a suboptimal solution depending on initial centroid placement. Use `n_init=10` or higher to run multiple times and take the best result.

**3. Treating WCSS as a model quality metric**
WCSS always decreases as K increases. It cannot tell you whether clusters are meaningful — just whether they are tight. Always validate against domain knowledge and use silhouette scores.

**4. Reporting results from one run**
K-Means with `n_init=1` is unreliable. Always use `n_init >= 10` for production results.

**5. Assuming K-Means found the "true" clusters**
K-Means is an optimisation algorithm, not a truth detector. It finds the K-partition that minimises WCSS — which may or may not correspond to meaningful real-world groups. Always validate cluster assignments against domain expertise.
