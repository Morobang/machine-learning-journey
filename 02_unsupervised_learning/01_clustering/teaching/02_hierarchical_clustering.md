# Hierarchical Clustering — Complete Guide

## Table of Contents
1. [What is Hierarchical Clustering?](#what-is-hierarchical-clustering)
2. [How the Algorithm Works](#how-the-algorithm-works)
3. [Linkage Methods](#linkage-methods)
4. [Reading a Dendrogram](#reading-a-dendrogram)
5. [Advantages and Disadvantages](#advantages-and-disadvantages)
6. [K-Means vs Hierarchical Clustering](#k-means-vs-hierarchical-clustering)
7. [Real-World Applications](#real-world-applications)
8. [Implementation Steps](#implementation-steps)
9. [Common Pitfalls](#common-pitfalls)

---

## What is Hierarchical Clustering?

**Hierarchical Clustering** builds a tree of clusters — called a **dendrogram** — that shows the nested grouping structure of the data at every level of granularity.

Unlike K-Means, you do **not** need to specify the number of clusters upfront. You run the algorithm once, produce the full dendrogram, and then decide how many clusters you want by "cutting" the tree at the appropriate height.

There are two approaches:
- **Agglomerative (bottom-up):** Start with every point as its own cluster. Repeatedly merge the two closest clusters until everything is one cluster. This is by far the more common approach.
- **Divisive (top-down):** Start with one cluster containing all points. Repeatedly split the largest/most spread-out cluster. Rarely used in practice.

---

## How the Algorithm Works

### Agglomerative Clustering — Step by Step

1. **Initialise:** Each of the N data points is its own cluster (N clusters total).
2. **Compute distances:** Build an N×N distance matrix between all pairs of clusters.
3. **Merge:** Find the two closest clusters and merge them into one. Now N−1 clusters.
4. **Update distance matrix:** Recompute distances from the new merged cluster to all remaining clusters (using the chosen linkage method).
5. **Repeat steps 3–4** until only one cluster remains.
6. **Record all merges** — this sequence of merges is the dendrogram.

**Time complexity:** O(n³) in the naive implementation. Optimised implementations achieve O(n² log n). This is why hierarchical clustering is much slower than K-Means on large datasets.

**Space complexity:** O(n²) for the distance matrix — problematic for n > 10,000.

---

## Linkage Methods

The **linkage criterion** defines how the distance between two clusters is measured after a merge. This is the most important hyperparameter.

| Linkage | Distance between clusters A and B | Characteristics |
|---------|----------------------------------|-----------------|
| **Single** | Min distance between any point in A and any point in B | Creates long, "chained" clusters; sensitive to outliers |
| **Complete** | Max distance between any point in A and any point in B | Creates compact, equal-diameter clusters |
| **Average** | Mean distance between all pairs (one from A, one from B) | Compromise; moderate sensitivity to outliers |
| **Ward** | Increase in total WCSS from merging A and B | Creates clusters that minimise within-cluster variance; usually best for compact clusters |

**Ward linkage is the default choice for most tabular data.** It tends to produce clusters of roughly equal size and is the most similar to K-Means in spirit (both minimise within-cluster variance).

**Single linkage** can find elongated, non-spherical clusters that K-Means would miss — but is vulnerable to the "chaining effect" where one outlier bridges two natural clusters.

---

## Reading a Dendrogram

A dendrogram displays the merge history:
- **Leaves (bottom)** — individual data points
- **Height of horizontal line** — the distance at which two clusters were merged
- **Cutting the dendrogram at height h** — gives you all clusters that were formed before that height

**How to choose the number of clusters:**
Look for the largest vertical gap in the dendrogram — the longest horizontal line you can draw without crossing any merge lines. The number of vertical lines you cross at that height is your optimal K.

**Example:** If the dendrogram has three tall vertical segments before merging into one, cutting at the right height gives K=3.

**The dendrogram is the key advantage over K-Means:** it shows you the *structure* of the data — are there 2 natural clusters that each have 3 sub-clusters? K-Means cannot reveal this hierarchy.

---

## Advantages and Disadvantages

### Advantages

| Advantage | Detail |
|-----------|--------|
| No need to specify K upfront | The dendrogram shows natural groupings at all scales |
| Reveals cluster hierarchy | Sub-clusters and super-clusters are visible in one tree |
| Deterministic | No random initialisation — same data always produces same dendrogram |
| Works with any distance metric | Not limited to Euclidean distance |
| Single linkage finds non-convex clusters | Can detect clusters that K-Means cannot |

### Disadvantages

| Disadvantage | Detail |
|-------------|--------|
| Slow on large datasets | O(n²) space, O(n³) or O(n² log n) time — unusable for n > ~10,000 |
| Merges are irreversible | Once two clusters are merged, they can never be separated |
| Sensitive to outliers | Especially with single and complete linkage |
| Dendrogram interpretation is subjective | Different analysts may choose different cut heights |

---

## K-Means vs Hierarchical Clustering

| Property | K-Means | Hierarchical |
|----------|---------|--------------|
| Need to specify K? | Yes, required | No — choose after seeing dendrogram |
| Speed | O(n × K × iterations) — fast | O(n² log n) — slow for large n |
| Deterministic? | No (random initialisation) | Yes |
| Cluster shapes | Spherical only | More flexible (depends on linkage) |
| Reveals hierarchy? | No | Yes |
| Best for | Large datasets, compact clusters | Exploratory analysis, small/medium datasets, unknown K |

**In practice:** Use hierarchical clustering first for exploration — the dendrogram tells you how many clusters exist naturally. Then use K-Means on larger datasets where you already know K.

---

## Real-World Applications

**Gene Expression Analysis:**
Group genes with similar expression profiles across experimental conditions. The hierarchy reveals gene families and regulatory modules. Scientists read dendrograms as heatmaps with cluster trees alongside.

**Customer Segmentation (Exploration Phase):**
Before committing to a K for K-Means production segmentation, run hierarchical clustering on a sample to understand whether the natural structure is 3, 5, or 8 clusters.

**Document Organisation:**
Build a topic hierarchy for a document collection. The dendrogram shows that "Sports" sub-divides into "Football", "Tennis", "Golf", and "Football" further sub-divides into "Premier League", "Champions League", etc.

**Fraud Detection:**
Group transactions by spending behaviour. Unusual sub-clusters far from the main branches are candidates for fraud investigation.

---

## Implementation Steps

```python
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Plot dendrogram to choose K
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.axhline(y=7, color='r', linestyle='--')  # Cut line — adjust height based on gap
plt.show()

# 3. Train with chosen K (from dendrogram)
hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
y_hc = hc.fit_predict(X_scaled)

# 4. Inspect cluster assignments
import pandas as pd
pd.Series(y_hc).value_counts()  # Check cluster sizes
```

**Key parameters for `AgglomerativeClustering`:**
- `n_clusters` — set after inspecting the dendrogram
- `linkage` — `'ward'` (default, usually best), `'complete'`, `'average'`, `'single'`
- `metric` — distance metric (only Euclidean with Ward linkage)

---

## Common Pitfalls

**1. Using hierarchical clustering on large datasets**
With n > 5,000, the O(n²) distance matrix becomes too large to compute. Use K-Means or mini-batch K-Means instead.

**2. Ignoring the dendrogram structure**
Cutting the dendrogram at K=5 because you want 5 clusters misses the point. Look at the actual gap structure — if the natural cut is at K=3, forcing K=5 creates arbitrary sub-clusters.

**3. Using Ward linkage with non-Euclidean distances**
Ward linkage is only valid with Euclidean distance. For other metrics (cosine similarity for text, Jaccard for sets), use average or complete linkage.

**4. Forgetting to scale features**
The same requirement as K-Means — Euclidean distances are dominated by high-magnitude features without scaling.

**5. Treating the bottom of the dendrogram as signal**
The very bottom of the dendrogram (small heights) shows noise-level similarities. Focus on the large-gap region of the tree, not individual point merges.
