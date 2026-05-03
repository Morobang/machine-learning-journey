# Clustering

Clustering is an unsupervised learning technique that **groups similar data points together** without using labels. You do not tell the algorithm what the groups should be — it discovers the natural structure of the data on its own.

The goal: maximise similarity within groups (data points in the same cluster are as similar as possible) while maximising dissimilarity between groups (data points in different clusters are as different as possible).

---

## What Clustering Is and Is Not

**Clustering finds structure — it does not label it.** After running K-Means on customer data and getting 5 clusters, you still need domain expertise to interpret what each cluster means. "Cluster 3" is not a result — "Cluster 3 = high-income, low-spending customers who visit infrequently" is a result.

**There is no ground truth.** Unlike classification where you can check predictions against labels, clustering quality is inherently ambiguous. Two valid clusterings of the same data may lead to different but equally defensible business interpretations.

**The number of clusters is a decision, not a discovery** (for K-Means). You must decide K before running the algorithm, guided by the Elbow Method or silhouette scores, but ultimately confirmed by whether the clusters make business sense.

---

## Algorithms in This Section

### K-Means Clustering
**Notebook:** [01_k_means_clustering.ipynb](notebooks/01_k_means_clustering.ipynb) | **Guide:** [teaching/01_k_means_clustering.md](teaching/01_k_means_clustering.md)

K-Means partitions n data points into exactly K clusters by minimising the **within-cluster sum of squares (WCSS)** — the total squared distance from each point to its cluster's centroid.

The algorithm iterates between two steps until convergence:
1. Assign each point to the nearest centroid
2. Recompute each centroid as the mean of its assigned points

**Use when:**
- You have a rough idea of how many clusters to expect
- Clusters are roughly spherical and of similar size (K-Means assumes this)
- The dataset is large (K-Means is O(n × K × iterations) — fast)
- You need interpretable centroids (each centroid is a "typical member" of the cluster)

**Watch out for:** Sensitive to initialisation (use `k-means++` and `n_init=10`). Fails on non-spherical or unequal-size clusters. Always scale features first — Euclidean distance is dominated by high-magnitude features.

---

### Hierarchical Clustering
**Notebook:** [02_hierarchical_clustering.ipynb](notebooks/02_hierarchical_clustering.ipynb) | **Guide:** [teaching/02_hierarchical_clustering.md](teaching/02_hierarchical_clustering.md)

Hierarchical (agglomerative) clustering builds a **dendrogram** — a tree that shows the nested grouping structure at every level of granularity. Start with each point as its own cluster. Repeatedly merge the two closest clusters until everything is one cluster. The dendrogram records the entire merge history.

You do not need to specify K upfront. After building the dendrogram, you choose K by "cutting" the tree at the height where the largest gap in merge distances appears.

**Use when:**
- You do not know how many clusters exist and want the data to suggest it
- You want to understand the hierarchy of relationships (which sub-groups exist within larger groups?)
- The dataset is small to medium sized (≤ 5,000 rows — the O(n²) distance matrix becomes prohibitive for larger data)
- You need a deterministic result (hierarchical clustering has no random component)

**Watch out for:** Too slow for large datasets. Merges are irreversible — once two clusters are joined, they cannot be separated. Use Ward linkage by default; switch to single linkage only if you expect elongated, non-spherical clusters.

---

## K-Means vs Hierarchical — When to Use Each

| Decision | K-Means | Hierarchical |
|----------|---------|-------------|
| You know K approximately | Good | Good |
| You don't know K | Requires Elbow Method | Dendrogram shows it directly |
| Dataset size | Scales to millions of rows | Limited to ~5,000 rows |
| Cluster shapes | Spherical only | Flexible (depends on linkage) |
| Reproducibility | Run `n_init=10` for stability | Fully deterministic |
| Primary use | Production segmentation | Exploratory analysis |

**The recommended workflow:** Use hierarchical clustering first on a sample to understand the data's natural grouping structure. Use the dendrogram to determine K. Then run K-Means on the full dataset with that K.

---

## The Dataset Used Here

Both notebooks use **Mall Customers** data: 200 customers with `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`.

The task: segment customers by Annual Income and Spending Score into groups that can guide marketing strategy. Five natural segments typically emerge:
- High income, high spending (prime customers)
- High income, low spending (savers — potential to upsell)
- Moderate income, moderate spending (average customers)
- Low income, high spending (impulsive buyers — credit risk)
- Low income, low spending (budget-conscious)

---

## Evaluation Metrics for Clustering

Since there are no ground-truth labels, clustering quality is measured by internal metrics:

| Metric | What it measures | Better value |
|--------|-----------------|--------------|
| **WCSS** (Within-Cluster Sum of Squares) | Total squared distance from each point to its centroid | Lower |
| **Silhouette Score** | How well-separated each point is from its assigned cluster vs the next-nearest cluster | Higher (range: −1 to 1) |
| **Calinski-Harabasz Index** | Ratio of between-cluster to within-cluster variance | Higher |

**The Elbow Method** plots WCSS vs K. Look for the "elbow" — the point where adding more clusters gives diminishing WCSS reduction. This suggests the optimal K.

No single metric is definitive. Always validate cluster assignments against domain knowledge.

---

## What Each Teaching Guide Contains

[teaching/01_k_means_clustering.md](teaching/01_k_means_clustering.md) and [teaching/02_hierarchical_clustering.md](teaching/02_hierarchical_clustering.md) each cover:
- The algorithm step by step, with intuition
- The objective function being minimised
- How to choose the key parameters (K, linkage method)
- Advantages and disadvantages compared to the other method
- Real-world applications with worked examples
- Implementation code with key parameter explanations
- Common pitfalls to avoid
