# Association Rule Mining: Apriori and ECLAT — Complete Guide

## Table of Contents
1. [What is Association Rule Mining?](#what-is-association-rule-mining)
2. [Key Metrics: Support, Confidence, Lift](#key-metrics-support-confidence-lift)
3. [The Apriori Algorithm](#the-apriori-algorithm)
4. [The ECLAT Algorithm](#the-eclat-algorithm)
5. [Apriori vs ECLAT](#apriori-vs-eclat)
6. [Advantages and Disadvantages](#advantages-and-disadvantages)
7. [Real-World Applications](#real-world-applications)
8. [Implementation Steps](#implementation-steps)
9. [Common Pitfalls](#common-pitfalls)

---

## What is Association Rule Mining?

**Association Rule Mining** discovers relationships between items in large transaction datasets. The classic example: "customers who buy bread and butter also tend to buy milk."

This is a form of **unsupervised learning** — we are not predicting a target variable, we are discovering the structure of co-occurrence patterns in the data.

**The output is a set of rules of the form:**
```
{bread, butter} → {milk}
```
Which reads: "if a transaction contains bread and butter, it is likely to also contain milk."

**The challenge:** A supermarket with 10,000 products has 2^10,000 possible itemsets. We need a smarter approach than brute-force enumeration.

---

## Key Metrics: Support, Confidence, Lift

These three metrics evaluate the strength and usefulness of each rule.

### Support
The fraction of all transactions that contain the itemset:

```
Support(A) = (transactions containing A) / (total transactions)
```

**Why it matters:** Support filters out rare itemsets. A rule that applies to 0.001% of transactions is statistically fragile and actionable to almost nobody.

**The minimum support threshold** (e.g., `min_support=0.003`) is the primary lever for controlling how many rules you get. Lower it to find rare but potentially interesting patterns; raise it to focus on robust, frequent patterns.

### Confidence
Given that a transaction contains the antecedent A, the probability it also contains the consequent B:

```
Confidence(A → B) = Support(A ∪ B) / Support(A)
```

**Example:** If 40 out of 1000 transactions contain {bread, butter}, and 30 of those 40 also contain {milk}:
- Confidence({bread, butter} → {milk}) = 30/40 = 0.75

**Limitation:** Confidence ignores how common B is on its own. If 80% of all transactions contain milk regardless, a 75% confidence rule is actually *worse* than random.

### Lift
How much more likely B is given A, compared to the background rate of B:

```
Lift(A → B) = Confidence(A → B) / Support(B) = Support(A ∪ B) / (Support(A) × Support(B))
```

**Interpretation:**
- **Lift > 1** — A and B appear together more than chance; a genuine positive association
- **Lift = 1** — A and B are independent; the rule is not useful
- **Lift < 1** — A and B appear together *less* than chance; a negative association

**Lift is the most important metric for filtering useful rules.** Always filter for `lift > 1` (or a higher threshold like `lift > 1.5`) to ensure rules represent real patterns, not coincidence.

---

## The Apriori Algorithm

### The Apriori Principle
"Any subset of a frequent itemset must also be frequent."

If {bread, butter, milk} is frequent, then {bread, butter}, {bread, milk}, {butter, milk}, {bread}, {butter}, and {milk} must all be frequent too.

**Contrapositive:** If {bread} is infrequent (below min_support), then any itemset containing bread (e.g., {bread, butter, milk}) must also be infrequent. We can prune the entire search space containing bread.

### Algorithm Steps

1. **Find all frequent 1-itemsets** — individual items with support ≥ min_support
2. **Generate candidate 2-itemsets** from pairs of frequent 1-itemsets
3. **Prune** candidates that contain an infrequent subset (Apriori principle)
4. **Scan the database** to count support of remaining candidates
5. **Repeat** for 3-itemsets, 4-itemsets, etc. until no new frequent itemsets are found
6. **Generate rules** from all frequent itemsets that meet min_confidence

### Complexity
Apriori requires **multiple full scans of the transaction database** — one per itemset size. For datasets with many transactions, this is slow.

It generates **directional rules** (A → B is different from B → A), making it useful for recommendation systems where the direction matters ("because you bought X, buy Y").

---

## The ECLAT Algorithm

**ECLAT (Equivalence Class Clustering and bottom-up Lattice Traversal)** solves the same problem as Apriori but with a fundamentally different data representation.

### Vertical Data Format
Apriori uses a **horizontal format**: each row is a transaction, each column is an item.

ECLAT uses a **vertical format**: for each item, store the set of transaction IDs (TID-list) in which it appears.

```
Item    | TID-list
--------|------------------
bread   | {1, 2, 4, 7, 8}
butter  | {1, 3, 4, 6, 8}
milk    | {1, 2, 4, 5, 8}
```

### Computing Support with Set Intersection
To find support of {bread, butter}:

```
TID-list({bread, butter}) = TID-list(bread) ∩ TID-list(butter) = {1, 4, 8}
Support = |{1, 4, 8}| / total_transactions = 3/8
```

No database scan needed — just intersect two sets.

### ECLAT's Key Difference: Focus on Support, Not Direction
ECLAT is **symmetric** — it finds frequent itemsets but does not generate directional A → B rules. The question it answers is:

> "What items frequently appear together?"

Not:
> "Does buying A cause buying B?"

Use Apriori when you need directional rules (A causes B). Use ECLAT when you just want to know which items co-occur frequently (symmetric association, no direction needed).

---

## Apriori vs ECLAT

| Property | Apriori | ECLAT |
|----------|---------|-------|
| Data format | Horizontal (transaction rows) | Vertical (TID-lists per item) |
| Database scans | Multiple (one per itemset size) | Few — set intersections avoid repeated scans |
| Memory | Low | Higher (TID-lists can be large) |
| Output | Directional rules (A → B) with confidence and lift | Frequent itemsets with support only |
| Best for | Recommendation systems, market basket analysis with directional interest | Mining co-occurrence patterns, when direction doesn't matter |
| Speed | Slower on large databases | Faster when TID-lists fit in memory |

---

## Advantages and Disadvantages

### Advantages

| Advantage | Detail |
|-----------|--------|
| Interpretable output | Rules are human-readable: "if A then B" |
| No target variable needed | Works on raw transaction data |
| Scalable with pruning | Apriori's pruning dramatically reduces search space |
| Actionable | Rules directly inform product placement, recommendations, cross-selling |

### Disadvantages

| Disadvantage | Detail |
|-------------|--------|
| Computational cost | Still exponential in worst case; slow on high-dimensional data |
| Many trivial rules | Low thresholds produce thousands of rules, most uninteresting |
| Sensitive to threshold choices | min_support and min_confidence choices dramatically affect output |
| No causal claims | High lift means association, not causation |
| Rare item problem | Interesting rare patterns fall below min_support |

---

## Real-World Applications

**Retail Market Basket Analysis:**
Supermarkets analyse checkout data to find which products are bought together. Results inform shelf placement (put items near each other), promotional bundling (offer discount when both bought), and cross-selling recommendations.

**E-commerce Recommendations:**
"Customers who bought X also bought Y" — this is association rule mining at scale. Amazon's early recommendation engine was built on this principle.

**Medical Diagnosis:**
Find which combinations of symptoms co-occur with particular diagnoses. "Patients with symptoms A, B, and C often also have condition D" — flagging for further investigation.

**Web Usage Mining:**
Discover which web pages are visited together in the same session. "Users who visit the pricing page and the comparison page often then visit the checkout page" — useful for UX optimisation.

---

## Implementation Steps

```python
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# 1. Prepare transaction data (list of lists)
transactions = [
    ['bread', 'butter', 'milk'],
    ['bread', 'butter'],
    ['milk', 'eggs'],
    # ...
]

# 2. Encode as binary matrix
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

# 3. Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.003, use_colnames=True)

# 4. Generate association rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

# 5. Filter for useful rules
strong_rules = rules[
    (rules['lift'] >= 3) &
    (rules['confidence'] >= 0.5)
].sort_values('lift', ascending=False)
```

**For ECLAT** (using `apyori` library):
```python
from apyori import apriori as eclat_apriori

# Returns RelationRecord objects with support per itemset
results = list(eclat_apriori(transactions, min_support=0.003, min_confidence=0.5))

# Extract support-based itemset information
support_data = [(list(r.items), r.support) for r in results]
```

---

## Common Pitfalls

**1. Setting min_support too high**
You get only the most obvious, well-known associations (e.g., beer → chips). The interesting non-obvious patterns have lower support. Start lower and filter by lift instead.

**2. Ignoring lift and only looking at confidence**
High confidence can be misleading. A rule {bread} → {milk} with 80% confidence sounds impressive — until you realise milk appears in 85% of all transactions. Lift = 0.80/0.85 < 1 means this rule is actually *anti-correlated* relative to base rates.

**3. Confusing association with causation**
"Diapers and beer" is the famous spurious correlation (used as a teaching example — almost certainly not real). High lift means the items appear together more than chance, not that one causes the other.

**4. Not filtering the output**
Running with low thresholds and not filtering by lift ≥ 1.5 or confidence ≥ 0.5 produces thousands of meaningless rules. The output DataFrame needs filtering before you can act on it.

**5. Treating all rules as equally actionable**
A rule with support=0.001 applies to 1 in 1000 transactions. Even if lift=5, acting on it (e.g., shelf rearrangement) may not be cost-effective. Consider the absolute count, not just the ratio.
