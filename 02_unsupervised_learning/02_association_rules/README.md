# Association Rule Learning

Association rule learning discovers **relationships between items in transaction datasets** — which items frequently appear together, and how strong those relationships are. It is one of the most practically useful forms of unsupervised learning because its outputs are directly actionable: rules of the form "customers who buy X also tend to buy Y."

The classic application is market basket analysis: given a record of which products each customer bought, find which product combinations appear together more often than would be expected by chance.

---

## The Problem It Solves

You have a large collection of transactions. Each transaction is a set of items — a shopping basket, a medical record's symptom list, a user's session page views. You want to find which item combinations co-occur frequently, and how strong the relationship between items is.

Brute-force enumeration is impossible: a store with 10,000 products has 2^10,000 possible itemsets. Both Apriori and ECLAT use the **Apriori principle** — any subset of a frequent itemset must also be frequent — to prune the search space dramatically.

---

## The Three Metrics You Must Understand

Before looking at any algorithm, understand the three metrics that evaluate rules. Getting these wrong is the single most common mistake in association rule mining.

**Support** — how common is this itemset overall?
```
Support(A) = transactions containing A / total transactions
```
Filters out rare itemsets. Set `min_support` to avoid rules that apply to too few customers to be actionable.

**Confidence** — given A, how likely is B?
```
Confidence(A → B) = Support(A ∪ B) / Support(A)
```
The fraction of A-containing transactions that also contain B. Sounds good, but can be misleading if B is already very common.

**Lift** — is this association real, or just because B is common anyway?
```
Lift(A → B) = Confidence(A → B) / Support(B)
```
Lift > 1 means A and B appear together more than chance. **This is the most important metric for filtering meaningful rules.** A confidence of 0.80 for "bread → milk" looks impressive until you realise milk appears in 85% of all transactions — the lift would be less than 1, meaning bread actually slightly *suppresses* milk purchases relative to base rates.

---

## Algorithms in This Section

### Apriori
**Notebook:** [01_apriori.ipynb](notebooks/01_apriori.ipynb) | **Guide:** [teaching/03_apriori_and_eclat.md](teaching/03_apriori_and_eclat.md)

Apriori uses a **horizontal data format** (transactions as rows) and makes multiple passes through the database — one pass per itemset size — pruning candidates using the Apriori principle.

The output includes full **directional rules**: `{bread, butter} → {milk}` with support, confidence, and lift. The direction matters — `{milk} → {bread, butter}` is a different rule with a potentially different confidence.

**Use when:**
- You need directional rules for recommendations ("because you bought X, you might like Y")
- You want the complete set of association rules filtered by all three metrics
- The dataset has a manageable number of transactions

**Limitation:** Multiple database scans make it slow on very large datasets.

---

### ECLAT
**Notebook:** [02_eclat.ipynb](notebooks/02_eclat.ipynb) | **Guide:** [teaching/03_apriori_and_eclat.md](teaching/03_apriori_and_eclat.md)

ECLAT uses a **vertical data format**: instead of transaction rows, it stores a list of transaction IDs for each item. Finding the support of `{bread, butter}` requires only intersecting two sets — no repeated database scans.

ECLAT is **symmetric**: it finds which items frequently appear together, but does not generate directional rules (no confidence or lift). The question it answers is "what co-occurs?" not "what causes what?"

**Use when:**
- You want to know which items frequently appear together, without caring about direction
- Speed matters and TID-lists fit in memory
- The dataset is large and multiple Apriori scans are too slow

---

## When to Use Association Rule Learning

Association rule learning applies whenever you have **transaction data** — any dataset where each record is a set of items:

| Domain | Transactions | Items |
|--------|-------------|-------|
| Retail | Customer purchases | Products |
| Web analytics | User sessions | Pages visited |
| Medicine | Patient records | Symptoms or diagnoses |
| Bioinformatics | Gene expression experiments | Active genes |
| Cybersecurity | Network sessions | Events or packets |

The output — "A and B appear together more than expected" — is only useful if you act on it. Always filter by lift ≥ 1.5 (or higher) before deciding which rules are worth implementing.

---

## The Threshold Trade-off

The two parameters you set before running either algorithm determine everything about the output:

| Parameter | Lower value | Higher value |
|-----------|------------|--------------|
| `min_support` | More rules, includes rare patterns | Fewer rules, only robust patterns |
| `min_confidence` | More rules, weaker associations | Fewer rules, stronger associations |

A common starting point: `min_support=0.01` (appears in at least 1% of transactions), then filter the output by `lift ≥ 2.0`. Adjust based on dataset size and how many rules you can realistically act on.

---

## What the Teaching Guide Covers

[teaching/03_apriori_and_eclat.md](teaching/03_apriori_and_eclat.md) covers both algorithms in depth:
- The Apriori principle and how it prunes the search space
- Support, confidence, and lift — with examples showing why lift is essential
- Step-by-step walkthrough of both algorithms
- Side-by-side comparison of when to use each
- Real-world applications with worked examples
- Implementation code for both
- Common pitfalls (especially the lift trap)
