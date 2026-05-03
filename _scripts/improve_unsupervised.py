import json

# ── K-Means: cell 32 ─────────────────────────────────────────────────────────

km_path = r"c:\Users\User\Documents\Github\machine-learning-journey\02_unsupervised_learning\clustering\notebooks\01_k_means_clustering.ipynb"

with open(km_path, encoding="utf-8") as f:
    km_nb = json.load(f)

km_nb["cells"][32]["source"] = [
    "### Step 3: Visualise Hierarchical Clustering Results\n",
    "\n",
    "Plot the hierarchical cluster assignments alongside the K-Means results to compare how the two methods segment the customers.\n",
    "\n",
    "**What to look for in the comparison:**\n",
    "\n",
    "- Do both methods produce similar cluster boundaries? If yes, the segments are robust and not algorithm-dependent.\n",
    "- Are there customers that K-Means and Hierarchical Clustering assign to different groups? These are the ambiguous cases near cluster boundaries.\n",
    "- Does one method produce more natural-looking clusters in the income/spending space?\n",
    "\n",
    "**Key difference between the methods:**\n",
    "K-Means assumes clusters are roughly spherical and equal-sized. Hierarchical clustering makes no such assumption and can find clusters of irregular shape — though it is much slower on large datasets (O(n²) vs K-Means' O(n))."
]

with open(km_path, "w", encoding="utf-8") as f:
    json.dump(km_nb, f, indent=1, ensure_ascii=False)
print("Updated: 01_k_means_clustering.ipynb")


# ── Apriori: cell 16 ─────────────────────────────────────────────────────────

apriori_path = r"c:\Users\User\Documents\Github\machine-learning-journey\02_unsupervised_learning\association_rules\notebooks\01_apriori.ipynb"

with open(apriori_path, encoding="utf-8") as f:
    ap_nb = json.load(f)

ap_nb["cells"][16]["source"] = [
    "### Raw Output from the Apriori Function\n",
    "\n",
    "The `apriori()` function returns a generator of `RelationRecord` objects. Each record contains:\n",
    "\n",
    "- **`items`**: The itemset (e.g., `frozenset({'burger', 'eggs'})`)\n",
    "- **`support`**: Fraction of transactions containing this itemset\n",
    "- **`ordered_statistics`**: List of rules derived from this itemset, each with:\n",
    "  - `items_base`: The antecedent (\"if\" part)\n",
    "  - `items_add`: The consequent (\"then\" part)\n",
    "  - `confidence`: P(consequent | antecedent)\n",
    "  - `lift`: How much more likely the consequent is given the antecedent, vs randomly\n",
    "\n",
    "The raw output is not user-friendly. The next step converts it into a readable DataFrame."
]

with open(apriori_path, "w", encoding="utf-8") as f:
    json.dump(ap_nb, f, indent=1, ensure_ascii=False)
print("Updated: 01_apriori.ipynb")


# ── ECLAT: cell 16 ────────────────────────────────────────────────────────────

eclat_path = r"c:\Users\User\Documents\Github\machine-learning-journey\02_unsupervised_learning\association_rules\notebooks\02_eclat.ipynb"

with open(eclat_path, encoding="utf-8") as f:
    ec_nb = json.load(f)

ec_nb["cells"][16]["source"] = [
    "### Raw Output from the Association Rule Mining\n",
    "\n",
    "Same raw output format as the Apriori notebook. Each `RelationRecord` contains the itemset, support, and any rules derived from it.\n",
    "\n",
    "**The key difference for ECLAT:** We are primarily interested in the **support** values of frequent itemsets, not the full directional rules (confidence, lift) that Apriori emphasises.\n",
    "\n",
    "ECLAT focuses on finding **what items frequently appear together** — it is symmetric. Apriori additionally asks **in which direction** the association flows (does buying X cause buying Y, or vice versa?).\n",
    "\n",
    "The next step extracts only the support-based itemset information, filtering out the directional rule statistics."
]

with open(eclat_path, "w", encoding="utf-8") as f:
    json.dump(ec_nb, f, indent=1, ensure_ascii=False)
print("Updated: 02_eclat.ipynb")
