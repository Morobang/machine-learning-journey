import json

# ── Decision Tree Classification ─────────────────────────────────────────────

dt_path = r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\02_classification\notebooks\06_decision_tree_classification.ipynb"

with open(dt_path, encoding="utf-8") as f:
    dt_nb = json.load(f)

dt_nb["cells"][29]["source"] = [
    "### Decision Tree Feature Importance\n",
    "\n",
    "Decision Trees provide built-in feature importance scores: the total Gini impurity reduction attributable to each feature across all splits.\n",
    "\n",
    "A feature used near the root (early splits) typically has high importance because it partitions the largest portion of the data. "
    "Features only used in deep leaves contribute little.\n",
    "\n",
    "Access via `classifier.feature_importances_` after training. "
    "For this 2-feature dataset (Age and Salary), comparing the two scores tells you which dimension the tree relied on most to separate buyers from non-buyers.\n",
    "\n",
    "**Limitation:** Feature importance is not the same as causality. A feature can score high simply because it correlates with other predictive features."
]

dt_nb["cells"][34]["source"] = [
    "## Step 12: Summary and Key Findings\n",
    "\n",
    "**What the metrics tell you:**\n",
    "- Check confusion matrix for the relative balance of FP vs FN errors\n",
    "- If precision << recall: the model over-predicts buyers (too many false alarms)\n",
    "- If recall << precision: the model misses buyers (too conservative)\n",
    "\n",
    "**Decision Trees in practice:**\n",
    "\n",
    "| Strength | Limitation |\n",
    "|----------|------------|\n",
    "| Highly interpretable — you can print the tree and explain every decision | Unstable — small data changes cause large tree structure changes |\n",
    "| No scaling required | High variance — without depth limits, overfits aggressively |\n",
    "| Handles nonlinear boundaries naturally | Struggles with diagonal or curved boundaries (axis-aligned splits only) |\n",
    "\n",
    "**The blocky boundary** you see in the visualisation is characteristic of Decision Trees — every split is perpendicular to a feature axis. "
    "This produces rectangular decision regions, which can look odd on continuous data but work well in practice.\n",
    "\n",
    "**Next step:** Random Forest addresses the instability and overfitting by averaging many trees trained on random subsets — see `07_random_forest_classification.ipynb`."
]

with open(dt_path, "w", encoding="utf-8") as f:
    json.dump(dt_nb, f, indent=1, ensure_ascii=False)
print("Updated: 06_decision_tree_classification.ipynb")


# ── BoW / TF-IDF ─────────────────────────────────────────────────────────────

bow_path = r"c:\Users\User\Documents\Github\machine-learning-journey\05_natural_language_processing\notebooks\02_bag_of_words_tfidf.ipynb"

with open(bow_path, encoding="utf-8") as f:
    bow_nb = json.load(f)

bow_nb["cells"][19]["source"] = [
    "## Feature Importance Analysis\n",
    "\n",
    "Logistic Regression trained on TF-IDF features assigns a coefficient to each word. A large positive coefficient means the word strongly pushes predictions toward positive; a large negative coefficient pushes toward negative.\n",
    "\n",
    "This gives us interpretable word-level evidence for why the model makes a prediction — unlike neural networks, which are black boxes.\n",
    "\n",
    "**What to look for:**\n",
    "- Top positive words should be obvious sentiment indicators (\"excellent\", \"perfect\", \"loved\")\n",
    "- Top negative words should be equally obvious (\"terrible\", \"awful\", \"waste\")\n",
    "- Surprising words in the top features indicate the model has found non-obvious but real patterns in the training corpus\n",
    "\n",
    "**TF-IDF advantage:** Common words like \"movie\" or \"film\" get low weights because they appear in nearly every review (high df → low IDF). "
    "The top features are genuinely discriminative words."
]

bow_nb["cells"][21]["source"] = [
    "## Testing with New Examples\n",
    "\n",
    "Inference pipeline for a new text string:\n",
    "\n",
    "1. **Transform** the raw text using the **already-fitted** vectorizer (do NOT call `fit_transform` again — that would recompute the vocabulary from just the new examples, discarding everything learned during training)\n",
    "2. **Predict** — the model applies the learned coefficients to the new feature vector\n",
    "\n",
    "This is the same pipeline you would use in production: fit the vectorizer once on training data, then `transform` all new inputs at inference time.\n",
    "\n",
    "**Out-of-vocabulary (OOV) words:** If the new text contains words not seen during training, the vectorizer simply ignores them. "
    "The prediction is made purely from the words the model knows — a limitation of both BoW and TF-IDF compared to word embeddings."
]

with open(bow_path, "w", encoding="utf-8") as f:
    json.dump(bow_nb, f, indent=1, ensure_ascii=False)
print("Updated: 02_bag_of_words_tfidf.ipynb")
