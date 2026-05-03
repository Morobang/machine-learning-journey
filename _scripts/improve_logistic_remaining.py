import json

path = r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\02_classification\notebooks\01_logistic_regression.ipynb"

with open(path, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

cells[15]["source"] = [
    "## Step 6: Apply Feature Scaling\n",
    "\n",
    "We fit the `StandardScaler` on the training data only, then transform both sets.\n",
    "\n",
    "The output shows Age and Salary now expressed as standard deviations from the training mean — both features on the same scale. "
    "A value of 2.0 means \"2 standard deviations above the training mean\", regardless of whether the original unit was years or dollars."
]

cells[19]["source"] = [
    "## Step 7: Train the Logistic Regression Model\n",
    "\n",
    "Logistic Regression fits a sigmoid curve to the training data:\n",
    "\n",
    "$$P(y=1) = \\frac{1}{1 + e^{-(\\beta_0 + \\beta_1 \\cdot \\text{Age} + \\beta_2 \\cdot \\text{Salary})}}$$\n",
    "\n",
    "Training finds the coefficients β that maximise the likelihood of the observed outcomes (maximum likelihood estimation, solved via gradient descent).\n",
    "\n",
    "**`random_state=0`** — Logistic Regression with the default `lbfgs` solver is deterministic. The `random_state` parameter does not affect the result here, but is included for reproducibility if you later switch to a stochastic solver."
]

cells[21]["source"] = [
    "## Step 8: Make Predictions\n",
    "\n",
    "Two prediction types:\n",
    "\n",
    "**Single observation** — `[[30, 87000]]` (30-year-old, £87k salary). The double brackets create the 2D array shape `(1, 2)` that sklearn expects.\n",
    "\n",
    "**Note:** The input must be scaled with the same scaler used during training before passing to `predict()`. "
    "The code calls `sc.transform([[30, 87000]])` — this applies the training-set mean and std to the new point.\n",
    "\n",
    "**All test predictions** — `y_pred` contains 100 predictions. The side-by-side comparison with `y_test` lets you inspect individual misclassifications before looking at summary metrics."
]

cells[24]["source"] = [
    "## Step 9: Evaluate the Model\n",
    "\n",
    "Four metrics together give a complete picture:\n",
    "\n",
    "| Metric | Formula | What it answers |\n",
    "|--------|---------|----------------|\n",
    "| **Accuracy** | (TP + TN) / total | What fraction of all predictions were correct? |\n",
    "| **Precision** | TP / (TP + FP) | Of all predicted buyers, how many actually bought? |\n",
    "| **Recall** | TP / (TP + FN) | Of all actual buyers, how many did we catch? |\n",
    "| **F1** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |\n",
    "\n",
    "**Which metric to prioritise?** It is a business decision:\n",
    "- Maximise **recall** when missing a buyer is costly (you would rather send an extra ad than miss a sale)\n",
    "- Maximise **precision** when false positives are costly (e.g., an expensive personal sales call to a non-buyer)\n",
    "- Use **F1** when you need a single balanced number"
]

cells[26]["source"] = [
    "## Step 10: Visualise the Training Set Decision Boundary\n",
    "\n",
    "The plot shows two regions:\n",
    "- **Red region** — where the model predicts class 0 (did not purchase)\n",
    "- **Green region** — where the model predicts class 1 (purchased)\n",
    "\n",
    "The straight diagonal line separating the two regions is the **decision boundary** — the set of all points where P(y=1) = 0.5. "
    "Logistic Regression always produces a linear boundary in feature space.\n",
    "\n",
    "**What to check:** Training accuracy looks good by design — the model was fit to these points. "
    "A few red dots in the green region (and vice versa) indicate either overlapping classes or the linear boundary not being expressive enough for this data."
]

cells[29]["source"] = [
    "## Step 11: Visualise the Test Set Decision Boundary\n",
    "\n",
    "The same decision boundary from training, now evaluated against the 100 held-out test customers.\n",
    "\n",
    "**What to check:**\n",
    "- If the test plot looks as clean as the training plot → good generalisation, the linear boundary captures the true pattern\n",
    "- If the test plot has significantly more misclassifications → the model may be underfitting (the true boundary is nonlinear)\n",
    "\n",
    "For Social Network Ads data, Logistic Regression typically achieves ~85% test accuracy. "
    "The cluster of younger high-salary users who do not purchase (bottom-right of the bought region) is where a linear boundary consistently struggles — a kernel SVM or polynomial features would separate these better."
]

cells[32]["source"] = [
    "## Step 12: Summary and Interpretation\n",
    "\n",
    "The metrics table gives a quick diagnostic read:\n",
    "\n",
    "- **Accuracy ~85%** — strong for a linear model on this data\n",
    "- **Precision vs Recall trade-off** — check whether FP or FN dominate the error; this guides whether to adjust the classification threshold\n",
    "\n",
    "**When to choose Logistic Regression:**\n",
    "- Baseline for any binary classification problem — fast, interpretable, well-calibrated probabilities\n",
    "- When you need to explain model decisions to non-technical stakeholders (the coefficients directly show feature importance)\n",
    "- When the decision boundary is approximately linear\n",
    "\n",
    "**When to move to another model:**\n",
    "- Nonlinear boundary needed → try SVM with RBF kernel, Decision Tree, or Random Forest\n",
    "- Many features with complex interactions → gradient boosting (XGBoost, LightGBM)"
]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Updated: 01_logistic_regression.ipynb (remaining cells)")
