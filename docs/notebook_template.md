# Notebook Standard Template

Every notebook in this repository follows this structure.
Copy the raw markdown below as a starting skeleton for any new notebook.

---

## Why a standard template?

- Readers can predict where to find information across 60+ notebooks
- Beginners know they'll always get theory before code
- Exercises give learners something to do after reading
- Consistent structure makes the repo feel like a course, not a pile of files

---

## The Template

````markdown
# [Algorithm / Topic Name]

> **Module:** [e.g., Supervised Learning → Regression]
> **Difficulty:** Beginner | Intermediate | Advanced
> **Estimated reading time:** 20–40 min
> **Related theory:** [theory/NN_topic.md](../theory/NN_topic.md)
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](COLAB_LINK)

---

## Learning Objectives

By the end of this notebook you will be able to:

- [ ] Explain what [algorithm] does and when to use it
- [ ] Write out the key formula and explain each term
- [ ] Implement [algorithm] using scikit-learn
- [ ] Evaluate model performance and interpret the metrics
- [ ] Describe at least two limitations of this approach

---

## 1. The Problem We're Solving

[2–4 sentences. Set up a concrete, real business problem.
Never write "we will implement algorithm X." Instead: describe who
has a problem, what the problem is, and why solving it matters.]

**Example:**
> A real-estate startup wants to predict house prices from property
> features so buyers can assess whether a listed price is fair.
> Getting this wrong in either direction costs money: an overestimate
> misleads buyers; an underestimate costs the seller. We'll build a
> model to predict sale price from 7 property attributes.

---

## 2. The Theory

### What is [algorithm]?

[Plain English explanation. 2–4 paragraphs. Use an analogy.]

### The Mathematics

[Show the core equation. Use LaTeX. Explain every symbol.]

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n$$

| Symbol | Meaning |
|---|---|
| $\hat{y}$ | Predicted output |
| $\beta_0$ | Intercept (value when all features are zero) |
| $\beta_i$ | Weight for feature $x_i$ |

### What is the algorithm optimizing?

[Explain the loss function the algorithm minimizes.]

$$\text{RSS} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

### Key assumptions

- [Assumption 1]
- [Assumption 2]
- [When these are violated, the model fails because...]

---

## 3. Setup

```python
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn... import ...

# Reproducibility
RANDOM_STATE = 42
plt.style.use("seaborn-v0_8-whitegrid")
```

---

## 4. The Dataset

### Loading the data

```python
df = pd.read_csv("../../../data/small/dataset_name.csv")
df.head()
```

### What are we looking at?

[Describe the dataset: where it comes from, what each column means,
what the target variable is, how many rows/columns.]

| Column | Type | Description |
|---|---|---|
| `feature_1` | float | [meaning] |
| `target` | float | [what we're predicting] |

### Exploratory analysis

```python
# Shape and types
print(df.shape)
df.info()
df.describe()
```

[Show 1–2 plots that reveal the most important patterns.
Every plot must have: title, axis labels, brief caption below.]

---

## 5. Data Preparation

[Explain each preprocessing step and WHY it's needed here.
Don't just show code — explain the decision.]

```python
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
```

> We use a 80/20 train/test split. With [N] rows, the test set has
> [0.2*N] samples — large enough for stable metric estimates.

---

## 6. Training the Model

```python
from sklearn... import ModelName

model = ModelName()
model.fit(X_train, y_train)
```

[Explain the key hyperparameters of the model. What does each control?
What are the default values and are they reasonable here?]

---

## 7. Predictions and Visualization

```python
y_pred = model.predict(X_test)
```

[Plot actual vs predicted. Show the regression line / decision boundary.
Explain what the plot is telling you.]

---

## 8. Evaluation

```python
from sklearn.metrics import ...
```

[Show metrics. Then explain each one in plain English with context
specific to this dataset — not generic metric definitions.]

**Interpreting the results:**
> An R² of 0.92 means our model explains 92% of the variance in
> [target variable]. In practical terms, predictions on unseen data
> are off by an average of [MAE value] [units].

---

## 9. Assumptions Check

[Was it safe to use this algorithm on this data?
Check the assumptions you listed in Section 2.]

---

## 10. Limitations

- [Specific limitation on this dataset]
- [General limitation of the algorithm]
- [What would make this fail in production?]

---

## 11. Summary

- [Key takeaway 1]
- [Key takeaway 2]
- [Key takeaway 3]
- [When to use this algorithm]
- [When NOT to use this algorithm]

---

## 12. Exercises

**Beginner**
1. Change the `test_size` from 0.2 to 0.3. How do the metrics change?
2. What happens if you remove the most correlated feature?

**Intermediate**
1. Try [related algorithm] on the same dataset. Which performs better?
2. Add polynomial features. Does it improve or overfit?

**Advanced**
1. Implement the algorithm's core equation manually (without sklearn). Compare predictions.
2. Use cross-validation instead of a single train/test split. How does this change your confidence in the results?

---

## 13. What's Next?

- [Link to next notebook in sequence]
- [Link to related theory doc]

<details>
<summary>Interview Questions for This Topic</summary>

**Conceptual**
1. [Question]
2. [Question]

**Technical**
1. [Question]

**Case Study**
1. [Question]

</details>
````

---

## Checklist before committing a notebook

- [ ] Title, learning objectives, and problem framing at the top
- [ ] At least one formula with all symbols explained
- [ ] Every plot has a title, axis labels, and a caption
- [ ] Evaluation section interprets metrics in context — not just prints numbers
- [ ] Summary bullet points at the end
- [ ] At least 3 exercises (beginner / intermediate / advanced)
- [ ] Runs clean with `Kernel > Restart & Run All`
- [ ] `nbstripout` has stripped outputs (pre-commit handles this automatically)

---

## Common mistakes to avoid

| Mistake | Fix |
|---|---|
| Wall of code with no markdown | Add a markdown cell before every code cell |
| `print("Accuracy:", acc)` and nothing else | Explain what the number means |
| Generic dataset description ("this dataset has features and a target") | Name the columns, describe the domain, explain the problem |
| Skipping the math | Show at least the loss function |
| Exercises that are trivial (only "change a number") | Add one exercise that requires genuine thinking |
