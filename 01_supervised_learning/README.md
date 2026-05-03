# Supervised Learning

Supervised learning is the most common form of machine learning. You provide the algorithm with labelled training examples — input features paired with known output values — and it learns a mapping from inputs to outputs that generalises to new, unseen data.

The word "supervised" comes from the idea that the correct answers act as a teacher: the algorithm can compare its predictions against the known labels and adjust until its predictions improve.

## What Supervised Learning Solves

Every supervised learning problem has three components:

- **Features (X)** — the input variables used to make a prediction (age, salary, pixel values, word counts)
- **Target (y)** — the output the model should predict (house price, spam/not spam, disease diagnosis)
- **Training data** — labelled examples from which the model learns the relationship between X and y

The learned relationship is a function f(X) ≈ y. The goal is to find the f that generalises well to data the model has never seen, not just the training data it was fit on.

---

## The Two Types of Supervised Learning

### Regression

**Regression** predicts a continuous numerical value. There is no finite set of possible outputs — the answer can be any number on a scale.

**Examples:**
- Predict a house's sale price given its features (size, location, condition)
- Predict tomorrow's temperature given weather history
- Predict a patient's blood pressure given their lifestyle and medical history
- Predict how many units a product will sell next month

**How you know a problem is regression:** The target variable is a real number, and intermediate values are meaningful. A house price of £285,000 is a sensible answer — you would not say "that is between category 2 and category 3."

The `01_regression/` folder covers six regression algorithms, from the simplest linear model to ensemble methods:

| Algorithm | What makes it distinctive |
|-----------|--------------------------|
| Simple Linear Regression | One input, one straight-line relationship |
| Multiple Linear Regression | Multiple inputs, linear combination of features |
| Polynomial Regression | Curved relationships via polynomial feature expansion |
| Support Vector Regression | Fits within a tolerance tube; robust to outliers |
| Decision Tree Regression | Step-function prediction via recursive splits |
| Random Forest Regression | Ensemble average of many trees; reduces variance |

See [01_regression/README.md](01_regression/README.md) for a full explanation of each algorithm and when to use it.

### Classification

**Classification** predicts which category or class an input belongs to. The output is one of a finite set of possible labels.

**Examples:**
- Predict whether an email is spam or not spam (binary)
- Predict which handwritten digit (0–9) is in an image (multi-class)
- Predict whether a transaction is fraudulent (binary)
- Predict a customer's tier (Bronze / Silver / Gold) based on behaviour (multi-class)

**How you know a problem is classification:** The target variable is a discrete label, and ordering between labels may not be meaningful. Saying a new email is "between spam and not spam" makes no sense — it must be one or the other.

The `02_classification/` folder covers seven classification algorithms:

| Algorithm | What makes it distinctive |
|-----------|--------------------------|
| Logistic Regression | Linear decision boundary; outputs calibrated probabilities |
| K-Nearest Neighbors | Classifies by majority vote among nearest training examples |
| Support Vector Machine (SVM) | Finds the maximum-margin decision boundary |
| Kernel SVM | SVM with nonlinear kernel for curved boundaries |
| Naive Bayes | Probabilistic classifier based on Bayes' theorem |
| Decision Tree Classification | Interpretable tree of if-then rules |
| Random Forest Classification | Ensemble of trees; reduces overfitting vs single tree |

See [02_classification/README.md](02_classification/README.md) for a full explanation of each algorithm.

---

## Regression vs Classification — How to Decide

| Question | If yes |
|----------|--------|
| Is the output a real number (price, temperature, score)? | Regression |
| Is the output one of a fixed set of categories? | Classification |
| Does it make sense to average two outputs? | Regression |
| Would you describe the output as a "label" or "type"? | Classification |

Note: some problems sit on the boundary. Predicting credit risk as a probability is technically classification (approved/declined) but the probability output itself is continuous. Age can be predicted as a number (regression) or as an age group (classification) depending on what the downstream decision requires.

---

## The Supervised Learning Workflow

Every supervised learning project follows the same core steps regardless of which algorithm you choose:

1. **Load and inspect the data** — understand the features, check for missing values, examine the target distribution
2. **Exploratory Data Analysis (EDA)** — visualise relationships, check class balance, identify outliers
3. **Preprocessing** — handle missing values, encode categorical features, split into train/test sets, scale features
4. **Train the model** — fit the chosen algorithm on training data
5. **Evaluate** — measure performance on the held-out test set using the right metrics (R² and RMSE for regression; accuracy, precision, recall, F1 for classification)
6. **Iterate** — adjust preprocessing, try other algorithms, tune hyperparameters

This workflow is demonstrated step by step in each notebook throughout this section.

---

## Evaluation Metrics

### For Regression
| Metric | Formula | What it measures |
|--------|---------|-----------------|
| MAE | mean(|y - ŷ|) | Average absolute error; same units as target |
| RMSE | √mean((y - ŷ)²) | Penalises large errors more; same units as target |
| R² | 1 − SS_res/SS_tot | Fraction of variance explained; 1.0 is perfect, 0 is baseline |

### For Classification
| Metric | Formula | What it measures |
|--------|---------|-----------------|
| Accuracy | (TP + TN) / total | Overall fraction correct |
| Precision | TP / (TP + FP) | Of predicted positives, how many were actually positive |
| Recall | TP / (TP + FN) | Of actual positives, how many were correctly predicted |
| F1 | 2 × P × R / (P + R) | Harmonic mean of precision and recall |

---

## Key Concepts to Understand Before Starting

**Overfitting vs underfitting** — A model that memorises training data performs perfectly on training but poorly on new data (overfitting). A model that is too simple fails even on training data (underfitting). Both are failure modes.

**The train/test split** — Always hold out a portion of data that the model never sees during training. This gives an honest estimate of real-world performance.

**Feature scaling** — Distance-based algorithms (KNN, SVM) and gradient descent (logistic regression, neural networks) require features to be on comparable scales. Tree-based methods (decision trees, random forests) do not.

**Data leakage** — Any preprocessing step that uses information from the test set (fitting a scaler on all data, computing statistics across all rows before splitting) produces an optimistic estimate of performance that will not hold in production. Split first, then fit all transformers on training data only.
