# Learning Roadmap

A concrete, week-by-week guide through the repository.
Estimated time assumes ~1 hour per day.

---

## Phase 0 — Before You Start (Week 1)

**Goal:** Make sure you have the right foundation so nothing in the curriculum blindsides you.

### Check your prerequisites
- [ ] Python: can you write a function, use list comprehensions, import a library?
- [ ] NumPy: can you create arrays, slice them, do element-wise operations?
- [ ] Pandas: can you load a CSV, filter rows, handle missing values?
- [ ] Matplotlib: can you make a scatter plot and label its axes?

If any of the above is shaky, spend this week there first. Then come back.

### Set up your environment
```bash
git clone https://github.com/Morobang/machine-learning-journey.git
cd machine-learning-journey
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Full instructions: [docs/setup.md](setup.md)

---

## Phase 1 — Foundations (Weeks 2–3)

**Section:** [00_foundations/](../00_foundations/)

| Day | Activity |
|-----|----------|
| 1 | Read `theory/01_what_is_ml.md`, `02_types_of_ml.md` |
| 2 | Read `theory/04_ml_lifecycle.md`, `06_prerequisites.md` |
| 3 | Read `theory/08_math_fundamentals.md` — focus on linear algebra and calculus intuition |
| 4 | Run `notebooks/01_data_preprocessing.ipynb` — understand every cell |
| 5 | Read `theory/09_data_types_and_features.md`, `10_evaluation_metrics.md` |
| 6–7 | Read `theory/11_common_pitfalls.md`, `12_ml_ethics_and_bias.md` |

**Checkpoint:** Can you explain what supervised learning is, describe the ML lifecycle, and preprocess a dataset with missing values and categorical columns?

---

## Phase 2 — Supervised Learning: Regression (Weeks 4–5)

**Section:** [01_supervised_learning/regression/](../01_supervised_learning/regression/)

Work through notebooks in order. For each one: read the paired theory doc first, then run the notebook, then do at least the Beginner exercise.

| Notebook | Theory Doc | Time |
|----------|------------|------|
| `01_simple_linear_regression.ipynb` | `theory/01_simple_linear_regression.md` | 1–2 hrs |
| `02_multiple_linear_regression.ipynb` | `theory/02_multiple_linear_regression.md` | 1–2 hrs |
| `03_polynomial_regression.ipynb` | `theory/03_polynomial_regression.md` | 1 hr |
| `04_support_vector_regression.ipynb` | `theory/04_support_vector_regression.md` | 1–2 hrs |
| `05_decision_tree_regression.ipynb` | `theory/05_decision_tree_regression.md` | 1 hr |
| `06_random_forest_regression.ipynb` | `theory/06_random_forest_regression.md` | 1–2 hrs |

**Checkpoint:** Can you explain why R² alone isn't enough to evaluate a regression model? Can you detect overfitting from a learning curve?

---

## Phase 3 — Supervised Learning: Classification (Weeks 6–7)

**Section:** [01_supervised_learning/classification/](../01_supervised_learning/classification/)

| Notebook | Key concept to master |
|----------|----------------------|
| `01_logistic_regression.ipynb` | Sigmoid function, log-loss, probability outputs |
| `02_k_nearest_neighbors.ipynb` | Distance metrics, the curse of dimensionality |
| `03_support_vector_machine.ipynb` | Maximum margin, support vectors |
| `04_kernel_svm.ipynb` | Kernel trick, RBF kernel |
| `05_naive_bayes.ipynb` | Bayes theorem, conditional independence assumption |
| `06_decision_tree_classification.ipynb` | Information gain, Gini impurity, pruning |
| `07_random_forest_classification.ipynb` | Bagging, feature importance |

**Checkpoint:** Given a new classification problem, can you choose a reasonable first algorithm and justify the choice? Can you read a confusion matrix and explain the difference between precision and recall?

---

## Phase 4 — Unsupervised Learning (Week 8)

**Section:** [02_unsupervised_learning/](../02_unsupervised_learning/)

- K-Means: elbow method, inertia, silhouette score
- Hierarchical: dendrograms, linkage methods
- Apriori: support, confidence, lift — market basket interpretation
- ECLAT: how it differs from Apriori

**Checkpoint:** Can you cluster a dataset, choose k, and interpret what each cluster represents in business terms?

---

## Phase 5 — Deep Learning (Weeks 9–10)

**Section:** [04_deep_learning/](../04_deep_learning/)

Start here only if you are comfortable with gradient descent and the chain rule. If not, re-read `theory/08_math_fundamentals.md`.

| Notebook | What you're learning |
|----------|---------------------|
| `01_artificial_neural_network.ipynb` | Forward pass, backprop, activation functions, dropout |
| `02_convolutional_neural_network.ipynb` | Filters, pooling, feature maps, image classification |

**Time warning:** Deep learning notebooks take longer to run. Use GPU runtime in Google Colab if your machine is slow.

**Checkpoint:** Can you explain why a deeper network isn't always better? Can you diagnose overfitting in a neural network and name two remedies?

---

## Phase 6 — NLP (Week 11)

**Section:** [05_natural_language_processing/](../05_natural_language_processing/)

- Text preprocessing: tokenisation, stopwords, stemming vs lemmatisation
- Bag of Words and TF-IDF: what they encode, what they lose
- Sentiment analysis: classification applied to text
- Word embeddings: why Word2Vec outperforms BoW

**Checkpoint:** Can you build a sentiment classifier from raw text data? Can you explain what TF-IDF weights represent?

---

## Phase 7 — Ensemble Methods (Week 12)

**Section:** [06_ensemble_methods/](../06_ensemble_methods/)

This is the most interview-relevant section. XGBoost appears in the majority of winning Kaggle solutions and is asked about in nearly every DS interview.

| Topic | Key questions to be able to answer |
|-------|-----------------------------------|
| Bagging | How does averaging predictions reduce variance? |
| Gradient Boosting | What does "fitting the residuals" mean? |
| XGBoost | What regularisation terms does XGBoost add vs vanilla GBM? |
| LightGBM | Why is leaf-wise growth faster? When does it overfit? |
| Stacking | What is a meta-learner? Why use cross-val predictions as features? |

---

## Phase 8 — Dimensionality Reduction (Week 13)

**Section:** [08_dimensionality_reduction/](../08_dimensionality_reduction/)

- PCA: eigenvalues, explained variance ratio, scree plot
- LDA: class-separating vs variance-maximising
- Kernel PCA: when linear PCA fails
- t-SNE / UMAP: visualisation only, not for features

**Checkpoint:** Can you explain why you should never use t-SNE output as features for a downstream model?

---

## Phase 9 — Model Selection and Evaluation (Week 14)

**Section:** [09_model_selection_and_evaluation/](../09_model_selection_and_evaluation/)

The capstone of the core curriculum. Everything you've learned feeds here.

- K-fold cross-validation: why a single train/test split misleads you
- Grid search vs random search: when to use each
- Bias-variance tradeoff: diagnosing under- vs over-fitting
- SHAP: making black-box models explainable

**Checkpoint:** Can you design a proper evaluation pipeline that avoids data leakage? Can you produce a SHAP summary plot and interpret it?

---

## Phase 10 — Projects (Weeks 15–18)

**Section:** [projects/](../projects/)

Now build something end-to-end. Work through at least one project at each difficulty level.

| Project | What it practises |
|---------|------------------|
| [Titanic](../projects/beginner/01_titanic_eda/) | EDA, feature engineering, classification |
| [COVID-19 Analysis](../projects/beginner/02_covid19_analysis/) | Data wrangling, time series plots |
| Customer Churn | Class imbalance, SHAP, model serialisation |
| House Price Prediction | Full pipeline, feature engineering, ensembling |

A complete project has: a clear problem statement, EDA, preprocessing pipeline, model comparison, evaluation with business interpretation, and a README.

---

## The Interview Preparation Track

If your goal is job interviews rather than depth, this compressed path covers the highest-frequency topics:

```
Week 1:  00_foundations (preprocessing, evaluation metrics)
Week 2:  Linear + Logistic Regression (the questions everyone asks)
Week 3:  Decision Trees + Random Forest (understand Gini, feature importance)
Week 4:  XGBoost (gradient boosting deep-dive — non-negotiable)
Week 5:  Cross-validation + bias-variance tradeoff
Week 6:  One complete project with SHAP interpretability
```

Then read `theory/11_common_pitfalls.md` and the interview prep sections (collapsible `<details>` blocks) at the bottom of each notebook.

---

## Tracking Your Progress

Make a copy of this checklist and tick items off as you go:

```
Phase 1 — Foundations
[ ] 01_what_is_ml.md
[ ] 08_math_fundamentals.md
[ ] data_preprocessing.ipynb

Phase 2 — Regression
[ ] simple_linear_regression
[ ] multiple_linear_regression
[ ] polynomial_regression
[ ] svr
[ ] decision_tree_regression
[ ] random_forest_regression

Phase 3 — Classification
[ ] logistic_regression
[ ] knn
[ ] svm
[ ] kernel_svm
[ ] naive_bayes
[ ] decision_tree
[ ] random_forest

Phase 4 — Unsupervised
[ ] kmeans
[ ] hierarchical
[ ] apriori
[ ] eclat

Phase 5 — Deep Learning
[ ] ann
[ ] cnn

Phase 6 — NLP
[ ] text_preprocessing
[ ] bag_of_words_tfidf
[ ] sentiment_analysis

Phase 7 — Ensemble Methods
[ ] bagging
[ ] gradient_boosting
[ ] xgboost_lightgbm
[ ] stacking

Phase 8 — Dimensionality Reduction
[ ] pca
[ ] lda
[ ] kernel_pca

Phase 9 — Model Selection
[ ] cross_validation
[ ] hyperparameter_tuning
[ ] bias_variance
[ ] shap_interpretability

Phase 10 — Projects
[ ] titanic (complete)
[ ] customer_churn (complete)
```