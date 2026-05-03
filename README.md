# Machine Learning Journey

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Notebooks](https://img.shields.io/badge/Notebooks-65%2B-orange?logo=jupyter)](https://jupyter.org/)
[![Sections](https://img.shields.io/badge/Sections-10-blueviolet)](#learning-roadmap)
[![Last Commit](https://img.shields.io/github/last-commit/Morobang/machine-learning-journey)](https://github.com/Morobang/machine-learning-journey/commits/main)
[![CI](https://github.com/Morobang/machine-learning-journey/actions/workflows/ci.yml/badge.svg)](https://github.com/Morobang/machine-learning-journey/actions)

---

> A structured, beginner-to-advanced machine learning curriculum.
> Every topic includes **theory**, a **runnable notebook**, and **exercises** —
> so you understand the algorithm, not just the API call.

---

## Who is this for?

| You are... | Start here |
|---|---|
| New to ML, solid Python basics | [00 — Foundations](00_foundations/) |
| Know some ML, want depth | [01 — Supervised Learning](01_supervised_learning/) |
| Preparing for ML interviews | [09 — Model Selection & Evaluation](09_model_selection_and_evaluation/) + interview prep sections in each notebook |
| Building a portfolio project | [projects/](projects/) |
| Looking for reusable ML utilities | [src/](src/) |

---

## Learning Roadmap

Work through sections in order, or jump to any topic you need.

```
┌─────────────────────────────────────────────────────────────┐
│                   START HERE (everyone)                     │
│              00 · Foundations & Preprocessing               │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────┐             ┌──────────────────┐
│  01 · Supervised│             │ 02 · Unsupervised │
│     Learning    │             │     Learning      │
│ Regression      │             │ Clustering        │
│ Classification  │             │ Association Rules │
└────────┬────────┘             └────────┬──────────┘
         │                               │
         └───────────────┬───────────────┘
                         ▼
         ┌───────────────────────────────┐
         │   03 · Reinforcement Learning  │
         └───────────────┬───────────────┘
                         ▼
         ┌───────────────────────────────┐
         │      04 · Deep Learning        │
         │   ANN · CNN · RNN · Transfer   │
         └───────────────┬───────────────┘
                         ▼
         ┌───────────────────────────────┐
         │  05 · Natural Language Proc.   │
         │  Text · Sentiment · Transformers│
         └───────────────┬───────────────┘
                         ▼
    ┌────────────────────┼────────────────────┐
    ▼                    ▼                    ▼
┌────────────┐  ┌──────────────┐  ┌──────────────────┐
│06 · Ensemble│  │07 · Time     │  │08 · Dimensionality│
│   Methods  │  │   Series     │  │    Reduction      │
└────────────┘  └──────────────┘  └──────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  09 · Model Selection &        │
         │      Evaluation               │
         └───────────────┬───────────────┘
                         ▼
         ┌───────────────────────────────┐
         │  10 · MLOps & Deployment  🚀  │
         └───────────────────────────────┘
```

---

## What's Inside

| # | Section | Topics | Notebooks |
|---|---|---|---|
| [00](00_foundations/) | **Foundations** | Data types, preprocessing, EDA, feature engineering, math basics | 4 |
| [01](01_supervised_learning/) | **Supervised Learning** | Linear/Polynomial/SVR/Tree/Forest Regression · Logistic/KNN/SVM/Naive Bayes/Tree/Forest Classification | 14 |
| [02](02_unsupervised_learning/) | **Unsupervised Learning** | K-Means, Hierarchical, DBSCAN · Apriori, ECLAT | 5 |
| [03](03_reinforcement_learning/) | **Reinforcement Learning** | UCB, Thompson Sampling, Q-Learning | 3 |
| [04](04_deep_learning/) | **Deep Learning** | ANN, CNN, RNN/LSTM, Transfer Learning | 4 |
| [05](05_natural_language_processing/) | **NLP** | Text preprocessing, BoW/TF-IDF, Sentiment, Word2Vec, Transformers | 5 |
| [06](06_ensemble_methods/) | **Ensemble Methods** | Bagging, XGBoost, LightGBM, Gradient Boosting, Stacking | 4 |
| [07](07_time_series_analysis/) | **Time Series** | ARIMA, Exponential Smoothing, LSTM Forecasting | 3 |
| [08](08_dimensionality_reduction/) | **Dimensionality Reduction** | PCA, LDA, Kernel PCA, t-SNE/UMAP | 4 |
| [09](09_model_selection_and_evaluation/) | **Model Selection & Evaluation** | Cross-validation, GridSearch, Bias-Variance, SHAP | 4 |
| [10](10_mlops_and_deployment/) | **MLOps & Deployment** | Model serialization, FastAPI, MLflow | 3 |

---

## Projects

End-to-end projects that put the theory to work.

| Project | Difficulty | Key Skills | Status |
|---|---|---|---|
| [Titanic Survival Prediction](projects/beginner/01_titanic_eda/) | Beginner | EDA, feature engineering, classification | Complete |
| [COVID-19 Analysis](projects/beginner/02_covid19_analysis/) | Beginner | Data wrangling, time series visualization | Complete |
| [Customer Churn Prediction](projects/intermediate/01_customer_churn/) | Intermediate | Imbalanced data, SHAP, model serialization | In Progress |
| [House Price Prediction](projects/intermediate/02_house_price_prediction/) | Intermediate | Full pipeline, feature engineering, ensembling | Planned |
| [Sentiment Analysis API](projects/advanced/01_sentiment_api/) | Advanced | NLP, FastAPI, Docker | Planned |

---

## Quick Start

```bash
# Clone
git clone https://github.com/Morobang/machine-learning-journey.git
cd machine-learning-journey

# Set up environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

See [docs/setup.md](docs/setup.md) for full setup instructions, conda alternative, and troubleshooting.

---

## Repository Structure

```
machine-learning-journey/
├── 00_foundations/          # ML basics, data preprocessing, EDA
├── 01_supervised_learning/  # Regression + Classification algorithms
├── 02_unsupervised_learning/# Clustering + Association Rules
├── 03_reinforcement_learning/
├── 04_deep_learning/
├── 05_natural_language_processing/
├── 06_ensemble_methods/
├── 07_time_series_analysis/
├── 08_dimensionality_reduction/
├── 09_model_selection_and_evaluation/
├── 10_mlops_and_deployment/
├── projects/                # End-to-end ML projects (beginner → advanced)
│   ├── beginner/
│   ├── intermediate/
│   └── advanced/
├── src/                     # Reusable Python utilities (preprocessing, evaluation, plots)
├── data/                    # Centralized datasets (see data/README.md for sources)
│   ├── small/               # <1 MB — committed to git
│   ├── medium/              # 1–50 MB — committed to git
│   └── external/            # >50 MB — run download_datasets.py
├── docs/                    # Setup guide, notebook template, contributing info
├── tests/                   # Unit tests for src/ utilities
└── .github/                 # CI workflows, issue templates, PR template
```

---

## How Each Section is Organized

```
NN_section_name/
├── README.md           ← Overview, contents table, prerequisites
├── theory/             ← Markdown files: math, intuition, use cases
│   └── NN_topic.md
└── notebooks/          ← Jupyter notebooks: implementation + exercises
    └── NN_topic.ipynb
```

Theory docs and notebooks are paired. The theory explains the *why*; the notebook implements the *how*.

---

## Contributing

Contributions are welcome — from fixing a typo to adding a full new section.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

Quick checklist for notebooks:
- Follows the [standard notebook template](docs/notebook_template.md)
- Runs clean with `Kernel > Restart & Run All`
- No datasets larger than 5 MB committed

---

## License

[MIT License](LICENSE) — free to use, modify, and share with attribution.

---

*Built to understand machine learning, not just use it.*
