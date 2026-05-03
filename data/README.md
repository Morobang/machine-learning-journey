# Dataset Registry

This file is the single source of truth for every dataset in the repository.

**Rules:**
- Files in `small/` (<1 MB) are committed to git — safe for cloning
- Files in `medium/` (1–50 MB) are committed to git — use `git lfs` if the repo grows large
- Files in `external/` are **not committed** — run `python data/external/download_datasets.py` to fetch them
- Never commit a file larger than 50 MB. Use a download script instead.

---

## small/ — Teaching Datasets

| File | Size | Source | License | Used In |
|---|---|---|---|---|
| `salary_data.csv` | ~3 KB | [Superdatascience](https://www.superdatascience.com) | Public domain | `01_supervised_learning/regression/notebooks/01_simple_linear_regression.ipynb` |
| `position_salaries.csv` | ~0.4 KB | Superdatascience | Public domain | `01_supervised_learning/regression/notebooks/05_decision_tree_regression.ipynb` |
| `startups_data.csv` | ~2 KB | Superdatascience | Public domain | `01_supervised_learning/regression/notebooks/02_multiple_linear_regression.ipynb` |
| `social_network_ads.csv` | ~4.9 KB | Superdatascience | Public domain | All classification notebooks |
| `mall_customers.csv` | ~4.3 KB | [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) | CC0 Public Domain | `02_unsupervised_learning/clustering/` |
| `wine.csv` | ~11.5 KB | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine) | CC BY 4.0 | `08_dimensionality_reduction/` |
| `iris.csv` | ~4 KB | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/iris) | Public domain | `projects/beginner/07_iris_classification/` |

## medium/ — Larger Datasets

| File | Size | Source | License | Used In |
|---|---|---|---|---|
| `churn_modelling.csv` | ~650 KB | [Kaggle](https://www.kaggle.com/datasets/shubh0799/churn-modelling) | Public | `04_deep_learning/notebooks/01_artificial_neural_network.ipynb` |
| `market_basket_optimization.csv` | ~300 KB | Superdatascience | Public domain | `02_unsupervised_learning/association_rules/` |
| `restaurant_reviews.tsv` | ~61 KB | Superdatascience | Public domain | `05_natural_language_processing/` |
| `ads_ctr_optimization.csv` | ~200 KB | Superdatascience | Public domain | `03_reinforcement_learning/` |

## external/ — Large Datasets (not committed)

Run this to download:
```bash
python data/external/download_datasets.py
```

| File | Size | Source | Used In |
|---|---|---|---|
| `usa_county_wise_covid19.csv` | ~66 MB | [Kaggle COVID-19](https://www.kaggle.com/datasets/sudalairajkumar/covid19-in-usa) | Public | `projects/beginner/02_covid19_analysis/` |
| `titanic_train.csv` | ~61 KB | [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data) | Competition rules | `projects/beginner/01_titanic_eda/` |

---

## Adding a new dataset

1. Check the license — only use datasets with a clear permissive license (CC0, CC BY, Public Domain)
2. Place it in the correct subfolder based on size
3. Add a row to this table with: filename, size, source URL, license, and which notebook uses it
4. If >50 MB, add a download function to `data/external/download_datasets.py` instead
