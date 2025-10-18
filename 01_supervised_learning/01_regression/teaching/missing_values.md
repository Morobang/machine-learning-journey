# Handling Missing Values

When to handle missing values:
- Before model training and split, usually during preprocessing step.
- For experiments, you may keep a copy of raw data for auditing.

Common strategies:
- Drop rows/columns with many missing values
- Impute with mean/median (numerical) or mode (categorical)
- Use model-based imputation (KNN, IterativeImputer)

Notes:
- For time-series follow forward/backward fill methods.
- Always fit imputer on training set and apply to test set to avoid leakage.

See notebook step: "Step X — Missing values" for where this fits in the compact template.