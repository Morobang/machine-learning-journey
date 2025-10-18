# Feature Scaling

When to scale:
- Linear models, k-NN, SVM, and gradient-based methods need scaling.
- Tree-based models (Decision Trees, Random Forests) usually don't require scaling.

Common scalers:
- `StandardScaler` (mean 0, std 1)
- `MinMaxScaler` (0-1 range)
- `RobustScaler` (robust to outliers)

Practical advice:
- Fit scaler on training data only
- Use Pipelines to prevent leakage
- Document why you chose a scaler in the teaching notes
