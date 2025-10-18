# Feature Engineering for Regression

Topics:
- Polynomial features and interactions
- Log-transformations for skewed targets
- Binning continuous features
- Feature selection: correlation filter, recursive feature elimination

When to apply:
- After missing-value handling and basic encoding
- Before scaling (if scaling is needed)

Practical workflows:
- Use `PolynomialFeatures` in a Pipeline when testing polynomial regression
- Keep track of feature names when transforming (ColumnTransformer helps)
