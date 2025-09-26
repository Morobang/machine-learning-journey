# Customer Purchase Prediction Dataset

## Overview
This dataset contains customer demographic and financial information to predict whether a customer will make a purchase (binary classification problem). The dataset is particularly well-suited for demonstrating fundamental classification algorithms and feature importance analysis.

## Dataset Description

### Features:
1. **Age**: Numerical feature representing customer age (18-49 years)
   - Range: 18 to 49 years
   - Important for purchase behavior as different age groups have different spending patterns

2. **EstimatedSalary**: Numerical feature representing annual salary (in USD)
   - Range: $19,000 to $150,000
   - Strong indicator of purchasing power

### Target Variable:
- **Purchased**: Binary outcome variable
  - `0`: Customer did NOT make a purchase
  - `1`: Customer made a purchase

## Dataset Characteristics
- Samples: 26 (small but balanced for demonstration)
- Features: 2 (both numerical)
- Target: Binary (0/1)
- No missing values
- Contains clear decision boundaries (visible in EDA)

## Why This Dataset Works Well for Classification

### 1. Clear Feature-Target Relationships
- **Age Pattern**: Customers aged 45+ predominantly purchase (9/10 positive cases)
- **Salary Threshold**: Purchases occur above ~$150k salary or with older age at lower salaries

### 2. Balanced Complexity
- Simple enough for linear models to perform reasonably
- Complex enough to benefit from non-linear models (decision trees, SVM with RBF kernel)

### 3. Ideal for Demonstrating Key Concepts:
- Feature scaling importance (large salary vs age ranges)
- Decision boundaries visualization
- Feature importance analysis
- Class imbalance handling (slight 60:40 distribution)
