# 📈 Regression Algorithms

Regression is a supervised learning technique used to predict continuous numerical values. It's one of the most fundamental areas in machine learning and serves as the foundation for understanding more complex algorithms.

## 🎯 What is Regression?

Regression algorithms learn the relationship between input features (X) and a continuous target variable (Y). The goal is to predict numerical outcomes such as prices, temperatures, sales figures, or any other continuous values.

## 📊 Algorithms in This Section

### 1. **Simple Linear Regression** 
- **File**: `01_simple_linear_regression.ipynb`
- **Concept**: Finding the best straight line through data points
- **Use Case**: Understanding basic relationships between two variables
- **Example**: Predicting salary based on years of experience

### 2. **Multiple Linear Regression**
- **File**: `02_multiple_linear_regression.ipynb`
- **Concept**: Linear relationships with multiple input features
- **Use Case**: More realistic scenarios with multiple factors
- **Example**: Predicting startup profit based on R&D spend, administration, marketing, and state

### 3. **Polynomial Regression**
- **File**: `03_polynomial_regression.ipynb`
- **Concept**: Capturing non-linear relationships using polynomial features
- **Use Case**: When relationships are curved rather than straight
- **Example**: Predicting salary based on position level (non-linear relationship)

### 4. **Support Vector Regression (SVR)**
- **File**: `04_support_vector_regression.ipynb`
- **Concept**: Using support vector machines for regression tasks
- **Use Case**: Robust regression with outliers, high-dimensional data
- **Example**: Position-salary prediction with robust handling of outliers

### 5. **Decision Tree Regression**
- **File**: `05_decision_tree_regression.ipynb`
- **Concept**: Tree-based decisions for prediction
- **Use Case**: Interpretable models, handling non-linear patterns
- **Example**: Salary prediction with clear decision rules

### 6. **Random Forest Regression**
- **File**: `06_random_forest_regression.ipynb`
- **Concept**: Ensemble of decision trees
- **Use Case**: High accuracy, handling complex patterns, robust predictions
- **Example**: Improved salary prediction using multiple decision trees

## 📁 Datasets

- **`salary_data.csv`**: Simple relationship between experience and salary
- **`startups_data.csv`**: Startup financial data with multiple features
- **`position_salaries.csv`**: Position levels and corresponding salaries

## 🔄 Learning Progression

1. **Start with Simple Linear Regression**: Understand the basics of fitting a line
2. **Move to Multiple Linear Regression**: Learn to handle multiple features
3. **Explore Polynomial Regression**: Understand non-linear relationships
4. **Try SVR**: Learn robust regression techniques
5. **Practice Decision Trees**: Understand tree-based approaches
6. **Master Random Forest**: Learn ensemble methods

## 📏 Key Evaluation Metrics

- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values
- **Mean Squared Error (MSE)**: Average squared difference (penalizes larger errors more)
- **Root Mean Squared Error (RMSE)**: Square root of MSE (same units as target)
- **R² Score (Coefficient of Determination)**: Proportion of variance explained (0-1, higher is better)

## 💡 Key Concepts to Master

- **Linear vs Non-linear Relationships**: When to use which approach
- **Overfitting and Underfitting**: Model complexity trade-offs
- **Feature Scaling**: Importance of normalizing inputs
- **Residual Analysis**: Understanding prediction errors
- **Cross-Validation**: Proper model evaluation

## 🚀 Getting Started

1. Open `01_simple_linear_regression.ipynb` to start your regression journey
2. Work through each notebook in order
3. Experiment with different parameters
4. Compare results across different algorithms
5. Practice interpreting model performance metrics

---

**Ready to predict the future?** Start with simple linear regression and build your way up! 📈