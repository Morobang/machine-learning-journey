# 🧹 Data Cleaning & Preprocessing Masterclass

## 🎯 Project Overview
Master the art of data cleaning and preprocessing by working with intentionally messy datasets. This project teaches essential skills for handling real-world data quality issues that every data scientist encounters.

## 📊 What You'll Learn
- **Missing Value Handling**: Multiple strategies for dealing with null values
- **Data Type Conversion**: Proper data type assignment and conversion
- **Outlier Detection**: Statistical methods to identify and handle outliers
- **Data Validation**: Techniques to ensure data quality and consistency
- **Text Cleaning**: String manipulation and standardization
- **Date/Time Processing**: Parsing and standardizing temporal data
- **Duplicate Handling**: Identifying and managing duplicate records

## 🎯 Project Objectives
1. **Data Quality Assessment**: Learn to systematically evaluate data quality
2. **Cleaning Techniques**: Master various preprocessing methods
3. **Automation**: Create reusable cleaning pipelines
4. **Documentation**: Document all cleaning decisions and their rationale
5. **Validation**: Implement quality checks for cleaned data

## 🗃️ Datasets Used
We'll work with multiple intentionally messy datasets to practice different scenarios:

### 📋 Dataset 1: Customer Database (messy_customers.csv)
- **Issues**: Missing values, inconsistent formatting, duplicate records
- **Size**: ~10,000 records
- **Columns**: customer_id, name, email, phone, address, registration_date

### 📊 Dataset 2: Sales Transactions (messy_sales.csv)
- **Issues**: Wrong data types, outliers, invalid dates
- **Size**: ~50,000 transactions
- **Columns**: transaction_id, customer_id, product, amount, date, category

### 📈 Dataset 3: Employee Records (messy_employees.csv)
- **Issues**: Encoding problems, mixed formats, missing critical data
- **Size**: ~5,000 employees
- **Columns**: emp_id, name, department, salary, hire_date, performance_score

## 📁 Project Structure
```
04_data_cleaning/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── data/
│   ├── raw/                     # Original messy datasets
│   ├── processed/               # Cleaned datasets
│   └── sample/                  # Sample datasets for practice
├── notebooks/
│   ├── 01_data_assessment.ipynb     # Initial data quality assessment
│   ├── 02_missing_values.ipynb      # Missing value handling strategies
│   ├── 03_data_types_outliers.ipynb # Data types and outlier detection
│   ├── 04_text_cleaning.ipynb       # String cleaning and standardization
│   └── 05_final_pipeline.ipynb      # Complete cleaning pipeline
├── src/
│   ├── data_assessor.py         # Data quality assessment tools
│   ├── cleaners.py              # Cleaning function library
│   ├── validators.py            # Data validation functions
│   └── pipeline.py              # Automated cleaning pipeline
├── reports/
│   ├── data_quality_report.md   # Before/after quality assessment
│   └── cleaning_decisions.md    # Documentation of cleaning choices
└── tests/
    └── test_cleaning_functions.py # Unit tests for cleaning functions
```

## 🛠️ Technologies Used
- **Python 3.8+**
- **Pandas** - Data manipulation and cleaning
- **NumPy** - Numerical operations
- **Matplotlib/Seaborn** - Data visualization
- **Scikit-learn** - Outlier detection and preprocessing
- **Regular Expressions** - Text pattern matching and cleaning
- **Jupyter Notebook** - Interactive development

## 🔍 Common Data Quality Issues

### 🚫 Missing Values
- **Completely missing**: NaN, None, empty strings
- **Placeholder values**: "N/A", "Unknown", -999, 0
- **Inconsistent nulls**: Mix of different null representations

### 📝 Data Type Issues
- **Wrong types**: Numbers stored as strings
- **Mixed types**: Inconsistent data types in same column
- **Date parsing**: Various date formats in same column

### 🎯 Inconsistent Formatting
- **Case variations**: "John Smith" vs "john smith" vs "JOHN SMITH"
- **Whitespace**: Leading/trailing spaces, multiple spaces
- **Special characters**: Unexpected symbols, encoding issues

### 🔄 Duplicate Records
- **Exact duplicates**: Identical rows
- **Near duplicates**: Similar but not identical records
- **Partial duplicates**: Same entity with different information

## 📊 Cleaning Strategies & Techniques

### 1️⃣ Missing Value Strategies
```python
# Different approaches for different scenarios
- Drop rows/columns with too many missing values
- Forward/backward fill for time series
- Mean/median imputation for numerical data
- Mode imputation for categorical data
- Advanced imputation (KNN, iterative)
```

### 2️⃣ Outlier Handling
```python
# Statistical methods
- Z-score method (>3 standard deviations)
- IQR method (beyond Q1-1.5*IQR or Q3+1.5*IQR)
- Isolation Forest for multivariate outliers
- Domain knowledge validation
```

### 3️⃣ Text Standardization
```python
# Common text cleaning operations
- Remove leading/trailing whitespace
- Standardize case (upper/lower/title)
- Remove or replace special characters
- Standardize phone numbers, emails
- Handle encoding issues
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter regex
```

### Sample Data Generation
If you don't have messy data, we'll create some:
```python
# Generate messy sample data for practice
python src/generate_messy_data.py
```

### Running the Project
1. **Data Assessment**: `01_data_assessment.ipynb`
2. **Missing Values**: `02_missing_values.ipynb`
3. **Data Types & Outliers**: `03_data_types_outliers.ipynb`
4. **Text Cleaning**: `04_text_cleaning.ipynb`
5. **Complete Pipeline**: `05_final_pipeline.ipynb`

## 📈 Expected Outcomes

### Before Cleaning
- Data quality score: ~40-60%
- Missing values: 15-30%
- Inconsistent formats: High
- Duplicate records: 5-10%

### After Cleaning
- Data quality score: >90%
- Missing values: <5% (handled appropriately)
- Consistent formatting: 100%
- Duplicate records: 0%

## 🎯 Success Criteria
- [ ] Complete data quality assessment with metrics
- [ ] Handle all missing values with appropriate strategies
- [ ] Standardize all text fields and formats
- [ ] Detect and handle outliers appropriately
- [ ] Remove or merge duplicate records
- [ ] Create automated cleaning pipeline
- [ ] Document all cleaning decisions
- [ ] Validate cleaned data quality

## 📝 Learning Outcomes
By completing this project, you will:
- Master pandas for data manipulation
- Understand different missing value strategies
- Learn outlier detection techniques
- Gain text processing and regex skills
- Build automated data pipelines
- Develop data validation practices
- Create comprehensive data documentation

## 🔧 Advanced Techniques

### 🤖 Automated Pipeline
```python
# Example pipeline structure
class DataCleaningPipeline:
    def __init__(self):
        self.steps = []
    
    def add_step(self, step_function, **kwargs):
        self.steps.append((step_function, kwargs))
    
    def execute(self, df):
        for step_func, params in self.steps:
            df = step_func(df, **params)
        return df
```

### 📊 Quality Metrics
Track improvement with quantitative metrics:
- **Completeness**: % of non-null values
- **Validity**: % of values matching expected patterns
- **Consistency**: % of standardized formats
- **Uniqueness**: % of unique records

## 🔗 Resources & References
- [Pandas Data Cleaning Guide](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Data Cleaning Best Practices](https://towardsdatascience.com/data-cleaning-with-python-and-pandas-detecting-missing-values-3e9c6ebcf78b)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Regular Expressions Tutorial](https://regexr.com/)

## ⚠️ Best Practices
1. **Always keep original data**: Never overwrite raw data
2. **Document decisions**: Record why you made each cleaning choice
3. **Validate results**: Check data after each cleaning step
4. **Consider domain context**: Not all "outliers" are errors
5. **Test your pipeline**: Ensure it works on new data
6. **Monitor data quality**: Set up ongoing quality checks

## 🚀 Extensions & Next Steps
- **Real-time cleaning**: Stream processing for live data
- **Machine learning**: Predictive imputation methods
- **Data profiling**: Automated quality assessment tools
- **Monitoring dashboards**: Track data quality over time
- **Integration**: Connect with databases and APIs

---
*Duration: 2-3 days | Difficulty: Beginner | Skills: Data Preprocessing, Quality Assessment, Pipeline Development*