# **Data Types and Features in Machine Learning**

Understanding your data is the first step to building successful ML models. This guide covers the different types of data you'll encounter and how to work with them effectively.

---

## **🗂️ Types of Data**

### **1. Numerical Data (Numbers)**

**Continuous (Decimal Numbers):**
- Can take any value within a range
- Examples: Height (175.5 cm), Temperature (23.7°C), Price ($299.99)
- **ML Use:** Perfect for regression problems, easy to work with

**Discrete (Whole Numbers):**
- Can only take specific integer values
- Examples: Number of kids (0, 1, 2, 3...), Website clicks, Exam scores
- **ML Use:** Can be treated like continuous or categorical depending on context

### **2. Categorical Data (Categories/Labels)**

**Nominal (No Order):**
- Categories with no natural ranking
- Examples: Colors (Red, Blue, Green), Countries, Brands
- **ML Encoding:** One-hot encoding (create separate columns for each category)

**Ordinal (Has Order):**
- Categories with meaningful ranking
- Examples: Education (High School < Bachelor's < Master's), Ratings (1-5 stars)
- **ML Encoding:** Label encoding (convert to numbers maintaining order)

### **3. Text Data (Words and Sentences)**
- Examples: Reviews, emails, social media posts, documents
- **ML Challenges:** Computers don't understand words directly
- **ML Solutions:** Convert to numbers using techniques like:
  - Bag of Words
  - TF-IDF
  - Word embeddings

### **4. Image Data (Pictures)**
- Examples: Photos, X-rays, satellite images, handwritten digits
- **ML Representation:** Arrays of pixel values (numbers!)
- **ML Use:** Computer vision, image classification, object detection

### **5. Time Series Data (Time-based)**
- Examples: Stock prices, weather data, sensor readings, website traffic
- **ML Challenges:** Order matters, trends and seasonality
- **ML Use:** Forecasting, anomaly detection

---

## **🔧 Feature Engineering Basics**

### **What are Features?**
Features are the **input variables** that describe your data. Think of them as the "questions" you ask about each data point.

**Example - Predicting House Prices:**
```
Raw Data: "3-bedroom house, built in 1995, 1500 sq ft, downtown"

Features:
- bedrooms: 3
- year_built: 1995  
- square_feet: 1500
- location: "downtown"
- house_age: 2025 - 1995 = 30 years (derived feature!)
```

### **Creating New Features (Feature Engineering)**

**Mathematical Combinations:**
- `total_rooms = bedrooms + bathrooms + living_rooms`
- `price_per_sqft = price / square_feet`
- `bmi = weight / (height^2)`

**Date/Time Features:**
From a date like "2025-09-26 14:30:00", extract:
- `year = 2025`
- `month = 9`
- `day_of_week = Friday`
- `hour = 14`
- `is_weekend = False`

**Text Features:**
From a product review, extract:
- `word_count = 50`
- `exclamation_count = 3`
- `sentiment_score = 0.8`
- `contains_negative_words = False`

### **Handling Categorical Data**

**One-Hot Encoding (for Nominal Data):**
```
Original: Color = ["Red", "Blue", "Red", "Green"]

Encoded:
Color_Red   Color_Blue   Color_Green
    1           0           0
    0           1           0  
    1           0           0
    0           0           1
```

**Label Encoding (for Ordinal Data):**
```
Original: Education = ["High School", "Bachelor's", "Master's", "PhD"]
Encoded:  Education = [1, 2, 3, 4]
```

---

## **📊 Data Quality Issues**

### **Missing Values**
When some data points are empty or unknown.

**Common Causes:**
- Survey respondents skip questions
- Sensor malfunctions
- Data entry errors
- Privacy concerns

**Solutions:**
1. **Remove:** Delete rows/columns with missing data
2. **Fill:** Replace with mean, median, or mode
3. **Predict:** Use other features to predict missing values
4. **Flag:** Create a new feature "is_missing"

### **Outliers**
Data points that are unusually different from others.

**Examples:**
- Person claiming to be 200 years old
- House priced at $1 (data entry error)
- Income of $10 million in a dataset of students

**Detection:**
- **Visual:** Box plots, scatter plots
- **Statistical:** Z-score > 3, IQR method
- **Domain Knowledge:** "This doesn't make sense"

**Handling:**
1. **Remove:** If clearly errors
2. **Transform:** Apply log transformation
3. **Keep:** If legitimate extreme values
4. **Cap:** Set maximum/minimum limits

### **Inconsistent Data**
Same information recorded differently.

**Examples:**
- "USA" vs "United States" vs "U.S.A"
- "Male" vs "M" vs "male" vs "MALE"
- Dates: "09/26/2025" vs "26-09-2025" vs "Sep 26, 2025"

**Solutions:**
- **Standardize:** Convert all to same format
- **Clean:** Remove extra spaces, fix capitalization
- **Validate:** Check against known valid values

---

## **🎯 Feature Selection**

### **Why Select Features?**
- **Reduce complexity:** Fewer features = simpler models
- **Improve performance:** Remove irrelevant/noisy features
- **Faster training:** Less data to process
- **Avoid overfitting:** Too many features can hurt generalization

### **Methods:**

**1. Filter Methods (Statistical Tests):**
- Correlation with target variable
- Mutual information
- Chi-square test

**2. Wrapper Methods (Try Different Combinations):**
- Forward selection: Start with one feature, add more
- Backward elimination: Start with all, remove worst
- Recursive feature elimination

**3. Embedded Methods (Built into Algorithms):**
- LASSO regression (automatically reduces features)
- Tree-based feature importance
- Neural network attention mechanisms

### **Feature Importance**
Understanding which features matter most:

**Tree-Based Models:**
- Count how often a feature is used for splitting
- Measure how much each split improves accuracy

**Linear Models:**
- Larger coefficients = more important features
- But be careful with different scales!

**Permutation Importance:**
- Shuffle one feature at a time
- See how much performance drops
- Bigger drop = more important feature

---

## **⚖️ Data Scaling and Normalization**

### **Why Scale Data?**
Different features have different ranges:
- Age: 0-100
- Income: $0-$1,000,000
- Height: 150-200 cm

Some algorithms (like KNN, SVM) are sensitive to scale.

### **Common Scaling Methods:**

**Min-Max Scaling (0 to 1):**
```
scaled_value = (value - min) / (max - min)
```
Use when: You know the min/max bounds

**Standard Scaling (Z-score):**
```
scaled_value = (value - mean) / standard_deviation
```
Use when: Data is normally distributed

**Robust Scaling:**
Uses median and quartiles instead of mean (less affected by outliers)

---

## **🔍 Exploratory Data Analysis (EDA)**

### **First Steps with New Data:**

**1. Basic Info:**
- How many rows and columns?
- What data types?
- Any missing values?

**2. Summary Statistics:**
- Mean, median, mode
- Min, max, range
- Standard deviation

**3. Visualizations:**
- **Histograms:** Distribution of numerical features
- **Bar charts:** Count of categorical features
- **Scatter plots:** Relationships between features
- **Correlation heatmaps:** Which features relate to each other

**4. Target Variable Analysis:**
- **Classification:** How balanced are the classes?
- **Regression:** What's the distribution of target values?

### **Key Questions to Ask:**
- What does each row represent?
- What are we trying to predict?
- Which features might be most important?
- Are there any obvious data quality issues?
- Do we need to create any new features?

---

## **💡 Practical Tips**

### **Start Simple:**
1. Use basic features first
2. Get a baseline model working
3. Then engineer more complex features

### **Domain Knowledge is Key:**
- Understand what the data represents
- Know what makes business sense
- Ask subject matter experts

### **Iterate and Experiment:**
- Try different feature combinations
- Test impact on model performance
- Keep track of what works

### **Common Mistakes to Avoid:**
- **Data leakage:** Using future information to predict the past
- **Overfitting:** Creating too many features for small datasets
- **Ignoring outliers:** Not investigating unusual values
- **Forgetting validation:** Not testing features on unseen data

---

## **🚀 Feature Engineering Workflow**

1. **Understand the Problem:** What are you trying to predict?
2. **Explore the Data:** What do you have to work with?
3. **Clean the Data:** Handle missing values, outliers, inconsistencies
4. **Create Features:** Combine, transform, derive new information
5. **Select Features:** Keep the most useful ones
6. **Scale/Encode:** Prepare for your chosen algorithm
7. **Validate:** Test on unseen data
8. **Iterate:** Refine based on results

---

## **📚 Key Takeaways**

- **Data understanding beats fancy algorithms:** Good features > complex models
- **Different algorithms need different data prep:** Know your algorithm's requirements
- **Start simple, then get creative:** Basic features first, then engineer new ones
- **Validate everything:** Always test changes on unseen data
- **Domain knowledge is crucial:** Understanding the problem helps create better features

Remember: **Garbage in, garbage out.** The quality of your features determines the ceiling of your model's performance!

---

**Ready to engineer some features?** The best way to learn is by working with real datasets! 🔧📊