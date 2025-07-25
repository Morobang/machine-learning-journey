# **Data Processing (Driving School Version)**

Think of Machine Learning as teaching a car how to drive.
Before the car can learn, we need to **prepare the driving lessons (data)**.

---

## **1. Importing Libraries**

**Why?**
Libraries are like the **car’s tools**:

* **Pandas** = GPS (organizes where everything goes)
* **NumPy** = Engine (does calculations fast)
* **Matplotlib** = Dashboard (shows speed & graphs)

**Example:**

```python
import pandas as pd  # "Start the GPS to read data"
```

---

## **2. Importing Dataset**

**Why?**
Imagine a **list of all students in a driving school**:

* **X (Features):** Things about the student (age, practice hours, eyesight)
* **y (Target):** Did they **pass** or **fail**?

**Example:**

```python
data = pd.read_csv('driving_school.csv')  
X = data[['age', 'practice_hours']]  # Features
y = data['passed_test']              # Target
```

---

## **3. Handling Missing Data**

**Why?**
Some students forgot to say how many practice hours they had.
We can:

* Throw away their form (remove row)
* Guess based on others (e.g., most students practiced 20 hrs)

**Example:**

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')  # "Fill missing hours with average"
X[['practice_hours']] = imputer.fit_transform(X[['practice_hours']])
```

---

## **4. Encoding Categorical Data**

**Why?**
Computers can’t understand **“Good” or “Bad eyesight”** – they only understand numbers.
We convert words to numbers:

* Good = 1
* Bad = 0

**Example:**

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['eyesight'] = le.fit_transform(X['eyesight'])
```

---

## **5. Train/Test Split**

**Why?**
To be fair, we **teach the car with some students’ data** and **test it on students it has never seen**.

* **Training set:** 80% → Driving lessons
* **Test set:** 20% → Final driving exam

**Example:**

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

---

## **6. Feature Scaling**

**Why?**
Different features have **different ranges**:

* Age = 16–60
* Practice Hours = 0–100

If we don’t scale, the car might think “practice hours” is 100x more important than “age”.

**Example:**

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

---

## **Key Takeaways**

* **Libraries** = Car’s tools (GPS, engine, dashboard)
* **Missing Data** = Fill in what students forgot
* **Encoding** = Turn words (Good/Bad eyesight) into numbers
* **Train/Test Split** = Don’t cheat on the driving test
* **Scaling** = Make sure all features are fair
