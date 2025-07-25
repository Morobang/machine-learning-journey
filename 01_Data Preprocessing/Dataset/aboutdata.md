# **About the Dataset**

This dataset is a **small, beginner-friendly dataset** used to practice **data preprocessing for Machine Learning**.

---

## **What does the data represent?**

* The dataset contains information about people from **France, Spain, and Germany**.
* Each row describes **one person** with details like their country, age, salary, and whether they purchased a product.

---

## **Columns (Features)**

1. **Country** *(Categorical)*

   * The person’s country: France, Spain, or Germany.

2. **Age** *(Numerical)*

   * The age of the person.
   * Some values are missing (blank cells).

3. **Salary** *(Numerical)*

   * The person’s annual salary in dollars.
   * One value is missing.

4. **Purchased** *(Categorical – Target)*

   * Whether the person bought the product:

     * **Yes** = Purchased
     * **No** = Did not purchase

---

## **Purpose of this Dataset**

This dataset is **not for real-world analysis**.
It is designed for **learning how to:**

* Handle **missing data**
* **Encode categorical values** (Country and Purchased)
* **Split data** into training and test sets
* **Scale numerical values** (Age, Salary)

These are the **essential first steps** before training any machine learning model.

---

## **File Details**

* Filename: `data.csv`
* Rows: 10
* Columns: 4

---

## **How This Dataset Will Be Used**

In the `01_Data_Processing` folder:

* We will:

  1. Import the dataset
  2. Handle missing values
  3. Encode text into numbers
  4. Split it into **Training** and **Test** sets
  5. Scale numerical values
* After these steps, the cleaned data will be ready for ML models.

