# Classification Metrics Explained

This document explains the key evaluation metrics used for classification models in machine learning.

## 📊 Confusion Matrix

The confusion matrix is a table that shows the performance of a classification model by comparing actual vs predicted values.

```
                Predicted
               Negative  Positive
Actual Negative   TN        FP
      Positive   FN        TP
```

- **True Positives (TP)**: Correctly predicted positive cases
- **True Negatives (TN)**: Correctly predicted negative cases
- **False Positives (FP)**: Negative cases incorrectly predicted as positive (Type I error)
- **False Negatives (FN)**: Positive cases incorrectly predicted as negative (Type II error)

## 📈 Accuracy Score

**Formula**: (TP + TN) / (TP + TN + FP + FN)

- Measures overall correctness of the model
- Percentage of all predictions that were correct
- Best for balanced datasets
- Can be misleading with imbalanced classes

## 🎯 Precision Score

**Formula**: TP / (TP + FP)

- Measures how many selected items are relevant
- Answers: "Of all predicted positives, how many are actually positive?"
- Important when FP costs are high (e.g., spam detection)
- High precision means few false positives

## 🔍 Recall Score (Sensitivity)

**Formula**: TP / (TP + FN)

- Measures how many relevant items are selected
- Answers: "Of all actual positives, how many did we predict correctly?"
- Important when FN costs are high (e.g., medical diagnosis)
- High recall means few false negatives

## ⚖️ F1 Score

**Formula**: 2 * (Precision * Recall) / (Precision + Recall)

- Harmonic mean of precision and recall
- Balances both metrics
- Useful when you need to consider both FP and FN
- Best for imbalanced datasets

## 📉 ROC AUC Score

**ROC Curve** plots True Positive Rate (Recall) vs False Positive Rate (FP / (FP + TN)) at different thresholds

**AUC (Area Under Curve)** measures the entire two-dimensional area under the ROC curve

- Evaluates model's ability to distinguish between classes
- AUC of 0.5 = random guessing
- AUC of 1.0 = perfect classifier
- Useful for comparing different models

## 📝 Classification Report

Provides a comprehensive summary of metrics:

```
              precision  recall  f1-score  support

     Class 0       0.85    0.90      0.87       100
     Class 1       0.80    0.70      0.75        60

    accuracy                           0.83       160
   macro avg       0.82    0.80      0.81       160
weighted avg       0.83    0.83      0.83       160
```

- Shows precision, recall, f1-score per class
- Includes support (number of actual occurrences)
- Provides macro-average (simple average)
- Provides weighted-average (accounts for class imbalance)

## When to Use Each Metric

| Metric | Best For | Watch Out For |
|--------|----------|---------------|
| Accuracy | Balanced classes | Imbalanced data |
| Precision | Minimizing FP | May sacrifice recall |
| Recall | Minimizing FN | May increase FP |
| F1 Score | Balanced view | Needs both P/R |
| ROC AUC | Overall performance | Needs probabilities |

Choose metrics based on your business objectives and the costs of different error types.