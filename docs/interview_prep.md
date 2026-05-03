# ML Interview Preparation Guide

A curated question bank organised by topic. Every question links back to the notebook or theory doc where the answer is covered in depth.

This is not a list of "trick questions" — these are the actual questions that appear in data science and ML engineering interviews at most companies.

---

## How to use this guide

1. Work through the curriculum first. Use this guide to **test yourself**, not to shortcut the learning.
2. For each question, close this file and write out your answer. Then open the linked resource and compare.
3. Questions marked **(P)** are particularly common in phone screens and take-home assessments.

---

## Foundations

**Data & Preprocessing**

1. **(P)** What is the difference between imputing with mean vs median? When would you use each?
   → [00_foundations/theory/09_data_types_and_features.md](../00_foundations/theory/09_data_types_and_features.md)

2. **(P)** What is data leakage? Give a concrete example involving feature scaling.
   → [00_foundations/theory/11_common_pitfalls.md](../00_foundations/theory/11_common_pitfalls.md)

3. Why should you fit your scaler on the training set only?
   → [src/preprocessing/scalers.py](../src/preprocessing/scalers.py)

4. What is the difference between one-hot encoding and label encoding? When is each appropriate?
   → [00_foundations/theory/09_data_types_and_features.md](../00_foundations/theory/09_data_types_and_features.md)

5. You have a feature with 95% missing values. What are your options?
   → [00_foundations/theory/11_common_pitfalls.md](../00_foundations/theory/11_common_pitfalls.md)

**Evaluation Metrics**

6. **(P)** What is the difference between precision and recall? Give an example where you would prioritise recall over precision.
   → [00_foundations/theory/10_evaluation_metrics.md](../00_foundations/theory/10_evaluation_metrics.md)

7. **(P)** Your model gets 97% accuracy on a fraud detection dataset. Is it a good model? What else do you need to know?

8. What does AUC-ROC measure? What does an AUC of 0.5 mean?
   → [00_foundations/theory/10_evaluation_metrics.md](../00_foundations/theory/10_evaluation_metrics.md)

9. When would you use the F1 score instead of accuracy?

10. What is the difference between RMSE and MAE? When is RMSE misleading?

---

## Supervised Learning — Regression

11. **(P)** What does R² actually measure? Can it be negative? What would that mean?
    → [01_supervised_learning/regression/notebooks/01_simple_linear_regression.ipynb](../01_supervised_learning/regression/notebooks/01_simple_linear_regression.ipynb)

12. What are the assumptions of linear regression? How do you check them?

13. What is multicollinearity? How does it affect linear regression? How do you detect it?

14. **(P)** What is regularisation? Explain L1 (Lasso) vs L2 (Ridge) in plain English. When would you use each?

15. Your regression model has high training R² but low test R². What is happening? Name two remedies.

---

## Supervised Learning — Classification

16. **(P)** Explain the sigmoid function. Why is it used in logistic regression?
    → [01_supervised_learning/classification/notebooks/01_logistic_regression.ipynb](../01_supervised_learning/classification/notebooks/01_logistic_regression.ipynb)

17. What is the decision boundary in logistic regression? What happens when classes are not linearly separable?

18. **(P)** Explain how a decision tree chooses which feature to split on. What is Gini impurity vs information gain?
    → [01_supervised_learning/classification/notebooks/06_decision_tree_classification.ipynb](../01_supervised_learning/classification/notebooks/06_decision_tree_classification.ipynb)

19. What is the "kernel trick"? Why does it allow SVMs to handle non-linear boundaries without explicitly computing high-dimensional features?
    → [01_supervised_learning/classification/notebooks/04_kernel_svm.ipynb](../01_supervised_learning/classification/notebooks/04_kernel_svm.ipynb)

20. Naive Bayes is called "naive" because of one key assumption. What is it, and when does it still work despite being wrong?
    → [01_supervised_learning/classification/notebooks/05_naive_bayes.ipynb](../01_supervised_learning/classification/notebooks/05_naive_bayes.ipynb)

---

## Model Evaluation and Selection

21. **(P)** What is k-fold cross-validation? Why is it better than a single train/test split?
    → [09_model_selection_and_evaluation/notebooks/01_cross_validation.ipynb](../09_model_selection_and_evaluation/notebooks/01_cross_validation.ipynb)

22. **(P)** Explain the bias-variance tradeoff. Draw the curve. Where does a decision tree tend to sit?

23. What is the difference between grid search and random search? When is random search preferred?
    → [09_model_selection_and_evaluation/notebooks/02_hyperparameter_tuning.ipynb](../09_model_selection_and_evaluation/notebooks/02_hyperparameter_tuning.ipynb)

24. You have a model with very high variance. Name four things you can try to reduce it.

25. What is stratified k-fold and when is it essential?

---

## Ensemble Methods

26. **(P)** What is the difference between bagging and boosting? Which reduces variance, which reduces bias?
    → [06_ensemble_methods/theory/01_gradient_boosting_intuition.md](../06_ensemble_methods/theory/01_gradient_boosting_intuition.md)

27. **(P)** How does Random Forest reduce overfitting compared to a single decision tree?

28. **(P)** Explain gradient boosting in plain English. What are the "residuals" each tree is fitting?
    → [06_ensemble_methods/theory/01_gradient_boosting_intuition.md](../06_ensemble_methods/theory/01_gradient_boosting_intuition.md)

29. **(P)** What two regularisation terms does XGBoost add that vanilla gradient boosting lacks?
    → [06_ensemble_methods/theory/02_xgboost_deep_dive.md](../06_ensemble_methods/theory/02_xgboost_deep_dive.md)

30. Your XGBoost model has training AUC 0.98 and validation AUC 0.82. Name three hyperparameters you would tune.

31. What is `scale_pos_weight` in XGBoost? Why would you set it to 4 on a dataset with 20% positive class?

32. When would you choose LightGBM over XGBoost?
    → [06_ensemble_methods/theory/02_xgboost_deep_dive.md](../06_ensemble_methods/theory/02_xgboost_deep_dive.md)

---

## Unsupervised Learning

33. **(P)** How does K-Means work? What are its limitations?
    → [02_unsupervised_learning/clustering/notebooks/01_kmeans_clustering.ipynb](../02_unsupervised_learning/clustering/notebooks/01_kmeans_clustering.ipynb)

34. How do you choose K in K-Means? Explain the elbow method and its limitations.

35. What is the difference between K-Means and hierarchical clustering? When would you use each?

36. In the Apriori algorithm, what do support, confidence, and lift measure?
    → [02_unsupervised_learning/association_rules/notebooks/01_apriori_algorithm.ipynb](../02_unsupervised_learning/association_rules/notebooks/01_apriori_algorithm.ipynb)

---

## Dimensionality Reduction

37. **(P)** What does PCA do? What does each principal component represent?
    → [08_dimensionality_reduction/notebooks/01_principal_component_analysis.ipynb](../08_dimensionality_reduction/notebooks/01_principal_component_analysis.ipynb)

38. How do you decide how many principal components to keep?

39. What is the difference between PCA and LDA? When is LDA preferred?

40. Why should you NOT use t-SNE output as input features for a downstream model?

---

## Deep Learning

41. **(P)** What is backpropagation? Explain the chain rule in the context of a neural network.

42. What is the vanishing gradient problem? Which activation functions suffer from it most?

43. What is dropout? How does it act as regularisation?

44. What is the difference between batch gradient descent, mini-batch, and stochastic gradient descent?

45. What makes a CNN architecture suitable for image data? Why would a plain neural network struggle with images?
    → [04_deep_learning/notebooks/02_convolutional_neural_network.ipynb](../04_deep_learning/notebooks/02_convolutional_neural_network.ipynb)

---

## NLP

46. What is TF-IDF? How is it different from raw word counts?
    → [05_natural_language_processing/notebooks/02_bag_of_words_tfidf.ipynb](../05_natural_language_processing/notebooks/02_bag_of_words_tfidf.ipynb)

47. What information does Bag of Words lose that matters for NLP tasks?

48. What problem do word embeddings (Word2Vec, GloVe) solve that BoW cannot?

---

## Production and Engineering

49. **(P)** What is data leakage? Give an example involving a feature that would only be available after the target event.

50. **(P)** Your model performs well in development but poorly in production. Name four possible causes.

51. How would you detect data drift in a deployed model?

52. Why is it important to version your model and your training data together?

53. What is a feature store and why do ML teams use them?

---

## System Design Questions (Senior / MLE Roles)

54. Design a real-time fraud detection system. What are the latency constraints? How do you handle imbalanced classes at scale?

55. You need to retrain your model monthly. What does a production retraining pipeline look like?

56. How would you A/B test a new ML model against the current production model?

57. Your model makes a prediction and a customer disputes it. How do you explain the prediction?
    → [09_model_selection_and_evaluation/](../09_model_selection_and_evaluation/) (SHAP notebooks)

---

## Quick-Fire Definitions

Can you define each of these in one sentence?

| Term | Definition |
|------|------------|
| Overfitting | |
| Regularisation | |
| Cross-validation | |
| Feature engineering | |
| Normalisation vs standardisation | |
| Ensemble method | |
| Precision | |
| Recall | |
| AUC-ROC | |
| Gradient descent | |
| Hyperparameter | |
| Stratified split | |
| Data leakage | |
| Confusion matrix | |
| Early stopping | |