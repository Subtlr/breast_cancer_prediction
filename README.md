# 🧠 Breast Cancer Prediction with Machine Learning

This repository showcases a comprehensive pipeline for predicting breast cancer diagnoses using various machine learning models, ranging from interpretable regressions to flexible neural networks. It is built on the [Breast Cancer Dataset](https://www.kaggle.com/datasets/wasiqaliyasir/breast-cancer-dataset) from Kaggle and focuses on balancing performance with interpretability.

## 🔍 Objective

The primary goal is to develop a classification system that prioritizes **recall**—minimizing false negatives—to support early breast cancer detection. Multiple models were trained and compared across performance metrics and extrapolation capabilities.

---

## 🧪 Data Preprocessing

- ✅ Removed collinearity in correlated features: `radius`, `perimeter`, and `area` columns.
- ✅ Standardized features using training set statistics to prevent data leakage.
- ✅ Engineered helper utilities to log model performances systematically.

<div align="center">
  <p>📊 <em>Correlation matrix before and after feature pruning</em></p>
  <img src="https://github.com/Subtlr/breast_cancer_prediction/blob/main/imgs/og_corr.png" alt="📊 Correlation matrix before feature pruning" width="40%"/>
  <img src="https://github.com/Subtlr/breast_cancer_prediction/blob/main/imgs/reduced_corr.png" alt="📊 Correlation matrix after feature pruning" width="40%"/>
</div>

---

## 🚀 Models Trained

| Model                  | Flexibility | Interpretability | Feature Importance | Overfitting Risk |
|-----------------------|-------------|------------------|--------------------|------------------|
| Logistic Regression   | Low         | High             | ✅                 | Low              |
| Support Vector Machine| Medium      | Medium           | ✅                 | High             |
| Random Forest         | High        | Medium           | ✅                 | Medium-Low       |
| XGBClassifier         | High        | Moderate         | ✅                 | Low              |
| **Neural Network (PyTorch)** | **Highest** | **Low** | ❌ | **Low (generalizes well!)** |

> 🧠 **Neural Network Highlight**  
> A compact PyTorch model with over **3000 learnable parameters** was designed and trained using BCE Loss with a dropout layer to reduce the risk of overfitting. Despite a higher loss on the validation set, its ability to generalize on unseen data makes it an attractive option in high-dimensional tasks. Custom architecture allowed flexibility while still handling a relatively small dataset.

---

## 📈 Evaluation Metrics

Each model was evaluated using:

- 📊 Confusion Matrices on both validation and test sets
- ✅ Recall (priority)
- ⚖️ Precision, F1 Score
- 🧮 Binary Cross Entropy Loss

<div align="center">
  <img src="https://github.com/Subtlr/breast_cancer_prediction/blob/main/imgs/model_evaluation_test.png", alt=" 📸 Comparison chart of metrics across models"/>
</div>

---

## 🔍 Key Insights

> ✅ **Tree-based models underperformed on validation, but generalized surprisingly well.**  
> ❌ **SVM, although had the highest recall score, made the most misclassifications.**  
> ⚖️ **Logistic Regression and Random Forest shared the same confusion matrix.**  
> 🔥 **Neural Network had highest BCE Loss but showed promising extrapolation ability.**  
> 🧾 **Random Forest remains a strong go-to model for practical applications—interpretability meets performance.**

---
## Photos

<div align="center">
  <p><em>Validation (above) and Test (below) Confusion Matrices</em></p>
  <img src="https://github.com/Subtlr/breast_cancer_prediction/blob/main/imgs/log_val.png" width="19%"/>
  <img src="https://github.com/Subtlr/breast_cancer_prediction/blob/main/imgs/forest_val.png" width="19%"/>
  <img src="https://github.com/Subtlr/breast_cancer_prediction/blob/main/imgs/svm_val.png" width="19%"/>
  <img src="https://github.com/Subtlr/breast_cancer_prediction/blob/main/imgs/xgb_test.png" width="19%"/>
  <img src="https://github.com/Subtlr/breast_cancer_prediction/blob/main/imgs/nn_val.png" width="19%"/>

  <p><em>I'm gonna be honest, I do not have the photo for the validation set confusion matrix for the XGBClassifier Model</em></p>
  <img src="https://github.com/Subtlr/breast_cancer_prediction/blob/main/imgs/log_test.png" width="19%"/>
  <img src="https://github.com/Subtlr/breast_cancer_prediction/blob/main/imgs/forest_test.png" width="19%"/>
  <img src="https://github.com/Subtlr/breast_cancer_prediction/blob/main/imgs/svm_test.png" width="19%"/>
  <img src="https://github.com/Subtlr/breast_cancer_prediction/blob/main/imgs/xgb_test.png" width="19%"/>
  <img src="https://github.com/Subtlr/breast_cancer_prediction/blob/main/imgs/nn_test.png" width="19%"/>
</div>
