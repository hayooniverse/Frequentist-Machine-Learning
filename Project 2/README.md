# ECE475 Frequentist Machine Learning - Project 2

## Author: Hailey Hayoon Chung

## Overview
This project replicates the analysis of the **South African heart disease dataset** from *The Elements of Statistical Learning* and applies logistic regression models to classify heart disease risk. Additionally, the analysis is extended to two other datasets: **Iris dataset** and **Wine Quality dataset**.

The primary techniques explored include:
1. **Logistic Regression with Stochastic Gradient Descent (SGD)** (No Regularization)
2. **Logistic Regression with L2 Regularization (Ridge Regression)**
3. **Logistic Regression with L1 Regularization (Lasso Regression)**
4. **Feature Selection using Forward Stepwise Regression**
5. **Comparison of Feature Importance across Stepwise and Lasso**

---

## 1. Data Preprocessing
### South African Heart Disease Dataset
- **Data Source**: `SAheart.txt`
- Removed unused columns (`adiposity`, `typea`).
- Converted `famhist` from categorical to binary (`Present=1, Absent=0`).
- Standardized features using training set statistics.
- Splitting Strategy:
  - **Training**: 80%
  - **Validation**: 10%
  - **Test**: 10%
- Scatter plot matrix generated (Figure 4.12) to visualize relationships between features.

### Iris Dataset
- **Data Source**: `iris.txt`
- Standardized numerical features.
- Splitting Strategy same as above.

### Wine Quality Dataset
- **Data Source**: `winequality-red.csv`
- Converted `quality` to binary: **Good Quality (≥6) → 1, Bad Quality (<6) → 0**.
- Standardized numerical features.
- Splitting Strategy same as above.

---

## 2. Logistic Regression Models
### 2.1 Logistic Regression with SGD (No Regularization)
- Implemented logistic regression using Stochastic Gradient Descent (SGD) without L1/L2 penalty.
- Computed percent accuracy on test set.

### 2.2 Logistic Regression with L2 Regularization (Ridge Regression)
- Added an L2 penalty term to reduce overfitting.
- Optimal **lambda** selected via cross-validation.

### 2.3 Logistic Regression with L1 Regularization (Lasso Regression)
- Added an L1 penalty term to encourage sparsity.
- Lambda was optimized through cross-validation.
- Generated **Lasso coefficient paths** (Figure 4.13) to visualize the impact of L1 regularization.

---

## 3. Feature Selection
### 3.1 Forward Stepwise Selection
- Incrementally selected features that maximized accuracy.
- Found that **tobacco** and **obesity** were the most important predictors for heart disease.
- For wine dataset, **volatile acidity, chlorides, total sulfur dioxide, and density** were the key features.

### 3.2 Comparison with Lasso
- Features selected via forward stepwise regression were compared with those retained under L1 regularization.
- High correlation observed between features with largest Lasso coefficients and those chosen by stepwise selection.

---

## 4. Results Summary
| Model | South African Heart Disease (Accuracy) | Wine Quality (Accuracy) |
|--------|--------------------------------|-------------------------|
| No Regularization | 80.85% | 66.88% |
| L2 Regularization | 80.85% | 65.63% |
| L1 Regularization | 68.08% | 66.88% |
| Stepwise Selection | 78.72% | 68.13% |

- **Stepwise selection slightly outperformed L1 regularization** in both datasets.
- **L2 regularization was stable but did not improve accuracy significantly.**
- **L1 regularization produced interpretable sparse models but had lower accuracy than stepwise.**

---

## Conclusion
- **Forward stepwise selection and L1 regularization identified similar important features**.
- **L2 regularization had minimal impact on accuracy**, suggesting the models were not heavily overfitting.
- **Stepwise selection provided the best accuracy in most cases**, indicating that a well-chosen subset of features can outperform full-model regularization.
- **Future Work**: Investigate more complex feature interactions and nonlinear transformations to improve classification performance.
