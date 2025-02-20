# ECE475 Frequentist Machine Learning - Project 1

## Author: Hailey Hayoon Chung

## Overview
This project explores various regression techniques applied to a prostate cancer dataset and a synchronous machine dataset. The main focus is on implementing and comparing different regression methods:

1. **Linear Regression (Ordinary Least Squares)**
2. **Ridge Regression**
3. **Lasso Regression**
4. **Application to a Synchronous Machine Dataset**
5. **Nonlinear and Interaction Terms**

Each method is evaluated based on the mean squared error (MSE) on the test dataset.

---

## 1. Data Handling
- **Dataset**: The `prostate.csv` dataset was used, which contains various predictors related to prostate cancer.
- **Preprocessing**: The dataset was standardized using mean and standard deviation computed from the training set.
- **Splitting Strategy**:
  - Training: 80%
  - Validation: 10%
  - Test: 10%

---

## 2. Linear Regression
- Implemented manually using the normal equation (Equation 3.6 from the textbook).
- Computed the correlation matrix of predictors (Table 3.1) and estimated coefficients (Table 3.2).
- MSE on the test dataset: **0.3507**

---

## 3. Ridge Regression
- Implemented using Equation 3.44.
- **Lambda Selection**: Cross-validation using the validation set.
- **Optimal Lambda**: **2.452** (selected based on validation MSE).
- **Test MSE**: **0.3301**
- Generated a Ridge plot showing coefficient shrinkage over different lambda values.

---

## 4. Lasso Regression
- Implemented using `sklearn`’s Lasso function.
- **Lambda Selection**: Cross-validation using the validation set.
- **Optimal Lambda**: **0.0001**
- **Test MSE**: **0.2683**
- Generated a Lasso plot showing coefficient shrinkage over different lambda values.

---

## 5. Application to Synchronous Machine Dataset
- **Dataset**: A real-world dataset from UCI, focusing on estimating the excitation current of synchronous motors.
- **Features**: Power factor (PF), power factor error (e), and other numerical predictors.
- **Regression Models & Performance**:
  - **Linear Regression Test MSE**: 0.5042
  - **Ridge Regression Test MSE**: 0.5101
  - **Lasso Regression Test MSE**: 0.5088
  - **Baseline MSE** (mean predictor): **0.9707**
- **Lasso Feature Selection**: The model primarily selected **If** and **PF** as the most significant predictors.

---

## 6. Nonlinear and Interaction Terms
- **Nonlinear Transformation**: Applied an exponential function to `pgg45`, reducing MSE for linear and lasso regression.
- **Interaction Term**: Created `lcavol * lbph`, reducing MSE across all regression types.
- **Final MSE Results**:
  - **Linear Regression**: 0.3319
  - **Ridge Regression**: 0.3160
  - **Lasso Regression**: 0.2546
  - **Baseline MSE**: 0.2119

---

## Conclusion
- Lasso regression outperformed Ridge and Linear Regression in both datasets by effectively selecting the most relevant features.
- Adding nonlinear and interaction terms improved model performance, demonstrating the importance of feature engineering.
- The synchronous machine dataset posed a greater challenge due to high correlation among features, but regression models still outperformed the baseline prediction.
