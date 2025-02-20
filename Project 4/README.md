# ECE475 Frequentist Machine Learning - Project 4

## Author: Hailey Hayoon Chung

## Overview
This project implements **Extreme Gradient Boosting (XGBoost)** for regression tasks on two datasets:
1. **California Housing Dataset**
2. **Wine Quality Dataset**

For both datasets, the following analyses were performed:
- Feature Engineering
- XGBoost Model Training and Tuning
- Evaluation of Training and Test Error Over Iterations
- Feature Importance Analysis
- Partial Dependence Plots for Model Interpretation

---

## 1. California Housing Dataset
### Dataset Description
- **Source**: 1990 Census data of California housing
- **Dependent Variable**: Log-transformed Median House Value
- **Predictors**:
  - Median Income, House Age, Population, Latitude, Longitude
  - Engineered Features: Average Rooms per Household, Average Bedrooms per Household, Average Occupancy
- **Preprocessing**:
  - Converted `total rooms`, `total bedrooms`, and `households` into engineered features.
  - Split into **80% training and 20% testing**.

### Model Training & Evaluation
- **Model**: XGBoost Regressor with the following hyperparameters:
  - `max_depth = 3`, `learning_rate = 0.1`, `colsample_bytree = 0.5`
  - `objective = "reg:pseudohubererror"`, `n_estimators = 800`, `alpha = 5`
  - `eval_metric = mean_absolute_error`
- **Performance Metrics**:
  - **Figure 10.13**: Training and test error vs. number of iterations.
  - **Figure 10.14**: Feature importance analysis.
  - **Figure 10.15**: Partial dependence plots for key predictors.
  - **Figure 10.16**: 3D partial dependence plot of `HouseAge` and `AveOccup`.

---

## 2. Wine Quality Dataset
### Dataset Description
- **Source**: Kaggle - Red Wine Quality Dataset
- **Dependent Variable**: Wine Quality Score
- **Predictors**: 11 physicochemical attributes including acidity, pH, alcohol, and sulfates.
- **Preprocessing**:
  - Directly used all numerical features.
  - Split into **80% training and 20% testing**.

### Model Training & Evaluation
- **Initial Model**:
  - `max_depth = 3`, `learning_rate = 0.1`, `colsample_bytree = 0.5`
  - **Result**: High variance between training and test errors.
- **Tuned Model**:
  - Reduced `learning_rate` to **0.02** and `n_estimators` to **400**.
  - Improved generalization with minimal performance tradeoff.
- **Performance Metrics**:
  - **Figure 10.13**: Training and test error vs. number of iterations.
  - **Figure 10.14**: Feature importance analysis.
  - **Figure 10.15**: Partial dependence plots for key predictors (`volatile acidity`, `total sulfur dioxide`, `pH`).
  - **Figure 10.16**: 3D partial dependence plot for `total sulfur dioxide` and `pH`.

---

## Key Takeaways
- **California Housing**: Median Income was the most influential predictor.
- **Wine Quality**: **Volatile Acidity**, **Total Sulfur Dioxide**, and **Alcohol** had the strongest effect on wine quality.
- **Tuning Hyperparameters**: Lowering the learning rate and controlling tree depth improved generalization.
- **Partial Dependence Analysis**: Provided insights into how individual features impact predictions, improving model interpretability.
```
