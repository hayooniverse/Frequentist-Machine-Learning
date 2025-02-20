# ECE475 Frequentist Machine Learning - Project 5

## Author: Hailey Hayoon Chung

## Overview
This project compares **Random Forest (RF) and Gradient Boosting Machines (GBM) using XGBoost** on two datasets:
1. **California Housing Dataset**
2. **Wine Quality Dataset**

For both datasets, the following analyses were performed:
- Feature Engineering
- Random Forest vs. XGBoost Model Training
- Hyperparameter Tuning
- Model Performance Evaluation
- Feature Importance Analysis

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
- **Random Forest (RF) Models**:
  - `m = 2`: Train/test absolute error over 100 trees.
  - `m = 6`: Train/test absolute error over 100 trees.
- **Gradient Boosting Machine (GBM) Models**:
  - `depth = 4`: 1000 estimators with `learning_rate = 0.05`
  - `depth = 6`: 1000 estimators with `learning_rate = 0.05`
- **Performance Metrics**:
  - **Figure 15.3**: RF and GBM test error comparison.
  - **Feature Importance**:
    - RF: Top predictors identified.
    - GBM: Relative predictor importance.

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
- **Random Forest (RF) Models**:
  - `m = 2`: Train/test absolute error over 100 trees.
  - `m = 6`: Train/test absolute error over 100 trees.
- **Gradient Boosting Machine (GBM) Models**:
  - `depth = 4`: 1000 estimators with `learning_rate = 0.05`
  - `depth = 6`: 1000 estimators with `learning_rate = 0.05`
- **Performance Metrics**:
  - **Figure 15.3**: RF and GBM test error comparison.
  - **Feature Importance**:
    - RF: Top predictors identified.
    - GBM: Relative predictor importance.

---

## Key Takeaways
- **California Housing**: RF performed comparably to GBM for lower depths, but GBM outperformed RF as tree depth increased.
- **Wine Quality**: GBM was more robust to noise, reducing test error faster than RF.
- **Feature Importance**: GBM highlighted **volatile acidity, total sulfur dioxide, and alcohol** as key predictors for wine quality.
- **Model Selection**: GBM outperformed RF in generalization and interpretability, particularly with deeper trees.
