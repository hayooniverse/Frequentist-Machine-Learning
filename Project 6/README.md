# ECE475 Frequentist Machine Learning - Project 6

## Author: Hailey Hayoon Chung

## Overview
This project explores **Collaborative Filtering using Non-negative Matrix Factorization (NMF)** on the **MovieLens 100K Dataset**. The main goal is to tune hyperparameters using **Grid Search** to minimize prediction errors.

### Key Steps:
- Load and preprocess the **MovieLens 100K** dataset.
- Implement **Non-negative Matrix Factorization (NMF)** for collaborative filtering.
- Use **GridSearchCV** to find optimal hyperparameters.
- Evaluate model performance using **RMSE** and **MAE**.

---

## Dataset: MovieLens 100K
### Description
- **Source**: MovieLens 100K dataset (builtin in Surprise library)
- **Data**: 100,000 ratings from 943 users on 1,682 movies.
- **Rating Scale**: 1 to 5 (integer values).

### Preprocessing
- The dataset is loaded using `surprise.Dataset.load_builtin('ml-100k')`.
- Split into training and testing sets using Surprise’s default split.

---

## Model: Non-negative Matrix Factorization (NMF)
### Hyperparameters Tuned
The following hyperparameters were optimized:
- **Number of latent factors (`n_factors`)**: {5, 10, 20}
- **Number of epochs (`n_epochs`)**: {5, 10}
- **Regularization parameters**:
  - **User feature (`reg_pu`)**: {0.01, 0.02, 0.03, 0.04}
  - **Item feature (`reg_qi`)**: {0.01, 0.02, 0.03, 0.04}
  - **User bias (`reg_bu`)**: {0.01, 0.02, 0.03, 0.04}
  - **Item bias (`reg_bi`)**: {0.01, 0.02, 0.03, 0.04}

### Grid Search Implementation
- Performed a grid search using **5-fold cross-validation** (`cv=5`).
- Evaluated the model using **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**.

```python
from surprise import Dataset, NMF
from surprise.model_selection import GridSearchCV

# Load MovieLens 100K dataset
data = Dataset.load_builtin('ml-100k')

# Define parameter grid
param_grid = {
    'n_factors': [5, 10, 20],
    'n_epochs': [5, 10],
    'reg_pu': [0.01, 0.02, 0.03, 0.04],
    'reg_qi': [0.01, 0.02, 0.03, 0.04],
    'reg_bu': [0.01, 0.02, 0.03, 0.04],
    'reg_bi': [0.01, 0.02, 0.03, 0.04]
}

# Run grid search
grid = GridSearchCV(NMF, param_grid, cv=5)
grid.fit(data)

# Best parameters for RMSE
print("Best parameters (RMSE):", grid.best_params['rmse'])

# Best parameters for MAE
print("Best parameters (MAE):", grid.best_params['mae'])
