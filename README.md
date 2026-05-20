# Linear Regression from Scratch (Closed Form + Gradient Descent)

## Overview
This project implements **Linear Regression from scratch in Python**, including both:

- Closed-form solution (Normal Equation)
- Gradient Descent optimization

It also includes comparisons with `scikit-learn` implementations and experiments on real-world datasets.

The goal is to deeply understand how linear regression works mathematically and computationally.

---

## Implemented Methods

### 1. Closed-Form Linear Regression (Normal Equation)
Implemented using the analytical solution:

\[
\theta = (X^T X)^{-1} X^T y
\]

Features:
- Manual bias term handling
- Matrix-based solution using NumPy
- Direct computation without iteration

---

### 2. Gradient Descent Linear Regression
Implemented from scratch using iterative optimization:

- Mean Squared Error loss
- Manual gradient computation
- Parameter updates for slope and intercept
- Learning rate tuning and convergence tracking

---

## Data Preprocessing

### Datasets used:
- Salary dataset (`Salary_Data.csv`)
- Startup dataset (`50_Startups.csv`)
- California Housing dataset (`sklearn`)

### Feature engineering:
- Manual one-hot encoding for categorical variables (`State`)
- Bias term handling in closed-form regression

---

## Evaluation

Models are evaluated using:

- R² Score (`sklearn.metrics.r2_score`)
- Visual regression plots (Matplotlib)
- Comparison against `scikit-learn.LinearRegression`

---

## Key Features

- Linear Regression implemented from scratch
- Gradient descent optimization with manual updates
- Closed-form solution using matrix algebra
- Manual categorical encoding (one-hot encoding)
- Visualization of regression fit
- Comparison with standard ML library implementations

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn (for comparison only)


