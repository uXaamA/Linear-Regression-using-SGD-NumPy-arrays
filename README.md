# Linear Regression using Stochastic Gradient Descent (SGD) — California Housing

This project implements a **Linear Regression model from scratch using NumPy** and trains it via **Stochastic Gradient Descent (SGD)** on the California Housing Dataset.

---

## Project Overview

**Goal:** Predict the median house value using 8 numerical features such as income, house age, rooms, etc.

### 💥 Problem Faced: Exploding Loss

Initially, the training loss exploded to values like `4.18 × 10^80` and later became `NaN`. After deep debugging, the root cause was found to be:

> ❗ **Missing gradient normalization by batch size** during weight updates.

---

## ✅ Final Fix

```python
# Incorrect gradient (causes explosion)
grad = X_batch @ error

# Corrected version (divided by batch size)
grad = (X_batch @ error) / len(y_batch)
