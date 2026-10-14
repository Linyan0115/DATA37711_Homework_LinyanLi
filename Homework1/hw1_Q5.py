
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from scipy import stats
from scipy.stats import normaltest

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Question 5

# Load data
data = load_breast_cancer()
X = data.data
y = data.target

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Baseline logistic regression
baseline = LogisticRegression(max_iter=5000)
baseline.fit(X_train, y_train)

y_pred = baseline.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred)

print("Baseline Logistic Regression Accuracy:", baseline_acc)


# Convert data to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# Parameters
w = torch.zeros(X_train.shape[1], requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# SGD optimizer
optimizer = torch.optim.SGD([w, b], lr=0.01)

# Training
for epoch in range(5000):
    optimizer.zero_grad()

    # Logistic regression
    prob = torch.sigmoid(X_train_t @ w + b)

    # Clip probabilities for numerical stability
    prob = torch.clamp(prob, 1e-7, 1 - 1e-7)

    # Binary cross-entropy loss
    loss = -torch.mean(
        y_train_t * torch.log(prob)
        + (1 - y_train_t) * torch.log(1 - prob)
    )

    # Automatic differentiation
    loss.backward()

    # SGD update
    optimizer.step()

# Test predictions
with torch.no_grad():
    test_prob = torch.sigmoid(X_test_t @ w + b)
    test_pred = (test_prob >= 0.5).float()

    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    accuracy = (test_pred == y_test_t).float().mean()

print("Custom Logistic Regression Accuracy:", accuracy.item())

# Probit Regression
w_probit = torch.zeros(X_train.shape[1], requires_grad=True)
b_probit = torch.zeros(1, requires_grad=True)

optimizer_probit = torch.optim.SGD(
    [w_probit, b_probit], lr=0.01
)

normal = torch.distributions.Normal(0, 1)

for epoch in range(5000):
    optimizer_probit.zero_grad()

    z = X_train_t @ w_probit + b_probit
    prob = normal.cdf(z)
    prob = torch.clamp(prob, 1e-7, 1 - 1e-7)

    loss = -torch.mean(
        y_train_t * torch.log(prob)
        + (1 - y_train_t) * torch.log(1 - prob)
    )

    loss.backward()
    optimizer_probit.step()

# Test accuracy
with torch.no_grad():
    z_test = X_test_t @ w_probit + b_probit
    test_prob = normal.cdf(z_test)
    test_pred = (test_prob >= 0.5).float()

    accuracy_probit = (
        test_pred == y_test_t
    ).float().mean()

print("Probit Regression Accuracy:", accuracy_probit.item())