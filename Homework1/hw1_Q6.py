import pandas as pd

import torch

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Load ALE dataset
df = pd.read_csv("mcs_ds_edited_iter_shuffled.csv")

# First four variables as predictors
X = df.iloc[:, :4].values
y = df["ale"].values

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Baseline linear regression
baseline = LinearRegression()
baseline.fit(X_train, y_train)

y_pred = baseline.predict(X_test)
baseline_mae = mean_absolute_error(y_test, y_pred)

print("Baseline Linear Regression MAE:", baseline_mae)


# Convert data to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# Parameters
w = torch.zeros(X_train.shape[1], requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# SGD optimizer
optimizer = torch.optim.SGD([w, b], lr=0.01)

# Training
for epoch in range(5000):
    optimizer.zero_grad()

    # Linear regression
    y_pred_train = X_train_t @ w + b

    # Mean squared error
    loss = torch.mean((y_train_t - y_pred_train) ** 2)

    # Automatic differentiation
    loss.backward()
    optimizer.step()

# Test
with torch.no_grad():
    y_pred_test = X_test_t @ w + b

    mae = torch.mean(
        torch.abs(y_test_t - y_pred_test)
    )

print("Custom Linear Regression MAE:", mae.item())

# Student-t Regression
w_t = torch.zeros(X_train.shape[1], requires_grad=True)
b_t = torch.zeros(1, requires_grad=True)

optimizer_t = torch.optim.SGD([w_t, b_t], lr=0.01)

for epoch in range(5000):
    optimizer_t.zero_grad()

    y_pred_t = X_train_t @ w_t + b_t
    residuals = y_train_t - y_pred_t

    # Student-t distribution with nu = 5
    dist = torch.distributions.StudentT(df=5.0)

    # Negative log-likelihood
    loss_t = -dist.log_prob(residuals).mean()

    loss_t.backward()
    optimizer_t.step()

# Test
with torch.no_grad():
    y_pred_test_t = X_test_t @ w_t + b_t

    mae_t = torch.mean(
        torch.abs(y_test_t - y_pred_test_t)
    )

print("Student-t Regression MAE:", mae_t.item())