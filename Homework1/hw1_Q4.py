# Question 4 - ALE Residual Normality

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Load the dataset
df = pd.read_csv("mcs_ds_edited_iter_shuffled.csv")

# Check the dataset
print(df.head())
print(df.columns)

# -------------------------
# Linear Regression
# -------------------------

# First 4 variables as predictors
X = df.iloc[:, :4]

# Target variable
y = df["ale"]

# Fit linear regression
linear_model = LinearRegression()
linear_model.fit(X, y)

# Predictions
y_pred = linear_model.predict(X)

# Residuals
residuals = y - y_pred

# Normality test
statistic, p_value = normaltest(residuals)

print("\nLinear Regression")
print("Normality test statistic:", statistic)
print("p-value:", p_value)

# QQ plot
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot - Linear Regression Residuals")
plt.show()



# -------------------------
# Support Vector Regression
# -------------------------

svr_model = SVR()
svr_model.fit(X, y)

# Predictions
y_pred_svr = svr_model.predict(X)

# Residuals
residuals_svr = y - y_pred_svr

# Normality test
statistic_svr, p_value_svr = normaltest(residuals_svr)

print("\nSupport Vector Regression")
print("Normality test statistic:", statistic_svr)
print("p-value:", p_value_svr)

# QQ plot
stats.probplot(residuals_svr, dist="norm", plot=plt)
plt.title("QQ Plot - SVR Residuals")
plt.show()

# -------------------------
# Decision Tree Regression with max_depth
# -------------------------

tree_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

tree_model.fit(X, y)

y_pred_tree = tree_model.predict(X)
residuals_tree = y - y_pred_tree

statistic_tree, p_value_tree = normaltest(residuals_tree)

print("\nDecision Tree Regression")
print("Normality test statistic:", statistic_tree)
print("p-value:", p_value_tree)

stats.probplot(residuals_tree, dist="norm", plot=plt)
plt.title("QQ Plot - Decision Tree Residuals")
plt.show()


# -------------------------
# MLP Neural Network
# -------------------------

mlp_model = MLPRegressor(
    hidden_layer_sizes=(50,),
    max_iter=5000,
    random_state=42
)

mlp_model.fit(X, y)

# Predictions
y_pred_mlp = mlp_model.predict(X)

# Residuals
residuals_mlp = y - y_pred_mlp

# Normality test
statistic_mlp, p_value_mlp = normaltest(residuals_mlp)

print("\nMLP Neural Network")
print("Normality test statistic:", statistic_mlp)
print("p-value:", p_value_mlp)

# QQ plot
stats.probplot(residuals_mlp, dist="norm", plot=plt)
plt.title("QQ Plot - MLP Neural Network Residuals")
plt.show()



