"""Completed teacher-style HW2 Q3 models; settings match the earlier notebook."""
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


class RandomKitchenSinks:
    def __init__(self, D=1000, sigma=1.0, ridge_lambda=1e-3,
                 random_state=42):
        if D <= 0 or D % 2 or sigma <= 0 or ridge_lambda <= 0:
            raise ValueError("D must be positive and even; sigma and lambda positive.")
        self.D = D
        self.sigma = sigma
        self.ridge_lambda = ridge_lambda
        self.random_state = random_state

    def _features(self, X):
        angles = np.asarray(X) @ self.W + self.b
        return np.stack([np.sin(angles), np.cos(angles)], axis=-1).reshape(len(X), self.D)

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.W = rng.normal(0.0, 1.0 / self.sigma, size=(X.shape[1], self.D // 2))
        self.b = rng.uniform(0.0, 2 * np.pi, size=self.D // 2)
        phi = self._features(X)
        # Mean squared error + lambda * ||beta||^2; intercept is unpenalized.
        self.ridge_ = Ridge(alpha=len(X) * self.ridge_lambda, fit_intercept=True)
        self.ridge_.fit(phi, y)
        self.coef_ = self.ridge_.coef_
        self.intercept_ = self.ridge_.intercept_
        return self

    def predict(self, X):
        phi = self._features(X)
        return phi @ self.coef_ + self.intercept_


def sweep_rks_model(sweep_values, X_train, y_train, X_val, y_val,
                    parameter="D", D=512, sigma=1.0, ridge_lambda=1e-3,
                    random_state=42):
    """Teacher-style sweep, extended to vary D, sigma, or ridge_lambda.

    Returns (best_model, best_validation_mse, best_parameter_value).
    The full table is stored as best_model.sweep_results_.
    """
    if parameter not in {"D", "sigma", "ridge_lambda"}:
        raise ValueError("Unknown sweep parameter.")
    records = []
    best_model, best_val_mse, best_value = None, float("inf"), None
    for value in sweep_values:
        params = dict(D=D, sigma=sigma, ridge_lambda=ridge_lambda,
                      random_state=random_state)
        params[parameter] = value
        model = RandomKitchenSinks(**params).fit(X_train, y_train)
        train_mse = mean_squared_error(y_train, model.predict(X_train))
        val_mse = mean_squared_error(y_val, model.predict(X_val))
        records.append({parameter: value, "Train MSE": train_mse,
                        "Validation MSE": val_mse})
        if val_mse < best_val_mse:
            best_model, best_val_mse, best_value = model, val_mse, value
    if best_model is None:
        raise ValueError("Sweep values cannot be empty.")
    best_model.sweep_results_ = pd.DataFrame(records)
    return best_model, best_val_mse, best_value


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden=128, d_out=1, lr=1e-3,
                 batch_size=128, activation="relu", device="cpu", seed=42):
        super().__init__()
        if activation != "relu" or d_out != 1:
            raise ValueError("This implementation uses ReLU and one regression output.")
        torch.manual_seed(seed)
        self.device = device
        self.lr, self.batch_size, self.seed = lr, batch_size, seed
        self.d_hidden = d_hidden
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(),
                                 nn.Linear(d_hidden, d_out))
        self.to(device)

    def forward(self, X):
        return self.net(X)

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            return self.forward(X).cpu().numpy().reshape(-1)

    def fit(self, X, y, X_val, y_val, num_epochs=3000, patience=200):
        """Validation arguments extend the skeleton to support early stopping."""
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1)
        Xv = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
        yv = torch.as_tensor(y_val, dtype=torch.float32, device=self.device).reshape(-1, 1)
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size,
                            shuffle=True, generator=generator)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        best_loss, best_state, wait = float("inf"), None, 0
        for epoch in range(1, num_epochs + 1):
            self.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(self(xb), yb)
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Nonfinite loss: lower the learning rate.")
                loss.backward()
                optimizer.step()
            self.eval()
            with torch.no_grad():
                val_loss = loss_fn(self(Xv), yv).item()
            if not np.isfinite(val_loss):
                raise RuntimeError("Nonfinite validation loss.")
            if val_loss < best_loss:
                best_loss, best_state = val_loss, copy.deepcopy(self.state_dict())
                self.best_epoch_, wait = epoch, 0
            else:
                wait += 1
            if wait >= patience:
                break
        self.epochs_trained_ = epoch
        self.load_state_dict(best_state)
        self.eval()
        with torch.no_grad():
            self.train_mse_ = loss_fn(self(X), y).item()
            self.val_mse_ = loss_fn(self(Xv), yv).item()
        return self


def mlp_sweep(sweep_values, X_train, y_train, X_val, y_val,
              lr=0.01, batch_size=64, num_epochs=3000,
              patience=200, seed=42, device="cpu"):
    """Vary hidden width only; return the best model, loss, and width."""
    records = []
    best_model, best_loss, best_width = None, float("inf"), None
    for width in sweep_values:
        model = MLP(X_train.shape[1], d_hidden=width, lr=lr,
                    batch_size=batch_size, seed=seed, device=device)
        model.fit(X_train, y_train, X_val, y_val, num_epochs, patience)
        records.append({"Hidden width": width, "Train MSE": model.train_mse_,
                        "Validation MSE": model.val_mse_,
                        "Best epoch": model.best_epoch_,
                        "Epochs trained": model.epochs_trained_})
        print(f"H={width}: train={model.train_mse_:.6f}, "
              f"validation={model.val_mse_:.6f}, best epoch={model.best_epoch_}")
        if model.val_mse_ < best_loss:
            best_model, best_loss, best_width = model, model.val_mse_, width
    if best_model is None:
        raise ValueError("Sweep values cannot be empty.")
    best_model.sweep_results_ = pd.DataFrame(records)
    return best_model, best_loss, best_width
