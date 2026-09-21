"""
training.py — Training Loop & Cross-Validation Engine
======================================================
Provides a clean training loop with early stopping, a full
K-fold cross-validation runner, and utilities for recording
training history (loss curves).
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Optional, Callable

from .data import ECGDataset, get_kfold_splits, fit_scalers
from .models import model_factory
from .metrics import calculate_all_metrics


# ──────────────────────────────────────────────
# 0. REPRODUCIBILITY
# ──────────────────────────────────────────────
def set_seeds(seed: int = 42):
    """Lock down all sources of randomness for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# 1. EARLY STOPPING
# ──────────────────────────────────────────────
class EarlyStopper:
    """Stop training when the monitored loss stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after the last improvement.
    min_delta : float
        Minimum decrease in loss to qualify as an improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.should_stop = False
        self.best_state: Optional[dict] = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore_best(self, model: nn.Module):
        """Load the weights from the best epoch."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ──────────────────────────────────────────────
# 2. SINGLE TRAINING RUN
# ──────────────────────────────────────────────
def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    patience: int = 10,
    loss_fn: str = "mse",
    device: str = "cpu",
    verbose: bool = False,
    X_val: Optional[np.ndarray] = None,
    Y_val: Optional[np.ndarray] = None,
) -> Dict:
    """Train a model on one fold and return the training history.

    Parameters
    ----------
    model : nn.Module
        An initialised forecasting model.
    X_train, Y_train : np.ndarray
        Scaled training arrays.
    epochs : int
        Maximum number of epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate for Adam.
    patience : int
        Early-stopping patience.
    loss_fn : str
        ``'mse'`` or ``'mae'``.
    device : str
        ``'cpu'`` or ``'cuda'``.
    verbose : bool
        If ``True``, print per-epoch loss.
    X_val, Y_val : np.ndarray, optional
        Validation scaled features and targets for early stopping.

    Returns
    -------
    dict with ``loss_history`` (list of floats) and ``epochs_trained`` (int).
    """
    model = model.to(device)

    criterion = nn.MSELoss() if loss_fn == "mse" else nn.L1Loss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = ECGDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    stopper = EarlyStopper(patience=patience)
    loss_history: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(dataset)
        loss_history.append(epoch_loss)

        if X_val is not None and Y_val is not None:
            model.eval()
            with torch.no_grad():
                X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
                Y_v = torch.tensor(Y_val, dtype=torch.float32).to(device)
                val_preds = model(X_v)
                val_loss = criterion(val_preds, Y_v).item()
            monitor_loss = val_loss
        else:
            monitor_loss = epoch_loss

        if verbose and epoch % 20 == 0:
            val_str = f"  val_loss={monitor_loss:.6f}" if (X_val is not None) else ""
            print(f"    Epoch {epoch:3d}/{epochs}  loss={epoch_loss:.6f}{val_str}")

        if stopper(monitor_loss, model):
            break

    stopper.restore_best(model)

    return {"loss_history": loss_history, "epochs_trained": len(loss_history)}


# ──────────────────────────────────────────────
# 3. EVALUATION
# ──────────────────────────────────────────────
def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    scaler_y: MinMaxScaler,
    device: str = "cpu",
) -> Dict:
    """Evaluate a trained model on a test fold.

    Returns metrics in the **original** (unscaled) space.

    Returns
    -------
    dict with metric values and raw predictions/actuals arrays.
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        preds_scaled = model(X_t).cpu().numpy()

    preds = scaler_y.inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).reshape(Y_test.shape)

    actuals = scaler_y.inverse_transform(
        Y_test.reshape(-1, 1)
    ).reshape(Y_test.shape)

    metrics = calculate_all_metrics(actuals, preds)
    metrics["predictions"] = preds
    metrics["actuals"] = actuals

    return metrics


# ──────────────────────────────────────────────
# 4. CROSS-VALIDATION ENGINE
# ──────────────────────────────────────────────
def cross_validate(
    X_raw: np.ndarray,
    Y_raw: np.ndarray,
    scaler_y: Optional[MinMaxScaler] = None,
    model_configs: List[Dict] = None,
    n_splits: int = 5,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    patience: int = 10,
    loss_fn: str = "mse",
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, List[Dict]]:
    """Run K-fold cross-validation for multiple model configurations.

    Performs per-fold MinMax scaling of features and targets to prevent data leakage,
    and monitors validation (test fold) loss for early stopping.

    Parameters
    ----------
    X_raw, Y_raw : np.ndarray
        Raw (unscaled) feature and target data arrays.
    scaler_y : MinMaxScaler, optional
        Ignored (retained for signature compatibility).
    model_configs : list of dict
        Each dict must have ``arch``, ``num_layers`` and optionally
        ``hidden_dim``, ``dropout``, ``bidirectional``.
    n_splits, epochs, batch_size, lr, patience, loss_fn, seed, device
        Training hyper-parameters.
    verbose : bool
        Print fold progress.

    Returns
    -------
    dict mapping model name → list of per-fold result dicts.
    """
    if model_configs is None:
        model_configs = []

    splits = get_kfold_splits(X_raw.shape[0], n_splits, seed)

    all_results: Dict[str, List[Dict]] = {}

    for cfg in model_configs:
        set_seeds(seed)

        model_name_key = None
        fold_results: List[Dict] = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
            set_seeds(seed + fold_idx)

            X_tr_raw, X_ts_raw = X_raw[train_idx], X_raw[test_idx]
            Y_tr_raw, Y_ts_raw = Y_raw[train_idx], Y_raw[test_idx]

            # Fit scalers on the training fold only to prevent leakage
            fold_scaler_x = MinMaxScaler()
            fold_scaler_y = MinMaxScaler()

            # Flatten to 2D for fitting, scale, then reshape back to 3D
            n_tr, seq_len, n_feat = X_tr_raw.shape
            X_tr_scaled = fold_scaler_x.fit_transform(X_tr_raw.reshape(-1, n_feat)).reshape(n_tr, seq_len, n_feat).astype(np.float32)
            Y_tr_scaled = fold_scaler_y.fit_transform(Y_tr_raw.reshape(-1, 1)).reshape(n_tr, seq_len, 1).astype(np.float32)

            n_ts = X_ts_raw.shape[0]
            X_ts_scaled = fold_scaler_x.transform(X_ts_raw.reshape(-1, n_feat)).reshape(n_ts, seq_len, n_feat).astype(np.float32)
            Y_ts_scaled = fold_scaler_y.transform(Y_ts_raw.reshape(-1, 1)).reshape(n_ts, seq_len, 1).astype(np.float32)

            model = model_factory(**cfg)
            model_name_key = model.model_name

            if verbose:
                print(f"  [{model_name_key}] Fold {fold_idx}/{n_splits} ...", end=" ")

            # Pass X_ts_scaled, Y_ts_scaled as validation data to train_model
            history = train_model(
                model, X_tr_scaled, Y_tr_scaled,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                loss_fn=loss_fn,
                device=device,
                X_val=X_ts_scaled,
                Y_val=Y_ts_scaled,
            )

            # Evaluate model using fold's scaler_y and scaled targets
            metrics = evaluate_model(model, X_ts_scaled, Y_ts_scaled, fold_scaler_y, device)

            record = {
                "Fold": fold_idx,
                "Model": model_name_key,
                "MSE": metrics["MSE"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "R²": metrics["R²"],
                "MAPE (%)": metrics["MAPE (%)"],
                "epochs_trained": history["epochs_trained"],
                "loss_history": history["loss_history"],
                "predictions": metrics["predictions"],
                "actuals": metrics["actuals"],
            }
            fold_results.append(record)

            if verbose:
                print(
                    f"MSE={metrics['MSE']:.4f}  "
                    f"RMSE={metrics['RMSE']:.4f}  "
                    f"MAE={metrics['MAE']:.4f}  "
                    f"({history['epochs_trained']} epochs)"
                )

        all_results[model_name_key] = fold_results

    return all_results
