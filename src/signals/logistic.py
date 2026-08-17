"""Train-only logistic classifier for S-2 ({-1, 0, +1})."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def logistic_positions(
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Fit on train rows with finite features+labels; predict the full series.

    Returns 0 everywhere if train has <2 classes or too few rows.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    train = np.asarray(train_mask, dtype=bool)
    n = len(y)
    pos = np.zeros(n, dtype=np.float64)
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
    fit_idx = np.where(train & finite)[0]
    if fit_idx.size < 50:
        return pos
    y_fit = y[fit_idx]
    classes = np.unique(y_fit)
    if classes.size < 2:
        return pos
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x[fit_idx])
    clf = LogisticRegression(
        max_iter=400,
        solver="lbfgs",
        class_weight="balanced",
        random_state=int(seed),
    )
    try:
        clf.fit(x_fit, y_fit)
    except ValueError:
        return pos
    pred_idx = np.where(finite)[0]
    if pred_idx.size == 0:
        return pos
    x_pred = scaler.transform(x[pred_idx])
    pred = clf.predict(x_pred)
    pos[pred_idx] = pred.astype(np.float64)
    return pos
