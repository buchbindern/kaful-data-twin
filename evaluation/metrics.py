"""
Prognostic metrics (M8) — pure functions over prediction/truth arrays.

Two families:
  * generic error/coverage (rmse, mae, coverage) — used for BOTH wear and RUL.
  * RUL-specific prognostic metrics (alpha_lambda_accuracy, prognostic_horizon)
    that judge how well and how EARLY the RUL estimate locks onto the truth.

None of these know where "truth" comes from; the caller decides whether truth is
observed (wear labels) or extrapolated (RUL). That separation is what lets the
validation report keep the trustworthy and the flagged metrics cleanly apart.
"""

from __future__ import annotations

import numpy as np


def rmse(pred, true) -> float:
    pred = np.asarray(pred, float); true = np.asarray(true, float)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred, true) -> float:
    pred = np.asarray(pred, float); true = np.asarray(true, float)
    return float(np.mean(np.abs(pred - true)))


def coverage(true, lower, upper) -> float:
    """Fraction of points whose true value falls inside [lower, upper]."""
    true = np.asarray(true, float); lower = np.asarray(lower, float); upper = np.asarray(upper, float)
    return float(np.mean((true >= lower) & (true <= upper)))


def alpha_lambda_accuracy(pred_rul, true_rul, alpha: float = 0.2) -> float:
    """Fraction of predictions within an alpha-cone: [(1-a)*true, (1+a)*true]."""
    pred = np.asarray(pred_rul, float); true = np.asarray(true_rul, float)
    lo = (1 - alpha) * true
    hi = (1 + alpha) * true
    return float(np.mean((pred >= lo) & (pred <= hi)))


def prognostic_horizon(cuts, pred_rul, true_rul, alpha: float = 0.2):
    """Earliest cut from which ALL later predictions stay in the alpha-cone.
    Returns that cut (or None if it never locks on)."""
    cuts = np.asarray(cuts); pred = np.asarray(pred_rul, float); true = np.asarray(true_rul, float)
    in_band = (pred >= (1 - alpha) * true) & (pred <= (1 + alpha) * true)
    for i in range(len(cuts)):
        if in_band[i:].all():
            return int(cuts[i])
    return None
