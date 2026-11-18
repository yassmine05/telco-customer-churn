"""Metric helpers shared by train.py and evaluate.py."""
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config


def classification_metrics(y_true, proba, threshold: float) -> dict:
    """Precision / recall on churners at `threshold`, plus threshold-free AUCs."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def expected_cost(y_true, proba, threshold: float) -> float:
    """Total cost of acting at `threshold` under the assumptions in config.py."""
    pred = np.asarray(proba) >= threshold
    y = np.asarray(y_true).astype(bool)
    fn = np.sum(y & ~pred)
    fp = np.sum(~y & pred)
    return config.COST_FALSE_NEGATIVE * fn + config.COST_FALSE_POSITIVE * fp


def cost_curve(y_true, proba, grid=None):
    """Cost at every candidate threshold. Returns (thresholds, costs)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    return grid, np.array([expected_cost(y_true, proba, t) for t in grid])


def best_cost_threshold(y_true, proba) -> float:
    """Threshold with the lowest expected cost."""
    grid, costs = cost_curve(y_true, proba)
    return float(grid[int(np.argmin(costs))])


def bayes_threshold() -> float:
    """Cost-optimal threshold for a perfectly calibrated probability p.

    Flagging costs (1 - p) * C_FP in expectation, not flagging costs p * C_FN,
    so flag when p > C_FP / (C_FP + C_FN).
    """
    return config.COST_FALSE_POSITIVE / (config.COST_FALSE_POSITIVE
                                         + config.COST_FALSE_NEGATIVE)


def brier_score(y_true, proba) -> float:
    """Mean squared error between predicted probability and the 0/1 outcome.

    0 is perfect; always predicting the churn rate r scores r * (1 - r).
    """
    return float(brier_score_loss(np.asarray(y_true), np.asarray(proba)))


def calibration_bins(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Reliability-diagram points: mean predicted probability vs observed churn
    rate in `n_bins` equal-count (quantile) bins.

    Bins whose edges coincide, which happens when isotonic calibration produces
    tied scores, are merged, so fewer than `n_bins` rows can come back.
    """
    observed, predicted = calibration_curve(np.asarray(y_true), np.asarray(proba),
                                            n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": predicted, "observed_rate": observed})


def capture_at_top(y_true, proba, fraction: float) -> dict:
    """If we contact only the riskiest `fraction` of customers, how many of all
    churners do we reach (capture), and how much better is that than random (lift)?"""
    y = np.asarray(y_true)
    order = np.argsort(-np.asarray(proba), kind="stable")
    k = int(round(len(y) * fraction))
    hits = int(y[order][:k].sum())
    capture = hits / y.sum()
    return {"contacted": k, "churners_found": hits, "capture": capture,
            "precision_in_group": hits / k, "lift": capture / fraction}


def log_metrics(rows: list[dict]):
    """Append rows to results/metrics.csv (one row per model per evaluation)."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([{"run_time": stamp, "seed": config.SEED, **r} for r in rows])
    path = config.METRICS_PATH
    if path.exists():
        columns = list(pd.read_csv(path, nrows=0).columns)
        new = [c for c in df.columns if c not in columns]
        if new:
            # A metric column was added since the file was written: rewrite it
            # with the extra columns (blank for older rows) so rows stay aligned.
            old = pd.read_csv(path, dtype=str, keep_default_na=False)
            columns += new
            old.reindex(columns=columns, fill_value="").to_csv(path, index=False)
        df = df.reindex(columns=columns)
    df.to_csv(path, mode="a", index=False, header=not path.exists(), float_format="%.4f")
    print(f"  logged {len(df)} row(s) to {config.METRICS_PATH.relative_to(config.ROOT)}")
