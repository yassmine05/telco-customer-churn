"""Metric helpers shared by train.py and evaluate.py."""
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)

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
    df.to_csv(config.METRICS_PATH, mode="a", index=False,
              header=not config.METRICS_PATH.exists(), float_format="%.4f")
    print(f"  logged {len(df)} row(s) to {config.METRICS_PATH.relative_to(config.ROOT)}")
