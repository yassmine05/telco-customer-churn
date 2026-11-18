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


def qini_curve(churn, treatment, score):
    """Qini curve for a retention campaign: incremental customers retained when
    the customers with the highest `score` are contacted first.

    After contacting the top k customers, with N_T / N_C treated / control
    customers and C_T / C_C churners among them, the Qini value is

        Q(k) = C_C * N_T / N_C - C_T

    i.e. the churners we would expect among the treated customers had they
    behaved like the controls, minus the churners actually seen. Q(k) = 0 while
    no control customer has been reached. Ties in `score` keep the input order.

    Returns (fraction contacted, Q), both of length n + 1, starting at (0, 0).
    """
    churn = np.asarray(churn, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    t = treatment[order]
    c = churn[order]
    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    churn_t = np.cumsum(c * t)
    churn_c = np.cumsum(c * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(n_c > 0, churn_c * n_t / n_c, 0.0) - churn_t
    fraction = np.arange(len(churn) + 1) / len(churn)
    return fraction, np.concatenate([[0.0], q])


def qini_coefficient(churn, treatment, score) -> float:
    """Area between the Qini curve and the random-targeting line, per customer.

    Both axes are scaled to [0, 1] of the population (x: share contacted,
    y: incremental customers retained / n), so the value does not grow with n.
    The random line runs straight from (0, 0) to the curve's end point. A
    random ordering scores about 0; a useful ordering scores above 0.
    """
    fraction, q = qini_curve(churn, treatment, score)
    gain = q / len(q[1:])
    area = float(np.sum((gain[1:] + gain[:-1]) / 2 * np.diff(fraction)))
    return area - gain[-1] / 2


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
