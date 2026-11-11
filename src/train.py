"""Train the baseline and the tuned XGBoost model.

Everything in this file uses the TRAINING set only:
  * preprocessing lives inside a scikit-learn Pipeline, so in cross-validation
    it is re-fitted on each training fold (no leakage into the validation fold);
  * hyperparameters are chosen by 5-fold stratified cross-validation;
  * the decision threshold is chosen from out-of-fold predictions.

The test set is not touched here. See evaluate.py.

Run:  python src/train.py
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import config
import metrics
import plotting
from data import load_train_test


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """One-hot encode categoricals; optionally standardise numeric columns.

    Two-level columns (Yes/No, Male/Female) become a single 0/1 column.
    """
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_FEATURES),
            ("bin", "passthrough", config.PASSTHROUGH_FEATURES),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                  sparse_output=False), config.CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def make_baseline() -> Pipeline:
    """Baseline: one-hot encoding + logistic regression."""
    return Pipeline([
        ("prep", make_preprocessor(scale_numeric=True)),
        ("model", LogisticRegression(max_iter=2000, random_state=config.SEED)),
    ])


def make_xgboost(scale_pos_weight: float, **params) -> Pipeline:
    """XGBoost with class re-weighting for the minority (churn) class."""
    return Pipeline([
        ("prep", make_preprocessor(scale_numeric=False)),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.SEED,
            n_jobs=1,          # parallelism happens across CV fits instead
            **params,
        )),
    ])


def plot_cost_curves(y, oof: dict, thresholds: dict):
    """Expected cost vs threshold for both models (out-of-fold, training set)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    colours = {"baseline_logreg": plotting.ORANGE, "xgboost": plotting.BLUE}
    labels = {"baseline_logreg": "Logistic regression", "xgboost": "XGBoost"}
    for name, proba in oof.items():
        grid, costs = metrics.cost_curve(y, proba)
        ax.plot(grid, costs, color=colours[name], label=labels[name])
        t = thresholds[name]
        c = metrics.expected_cost(y, proba, t)
        ax.scatter([t], [c], s=60, color=colours[name], edgecolor=plotting.SURFACE,
                   linewidth=2, zorder=3)
        ax.annotate(f"min at {t:.2f}", (t, c), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=9.5)
    ax.set_ylim(0, None)
    ax.axvline(0.5, color=plotting.GREY, ls="--", lw=1.2)
    ax.text(0.51, ax.get_ylim()[1] * 0.97, "default 0.50", color=plotting.INK_SOFT,
            fontsize=9, va="top")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel(f"Cost units ({config.COST_FALSE_NEGATIVE:g} per missed churner, "
                  f"{config.COST_FALSE_POSITIVE:g} per wasted offer)")
    ax.set_title("Expected cost by threshold (out-of-fold, training set)")
    ax.legend(loc="upper left")
    plotting.save(fig, "05_threshold_cost_curve.png")


def main():
    plotting.apply_style()
    X_train, _, y_train, _ = load_train_test()
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.SEED)

    # ---------------------------------------------------------------- baseline
    print("[1/3] Baseline: one-hot + logistic regression")
    baseline = make_baseline()
    oof_base = cross_val_predict(baseline, X_train, y_train, cv=cv,
                                 method="predict_proba")[:, 1]

    # ----------------------------------------------------------------- xgboost
    n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = n_neg / n_pos
    n_combos = int(np.prod([len(v) for v in config.XGB_PARAM_GRID.values()]))
    print(f"[2/3] XGBoost: scale_pos_weight = {n_neg}/{n_pos} = {spw:.3f}; "
          f"grid search over {n_combos} combinations x {config.CV_FOLDS} folds")
    search = GridSearchCV(
        make_xgboost(spw),
        config.XGB_PARAM_GRID,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
        refit="pr_auc",        # select on PR-AUC: it focuses on the churn class
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    print(f"      best params: {best_params}")

    # Keep the full search table so the tuning is auditable.
    cv_table = (pd.DataFrame(search.cv_results_)
                .filter(regex="^(param_|mean_test_|std_test_|rank_test_)")
                .sort_values("rank_test_pr_auc"))
    cv_table.columns = [c.replace("param_model__", "") for c in cv_table.columns]
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_table.to_csv(config.RESULTS_DIR / "xgb_cv_results.csv", index=False,
                    float_format="%.5f")

    # Out-of-fold predictions for the chosen configuration.
    xgb = make_xgboost(spw, **best_params)
    oof_xgb = cross_val_predict(xgb, X_train, y_train, cv=cv,
                                method="predict_proba")[:, 1]

    # ------------------------------------------------- cost-based thresholds
    print("[3/3] Choosing decision thresholds from out-of-fold predictions")
    oof = {"baseline_logreg": oof_base, "xgboost": oof_xgb}
    thresholds = {name: metrics.best_cost_threshold(y_train, p)
                  for name, p in oof.items()}
    plot_cost_curves(y_train, oof, thresholds)

    rows = []
    for name, proba in oof.items():
        for rule, t in (("default_0.5", 0.5), ("min_cost", thresholds[name])):
            m = metrics.classification_metrics(y_train, proba, t)
            rows.append({"model": name, "data": "train_cv_out_of_fold",
                         "threshold_rule": rule, **m,
                         "top10_capture": metrics.capture_at_top(y_train, proba, 0.10)["capture"],
                         "top20_capture": metrics.capture_at_top(y_train, proba, 0.20)["capture"],
                         "params": json.dumps(best_params) if name == "xgboost" else "{}"})
    metrics.log_metrics(rows)
    print(pd.DataFrame(rows)[["model", "threshold_rule", "threshold", "precision",
                              "recall", "roc_auc", "pr_auc"]].round(4).to_string(index=False))

    # ------------------------------------------- fit final models on all train
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline.fit(X_train, y_train), config.MODELS_DIR / "baseline_logreg.joblib")
    joblib.dump(xgb.fit(X_train, y_train), config.MODELS_DIR / "xgboost.joblib")
    meta = {
        "seed": config.SEED,
        "scale_pos_weight": spw,
        "xgb_best_params": best_params,
        "thresholds": thresholds,
        "cost_false_negative": config.COST_FALSE_NEGATIVE,
        "cost_false_positive": config.COST_FALSE_POSITIVE,
        "cv_roc_auc": {r["model"]: r["roc_auc"] for r in rows},
        "cv_pr_auc": {r["model"]: r["pr_auc"] for r in rows},
    }
    (config.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("  saved models/ (baseline_logreg.joblib, xgboost.joblib, metadata.json)")


if __name__ == "__main__":
    main()
