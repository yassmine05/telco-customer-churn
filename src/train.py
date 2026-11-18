"""Train the baseline and the tuned XGBoost model.

Everything in this file uses the TRAINING set only:
  * preprocessing lives inside a scikit-learn Pipeline, so in cross-validation
    it is re-fitted on each training fold (no leakage into the validation fold);
  * hyperparameters are chosen by 5-fold stratified cross-validation;
  * the decision threshold is chosen from out-of-fold predictions;
  * the tuned XGBoost model is calibrated (isotonic or sigmoid, chosen by
    out-of-fold Brier score), because scale_pos_weight inflates its scores;
  * nested cross-validation (an outer loop around the whole tuning procedure)
    gives an unbiased performance estimate. It only reports a score: the final
    model and threshold are chosen exactly as above.

The test set is not touched here. See evaluate.py.

Run:  python src/train.py
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
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


CALIBRATION_METHODS = ("isotonic", "sigmoid")


def make_calibrated(estimator, method: str, cv) -> CalibratedClassifierCV:
    """Wrap `estimator` in a calibrator fitted by internal cross-validation.

    For each of the cv folds, a copy of the pipeline is fitted on the other
    folds and the score -> probability mapping is learnt on the held-out fold.
    predict_proba averages the calibrated copies.
    """
    return CalibratedClassifierCV(estimator, method=method, cv=cv)


def compare_calibration(estimator, X, y, cv) -> tuple[str, dict, dict]:
    """Out-of-fold comparison of the calibration methods on the training set.

    The outer cross_val_predict and the calibrator's inner cv use the same
    seeded splitter, so a calibrator never sees the rows it is scored on.
    Lower Brier score wins; log loss breaks a tie and is reported alongside.
    Returns (chosen method, scores per method, out-of-fold probas per method).
    """
    oof, scores = {}, {}
    for method in CALIBRATION_METHODS:
        proba = cross_val_predict(make_calibrated(estimator, method, cv), X, y,
                                  cv=cv, method="predict_proba")[:, 1]
        oof[method] = proba
        scores[method] = {"brier": metrics.brier_score(y, proba),
                          "log_loss": float(log_loss(y, proba))}
    chosen = min(CALIBRATION_METHODS,
                 key=lambda m: (scores[m]["brier"], scores[m]["log_loss"]))
    return chosen, scores, oof


def make_search(scale_pos_weight: float, cv) -> GridSearchCV:
    """The XGBoost hyperparameter search, used both for the final model and
    inside each outer fold of the nested cross-validation."""
    return GridSearchCV(
        make_xgboost(scale_pos_weight),
        config.XGB_PARAM_GRID,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
        refit="pr_auc",        # select on PR-AUC: it focuses on the churn class
        cv=cv,
        n_jobs=-1,
    )


def nested_cv(X, y) -> pd.DataFrame:
    """Nested cross-validation on the training set.

    For each outer fold, the complete tuning procedure (scale_pos_weight +
    inner grid search) runs on the outer-training part only, and the refitted
    best model is scored on the outer-validation part, which the search never
    sees. The baseline has no hyperparameters, so it is simply fitted on the
    outer-training part and scored on the same outer-validation part.

    The outer loop is sequential; only the inner GridSearchCV runs in parallel.
    Returns one row per outer fold per model.
    """
    outer = StratifiedKFold(n_splits=config.OUTER_CV_FOLDS, shuffle=True,
                            random_state=config.SEED + 1)
    inner = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                            random_state=config.SEED)
    rows = []
    for fold, (tr, va) in enumerate(outer.split(X, y), start=1):
        X_tr, X_va, y_tr, y_va = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
        spw = int((y_tr == 0).sum()) / int((y_tr == 1).sum())
        search = make_search(spw, inner).fit(X_tr, y_tr)
        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        fitted = {"baseline_logreg": make_baseline().fit(X_tr, y_tr),
                  "xgboost": search.best_estimator_}
        for name, model in fitted.items():
            proba = model.predict_proba(X_va)[:, 1]
            rows.append({"fold": fold, "model": name,
                         "roc_auc": roc_auc_score(y_va, proba),
                         "pr_auc": average_precision_score(y_va, proba),
                         "best_params": json.dumps(params) if name == "xgboost" else "{}"})
        print(f"      outer fold {fold}/{config.OUTER_CV_FOLDS}: XGBoost ROC-AUC "
              f"{rows[-1]['roc_auc']:.4f}, PR-AUC {rows[-1]['pr_auc']:.4f}, "
              f"baseline ROC-AUC {rows[-2]['roc_auc']:.4f}, PR-AUC {rows[-2]['pr_auc']:.4f}; "
              f"params {params}", flush=True)
    return pd.DataFrame(rows)


def summarise_nested_cv(table: pd.DataFrame) -> dict:
    """Mean and sample standard deviation (ddof=1) per model, plus params per fold."""
    summary = {"outer_folds": config.OUTER_CV_FOLDS, "inner_folds": config.CV_FOLDS}
    for name, group in table.groupby("model"):
        summary[name] = {f"{m}_{stat}": float(getattr(group[m], stat)())
                         for m in ("roc_auc", "pr_auc") for stat in ("mean", "std")}
    xgb = table[table["model"] == "xgboost"]
    summary["xgb_params_by_fold"] = [json.loads(p) for p in xgb["best_params"]]
    return summary


def plot_cost_curves(y, oof: dict, thresholds: dict):
    """Expected cost vs threshold for both models (out-of-fold, training set)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    colours = {"baseline_logreg": plotting.ORANGE, "xgboost": plotting.BLUE,
               "xgboost_calibrated": plotting.GREEN}
    labels = {"baseline_logreg": "Logistic regression", "xgboost": "XGBoost (raw score)",
              "xgboost_calibrated": "XGBoost (calibrated)"}
    for i, (name, proba) in enumerate(oof.items()):
        grid, costs = metrics.cost_curve(y, proba)
        ax.plot(grid, costs, color=colours[name], label=labels[name])
        t = thresholds[name]
        c = metrics.expected_cost(y, proba, t)
        ax.scatter([t], [c], s=60, color=colours[name], edgecolor=plotting.SURFACE,
                   linewidth=2, zorder=3)
        # Stagger the labels: two models can share the same minimum.
        ax.annotate(f"min at {t:.2f}", (t, c), textcoords="offset points",
                    xytext=(0, -16 - 14 * i), ha="center", fontsize=9.5,
                    color=colours[name])
    ax.set_ylim(0, None)
    bayes = metrics.bayes_threshold()
    ax.axvline(bayes, color=plotting.GREEN, ls=":", lw=1.2)
    ax.text(bayes + 0.01, ax.get_ylim()[1] * 0.04, f"theory {bayes:.3f}",
            color=plotting.INK_SOFT, fontsize=9)
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
    print("[1/5] Baseline: one-hot + logistic regression")
    baseline = make_baseline()
    oof_base = cross_val_predict(baseline, X_train, y_train, cv=cv,
                                 method="predict_proba")[:, 1]

    # ----------------------------------------------------------------- xgboost
    n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = n_neg / n_pos
    n_combos = int(np.prod([len(v) for v in config.XGB_PARAM_GRID.values()]))
    print(f"[2/5] XGBoost: scale_pos_weight = {n_neg}/{n_pos} = {spw:.3f}; "
          f"grid search over {n_combos} combinations x {config.CV_FOLDS} folds")
    search = make_search(spw, cv)
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

    # ------------------------------------------------------------- calibration
    print(f"[3/5] Calibrating XGBoost ({', '.join(CALIBRATION_METHODS)}; "
          f"cv={config.CV_FOLDS} inside the calibrator, scored out-of-fold)")
    method, cal_scores, cal_oof = compare_calibration(xgb, X_train, y_train, cv)
    cal_scores = {"uncalibrated": {"brier": metrics.brier_score(y_train, oof_xgb),
                                   "log_loss": float(log_loss(y_train, oof_xgb))},
                  **cal_scores}
    for name, sc in cal_scores.items():
        print(f"      {name:>12}: Brier {sc['brier']:.4f}, log loss {sc['log_loss']:.4f}")
    print(f"      chosen: {method}")
    oof_cal = cal_oof[method]

    # ------------------------------------------------- cost-based thresholds
    print("[4/5] Choosing decision thresholds from out-of-fold predictions")
    oof = {"baseline_logreg": oof_base, "xgboost": oof_xgb, "xgboost_calibrated": oof_cal}
    thresholds = {name: metrics.best_cost_threshold(y_train, p)
                  for name, p in oof.items()}
    bayes = metrics.bayes_threshold()
    print(f"      calibrated XGBoost: empirical best threshold "
          f"{thresholds['xgboost_calibrated']:.2f}, theoretical "
          f"C_FP / (C_FP + C_FN) = {bayes:.3f}")
    plot_cost_curves(y_train, oof, thresholds)

    rows = []
    for name, proba in oof.items():
        for rule, t in (("default_0.5", 0.5), ("min_cost", thresholds[name])):
            m = metrics.classification_metrics(y_train, proba, t)
            rows.append({"model": name, "data": "train_cv_out_of_fold",
                         "threshold_rule": rule, **m,
                         "brier": metrics.brier_score(y_train, proba),
                         "top10_capture": metrics.capture_at_top(y_train, proba, 0.10)["capture"],
                         "top20_capture": metrics.capture_at_top(y_train, proba, 0.20)["capture"],
                         "params": "{}" if name == "baseline_logreg" else json.dumps(best_params)})
    metrics.log_metrics(rows)
    print(pd.DataFrame(rows)[["model", "threshold_rule", "threshold", "precision",
                              "recall", "roc_auc", "pr_auc", "brier"]]
          .round(4).to_string(index=False))

    # ------------------------------------------- fit final models on all train
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline.fit(X_train, y_train), config.MODELS_DIR / "baseline_logreg.joblib")
    joblib.dump(xgb.fit(X_train, y_train), config.MODELS_DIR / "xgboost.joblib")
    # The calibrator clones and refits its own copies of the pipeline, so the
    # booster saved above (the one SHAP explains) is left as it is.
    joblib.dump(make_calibrated(xgb, method, cv).fit(X_train, y_train),
                config.MODELS_DIR / "xgboost_calibrated.joblib")

    # ---------------------------------------------- nested CV (estimate only)
    # Runs after the final models are fitted and saved, and only writes its own
    # outputs, so it cannot change the final model or threshold.
    n_outer_fits = config.OUTER_CV_FOLDS * (n_combos * config.CV_FOLDS + 1)
    print(f"[5/5] Nested CV: {config.OUTER_CV_FOLDS} outer x {config.CV_FOLDS} inner folds, "
          f"{n_combos} combinations ({n_outer_fits} XGBoost fits)")
    nested = nested_cv(X_train, y_train)
    nested.to_csv(config.RESULTS_DIR / "nested_cv_results.csv", index=False,
                  float_format="%.5f")
    print(f"  saved {(config.RESULTS_DIR / 'nested_cv_results.csv').relative_to(config.ROOT)}")
    print(nested.groupby("model")[["roc_auc", "pr_auc"]].agg(["mean", "std"]).round(4)
          .to_string())

    meta = {
        "seed": config.SEED,
        "scale_pos_weight": spw,
        "xgb_best_params": best_params,
        "thresholds": thresholds,
        "cost_false_negative": config.COST_FALSE_NEGATIVE,
        "cost_false_positive": config.COST_FALSE_POSITIVE,
        "cv_roc_auc": {r["model"]: r["roc_auc"] for r in rows},
        "cv_pr_auc": {r["model"]: r["pr_auc"] for r in rows},
        "calibration": {
            "method": method,
            "cv_folds": config.CV_FOLDS,
            "selection": "lowest out-of-fold Brier score (log loss breaks ties)",
            "oof_scores": cal_scores,
            "theoretical_threshold": bayes,
            "empirical_threshold": thresholds["xgboost_calibrated"],
        },
        "nested_cv": summarise_nested_cv(nested),
    }
    (config.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("  saved models/ (baseline_logreg.joblib, xgboost.joblib, "
          "xgboost_calibrated.joblib, metadata.json)")


if __name__ == "__main__":
    main()
