"""Final evaluation on the held-out test set.

This is the only script that reads the test labels. Models, hyperparameters and
thresholds were all frozen by train.py before this runs.

Run:  python src/evaluate.py     (after python src/train.py)
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, precision_recall_curve

import config
import metrics
import plotting
from data import load_train_test

MODELS = {"baseline_logreg": "Logistic regression", "xgboost": "XGBoost",
          "xgboost_calibrated": "XGBoost (calibrated)"}
COLOURS = {"baseline_logreg": plotting.ORANGE, "xgboost": plotting.BLUE,
           "xgboost_calibrated": plotting.GREEN}
# Ranking views (PR curve, gains): calibration barely changes the ordering, so
# these keep to the baseline and the raw XGBoost score.
RANKING_MODELS = {k: MODELS[k] for k in ("baseline_logreg", "xgboost")}


def plot_confusion_matrices(y, probas, thresholds):
    """Side-by-side confusion matrices at each model's cost-based threshold."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.5 * len(MODELS), 4))
    for ax, name in zip(axes, MODELS, strict=True):
        m = metrics.classification_metrics(y, probas[name], thresholds[name])
        grid = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=grid.max() * 1.25)
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=14,
                    color="white" if v > grid.max() * 0.6 else plotting.INK)
        ax.set_xticks([0, 1], ["Predicted stay", "Predicted churn"])
        ax.set_yticks([0, 1], ["Stayed", "Churned"])
        ax.set_title(f"{MODELS[name]} (threshold {thresholds[name]:.2f})", fontsize=11)
        ax.grid(False)
    fig.suptitle("Confusion matrices on the test set", x=0.02, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    plotting.save(fig, "06_confusion_matrices.png")


def plot_pr_curves(y, probas, thresholds):
    """Precision-recall curves with the chosen operating points marked."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name, label in RANKING_MODELS.items():
        precision, recall, _ = precision_recall_curve(y, probas[name])
        m = metrics.classification_metrics(y, probas[name], thresholds[name])
        ax.plot(recall, precision, color=COLOURS[name],
                label=f"{label} (PR-AUC {m['pr_auc']:.3f})")
        ax.scatter([m["recall"]], [m["precision"]], s=60, color=COLOURS[name],
                   edgecolor=plotting.SURFACE, linewidth=2, zorder=3)
    ax.axhline(np.mean(y), color=plotting.GREY, ls="--", lw=1.2)
    ax.text(0.01, np.mean(y) + 0.015, f"no-skill = churn rate {np.mean(y):.2f}",
            color=plotting.INK_SOFT, fontsize=9)
    ax.set_xlabel("Recall (share of churners caught)")
    ax.set_ylabel("Precision (share of flagged who churn)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall on the test set (dots = chosen thresholds)")
    ax.legend(loc="upper right")
    plotting.save(fig, "07_precision_recall_curve.png")


def plot_gains(y, probas):
    """Cumulative gains: contact the riskiest x% of customers, reach y% of churners."""
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    x = np.arange(0, len(y) + 1) / len(y) * 100
    for name, label in RANKING_MODELS.items():
        order = np.argsort(-probas[name], kind="stable")
        gains = np.concatenate([[0], np.cumsum(y[order])]) / y.sum() * 100
        ax.plot(x, gains, color=COLOURS[name], label=label)
    ax.plot([0, 100], [0, 100], color=plotting.GREY, ls="--", lw=1.2,
            label="Random targeting")

    # Annotate the XGBoost curve at the two operating points a team would use.
    for frac in (0.10, 0.20):
        c = metrics.capture_at_top(y, probas["xgboost"], frac)
        ax.scatter([frac * 100], [c["capture"] * 100], s=60, color=plotting.BLUE,
                   edgecolor=plotting.SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"top {frac:.0%}: {c['capture']:.1%} of churners\n(lift {c['lift']:.2f}x)",
                    (frac * 100, c["capture"] * 100), textcoords="offset points",
                    xytext=(14, -22), fontsize=9.5)
    ax.set_xlabel("Customers contacted, riskiest first (%)")
    ax.set_ylabel("Churners reached (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.set_title("Cumulative gains on the test set")
    ax.legend(loc="lower right")
    plotting.save(fig, "08_gains_chart.png")


def plot_reliability(y, probas):
    """Reliability diagram: predicted probability vs observed churn rate in
    equal-count bins. Points on the diagonal mean the scores are calibrated."""
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], color=plotting.GREY, ls="--", lw=1.2,
            label="Perfectly calibrated")
    for name, label in MODELS.items():
        bins = metrics.calibration_bins(y, probas[name])
        brier = metrics.brier_score(y, probas[name])
        ax.plot(bins["mean_predicted"], bins["observed_rate"], color=COLOURS[name],
                marker="o", markersize=5, label=f"{label} (Brier {brier:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted churn probability (10 equal-count bins)")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Reliability diagram on the test set")
    ax.legend(loc="upper left")
    plotting.save(fig, "11_calibration_curve.png")


def main():
    plotting.apply_style()
    _, X_test, _, y_test = load_train_test()
    meta = json.loads((config.MODELS_DIR / "metadata.json").read_text())
    thresholds = meta["thresholds"]

    probas = {name: joblib.load(config.MODELS_DIR / f"{name}.joblib")
              .predict_proba(X_test)[:, 1] for name in MODELS}

    print(f"[TEST] {len(y_test):,} customers, {int(y_test.sum())} churners "
          f"({y_test.mean():.2%})")
    rows = []
    for name in MODELS:
        top10 = metrics.capture_at_top(y_test, probas[name], 0.10)
        top20 = metrics.capture_at_top(y_test, probas[name], 0.20)
        for rule, t in (("default_0.5", 0.5), ("min_cost", thresholds[name])):
            m = metrics.classification_metrics(y_test, probas[name], t)
            rows.append({"model": name, "data": "test", "threshold_rule": rule, **m,
                         "top10_capture": top10["capture"],
                         "top20_capture": top20["capture"],
                         "brier": metrics.brier_score(y_test, probas[name]),
                         "params": "{}" if name == "baseline_logreg"
                         else json.dumps(meta["xgb_best_params"])})
        print(f"  {name}: top 10% -> {top10['churners_found']}/{int(y_test.sum())} churners "
              f"({top10['capture']:.1%}, lift {top10['lift']:.2f}x, "
              f"precision {top10['precision_in_group']:.1%}); "
              f"top 20% -> {top20['churners_found']} ({top20['capture']:.1%}, "
              f"lift {top20['lift']:.2f}x, precision {top20['precision_in_group']:.1%})")
    metrics.log_metrics(rows)
    table = pd.DataFrame(rows).drop(columns=["data", "params"])
    print(table.round(4).to_string(index=False))

    print("[TEST] calibration (lower is better):")
    for name, label in MODELS.items():
        print(f"  {label:>22}: Brier {metrics.brier_score(y_test, probas[name]):.4f}, "
              f"log loss {log_loss(y_test, probas[name]):.4f}, "
              f"mean predicted {probas[name].mean():.3f} vs churn rate {y_test.mean():.3f}")

    # Cost under the stated assumptions, to show what the threshold buys.
    for name in MODELS:
        c05 = metrics.expected_cost(y_test, probas[name], 0.5)
        ct = metrics.expected_cost(y_test, probas[name], thresholds[name])
        print(f"  {name}: cost at 0.50 = {c05:.0f} units, "
              f"at {thresholds[name]:.2f} = {ct:.0f} units")

    plot_confusion_matrices(y_test, probas, thresholds)
    plot_pr_curves(y_test, probas, thresholds)
    plot_gains(y_test, probas)
    plot_reliability(y_test, probas)

    # The deliverable for a retention team: every test customer, riskiest first.
    # churn_probability is the calibrated model's output; raw_score (the
    # uncalibrated booster, which SHAP explains) breaks ties between customers
    # that isotonic calibration maps to the same probability.
    ranked = X_test.assign(churn_probability=probas["xgboost_calibrated"].round(4),
                           raw_score=probas["xgboost"].round(4),
                           actual_churn=y_test.to_numpy())
    ranked = ranked.sort_values(["churn_probability", "raw_score"], ascending=False,
                                kind="stable")
    ranked.insert(0, "risk_rank", range(1, len(ranked) + 1))
    ranked["flagged_for_offer"] = (ranked["churn_probability"]
                                   >= thresholds["xgboost_calibrated"])
    path = config.RESULTS_DIR / "ranked_customers_test.csv"
    ranked.to_csv(path, index_label="row_id")
    print(f"  saved {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
