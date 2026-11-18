"""Uplift evaluation on a SIMULATED retention campaign.

The Telco data has no treatment column, so the campaign, its random assignment
and the true effect of the offer are all invented (see simulate.py and the
uplift settings in config.py). The numbers show whether the method recovers a
known effect; they are not facts about real customers.

Steps:
  1. simulate the campaign on the training and test customers;
  2. fit a T-learner on the simulated training customers only;
  3. on the simulated test customers, compare targeting rules at the same
     budget (top UPLIFT_TARGET_FRACTION contacted), scored with the true effect,
     and draw Qini curves.

Run:  python src/uplift/evaluate.py     (after python src/train.py)
"""
import json
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    # Run as a script: put src/ (not src/uplift/) first on the import path.
    sys.path[0] = str(Path(__file__).resolve().parents[1])

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import config
import metrics
import plotting
from data import load_train_test
from uplift.models import TLearner
from uplift.simulate import segment_summary, simulate

RULES = {
    "random": "Random",
    "churn_risk": "Highest churn risk (current approach)",
    "predicted_uplift": "Highest predicted uplift (T-learner)",
    "true_uplift": "Highest true uplift (simulation-only bound)",
}
COLOURS = {"random": plotting.GREY, "churn_risk": plotting.ORANGE,
           "predicted_uplift": plotting.BLUE, "true_uplift": plotting.GREEN}


def targeting_scores(test_sim: pd.DataFrame, predicted_uplift, seed: int) -> dict:
    """The score each rule ranks customers by (highest first)."""
    rng = np.random.default_rng(seed)
    return {
        "random": rng.random(len(test_sim)),
        "churn_risk": test_sim["p_control"].to_numpy(),
        "predicted_uplift": np.asarray(predicted_uplift),
        "true_uplift": test_sim["true_uplift"].to_numpy(),
    }


def compare_rules(test_sim: pd.DataFrame, scores: dict,
                  fraction: float = config.UPLIFT_TARGET_FRACTION) -> pd.DataFrame:
    """Incremental customers retained by contacting the top `fraction` under
    each rule, computed exactly from the true effect (sum of true uplift).

    The random rule reports its expected value (fraction x total true uplift)
    and the expected segment mix, not one draw. The Qini coefficient is
    computed from the observed (simulated) outcomes, as it would be in a real
    campaign; for the random rule it uses one seeded random ordering.
    """
    n = len(test_sim)
    k = int(round(n * fraction))
    tau = test_sim["true_uplift"].to_numpy()
    segment = test_sim["segment"].to_numpy()
    best = np.sort(tau)[::-1][:k].sum()
    rows = []
    for rule, score in scores.items():
        if rule == "random":
            gained = fraction * tau.sum()
            mix = {s: fraction * (segment == s).sum() for s in config.UPLIFT_SEGMENT_EFFECTS}
        else:
            top = np.argsort(-np.asarray(score), kind="stable")[:k]
            gained = tau[top].sum()
            mix = {s: float((segment[top] == s).sum()) for s in config.UPLIFT_SEGMENT_EFFECTS}
        rows.append({
            "rule": rule, "contacted": k,
            "incremental_retained": gained,
            "share_of_best": gained / best,
            "qini_coefficient": metrics.qini_coefficient(test_sim["churn"],
                                                         test_sim["treatment"], score),
            **{f"n_{s}": v for s, v in mix.items()},
        })
    return pd.DataFrame(rows)


def plot_qini(test_sim: pd.DataFrame, scores: dict, fraction: float):
    """Left: Qini curves from the observed outcomes (what a real campaign would
    show). Right: the exact incremental customers retained, from the true effect."""
    fig, (ax_obs, ax_true) = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    n = len(test_sim)
    tau = test_sim["true_uplift"].to_numpy()
    for rule, score in scores.items():
        style = {"color": COLOURS[rule], "label": RULES[rule],
                 "ls": "--" if rule == "random" else "-",
                 "lw": 1.4 if rule == "random" else 2}
        x, q = metrics.qini_curve(test_sim["churn"], test_sim["treatment"], score)
        ax_obs.plot(x, q, **style)
        if rule == "random":
            ax_true.plot([0, 1], [0, tau.sum()], **style)
        else:
            order = np.argsort(-np.asarray(score), kind="stable")
            ax_true.plot(np.arange(n + 1) / n, np.concatenate([[0], np.cumsum(tau[order])]),
                         **style)
    for ax in (ax_obs, ax_true):
        ax.axvline(fraction, color=plotting.INK_SOFT, ls=":", lw=1)
        ax.axhline(0, color=plotting.GREY, lw=0.8)
        ax.set_xlabel("Share of test customers contacted (highest score first)")
        ax.text(fraction + 0.01, ax.get_ylim()[0] + 0.03 * np.ptp(ax.get_ylim()),
                f"budget {fraction:.0%}", color=plotting.INK_SOFT, fontsize=9)
    ax_obs.set_ylabel("Incremental customers retained")
    ax_obs.set_title("Qini curve (observed simulated outcomes)", fontsize=11.5)
    ax_true.set_title("Exact gain (sum of true simulated effect)", fontsize=11.5)
    ax_true.legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Uplift on a SIMULATED retention campaign ({n:,} test customers)",
                 x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plotting.save(fig, "12_qini_curve.png")


def main():
    start = time.perf_counter()
    plotting.apply_style()
    print("NOTE: the retention campaign below is SIMULATED; the data has no treatment "
          "column.")
    X_train, X_test, y_train, _ = load_train_test()
    calibrated = joblib.load(config.MODELS_DIR / "xgboost_calibrated.joblib")
    params = json.loads((config.MODELS_DIR / "metadata.json").read_text())["xgb_best_params"]

    print("[1/3] Simulating the campaign (control-arm churn probability: calibrated "
          "XGBoost, out-of-fold on training rows)")
    train_sim, test_sim = simulate(calibrated, X_train, y_train, X_test)
    for name, sim in (("training", train_sim), ("test", test_sim)):
        print(f"      {name}: {len(sim)} customers, {sim['treatment'].mean():.1%} treated")
        print(segment_summary(sim).round(3).to_string(), "\n")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    segment_summary(test_sim).to_csv(config.RESULTS_DIR / "uplift_segments_test.csv",
                                     float_format="%.4f")

    print(f"[2/3] T-learner on the simulated training customers (XGBoost {params})")
    tlearner = TLearner(**params).fit(X_train, train_sim["treatment"], train_sim["churn"])
    predicted = tlearner.predict_uplift(X_test)
    pearson = float(pearsonr(predicted, test_sim["true_uplift"]).statistic)
    spearman = float(spearmanr(predicted, test_sim["true_uplift"]).statistic)
    print(f"      predicted vs true uplift on test: Pearson {pearson:.3f}, "
          f"Spearman {spearman:.3f}")
    by_segment = (test_sim.assign(predicted_uplift=predicted).groupby("segment")
                  [["true_uplift", "predicted_uplift"]].mean())
    print(by_segment.round(3).to_string())

    fraction = config.UPLIFT_TARGET_FRACTION
    print(f"[3/3] Targeting rules: top {fraction:.0%} of test customers contacted")
    scores = targeting_scores(test_sim, predicted, config.UPLIFT_SEED)
    table = compare_rules(test_sim, scores, fraction)
    print(table.round(3).to_string(index=False))
    table["uplift_corr_pearson"] = np.where(table["rule"] == "predicted_uplift", pearson,
                                            np.nan)
    table["uplift_corr_spearman"] = np.where(table["rule"] == "predicted_uplift", spearman,
                                             np.nan)
    table.to_csv(config.RESULTS_DIR / "uplift_targeting.csv", index=False,
                 float_format="%.4f")
    print(f"  saved {(config.RESULTS_DIR / 'uplift_targeting.csv').relative_to(config.ROOT)}")
    # Per-customer output (row index from the raw file, no features).
    test_sim.assign(predicted_uplift=predicted).to_csv(
        config.RESULTS_DIR / "uplift_campaign_test.csv", index_label="row",
        float_format="%.4f")
    plot_qini(test_sim, scores, fraction)
    print(f"  uplift evaluation took {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
