"""Simulate a randomised retention campaign on the Telco customers.

THIS IS A SIMULATION. The real data records who churned, not who was sent an
offer, so there is nothing to learn a treatment effect from. Here we invent one:

  1. Each customer gets a control-arm churn probability p0 = P(churn | no offer)
     from the calibrated XGBoost model. Training customers get out-of-fold
     predictions (a clone of the model is refitted on the other folds), because
     the saved model was fitted on them and would be overconfident; test
     customers get the saved model's predictions.
  2. Each customer is put in a segment by the rules in config.py, and the
     segment fixes the true effect of the offer, tau. The treated-arm
     probability is p1 = clip(p0 - tau, 0, 1), and the true individual uplift
     is p0 - p1 (after clipping).
  3. Half the customers, chosen at random, are treated. The observed outcome is
     a single draw from p1 for treated customers and from p0 for the others.

Because p0 and p1 are known for everyone, any targeting rule can be scored
exactly: the expected number of extra customers kept by contacting a group is
the sum of their true uplift.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import config

SEGMENTS = list(config.UPLIFT_SEGMENT_EFFECTS)


def control_probabilities(calibrated_model, X_train, y_train, X_test, cv=None):
    """P(churn | no offer) for training and test customers.

    Training rows: out-of-fold predictions of an unfitted clone of the saved
    model (same pipeline, parameters and calibration method). Test rows: the
    saved model itself, which never saw them.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                             random_state=config.SEED)
    p_train = cross_val_predict(clone(calibrated_model), X_train, y_train, cv=cv,
                                method="predict_proba")[:, 1]
    p_test = calibrated_model.predict_proba(X_test)[:, 1]
    return p_train, p_test


def assign_segments(X: pd.DataFrame) -> pd.Series:
    """Segment label per customer; the first matching rule wins (see config.py)."""
    mtm = X["Contract"] == "Month-to-month"
    rules = [
        ("sure_thing", X["Contract"] == "Two year"),
        ("sleeping_dog", mtm & (X["tenure"] >= config.UPLIFT_SLEEPING_DOG_MIN_TENURE)),
        ("lost_cause", mtm & (X["InternetService"] == "Fiber optic")
         & (X["PaymentMethod"] == "Electronic check")),
        ("persuadable", mtm & (X["tenure"] <= config.UPLIFT_PERSUADABLE_MAX_TENURE)
         & (X["TechSupport"] != "Yes")),
    ]
    segment = pd.Series("other", index=X.index, dtype=object)
    assigned = pd.Series(False, index=X.index)
    for name, mask in rules:
        segment[mask & ~assigned] = name
        assigned |= mask
    return segment


def simulate_campaign(X: pd.DataFrame, p_control, seed: int,
                      effects: dict | None = None) -> pd.DataFrame:
    """One simulated campaign. Returns a frame aligned with X:

    treatment      1 = sent the offer (random, UPLIFT_TREATMENT_SHARE of rows)
    segment        ground-truth segment
    p_control      P(churn | no offer)
    p_treated      P(churn | offer)
    true_uplift    p_control - p_treated (positive = the offer helps)
    churn          observed outcome in the arm the customer was assigned to
    """
    effects = config.UPLIFT_SEGMENT_EFFECTS if effects is None else effects
    rng = np.random.default_rng(seed)
    n = len(X)
    p0 = np.asarray(p_control, dtype=float)
    segment = assign_segments(X)
    tau = segment.map(effects).to_numpy(dtype=float)
    p1 = np.clip(p0 - tau, 0.0, 1.0)

    # Complete randomisation: exactly round(n * share) treated, in random order.
    treatment = np.zeros(n, dtype=int)
    treatment[: int(round(n * config.UPLIFT_TREATMENT_SHARE))] = 1
    rng.shuffle(treatment)

    p_assigned = np.where(treatment == 1, p1, p0)
    churn = (rng.random(n) < p_assigned).astype(int)
    return pd.DataFrame({"treatment": treatment, "segment": segment.to_numpy(),
                         "p_control": p0, "p_treated": p1, "true_uplift": p0 - p1,
                         "churn": churn}, index=X.index)


def simulate(calibrated_model, X_train, y_train, X_test, seed=config.UPLIFT_SEED,
             effects=None, cv=None):
    """Simulated campaigns on the training and test customers (separate seeds)."""
    p_train, p_test = control_probabilities(calibrated_model, X_train, y_train, X_test, cv)
    train = simulate_campaign(X_train, p_train, seed, effects)
    test = simulate_campaign(X_test, p_test, seed + 1, effects)
    return train, test


def segment_summary(sim: pd.DataFrame) -> pd.DataFrame:
    """Size, share and true average effect of each segment."""
    g = sim.groupby("segment")
    table = pd.DataFrame({
        "customers": g.size(),
        "share": g.size() / len(sim),
        "mean_p_control": g["p_control"].mean(),
        "mean_true_uplift": g["true_uplift"].mean(),
    }).reindex(SEGMENTS).dropna(how="all")
    table.loc["all"] = [len(sim), 1.0, sim["p_control"].mean(), sim["true_uplift"].mean()]
    table["customers"] = table["customers"].astype(int)
    return table
