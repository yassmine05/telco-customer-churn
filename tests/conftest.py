"""Shared fixtures.

Nothing here reads the Kaggle CSV or touches the real reports/, results/ or
models/ folders. Tests run on a small synthetic dataset with the same columns,
dtypes and quirks as the real Telco file (TotalCharges stored as text with
blank values for brand-new customers, Churn as "Yes"/"No").
"""
import hashlib
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

N_ROWS = 200
N_CHURNERS = 54            # 27% churn, close to the real 26.5%
N_NEW_CUSTOMERS = 6        # tenure 0 -> blank TotalCharges
FIXTURE_SEED = 2024

RAW_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
          "StreamingTV", "StreamingMovies"]

# Two configurations instead of the real 100, so the grid search takes seconds.
TINY_GRID = {
    "model__max_depth": [2, 3],
    "model__learning_rate": [0.1],
    "model__n_estimators": [30],
}

# Small, fixed XGBoost settings for unit tests that need a fitted model.
SMALL_XGB_PARAMS = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 30}


def make_raw_telco(n=N_ROWS, n_churners=N_CHURNERS, seed=FIXTURE_SEED) -> pd.DataFrame:
    """Synthetic customers in the exact raw format of the Kaggle CSV."""
    rng = np.random.default_rng(seed)
    yes_no = lambda p: np.where(rng.random(n) < p, "Yes", "No")  # noqa: E731

    tenure = rng.integers(1, 73, n)
    tenure[rng.choice(n, N_NEW_CUSTOMERS, replace=False)] = 0

    phone = yes_no(0.9)
    multiple = np.where(phone == "Yes", yes_no(0.45), "No phone service")
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    addons = {col: np.where(internet == "No", "No internet service", yes_no(0.4))
              for col in ADDONS}
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n,
                          p=[0.55, 0.21, 0.24])
    payment = rng.choice(["Electronic check", "Mailed check",
                          "Bank transfer (automatic)", "Credit card (automatic)"], n)

    n_addons = sum((v == "Yes").astype(int) for v in addons.values())
    monthly = (20 + 25 * (internet == "DSL") + 50 * (internet == "Fiber optic")
               + 5 * n_addons + 5 * (multiple == "Yes") + rng.normal(0, 3, n))
    monthly = np.round(np.clip(monthly, 18.25, 118.75), 2)
    # Text column, blank (" ") for customers who have not been billed yet,
    # exactly like the real file.
    total = [" " if t == 0 else str(round(m * t, 2)) for m, t in zip(monthly, tenure)]

    senior = (rng.random(n) < 0.16).astype(int)
    risk = (1.5 * (contract == "Month-to-month") + 1.0 * (internet == "Fiber optic")
            + 0.8 * (payment == "Electronic check") - 0.04 * tenure + 0.5 * senior
            + rng.normal(0, 1, n))
    churn = np.full(n, "No", dtype=object)
    churn[np.argsort(-risk)[:n_churners]] = "Yes"

    letters = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    ids = [f"{i:04d}-{''.join(rng.choice(letters, 5))}" for i in range(n)]

    df = pd.DataFrame({
        "customerID": ids,
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": senior,
        "Partner": yes_no(0.48),
        "Dependents": yes_no(0.3),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        **addons,
        "Contract": contract,
        "PaperlessBilling": yes_no(0.59),
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": churn,
    })
    return df[RAW_COLUMNS]


def file_digest(path):
    """sha256 of a file, or None if it does not exist."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


class Untouchable:
    """Stand-in for a data split that a script must never read.

    Any attribute access, indexing, iteration or array conversion fails the
    test, so a script that silently used it would be caught.
    """

    def __init__(self, name):
        object.__setattr__(self, "_name", name)

    def _fail(self, *_, **__):
        raise AssertionError(f"{self._name} was accessed by a script that must not read it")

    def __getattr__(self, attr):
        self._fail()

    __len__ = __iter__ = __getitem__ = __array__ = _fail


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def raw_df():
    return make_raw_telco()


@pytest.fixture(scope="session")
def raw_csv(tmp_path_factory, raw_df):
    """The synthetic data written to disk, so load_and_clean reads a real CSV."""
    path = tmp_path_factory.mktemp("data") / "telco_synthetic.csv"
    raw_df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def clean_df(raw_csv):
    from data import load_and_clean
    return load_and_clean(raw_csv)


@pytest.fixture(scope="session")
def split_data(clean_df):
    """(X_train, X_test, y_train, y_test) from the project's own split()."""
    from data import split
    return split(clean_df)


@pytest.fixture(scope="session")
def fitted_xgb(split_data):
    """A small XGBoost pipeline fitted on the synthetic training rows."""
    from train import make_xgboost
    X_train, _, y_train, _ = split_data
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    return make_xgboost(spw, **SMALL_XGB_PARAMS).fit(X_train, y_train)


@pytest.fixture(scope="session")
def pipeline_run(tmp_path_factory, raw_csv):
    """Run eda -> train -> evaluate -> explain (the run_all.py order) once.

    * Every output path in config is redirected to a temporary project folder.
    * The grid search uses TINY_GRID instead of the real 100 combinations.
    * eda.py and train.py receive Untouchable objects in place of the test
      split, and explain.py in place of the test labels, so any use of them
      fails the run.
    * metrics.best_cost_threshold is wrapped to record exactly which rows and
      scores it receives.
    """
    import config
    import eda
    import evaluate
    import explain
    import metrics
    import train
    from data import load_and_clean, split

    root = tmp_path_factory.mktemp("project")
    guarded = [config.METRICS_PATH, config.MODELS_DIR / "metadata.json"]
    digests_before = {p: file_digest(p) for p in guarded}

    def loader(hide_test_features, hide_test_labels):
        def load_train_test():
            X_train, X_test, y_train, y_test = split(load_and_clean(raw_csv))
            if hide_test_features:
                X_test = Untouchable("X_test")
            if hide_test_labels:
                y_test = Untouchable("y_test")
            return X_train, X_test, y_train, y_test
        return load_train_test

    threshold_calls = []
    real_best_cost_threshold = metrics.best_cost_threshold

    def spy_best_cost_threshold(y_true, proba):
        chosen = real_best_cost_threshold(y_true, proba)
        threshold_calls.append({"y": y_true.copy(), "proba": np.asarray(proba).copy(),
                                "threshold": chosen})
        return chosen

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "ROOT", root)
        mp.setattr(config, "REPORTS_DIR", root / "reports")
        mp.setattr(config, "FIGURES_DIR", root / "reports" / "figures")
        mp.setattr(config, "RESULTS_DIR", root / "results")
        mp.setattr(config, "METRICS_PATH", root / "results" / "metrics.csv")
        mp.setattr(config, "MODELS_DIR", root / "models")
        mp.setattr(config, "XGB_PARAM_GRID", TINY_GRID)
        mp.setattr(metrics, "best_cost_threshold", spy_best_cost_threshold)
        mp.setattr(eda, "load_train_test", loader(True, True))
        mp.setattr(train, "load_train_test", loader(True, True))
        mp.setattr(evaluate, "load_train_test", loader(False, False))
        mp.setattr(explain, "load_train_test", loader(False, True))
        # GridSearchCV(n_jobs=-1) would spawn worker processes, which costs
        # ~8 s on Windows. Threads give identical results here: each fit is
        # independent and seeded, and XGBoost itself runs with n_jobs=1.
        with joblib.parallel_config(backend="threading"):
            for step in (eda, train, evaluate, explain):
                step.main()

    X_train, X_test, y_train, y_test = split(load_and_clean(raw_csv))
    return SimpleNamespace(
        root=root,
        meta=json.loads((root / "models" / "metadata.json").read_text()),
        threshold_calls=threshold_calls,
        guarded_digests_before=digests_before,
        guarded_digests_after={p: file_digest(p) for p in guarded},
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    )
