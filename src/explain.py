"""SHAP explanations for the tuned XGBoost model.

SHAP values describe what the MODEL relies on to produce a score. They are not
causal effects: a large SHAP value for "month-to-month contract" does not prove
that moving a customer to a yearly contract would stop them leaving.

Run:  python src/explain.py     (after python src/train.py)
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import config
import plotting
from data import load_train_test


def explain(pipeline, X: pd.DataFrame) -> shap.Explanation:
    """SHAP values (in log-odds) for rows of raw customer data `X`.

    The pipeline's own preprocessing is applied first, so the explanation is
    over the one-hot encoded columns the model actually sees.
    """
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    X_enc = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out(),
                         index=X.index)
    return shap.TreeExplainer(model)(X_enc)


def main():
    _, X_test, _, _ = load_train_test()   # features only; labels are not used
    pipeline = joblib.load(config.MODELS_DIR / "xgboost.joblib")
    explanation = explain(pipeline, X_test)

    # Sanity check: base value + SHAP values must reproduce the model's log-odds.
    proba = pipeline.predict_proba(X_test)[:, 1]
    rebuilt = explanation.base_values + explanation.values.sum(axis=1)
    assert np.allclose(rebuilt, np.log(proba / (1 - proba)), atol=1e-3), \
        "SHAP values do not add up to the model output"

    # ------------------------------------------------ global: summary plot
    plt.figure()
    shap.summary_plot(explanation, max_display=12, show=False, plot_size=(8.5, 6))
    plt.title("SHAP summary: what the model relies on",
              loc="left", fontweight="bold", fontsize=13)
    plotting.save(plt.gcf(), "09_shap_summary.png")

    importance = (pd.Series(np.abs(explanation.values).mean(axis=0),
                            index=explanation.feature_names)
                  .sort_values(ascending=False))
    importance.rename("mean_abs_shap").round(4).to_csv(
        config.RESULTS_DIR / "shap_importance.csv", index_label="feature")
    print("[SHAP] top drivers (mean |SHAP|, log-odds):")
    print(importance.head(8).round(3).to_string())

    # --------------------------------------- local: the riskiest test customer
    idx = int(np.argmax(proba))
    print(f"[SHAP] waterfall for the highest-risk test customer "
          f"(row {X_test.index[idx]}, predicted probability {proba[idx]:.3f})")
    print(X_test.iloc[idx].to_string())
    plt.figure()
    shap.plots.waterfall(explanation[idx], max_display=10, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 5.5)
    plt.title(f"Why this customer scores {proba[idx]:.0%} churn risk "
              "(contributions in log-odds)", loc="left", fontweight="bold", fontsize=12,
              pad=30)
    plotting.save(fig, "10_shap_waterfall_customer.png")


if __name__ == "__main__":
    main()
