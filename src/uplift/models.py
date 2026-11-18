"""T-learner: one churn model per arm of the (simulated) campaign.

    uplift(x) = P(churn | x, no offer) - P(churn | x, offer)

Positive uplift means the offer is predicted to reduce churn for that customer.
Each arm's model is the project's own preprocessing + XGBoost pipeline
(train.make_xgboost). scale_pos_weight is left at 1: re-weighting inflates
predicted probabilities by an amount that depends on each arm's churn rate, and
the uplift is a difference of probabilities, so that would bias it.
"""
import numpy as np

from train import make_xgboost


class TLearner:
    """Fit separate classifiers on treated and control customers."""

    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params

    def fit(self, X, treatment, y):
        treatment = np.asarray(treatment)
        y = np.asarray(y)
        self.models_ = {}
        for arm in (0, 1):
            rows = treatment == arm
            self.models_[arm] = make_xgboost(1.0, **self.xgb_params).fit(X[rows], y[rows])
        return self

    def predict_arms(self, X):
        """(P(churn | no offer), P(churn | offer)) for each customer."""
        return (self.models_[0].predict_proba(X)[:, 1],
                self.models_[1].predict_proba(X)[:, 1])

    def predict_uplift(self, X):
        p_control, p_treated = self.predict_arms(X)
        return p_control - p_treated
