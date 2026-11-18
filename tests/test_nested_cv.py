"""Tests for nested cross-validation in src/train.py."""
import json

import joblib
import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

import config
import train

from conftest import TINY_GRID


@pytest.fixture(scope="module")
def nested_run(split_data):
    """Run nested_cv on the synthetic training set with the tiny grid, recording
    exactly which rows every inner grid search is fitted on."""
    X_train, _, y_train, _ = split_data
    seen_by_search = []

    class RecordingGridSearchCV(train.GridSearchCV):
        def fit(self, X, y=None, **kwargs):
            seen_by_search.append(X.index.copy())
            return super().fit(X, y, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "XGB_PARAM_GRID", TINY_GRID)
        mp.setattr(train, "GridSearchCV", RecordingGridSearchCV)
        with joblib.parallel_config(backend="threading"):
            table = train.nested_cv(X_train, y_train)
    return table, seen_by_search


def outer_splits(X, y):
    outer = StratifiedKFold(n_splits=config.OUTER_CV_FOLDS, shuffle=True,
                            random_state=config.SEED + 1)
    return [(X.index[tr], X.index[va]) for tr, va in outer.split(X, y)]


def test_one_row_per_outer_fold_per_model(nested_run):
    table, _ = nested_run
    assert len(table) == 2 * config.OUTER_CV_FOLDS
    pairs = set(zip(table["fold"], table["model"]))
    assert pairs == {(f, m) for f in range(1, config.OUTER_CV_FOLDS + 1)
                     for m in ("baseline_logreg", "xgboost")}
    assert list(table.columns) == ["fold", "model", "roc_auc", "pr_auc", "best_params"]


def test_scores_are_between_0_and_1(nested_run):
    table, _ = nested_run
    for col in ("roc_auc", "pr_auc"):
        assert np.isfinite(table[col]).all()
        assert table[col].between(0, 1).all()


def test_inner_search_never_sees_outer_validation_rows(nested_run, split_data):
    X_train, _, y_train, _ = split_data
    _, seen_by_search = nested_run
    splits = outer_splits(X_train, y_train)
    assert len(seen_by_search) == len(splits) == config.OUTER_CV_FOLDS
    for seen, (outer_train, outer_valid) in zip(seen_by_search, splits):
        assert seen.intersection(outer_valid).empty
        assert set(seen) == set(outer_train)


def test_outer_folds_differ_from_inner_folds(split_data):
    # SEED + 1 for the outer loop: its validation folds are not the inner ones.
    X_train, _, y_train, _ = split_data
    inner = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                            random_state=config.SEED)
    inner_valid = [set(X_train.index[va]) for _, va in inner.split(X_train, y_train)]
    outer_valid = [set(va) for _, va in outer_splits(X_train, y_train)]
    assert inner_valid != outer_valid


def test_best_params_recorded_for_xgboost_only(nested_run):
    table, _ = nested_run
    for _, row in table.iterrows():
        params = json.loads(row["best_params"])
        if row["model"] == "xgboost":
            assert set(params) == {"max_depth", "learning_rate", "n_estimators"}
            assert params["max_depth"] in TINY_GRID["model__max_depth"]
        else:
            assert params == {}


def test_summary_matches_the_per_fold_table(nested_run):
    table, _ = nested_run
    summary = train.summarise_nested_cv(table)
    assert summary["outer_folds"] == config.OUTER_CV_FOLDS
    assert summary["inner_folds"] == config.CV_FOLDS
    for model in ("baseline_logreg", "xgboost"):
        scores = table[table["model"] == model]
        for metric in ("roc_auc", "pr_auc"):
            assert summary[model][f"{metric}_mean"] == pytest.approx(scores[metric].mean())
            assert summary[model][f"{metric}_std"] == pytest.approx(
                np.std(scores[metric], ddof=1))
    assert len(summary["xgb_params_by_fold"]) == config.OUTER_CV_FOLDS
    json.dumps(summary)     # must be JSON-serialisable for metadata.json
