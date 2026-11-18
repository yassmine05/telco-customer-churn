"""Guard the settings the published results depend on.

If one of these fails, results/metrics.csv no longer describes the code.
"""
import numpy as np

import config


def test_reproducibility_settings_unchanged():
    assert config.SEED == 42
    assert config.TEST_SIZE == 0.20
    assert config.CV_FOLDS == 5
    assert config.OUTER_CV_FOLDS == 5


def test_cost_ratio_is_five_to_one():
    assert config.COST_FALSE_NEGATIVE == 5.0
    assert config.COST_FALSE_POSITIVE == 1.0


def test_search_grid_is_the_published_100_combinations():
    assert config.XGB_PARAM_GRID == {
        "model__max_depth": [2, 3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__n_estimators": [100, 200, 300, 500, 800],
    }
    assert np.prod([len(v) for v in config.XGB_PARAM_GRID.values()]) == 100


def test_feature_lists_do_not_overlap_and_exclude_id_and_target():
    features = (config.NUMERIC_FEATURES + config.PASSTHROUGH_FEATURES
                + config.CATEGORICAL_FEATURES)
    assert len(features) == len(set(features)) == 19
    assert "customerID" not in features
    assert config.TARGET not in features
