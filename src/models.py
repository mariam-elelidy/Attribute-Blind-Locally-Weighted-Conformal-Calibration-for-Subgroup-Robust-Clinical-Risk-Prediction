"""
models.py
=========
Base clinical risk predictor: LightGBM binary classifier for 30-day
hospital readmission risk. This is the "base predictor" f(x) -> p_hat(y|x)
on top of which all conformal / reliability machinery in this repo operates.

We deliberately use a real, standard, off-the-shelf gradient boosting
classifier rather than a custom architecture: the contribution of this
repository is the *reliability layer* (conformal calibration + selective
abstention + subgroup auditing), not a novel risk-prediction model, and the
reliability layer must work on top of an ordinary black-box classifier to
be practically relevant.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def train_base_model(X_train, y_train, seed: int = 0) -> lgb.LGBMClassifier:
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=seed, stratify=y_train
    )

    clf = lgb.LGBMClassifier(
        n_estimators=1500,
        num_leaves=31,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=40,
        reg_lambda=1.0,
        random_state=seed,
        verbosity=-1,
    )
    clf.fit(
        X_fit, y_fit,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)],
    )
    return clf


def three_way_split(X, y, subgroups, cal_size=0.25, test_size=0.25, seed=0):
    """train / calibration / test split, stratified on the label.

    The calibration set is what conformal quantiles are computed on; the
    test set is held out for evaluating coverage, effective coverage, and
    subgroup audits. Splitting is done once and stored so every experiment
    in this repo uses identical splits (see analysis/00_make_splits.py).
    """
    idx = np.arange(len(y))
    idx_train, idx_rest = train_test_split(
        idx, test_size=(cal_size + test_size), random_state=seed, stratify=y
    )
    rel_test = test_size / (cal_size + test_size)
    idx_cal, idx_test = train_test_split(
        idx_rest, test_size=rel_test, random_state=seed, stratify=y[idx_rest]
    )
    return idx_train, idx_cal, idx_test
