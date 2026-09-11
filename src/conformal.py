"""
conformal.py
============
Split conformal prediction for binary classification, using the standard
"least ambiguous set-valued classifier" (LAC) nonconformity score
(Sadinle, Lei & Wasserman, 2019):

    s(x, y) = 1 - p_hat(y | x)

Given a user-specified miscoverage level alpha, the (1-alpha) conformal
quantile qhat is the ceil((n+1)(1-alpha))/n empirical quantile of
calibration nonconformity scores {s(x_i, y_i)}. The prediction set is

    C(x) = { y in {0,1} : s(x, y) <= qhat }

This provides the textbook finite-sample marginal coverage guarantee
P(Y in C(X)) >= 1-alpha under exchangeability (Vovk, Gammerman & Shafer,
2005). Two baselines are implemented here:

  * GlobalConformal   -- one qhat for the whole calibration set.
  * MondrianConformal -- one qhat per pre-specified group (e.g. race),
    which achieves group-conditional coverage but REQUIRES group labels
    at both calibration and inference time (Vovk, Lindsay, Nouretdinov &
    Gammerman, 2003).

Our proposed method (src/local_conformal.py) is compared against both.
"""
from __future__ import annotations

import numpy as np


def nonconformity_scores(probs_pos: np.ndarray, y: np.ndarray) -> np.ndarray:
    """s(x,y) = 1 - p_hat(y|x), for binary y in {0,1}."""
    p_true = np.where(y == 1, probs_pos, 1 - probs_pos)
    return 1.0 - p_true


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def prediction_set_sizes(probs_pos: np.ndarray, qhat: float) -> np.ndarray:
    """Returns, for each point, the size of C(x) in {0, 1, 2} and which
    labels are included, given a scalar or per-point qhat."""
    s0 = 1 - (1 - probs_pos)  # s(x,0) = 1 - p_hat(0) = probs_pos
    s1 = 1 - probs_pos        # s(x,1) = 1 - p_hat(1) = 1 - probs_pos
    include_0 = s0 <= qhat
    include_1 = s1 <= qhat
    size = include_0.astype(int) + include_1.astype(int)
    return size, include_0, include_1


class GlobalConformal:
    name = "Global Conformal"

    def fit(self, probs_pos_cal, y_cal, alpha):
        self.alpha = alpha
        scores = nonconformity_scores(probs_pos_cal, y_cal)
        self.qhat = conformal_quantile(scores, alpha)
        return self

    def predict_sets(self, probs_pos_test):
        return prediction_set_sizes(probs_pos_test, self.qhat)

    def qhat_for(self, probs_pos_test, group_test=None):
        return np.full(len(probs_pos_test), self.qhat)


class MondrianConformal:
    """Group-conditional (Mondrian) conformal prediction. Requires group
    labels at calibration AND at test time -- this is the key practical
    limitation our proposed method (local_conformal.py) removes."""

    name = "Mondrian Conformal (uses group labels)"

    def fit(self, probs_pos_cal, y_cal, group_cal, alpha):
        self.alpha = alpha
        self.qhat_by_group = {}
        scores = nonconformity_scores(probs_pos_cal, y_cal)
        for g in np.unique(group_cal):
            mask = group_cal == g
            self.qhat_by_group[g] = conformal_quantile(scores[mask], alpha)
        # fallback for unseen groups at test time
        self.global_qhat = conformal_quantile(scores, alpha)
        return self

    def qhat_for(self, probs_pos_test, group_test):
        return np.array([
            self.qhat_by_group.get(g, self.global_qhat) for g in group_test
        ])

    def predict_sets(self, probs_pos_test, group_test):
        qhat = self.qhat_for(probs_pos_test, group_test)
        s0 = probs_pos_test
        s1 = 1 - probs_pos_test
        include_0 = s0 <= qhat
        include_1 = s1 <= qhat
        size = include_0.astype(int) + include_1.astype(int)
        return size, include_0, include_1
