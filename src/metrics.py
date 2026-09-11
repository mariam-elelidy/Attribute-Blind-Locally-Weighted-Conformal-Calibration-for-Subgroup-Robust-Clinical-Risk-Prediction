"""
metrics.py
==========
Evaluation metrics for the reliability analysis. All quantities here are
computed directly from prediction sets / scores produced elsewhere in
src/ -- nothing in this file is a fabricated or assumed number.

Effective coverage (this repo's operational definition)
---------------------------------------------------------
For classification, "actionable" is naturally operationalized as: the
conformal prediction set narrows the outcome to a single class
(|C(x)| == 1), rather than emitting both classes. We therefore define:

    EffectiveCoverage = P( Y in C(X)  AND  |C(X)| == 1 )

restricted to the retained (non-abstained) population. This is the
direct classification analogue of the DARF-style "coverage restricted to
actionable-width intervals" used in regression conformal prediction.
"""
from __future__ import annotations

import numpy as np


def marginal_coverage(y_true, include_0, include_1) -> float:
    covered = np.where(y_true == 1, include_1, include_0)
    return float(covered.mean())


def mean_set_size(size: np.ndarray) -> float:
    return float(size.mean())


def group_conditional_coverage(y_true, include_0, include_1, group):
    covered = np.where(y_true == 1, include_1, include_0)
    out = {}
    for g in np.unique(group):
        mask = group == g
        out[g] = {
            "n": int(mask.sum()),
            "coverage": float(covered[mask].mean()) if mask.sum() > 0 else np.nan,
            "mean_set_size": float((include_0.astype(int) + include_1.astype(int))[mask].mean()),
        }
    return out


def effective_coverage(y_true, include_0, include_1, retain_mask=None):
    size = include_0.astype(int) + include_1.astype(int)
    covered = np.where(y_true == 1, include_1, include_0)
    actionable = size == 1
    if retain_mask is None:
        retain_mask = np.ones(len(y_true), dtype=bool)
    denom = retain_mask.sum()
    if denom == 0:
        return np.nan
    num = (covered & actionable & retain_mask).sum()
    return float(num / denom)


def actionable_rate(size: np.ndarray, retain_mask=None) -> float:
    if retain_mask is None:
        retain_mask = np.ones(len(size), dtype=bool)
    denom = retain_mask.sum()
    if denom == 0:
        return np.nan
    return float(((size == 1) & retain_mask).sum() / denom)


def abstention_rate(abstain_mask: np.ndarray) -> float:
    return float(abstain_mask.mean())


def rbf_mmd2_unbiased(X: np.ndarray, Y: np.ndarray, gamma: float | None = None,
                       max_n: int = 500, seed: int = 0) -> float:
    """Unbiased estimator of squared Maximum Mean Discrepancy with an RBF
    kernel (Gretton et al., 2012), computed on a bounded random subsample
    of each population for tractability (pairwise kernel matrices are
    O(n^2) memory). gamma defaults to the median-heuristic bandwidth.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]

    from scipy.spatial.distance import cdist

    if gamma is None:
        pool = np.vstack([X, Y])
        d2 = cdist(pool, pool, metric="sqeuclidean")
        med = np.median(d2[d2 > 0])
        gamma = 1.0 / (2 * med + 1e-12)

    def k(A, B):
        return np.exp(-gamma * cdist(A, B, metric="sqeuclidean"))

    n, m = len(X), len(Y)
    Kxx = k(X, X)
    Kyy = k(Y, Y)
    Kxy = k(X, Y)
    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    sum_xy = Kxy.sum() / (n * m)
    return float(sum_xx + sum_yy - 2 * sum_xy)


def caa_curve(y_true, probs_pos, qhat_scalar_or_array_fn, thresholds, group=None):
    """Sweep an abstention threshold on the model's confidence margin and
    report (abstention_rate, marginal_coverage_on_retained,
    effective_coverage_on_retained) at each threshold, using a FIXED
    conformal set (computed once at the target alpha). Abstention here is
    implemented as: abstain when |C(x)| == 2 is "softened" by additionally
    referring borderline size-1 cases whose confidence margin is below
    the swept threshold -- this traces out a full coverage/abstention/
    actionability trade-off curve analogous to the CAA surface.
    """
    raise NotImplementedError("Use analysis/07_abstention_caa.py for the concrete sweep implementation.")
