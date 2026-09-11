"""
local_conformal.py
===================
PROPOSED METHOD: Attribute-Blind Locally-Weighted Conformal Calibration
with Selective Abstention (LWCC-A).

Research question this method targets
--------------------------------------
Mondrian conformal prediction (conformal.py: MondrianConformal) achieves
group-conditional coverage, but requires protected-attribute labels
(race, gender, age) at *both* calibration and inference time. This is
often clinically/legally undesirable or infeasible (protected attributes
may be unavailable, self-reported inconsistently, or excluded from the
model input by policy).

We test whether a substantial share of the group-conditional coverage
benefit of Mondrian conformal prediction can be recovered *without ever
using protected attributes*, by instead computing the conformal quantile
locally in the space of ordinary clinical features. The hypothesis is
that clinically under-represented demographic subgroups are also
feature-space outliers relative to the (majority-dominated) calibration
distribution -- i.e., that demographic under-representation and feature-
space local sparsity are correlated, so a feature-space-local calibration
naturally up-weights the calibration evidence that is actually relevant
to atypical patients, without ever being told which patients are atypical
because of race, gender, or age.

Method
------
For a test point x, we:
  1. Find its k nearest calibration neighbors in standardized clinical
     feature space (protected attributes EXCLUDED from the distance
     metric by construction).
  2. Weight each neighbor i by a Gaussian kernel of its distance,
     w_i = exp(-d_i^2 / (2 h(x)^2)), with an *adaptive local bandwidth*
     h(x) = median distance among x's k neighbors (Silverman-style local
     bandwidth; this makes the effective neighborhood size roughly
     comparable across dense and sparse regions of feature space).
  3. Compute a weighted empirical quantile of the neighbors'
     nonconformity scores at a finite-sample-corrected level using the
     Kish effective sample size n_eff = (sum w)^2 / sum(w^2), analogous
     to the (n+1)/n correction used in standard split conformal.

This is a heuristic, not a method with an exact finite-sample coverage
proof (this limitation is stated explicitly in RESULTS.md / the data
card; localized conformal methods in the literature -- e.g. Lei &
Wasserman 2014, Guan 2020, Han et al. 2022 -- share this property: they
trade an exact guarantee for adaptivity, and validity must be checked
empirically). We report exactly that empirical check.

Selective abstention
---------------------
On top of either conformal method, we add a selective abstention layer:
a query is *referred* (abstained on) if its resulting prediction set has
size 2 (i.e., the model, after calibration, cannot narrow the outcome to
a single class within the target confidence level). This exactly
mirrors clinical triage logic: "this patient's risk is genuinely
ambiguous given what we know -- flag for human review" rather than
silently emitting an uninformative prediction.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class LocalConformal:
    name = "LWCC (attribute-blind, local)"

    def __init__(self, k_neighbors: int = 400):
        self.k = k_neighbors

    def fit(self, X_cal_features: np.ndarray, scores_cal: np.ndarray, alpha: float):
        """X_cal_features: array of clinical features (NOT including
        protected attributes) used purely to define the distance metric.
        scores_cal: nonconformity scores s(x_i, y_i) on the calibration set.
        """
        self.alpha = alpha
        self.scaler = StandardScaler().fit(X_cal_features)
        Xs = self.scaler.transform(X_cal_features)
        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(Xs))).fit(Xs)
        self.scores_cal = np.asarray(scores_cal)
        self.n_cal = len(scores_cal)
        return self

    def _weighted_quantile_level(self, weights: np.ndarray) -> float:
        n_eff = (weights.sum() ** 2) / (np.sum(weights ** 2) + 1e-12)
        level = np.ceil((n_eff + 1) * (1 - self.alpha)) / n_eff
        return float(min(level, 1.0))

    def local_qhat(self, X_test_features: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X_test_features)
        dist, idx = self.nn.kneighbors(Xs)
        qhats = np.empty(len(Xs))
        for i in range(len(Xs)):
            d = dist[i]
            h = np.median(d) + 1e-6  # adaptive local bandwidth
            w = np.exp(-(d ** 2) / (2 * h ** 2))
            neigh_scores = self.scores_cal[idx[i]]
            order = np.argsort(neigh_scores)
            s_sorted = neigh_scores[order]
            w_sorted = w[order]
            cw = np.cumsum(w_sorted) / w_sorted.sum()
            level = self._weighted_quantile_level(w_sorted)
            pos = np.searchsorted(cw, level)
            pos = min(pos, len(s_sorted) - 1)
            qhats[i] = s_sorted[pos]
        return qhats

    def predict_sets(self, probs_pos_test: np.ndarray, X_test_features: np.ndarray):
        qhat = self.local_qhat(X_test_features)
        s0 = probs_pos_test
        s1 = 1 - probs_pos_test
        include_0 = s0 <= qhat
        include_1 = s1 <= qhat
        size = include_0.astype(int) + include_1.astype(int)
        return size, include_0, include_1, qhat


def abstain_mask(set_sizes: np.ndarray) -> np.ndarray:
    """Abstain (refer to clinician) exactly when the conformal set is
    uninformative (size 2, i.e. both classes covered)."""
    return set_sizes >= 2
