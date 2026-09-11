"""
analysis/04_abstention_sweep.py
=================================
Sweeps a selective-abstention threshold on the model's own confidence
margin (kappa = |p_hat(readmit) - 0.5|; abstain/refer-to-clinician when
margin < kappa, i.e. when the model itself is near-toss-up) and, at each
retained population, reports:
  - retained fraction (1 - abstention rate)
  - marginal coverage on the retained population
  - effective coverage on the retained population (coverage AND
    singleton/actionable conformal set), for Global, Mondrian(race), and
    LWCC-A.

This is the real, computed analogue of a Coverage-Abstention-
Actionability curve: nothing here is asserted, every point is read off
actual model outputs on the held-out test set.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.conformal import GlobalConformal, MondrianConformal, nonconformity_scores
from src.local_conformal import LocalConformal
from src import metrics as M

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"
ALPHA = 0.10
K_NEIGHBORS = 500
KAPPAS = np.linspace(0.0, 0.45, 19)


def main():
    clf = joblib.load(ROOT / "models" / "readmission_lgbm.joblib")
    X = pd.read_parquet(INTERIM / "X_full.parquet")
    y = np.load(INTERIM / "y_full.npy")
    idx_cal = np.load(INTERIM / "idx_cal.npy")
    idx_test = np.load(INTERIM / "idx_test.npy")

    protected_cols = [c for c in X.columns if c.startswith("race_") or c.startswith("gender_") or c == "age_ordinal"]
    dist_cols = [c for c in X.columns if c not in protected_cols]

    p_cal = np.load(INTERIM / "p_cal.npy")
    p_test = np.load(INTERIM / "p_test.npy")
    y_cal, y_test = y[idx_cal], y[idx_test]

    subgroups = pd.read_parquet(INTERIM / "subgroups_full.parquet")
    sg_cal = subgroups.iloc[idx_cal].reset_index(drop=True)
    sg_test = subgroups.iloc[idx_test].reset_index(drop=True)

    scores_cal = nonconformity_scores(p_cal, y_cal)

    gc = GlobalConformal().fit(p_cal, y_cal, ALPHA)
    size_g, inc0_g, inc1_g = gc.predict_sets(p_test)

    mc = MondrianConformal().fit(p_cal, y_cal, sg_cal["race"].values, ALPHA)
    size_m, inc0_m, inc1_m = mc.predict_sets(p_test, sg_test["race"].values)

    lc = LocalConformal(k_neighbors=K_NEIGHBORS).fit(X[dist_cols].values[idx_cal], scores_cal, ALPHA)
    size_l, inc0_l, inc1_l, _ = lc.predict_sets(p_test, X[dist_cols].values[idx_test])

    margin = np.abs(p_test - 0.5)

    methods = {
        "global": (inc0_g, inc1_g),
        "mondrian": (inc0_m, inc1_m),
        "lwcc_a": (inc0_l, inc1_l),
    }

    rows = []
    for kappa in KAPPAS:
        retain = margin >= kappa
        for name, (inc0, inc1) in methods.items():
            retained_n = int(retain.sum())
            if retained_n == 0:
                continue
            cov = M.marginal_coverage(y_test[retain], inc0[retain], inc1[retain])
            eff = M.effective_coverage(y_test, inc0, inc1, retain_mask=retain)
            act = M.actionable_rate((inc0.astype(int) + inc1.astype(int)), retain_mask=retain)
            rows.append({
                "kappa": kappa,
                "method": name,
                "abstention_rate": 1 - retain.mean(),
                "retained_n": retained_n,
                "coverage_retained": cov,
                "effective_coverage_retained": eff,
                "actionable_rate_retained": act,
            })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "tables" / "abstention_sweep.csv", index=False)
    print(out.pivot(index="kappa", columns="method", values="effective_coverage_retained").to_string())


if __name__ == "__main__":
    main()
