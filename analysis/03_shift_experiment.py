"""
analysis/03_shift_experiment.py
=================================
Real (not synthetic) covariate-shift experiment: the base model and all
conformal calibrations are fit using ONLY patients aged < 70 (train +
calibration), then evaluated at deployment on a series of held-out
populations of increasing age -- a genuine, clinically meaningful shift
(older patients differ systematically in comorbidity burden, medication
count, and utilization patterns; elderly patients are also a population
routinely under-represented in the clinical trials that many real risk
scores are originally calibrated on).

We report, at each shift level:
  - real MMD^2 (RBF kernel) between the calibration population's feature
    distribution and the deployment population's feature distribution
  - marginal coverage achieved by Global / Mondrian(race) / LWCC-A
  - mean set size / effective coverage

This directly tests whether the reliability advantage of LWCC-A found in
the in-distribution subgroup audit (02_local_conformal_method.py)
persists, strengthens, or disappears under real covariate shift.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.data_processing import build_design_matrix, PROCESSED_DIR
from src.models import train_base_model
from src.conformal import GlobalConformal, MondrianConformal, nonconformity_scores
from src.local_conformal import LocalConformal
from src import metrics as M

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
ALPHA = 0.10
SEED = 42
K_NEIGHBORS = 500

AGE_ORDER = [f"[{10*i}-{10*(i+1)})" for i in range(10)]


def main():
    df = pd.read_csv(PROCESSED_DIR / "cleaned_encounters.csv")
    X, y, subgroups = build_design_matrix(df)
    age_bucket = subgroups["age"].astype(str).values

    in_dist_mask = np.isin(age_bucket, AGE_ORDER[:6])  # ages < 70 : calibration population
    rng = np.random.default_rng(SEED)
    idx_in = np.where(in_dist_mask)[0]
    rng.shuffle(idx_in)
    n_train = int(0.55 * len(idx_in))
    n_cal = int(0.25 * len(idx_in))
    idx_train = idx_in[:n_train]
    idx_cal = idx_in[n_train:n_train + n_cal]
    idx_test_indist = idx_in[n_train + n_cal:]

    protected_cols = [c for c in X.columns if c.startswith("race_") or c.startswith("gender_") or c == "age_ordinal"]
    dist_cols = [c for c in X.columns if c not in protected_cols]

    clf = train_base_model(X.values[idx_train], y[idx_train], seed=SEED)
    p_cal = clf.predict_proba(X.values[idx_cal])[:, 1]
    y_cal = y[idx_cal]
    scores_cal = nonconformity_scores(p_cal, y_cal)

    gc = GlobalConformal().fit(p_cal, y_cal, ALPHA)
    mc = MondrianConformal().fit(p_cal, y_cal, subgroups["race"].values[idx_cal], ALPHA)
    lc = LocalConformal(k_neighbors=K_NEIGHBORS).fit(X[dist_cols].values[idx_cal], scores_cal, ALPHA)

    X_cal_dist = X[dist_cols].values[idx_cal]

    shift_buckets = {
        "In-dist (<70, held-out)": idx_test_indist,
        "Mild shift [70-80)": np.where(age_bucket == "[70-80)")[0],
        "Moderate shift [80-90)": np.where(age_bucket == "[80-90)")[0],
        "Severe shift [90-100)": np.where(age_bucket == "[90-100)")[0],
    }

    rows = []
    for label, idx_deploy in shift_buckets.items():
        if len(idx_deploy) < 5:
            continue
        p_dep = clf.predict_proba(X.values[idx_deploy])[:, 1]
        y_dep = y[idx_deploy]
        X_dep_dist = X[dist_cols].values[idx_deploy]

        mmd2 = M.rbf_mmd2_unbiased(X_cal_dist, X_dep_dist)

        size_g, inc0_g, inc1_g = gc.predict_sets(p_dep)
        size_m, inc0_m, inc1_m = mc.predict_sets(p_dep, subgroups["race"].values[idx_deploy])
        size_l, inc0_l, inc1_l, _ = lc.predict_sets(p_dep, X_dep_dist)

        rows.append({
            "deployment_population": label,
            "n": len(idx_deploy),
            "mmd2_from_calibration_dist": mmd2,
            "coverage_global": M.marginal_coverage(y_dep, inc0_g, inc1_g),
            "coverage_mondrian": M.marginal_coverage(y_dep, inc0_m, inc1_m),
            "coverage_lwcc_a": M.marginal_coverage(y_dep, inc0_l, inc1_l),
            "set_size_global": M.mean_set_size(size_g),
            "set_size_mondrian": M.mean_set_size(size_m),
            "set_size_lwcc_a": M.mean_set_size(size_l),
            "eff_coverage_global": M.effective_coverage(y_dep, inc0_g, inc1_g),
            "eff_coverage_mondrian": M.effective_coverage(y_dep, inc0_m, inc1_m),
            "eff_coverage_lwcc_a": M.effective_coverage(y_dep, inc0_l, inc1_l),
        })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "tables" / "shift_experiment.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
