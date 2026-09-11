"""
analysis/02_local_conformal_method.py
=======================================
Fits the proposed LWCC-A (attribute-blind, locally-weighted conformal
calibration) method and compares it against Global and Mondrian
conformal prediction on marginal coverage, subgroup coverage, and the
STABILITY of subgroup coverage (does it suffer the same small-sample
variance problem Mondrian shows for minority groups?).
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.conformal import GlobalConformal, MondrianConformal, nonconformity_scores
from src.local_conformal import LocalConformal, abstain_mask
from src import metrics as M

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"
ALPHA = 0.10
K_NEIGHBORS = 500


def main():
    clf = joblib.load(ROOT / "models" / "readmission_lgbm.joblib")
    X = pd.read_parquet(INTERIM / "X_full.parquet")
    y = np.load(INTERIM / "y_full.npy")
    subgroups = pd.read_parquet(INTERIM / "subgroups_full.parquet")
    idx_cal = np.load(INTERIM / "idx_cal.npy")
    idx_test = np.load(INTERIM / "idx_test.npy")

    protected_cols = [c for c in X.columns if c.startswith("race_") or c.startswith("gender_") or c == "age_ordinal"]
    dist_cols = [c for c in X.columns if c not in protected_cols]
    print(f"Distance-metric feature dims (attribute-blind): {len(dist_cols)} / {X.shape[1]} total")

    X_cal_dist = X[dist_cols].values[idx_cal]
    X_test_dist = X[dist_cols].values[idx_test]

    p_cal = clf.predict_proba(X.values[idx_cal])[:, 1]
    p_test = clf.predict_proba(X.values[idx_test])[:, 1]
    y_cal, y_test = y[idx_cal], y[idx_test]
    sg_cal = subgroups.iloc[idx_cal].reset_index(drop=True)
    sg_test = subgroups.iloc[idx_test].reset_index(drop=True)

    scores_cal = nonconformity_scores(p_cal, y_cal)

    t0 = time.time()
    lc = LocalConformal(k_neighbors=K_NEIGHBORS).fit(X_cal_dist, scores_cal, ALPHA)
    size_l, inc0_l, inc1_l, qhat_l = lc.predict_sets(p_test, X_test_dist)
    fit_time = time.time() - t0
    print(f"LocalConformal predict_sets wall time: {fit_time:.1f}s for {len(idx_test)} test points")

    result = {
        "marginal_coverage": M.marginal_coverage(y_test, inc0_l, inc1_l),
        "mean_set_size": M.mean_set_size(size_l),
        "effective_coverage": M.effective_coverage(y_test, inc0_l, inc1_l),
        "actionable_rate": M.actionable_rate(size_l),
        "qhat_mean": float(qhat_l.mean()),
        "qhat_std": float(qhat_l.std()),
    }
    print("LWCC-A marginal:", result)

    # subgroup audit + STABILITY comparison across methods
    gc = GlobalConformal().fit(p_cal, y_cal, ALPHA)
    size_g, inc0_g, inc1_g = gc.predict_sets(p_test)
    mc = MondrianConformal().fit(p_cal, y_cal, sg_cal["race"].values, ALPHA)
    size_m, inc0_m, inc1_m = mc.predict_sets(p_test, sg_test["race"].values)

    cal_counts_by_col = {c: sg_cal[c].astype(str).value_counts().to_dict() for c in ["race", "gender", "age"]}

    rows = []
    for subgroup_col in ["race", "gender", "age"]:
        gcov_g = M.group_conditional_coverage(y_test, inc0_g, inc1_g, sg_test[subgroup_col].astype(str).values)
        gcov_m = M.group_conditional_coverage(y_test, inc0_m, inc1_m, sg_test[subgroup_col].astype(str).values)
        gcov_l = M.group_conditional_coverage(y_test, inc0_l, inc1_l, sg_test[subgroup_col].astype(str).values)
        for g in gcov_g:
            rows.append({
                "subgroup_type": subgroup_col,
                "subgroup": g,
                "n": gcov_g[g]["n"],
                "n_cal": cal_counts_by_col[subgroup_col].get(g, 0),
                "coverage_global": gcov_g[g]["coverage"],
                "coverage_mondrian": gcov_m.get(g, {}).get("coverage", np.nan),
                "coverage_lwcc_a": gcov_l.get(g, {}).get("coverage", np.nan),
                "set_size_global": gcov_g[g]["mean_set_size"],
                "set_size_mondrian": gcov_m.get(g, {}).get("mean_set_size", np.nan),
                "set_size_lwcc_a": gcov_l.get(g, {}).get("mean_set_size", np.nan),
            })
    audit = pd.DataFrame(rows)
    audit["abs_dev_global"] = (audit["coverage_global"] - (1 - ALPHA)).abs()
    audit["abs_dev_mondrian"] = (audit["coverage_mondrian"] - (1 - ALPHA)).abs()
    audit["abs_dev_lwcc_a"] = (audit["coverage_lwcc_a"] - (1 - ALPHA)).abs()
    audit.to_csv(RESULTS / "tables" / "subgroup_audit_all_methods.csv", index=False)
    print(audit.to_string(index=False))

    # Race-only worst-case / stability summary (the key comparison)
    race_rows = audit[audit.subgroup_type == "race"]
    summary = {
        "target_coverage": 1 - ALPHA,
        "global": {
            "marginal_coverage": float(M.marginal_coverage(y_test, inc0_g, inc1_g)),
            "worst_group_coverage_race": float(race_rows["coverage_global"].min()),
            "max_abs_deviation_race": float(race_rows["abs_dev_global"].max()),
            "mean_abs_deviation_race": float(race_rows["abs_dev_global"].mean()),
        },
        "mondrian_race": {
            "marginal_coverage": float(M.marginal_coverage(y_test, inc0_m, inc1_m)),
            "worst_group_coverage_race": float(race_rows["coverage_mondrian"].min()),
            "max_abs_deviation_race": float(race_rows["abs_dev_mondrian"].max()),
            "mean_abs_deviation_race": float(race_rows["abs_dev_mondrian"].mean()),
        },
        "lwcc_a_attribute_blind": {
            "marginal_coverage": float(M.marginal_coverage(y_test, inc0_l, inc1_l)),
            "worst_group_coverage_race": float(race_rows["coverage_lwcc_a"].min()),
            "max_abs_deviation_race": float(race_rows["abs_dev_lwcc_a"].max()),
            "mean_abs_deviation_race": float(race_rows["abs_dev_lwcc_a"].mean()),
        },
    }
    with open(RESULTS / "tables" / "method_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # persist for downstream shift/abstention experiments
    np.save(INTERIM / "qhat_local_test.npy", qhat_l)
    np.save(INTERIM / "p_test.npy", p_test)
    np.save(INTERIM / "p_cal.npy", p_cal)


if __name__ == "__main__":
    main()
