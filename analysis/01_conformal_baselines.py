"""
analysis/01_conformal_baselines.py
====================================
Fits Global Split Conformal and Mondrian (group-conditional) Conformal
prediction on top of the trained base model, and reports:
  - marginal coverage / mean set size (sanity check: should hit target)
  - group-conditional coverage broken down by race, gender, age decile
This produces the empirical "subgroup masking" evidence (or lack of it)
directly from real data -- nothing here is asserted a priori.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.conformal import GlobalConformal, MondrianConformal, nonconformity_scores
from src import metrics as M

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"
ALPHA = 0.10


def main():
    clf = joblib.load(ROOT / "models" / "readmission_lgbm.joblib")
    X = pd.read_parquet(INTERIM / "X_full.parquet")
    y = np.load(INTERIM / "y_full.npy")
    subgroups = pd.read_parquet(INTERIM / "subgroups_full.parquet")
    idx_cal = np.load(INTERIM / "idx_cal.npy")
    idx_test = np.load(INTERIM / "idx_test.npy")

    p_cal = clf.predict_proba(X.values[idx_cal])[:, 1]
    p_test = clf.predict_proba(X.values[idx_test])[:, 1]
    y_cal, y_test = y[idx_cal], y[idx_test]
    sg_cal, sg_test = subgroups.iloc[idx_cal].reset_index(drop=True), subgroups.iloc[idx_test].reset_index(drop=True)

    results = {}

    # --- Global Conformal ---
    gc = GlobalConformal().fit(p_cal, y_cal, ALPHA)
    size_g, inc0_g, inc1_g = gc.predict_sets(p_test)
    results["global"] = {
        "qhat": gc.qhat,
        "marginal_coverage": M.marginal_coverage(y_test, inc0_g, inc1_g),
        "mean_set_size": M.mean_set_size(size_g),
        "effective_coverage": M.effective_coverage(y_test, inc0_g, inc1_g),
        "actionable_rate": M.actionable_rate(size_g),
    }

    # --- Mondrian Conformal (by race; uses labels at cal+test time) ---
    mc = MondrianConformal().fit(p_cal, y_cal, sg_cal["race"].values, ALPHA)
    size_m, inc0_m, inc1_m = mc.predict_sets(p_test, sg_test["race"].values)
    results["mondrian_race"] = {
        "qhat_by_group": mc.qhat_by_group,
        "marginal_coverage": M.marginal_coverage(y_test, inc0_m, inc1_m),
        "mean_set_size": M.mean_set_size(size_m),
        "effective_coverage": M.effective_coverage(y_test, inc0_m, inc1_m),
        "actionable_rate": M.actionable_rate(size_m),
    }

    # --- Subgroup audit tables (race, gender, age) for both methods ---
    audit_rows = []
    for subgroup_col in ["race", "gender", "age"]:
        gcov_g = M.group_conditional_coverage(y_test, inc0_g, inc1_g, sg_test[subgroup_col].astype(str).values)
        gcov_m = M.group_conditional_coverage(y_test, inc0_m, inc1_m, sg_test[subgroup_col].astype(str).values)
        for g in gcov_g:
            audit_rows.append({
                "subgroup_type": subgroup_col,
                "subgroup": g,
                "n": gcov_g[g]["n"],
                "coverage_global": gcov_g[g]["coverage"],
                "set_size_global": gcov_g[g]["mean_set_size"],
                "coverage_mondrian": gcov_m.get(g, {}).get("coverage", np.nan),
                "set_size_mondrian": gcov_m.get(g, {}).get("mean_set_size", np.nan),
            })
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(RESULTS / "tables" / "subgroup_audit_global_mondrian.csv", index=False)
    print(audit_df.to_string(index=False))

    with open(RESULTS / "tables" / "conformal_baselines_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))

    print("\nGlobal:", {k: v for k, v in results["global"].items() if k != "qhat"})
    print("Mondrian:", {k: v for k, v in results["mondrian_race"].items() if k != "qhat_by_group"})


if __name__ == "__main__":
    main()
