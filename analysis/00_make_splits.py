"""
analysis/00_make_splits.py
===========================
Builds the design matrix from the cleaned dataset, creates a single
canonical train / calibration / test split (used identically by every
downstream experiment), trains the base LightGBM risk model, and
persists everything needed by later scripts. Run this first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from src.data_processing import build_design_matrix, PROCESSED_DIR
from src.models import train_base_model, three_way_split

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
INTERIM_DIR = ROOT / "data" / "interim"
RESULTS_DIR = ROOT / "results"
SEED = 42
ALPHA = 0.10  # target 90% coverage, matches conventional conformal prediction default


def main():
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    INTERIM_DIR.mkdir(exist_ok=True, parents=True)
    (RESULTS_DIR / "tables").mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(PROCESSED_DIR / "cleaned_encounters.csv")
    X, y, subgroups = build_design_matrix(df)
    print(f"Design matrix: {X.shape}, positive rate: {y.mean():.4f}")

    idx_train, idx_cal, idx_test = three_way_split(X.values, y, subgroups, seed=SEED)
    print(f"train={len(idx_train)}  cal={len(idx_cal)}  test={len(idx_test)}")

    clf = train_base_model(X.values[idx_train], y[idx_train], seed=SEED)

    for name, idx in [("train", idx_train), ("cal", idx_cal), ("test", idx_test)]:
        p = clf.predict_proba(X.values[idx])[:, 1]
        auc = roc_auc_score(y[idx], p)
        brier = brier_score_loss(y[idx], p)
        print(f"[{name}] AUC={auc:.4f}  Brier={brier:.4f}  n={len(idx)}  pos_rate={y[idx].mean():.4f}")

    # Persist everything
    joblib.dump(clf, MODELS_DIR / "readmission_lgbm.joblib")
    X.to_parquet(INTERIM_DIR / "X_full.parquet")
    np.save(INTERIM_DIR / "y_full.npy", y)
    subgroups.to_parquet(INTERIM_DIR / "subgroups_full.parquet")
    np.save(INTERIM_DIR / "idx_train.npy", idx_train)
    np.save(INTERIM_DIR / "idx_cal.npy", idx_cal)
    np.save(INTERIM_DIR / "idx_test.npy", idx_test)

    feature_names = list(X.columns)
    with open(INTERIM_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    with open(RESULTS_DIR / "tables" / "base_model_performance.json", "w") as f:
        perf = {}
        for name, idx in [("train", idx_train), ("cal", idx_cal), ("test", idx_test)]:
            p = clf.predict_proba(X.values[idx])[:, 1]
            perf[name] = {
                "n": int(len(idx)),
                "positive_rate": float(y[idx].mean()),
                "auc": float(roc_auc_score(y[idx], p)),
                "brier": float(brier_score_loss(y[idx], p)),
            }
        json.dump(perf, f, indent=2)
    print("Saved splits, model, and performance metrics.")


if __name__ == "__main__":
    main()
