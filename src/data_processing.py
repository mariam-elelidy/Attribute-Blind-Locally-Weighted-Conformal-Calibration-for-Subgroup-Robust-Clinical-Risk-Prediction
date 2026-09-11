"""
data_processing.py
===================
Loads and preprocesses the UCI "Diabetes 130-US Hospitals for Years 1999-2008"
dataset (Strack et al., 2014) for the 30-day hospital readmission risk task.

This is REAL, de-identified, publicly available clinical data
(101,766 inpatient encounters, 130 US hospitals, 1999-2008). Source:
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

We use this dataset because it is one of the few public clinical datasets
that (a) is large enough for meaningful calibration/test splits, (b) contains
real demographic subgroup labels (race, gender, age decile) that are
routinely under-represented and clinically consequential, and (c) has a
well-studied, well-defined binary target (30-day readmission) that is used
as a real hospital quality metric (CMS Hospital Readmissions Reduction
Program), making "actionable, reliable risk prediction" a genuine clinical
reliability problem rather than a toy task.

All preprocessing decisions are documented in DATA_CARD.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "diabetic_data.csv"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Columns dropped for near-total missingness (documented, not silently discarded)
HIGH_MISSING_COLS = ["weight", "payer_code", "medical_specialty"]
# Identifier / leakage-risk / zero-variance columns
DROP_COLS = ["encounter_id", "examide", "citoglipton"]
# Discharge dispositions that mean the patient died or entered hospice
# (readmission is not a meaningful / actionable outcome for these encounters)
EXPIRED_HOSPICE_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}

DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

AGE_ORDER = [f"[{10*i}-{10*(i+1)})" for i in range(10)]


def _map_diag_to_category(code: str) -> str:
    """Map an ICD-9 diagnosis code to one of 9 broad clinical categories.

    This follows the grouping scheme introduced by Strack et al. (2014) and
    widely reused in follow-up work on this dataset.
    """
    if pd.isna(code) or code in ("?", ""):
        return "Missing"
    code = str(code)
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    try:
        val = float(code)
    except ValueError:
        return "Other"
    if 390 <= val <= 459 or val == 785:
        return "Circulatory"
    if 460 <= val <= 519 or val == 786:
        return "Respiratory"
    if 520 <= val <= 579 or val == 787:
        return "Digestive"
    if int(val) == 250 or (250 <= val < 251):
        return "Diabetes"
    if 800 <= val <= 999:
        return "Injury"
    if 710 <= val <= 739:
        return "Musculoskeletal"
    if 580 <= val <= 629 or val == 788:
        return "Genitourinary"
    if 140 <= val <= 239:
        return "Neoplasms"
    return "Other"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, na_values=["?"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. First encounter per patient only -> avoids patient-level leakage
    #    across calibration/test splits (a repeated patient's second visit
    #    is not independent of their first, which conformal prediction's
    #    exchangeability assumption requires).
    df = df.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first")

    # 2. Remove expired / hospice discharges (readmission undefined/meaningless)
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_HOSPICE_DISCHARGE_IDS)]

    # 3. Drop near-total-missingness + identifier + zero-variance columns
    df = df.drop(columns=[c for c in HIGH_MISSING_COLS + DROP_COLS if c in df.columns])

    # 4. Binary target: readmitted within 30 days
    df["target_readmit_30d"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # 5. Diagnosis code -> clinical category (3 diagnosis slots)
    for c in ["diag_1", "diag_2", "diag_3"]:
        df[c + "_cat"] = df[c].apply(_map_diag_to_category)
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

    # 6. Standardize remaining missing values
    df["race"] = df["race"].fillna("Unknown")
    df["gender"] = df["gender"].replace("Unknown/Invalid", np.nan)
    df = df.dropna(subset=["gender"])  # only 3 rows, real UCI data quirk

    # 7. Age: UCI provides 10-year deciles as a categorical bucket (already
    #    de-identified at collection time -- this is the finest granularity
    #    available in the public release).
    df["age"] = pd.Categorical(df["age"], categories=AGE_ORDER, ordered=True)
    df["age_ordinal"] = df["age"].cat.codes

    # 8. Binary drug indicators (was the drug prescribed / changed at all)
    for c in DRUG_COLS:
        if c in df.columns:
            df[c + "_used"] = (~df[c].isin(["No"])).astype(int)
    df = df.drop(columns=[c for c in DRUG_COLS if c in df.columns])

    if "change" in df.columns:
        df["change_bin"] = (df["change"] == "Ch").astype(int)
        df = df.drop(columns=["change"])
    if "diabetesMed" in df.columns:
        df["diabetesMed_bin"] = (df["diabetesMed"] == "Yes").astype(int)
        df = df.drop(columns=["diabetesMed"])
    if "max_glu_serum" in df.columns:
        df = df.drop(columns=["max_glu_serum"])  # >94% missing/None
    if "A1Cresult" in df.columns:
        df = df.drop(columns=["A1Cresult"])  # >82% missing/None

    df = df.drop(columns=["patient_nbr"])
    return df.reset_index(drop=True)


FEATURE_COLS_NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "age_ordinal",
]

FEATURE_COLS_CATEGORICAL = [
    "race", "gender", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "diag_1_cat", "diag_2_cat", "diag_3_cat",
]

DRUG_USED_COLS = [c + "_used" for c in DRUG_COLS] + ["change_bin", "diabetesMed_bin"]

SUBGROUP_COLS = ["race", "gender", "age"]  # used for audit/masking analysis only


def build_design_matrix(df: pd.DataFrame):
    """One-hot encode categoricals, return (X, y, subgroup_frame)."""
    y = df["target_readmit_30d"].values
    subgroups = df[SUBGROUP_COLS].copy()

    cat_present = [c for c in FEATURE_COLS_CATEGORICAL if c in df.columns]
    num_present = [c for c in FEATURE_COLS_NUMERIC if c in df.columns]
    drug_present = [c for c in DRUG_USED_COLS if c in df.columns]

    X_cat = pd.get_dummies(df[cat_present].astype(str), prefix=cat_present)
    X_num = df[num_present].astype(float)
    X_drug = df[drug_present].astype(float)
    X = pd.concat([X_num, X_drug, X_cat], axis=1)
    return X, y, subgroups


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    cleaned = clean(raw)
    cleaned.to_csv(PROCESSED_DIR / "cleaned_encounters.csv", index=False)
    print(f"Raw rows: {len(raw):,}  ->  Cleaned rows (1 per patient): {len(cleaned):,}")
    print(f"Positive rate (readmit<30d): {cleaned['target_readmit_30d'].mean():.4f}")
    print("Race distribution:\n", cleaned["race"].value_counts())
    print("Saved -> ", PROCESSED_DIR / "cleaned_encounters.csv")


if __name__ == "__main__":
    main()
