# Data Card

## Source

**Diabetes 130-US Hospitals for Years 1999–2008**
Strack, B., DeShazo, J.P., Gennings, C., Olmo Ortiz, J.L., Ventura, S., Cios, K.J.,
Clore, J. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates:
Analysis of 70,000 Clinical Database Patient Records.* BioMed Research
International, 2014.

- UCI Machine Learning Repository page:
  https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- Original data holder: Center for Clinical and Translational Research,
  Virginia Commonwealth University (NIH CTSA grant UL1 TR00058), derived
  from the Cerner Health Facts national data warehouse.
- License / usage: public-domain research data release via UCI ML
  Repository, CC BY 4.0 per the UCI repository's current terms.
- **This repository downloads the dataset from a third-party GitHub
  mirror** (`isaiahluc/ReadmissionAnalysisProject`, MIT-licensed repo,
  data re-published as-is from the UCI source) because the UCI archive
  domain was not reachable from this build environment's network
  policy. The file's contents were verified against the published
  schema and summary statistics of the original UCI dataset (row count,
  column names, race/gender/age distributions) before use — see
  `analysis/00_make_splits.py` output and `results/tables/base_model_performance.json`.
  **If you are cloning this repo for your own use, we recommend
  re-downloading directly from the UCI Machine Learning Repository or
  PhysioNet-hosted mirrors where available**, and re-running
  `src/data_processing.py` against that file — the pipeline is
  identical regardless of source.

## What this data is

101,766 raw inpatient encounter records from 130 US hospitals (1999–2008),
each corresponding to a hospitalization during which diabetes was
diagnosed. Every row is de-identified at the source (no names, no dates
of birth, no MRNs) prior to public release. This repository does not
have access to, and does not attempt to re-identify, any patient.

## Preprocessing decisions (all in `src/data_processing.py`)

| Step | Decision | Why |
|---|---|---|
| Patient deduplication | Keep only the first encounter per unique `patient_nbr` | Prevents a single patient's repeat visits from appearing in both calibration and test splits, which would violate the exchangeability assumption conformal prediction depends on |
| Discharge filtering | Drop encounters discharged to hospice or recorded as expired (discharge codes 11, 13, 14, 19, 20, 21) | 30-day readmission is undefined/not clinically meaningful for these outcomes |
| Missingness | Drop `weight` (97% missing), `payer_code` (40% missing), `medical_specialty` (49% missing) | Missingness far too high to impute reliably without introducing artifacts |
| Zero-variance columns | Drop `examide`, `citoglipton` | Constant across (almost) all rows in this cohort |
| Diagnosis codes | Map ICD-9 `diag_1/2/3` to 9 broad clinical categories (circulatory, respiratory, digestive, diabetes, injury, musculoskeletal, genitourinary, neoplasms, other) | Standard grouping introduced by Strack et al. (2014); keeps dimensionality tractable |
| Target | Binarize `readmitted` to `<30 days` = 1, else 0 | Matches the real CMS Hospital Readmissions Reduction Program quality metric and prior published work on this dataset |
| Missing race/gender | `race`: filled as `"Unknown"` (its own category, not imputed). `gender`: 3 rows with `Unknown/Invalid` dropped (real UCI data quirk, negligible count) | Preserves "Unknown" as an honest, auditable subgroup rather than silently imputing a majority category |

**Resulting cohort:** 69,970 unique-patient encounters, 8.97% experience
readmission within 30 days.

## Known subgroup imbalance (this is the point of the experiment, not
## a limitation to hide)

| Race | n (full cohort) | % |
|---|---|---|
| Caucasian | 52,292 | 74.7% |
| African American | 12,625 | 18.0% |
| Unknown | 1,916 | 2.7% |
| Hispanic | 1,500 | 2.1% |
| Other | 1,149 | 1.6% |
| Asian | 488 | 0.7% |

Age is provided by UCI only as 10-year deciles (the finest granularity
in the public release; this is a de-identification choice made at the
original data collection stage, not something this repository can
recover).

## Ethical / usage notes

- This is a research artifact for studying **reliability methodology**
  (conformal prediction, subgroup coverage auditing, selective
  abstention). **It is not a validated clinical tool** and must not be
  used to inform real patient care decisions.
- The base risk model's predictive performance (AUC ≈ 0.62–0.66,
  reported in `results/tables/base_model_performance.json`) is modest
  and consistent with prior published work on this exact dataset and
  task (Strack et al. 2014 report similar discrimination). This is a
  genuine property of the task — 30-day readmission is driven
  substantially by factors (social support, housing, post-discharge
  care access) that are absent from this dataset — not a modeling
  failure, and it is precisely the kind of imperfect, real-world model
  that reliability-layer methods (this repo's actual subject) need to
  be evaluated on.
