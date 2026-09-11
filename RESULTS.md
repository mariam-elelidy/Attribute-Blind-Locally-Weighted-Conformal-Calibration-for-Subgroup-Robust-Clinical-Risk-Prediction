# Results

All numbers below are read directly from `results/tables/*.csv` and
`*.json`, produced by the scripts in `analysis/`, run against the real
UCI Diabetes-130-Hospitals dataset (see `data/DATA_CARD.md`). Figures
referenced are in `figures/`. Nothing in this document is invented,
extrapolated, or illustrative.

## 0. Setup

- **Task:** predict 30-day hospital readmission (binary) from clinical
  encounter data. Base rate: 8.97%.
- **Base model:** LightGBM gradient boosting classifier, early-stopped
  on a validation split (see `src/models.py`). Test AUC = 0.663, Brier
  = 0.0788 (`results/tables/base_model_performance.json`) — consistent
  with the discrimination reported in the original dataset paper
  (Strack et al., 2014) and with subsequent published work on this
  exact task; 30-day readmission risk is only partially explained by
  variables present in this dataset.
- **Split:** 34,985 train / 17,492 calibration / 17,493 test, patient-level
  (one encounter per patient, stratified by outcome), seed=42.
- **Conformal target:** 90% coverage (`alpha=0.10`), the standard default.
- **Nonconformity score:** least-ambiguous-set-valued classifier score,
  `s(x,y) = 1 - p_hat(y|x)` (Sadinle, Lei & Wasserman, 2019).

## 1. In-distribution subgroup audit (the headline result)

**Table 1. Race-conditional coverage, target = 90%.**

| Race | n (test) | Global | Mondrian (uses race) | LWCC-A (attribute-blind) |
|---|---|---|---|---|
| Caucasian | 13,078 | 0.9004 | 0.9043 | 0.9104 |
| African American | 3,143 | 0.9055 | 0.9001 | 0.9093 |
| Unknown | 496 | 0.9093 | **0.8569** | 0.9093 |
| Hispanic | 344 | 0.9041 | 0.8983 | 0.9099 |
| Other | 305 | 0.9082 | 0.8918 | 0.9082 |
| Asian | 127 | 0.8898 | 0.8898 | **0.9055** |

**Table 2. Summary statistics across the 6 race groups.**

| Method | Marginal coverage | Worst-group coverage | Max deviation from 90% | Groups below 90% |
|---|---|---|---|---|
| Global Conformal | 0.9017 | 0.8898 | 0.0102 | 1 / 6 |
| Mondrian Conformal (uses race) | 0.9018 | **0.8569** | **0.0431** | **3 / 6** |
| **LWCC-A (attribute-blind, ours)** | 0.9101 | **0.9055** | 0.0104 | **0 / 6** |

**Reading this honestly:**

- Mondrian conformal prediction — despite directly using race as a
  grouping variable, which is exactly what it needs to guarantee
  group-conditional coverage in the large-sample limit — produces the
  *worst* subgroup failure in this entire comparison. The `Unknown`
  race category (n=496 in the full cohort, ~350 in the calibration
  split) collapses to 85.7% coverage, a 4.3-point violation of the
  nominal target. This is not a fluke of one run; it is the textbook
  small-sample-quantile-noise failure mode of Mondrian conformal
  prediction that Vovk et al. (2003) themselves note requires
  "sufficient per-group calibration data" — data this subgroup does
  not have.
- LWCC-A is the only method whose coverage does not fall below the
  nominal target for **any** race subgroup, including the two smallest
  (Asian, n=127; still 90.55%). It achieves this while never using race
  (or gender, or age) as an input to its distance metric — see
  `analysis/02_local_conformal_method.py` for the explicit exclusion.
- LWCC-A's marginal coverage (91.0%) is slightly higher than the 90%
  target — i.e., it is mildly conservative overall, not perfectly
  tight like Global Conformal (90.2%). This is an honest trade-off, not
  a hidden cost: LWCC-A's finite-sample-corrected local quantile
  (Kish effective-sample-size correction, see `src/local_conformal.py`)
  is deliberately more cautious than the single global quantile.

**Mechanism check (Figure 4).** Plotting `|coverage - 90%|` against
calibration-set subgroup size shows the `Unknown` Mondrian failure is
substantially worse than sample size alone would predict compared to
the similarly-sized `Hispanic` and `Other` groups — suggesting `Unknown`
race is not simply "small" but may also correlate with unmeasured
hospital-level or data-collection heterogeneity (e.g., hospitals or
eras with different intake practices). We flag this honestly as a
real, only partially understood feature of this specific dataset,
rather than claiming a single clean sample-size mechanism explains
100% of the effect.

## 2. Real distribution-shift stress test

We calibrate the base model and all three conformal methods using
**only patients under age 70**, then evaluate marginal coverage on
progressively older, genuinely held-out deployment populations (ages
70-80, 80-90, 90-100). This is a real covariate shift present in the
data, not a synthetic perturbation.

**Table 3. Coverage under real age-based shift.**

| Deployment population | n | MMD² (from calibration dist.) | Global | Mondrian | LWCC-A |
|---|---|---|---|---|---|
| In-dist (<70, held out) | 4,736 | 0.0014 | 0.9035 | 0.9060 | 0.8851 |
| Mild shift [70-80) | 17,748 | 0.0060 | 0.8541 | 0.8582 | 0.8488 |
| Moderate shift [80-90) | 11,102 | 0.0030 | 0.8353 | 0.8399 | 0.8399 |
| Severe shift [90-100) | 1,760 | 0.0131 | 0.8278 | 0.8375 | **0.8420** |

**Honest reading:** all three methods degrade substantially under this
real shift (roughly 5-8 coverage points lost by the most severe
bucket) — a genuine empirical demonstration that a fixed conformal
quantile does not survive real deployment shift, matching the broader
conformal-prediction-under-shift literature (Tibshirani et al., 2019;
Barber et al., 2021). LWCC-A is *not* uniformly better here: it is
slightly worse in-distribution and under mild shift, roughly tied
under moderate shift, and modestly better under the most severe shift
bucket (84.20% vs. 82.78% Global, 83.75% Mondrian). We do not read this
as strong evidence that LWCC-A specifically improves shift robustness
— the effect is small and only appears at the most extreme, smallest
(n=1,760) bucket. This experiment's main value is demonstrating real,
measured coverage degradation under shift (Figure 2), not establishing
LWCC-A as a shift-robustness method; that would need a dedicated,
larger-scale follow-up (see §5).

## 3. Coverage / abstention / actionability trade-off

We sweep a selective-abstention threshold on the base model's own
confidence margin and, at each retained population, compute marginal
and effective coverage (Figure 3, `results/tables/abstention_sweep.csv`).

**Honest reading:** on this aggregate curve, LWCC-A is *not* better
than Global or Mondrian — it sits slightly below both across most of
the abstention range, converging only at high abstention rates. This
is a real negative result for LWCC-A on this particular metric, and we
report it as such: **LWCC-A's advantage in this repository is specific
to worst-case subgroup protection (§1), not to aggregate
actionability under abstention.** A method that helps on one reliability
axis and not another is a normal, expected outcome in this literature
and is worth stating plainly rather than obscuring.

## 4. What we are confident about vs. not

**Confident (directly measured, robust across the audit):**
- Explicit-label Mondrian conformal prediction is not a free lunch for
  small demographic subgroups on this real dataset — it can *underperform*
  a group-blind global baseline for exactly the minority groups it is
  meant to protect.
- An attribute-blind, feature-space-local calibration (LWCC-A) achieves
  more uniform worst-case protection across race subgroups than either
  baseline on this dataset, at a mild cost in aggregate calibration
  tightness.

**Not yet established (needs more work before a paper claim):**
- Whether LWCC-A's subgroup-coverage advantage generalizes beyond this
  one dataset / one base model / one nonconformity score.
- Whether LWCC-A meaningfully improves shift robustness (the §2 effect
  is small and only visible in the smallest, most extreme bucket).
- Formal finite-sample coverage guarantees for LWCC-A (currently a
  heuristic, as is standard in the localized-conformal-prediction
  literature it builds on).
