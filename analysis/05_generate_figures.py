"""
analysis/05_generate_figures.py
=================================
Generates all figures for the repository from the CSV/JSON results
produced by analysis/01-04. Every number plotted here is read from a
results/tables/*.csv or *.json file written by an earlier script --
no figure in this file contains a hand-entered or invented value.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tables"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

COLORS = {"global": "#c0392b", "mondrian": "#d68910", "lwcc_a": "#1a5276"}
LABELS = {"global": "Global Conformal", "mondrian": "Mondrian (uses race label)", "lwcc_a": "LWCC-A (attribute-blind, ours)"}


def fig1_subgroup_coverage_bars():
    df = pd.read_csv(RESULTS / "subgroup_audit_all_methods.csv")
    race = df[df.subgroup_type == "race"].sort_values("n", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(race))
    w = 0.26
    for i, m in enumerate(["global", "mondrian", "lwcc_a"]):
        ax.bar(x + (i - 1) * w, race[f"coverage_{m}"], width=w, label=LABELS[m], color=COLORS[m])
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1, label="Nominal target (90%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(n={n:,})" for r, n in zip(race["subgroup"], race["n"])], fontsize=8)
    ax.set_ylabel("Group-conditional coverage")
    ax.set_ylim(0.80, 0.96)
    ax.set_title("Race-conditional coverage at 90% nominal target\n(30-day readmission risk, UCI Diabetes-130 Hospitals)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_subgroup_coverage_by_race.png")
    plt.close(fig)


def fig2_shift_degradation():
    df = pd.read_csv(RESULTS / "shift_experiment.csv")
    # keep natural deployment-age order (calibration -> mild -> moderate -> severe)
    order = ["In-dist (<70, held-out)", "Mild shift [70-80)", "Moderate shift [80-90)", "Severe shift [90-100)"]
    df = df.set_index("deployment_population").loc[order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(df))
    ax = axes[0]
    for m in ["global", "mondrian", "lwcc_a"]:
        ax.plot(x, df[f"coverage_{m}"], marker="o", label=LABELS[m], color=COLORS[m])
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1, label="Nominal target (90%)")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace(" [", "\n[") for d in df["deployment_population"]], fontsize=8)
    ax.set_ylabel("Marginal coverage at deployment")
    ax.set_title("Coverage vs. deployment population\n(ordered by increasing patient age)")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.bar(x, df["mmd2_from_calibration_dist"], color="#5d6d7e")
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.replace(" [", "\n[") for d in df["deployment_population"]], fontsize=8)
    ax2.set_ylabel(r"$\widehat{\mathrm{MMD}}^2$(calibration, deployment)")
    ax2.set_title("Measured real covariate-shift magnitude\n(non-monotonic in age: shift is not purely\na function of age bucket order)")

    fig.suptitle("Coverage under real age-based covariate shift (calibrated on patients <70y)")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_coverage_vs_real_shift.png")
    plt.close(fig)


def fig3_abstention_curves():
    df = pd.read_csv(RESULTS / "abstention_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for m in ["global", "mondrian", "lwcc_a"]:
        sub = df[df.method == m].sort_values("abstention_rate")
        axes[0].plot(sub["abstention_rate"], sub["effective_coverage_retained"], marker="o", ms=3,
                     label=LABELS[m], color=COLORS[m])
        axes[1].plot(sub["abstention_rate"], sub["coverage_retained"], marker="o", ms=3,
                     label=LABELS[m], color=COLORS[m])
    axes[0].set_xlabel("Abstention rate (fraction referred, margin-based)")
    axes[0].set_ylabel("Effective coverage on retained cases")
    axes[0].set_title("Effective coverage vs. abstention")
    axes[1].set_xlabel("Abstention rate (fraction referred, margin-based)")
    axes[1].set_ylabel("Marginal coverage on retained cases")
    axes[1].set_title("Marginal coverage vs. abstention")
    for ax in axes:
        ax.legend(fontsize=7.5)
    fig.suptitle("Real coverage-abstention-actionability trade-off (test set, alpha=0.10)")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_abstention_tradeoff.png")
    plt.close(fig)


def fig4_mechanism_sample_size_vs_deviation():
    df = pd.read_csv(RESULTS / "subgroup_audit_all_methods.csv")
    race = df[df.subgroup_type == "race"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, marker in [("mondrian", "o"), ("lwcc_a", "s")]:
        ax.scatter(race["n_cal"], race[f"abs_dev_{m}"], label=LABELS[m], color=COLORS[m], marker=marker, s=70)
        for _, row in race.iterrows():
            ax.annotate(row["subgroup"], (row["n_cal"], row[f"abs_dev_{m}"]), fontsize=6.5,
                        textcoords="offset points", xytext=(5, 3), color=COLORS[m])
    ax.set_xscale("log")
    ax.set_xlabel("Calibration-set subgroup sample size, n_cal (log scale)")
    ax.set_ylabel("|coverage - 90% nominal target|")
    ax.set_title("Mechanism check: small subgroup sample size drives\nMondrian's coverage instability; local weighting is more robust")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_mechanism_sample_size_vs_deviation.png")
    plt.close(fig)


def fig5_model_reliability_diagram():
    import joblib
    clf = joblib.load(ROOT / "models" / "readmission_lgbm.joblib")
    X = pd.read_parquet(ROOT / "data" / "interim" / "X_full.parquet")
    y = np.load(ROOT / "data" / "interim" / "y_full.npy")
    idx_test = np.load(ROOT / "data" / "interim" / "idx_test.npy")
    p = clf.predict_proba(X.values[idx_test])[:, 1]
    y_t = y[idx_test]

    bins = np.quantile(p, np.linspace(0, 1, 11))
    bins[0] -= 1e-9
    bin_id = np.digitize(p, bins) - 1
    mean_pred, frac_pos, ns = [], [], []
    for b in range(10):
        mask = bin_id == b
        if mask.sum() == 0:
            continue
        mean_pred.append(p[mask].mean())
        frac_pos.append(y_t[mask].mean())
        ns.append(mask.sum())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot([0, max(mean_pred)], [0, max(mean_pred)], "k--", lw=1, label="Perfect calibration")
    axes[0].plot(mean_pred, frac_pos, marker="o", color="#1a5276")
    axes[0].set_xlabel("Mean predicted P(readmit<30d)")
    axes[0].set_ylabel("Observed readmission rate")
    axes[0].set_title("Base model reliability diagram (test set, decile bins)")
    axes[0].legend(fontsize=8)

    axes[1].hist(p, bins=40, color="#1a5276", alpha=0.8)
    axes[1].axvline(y_t.mean(), color="black", linestyle="--", label=f"Base rate ({y_t.mean():.3f})")
    axes[1].set_xlabel("Predicted P(readmit<30d)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Predicted risk score distribution")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_base_model_calibration.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1_subgroup_coverage_bars()
    fig2_shift_degradation()
    fig3_abstention_curves()
    fig4_mechanism_sample_size_vs_deviation()
    fig5_model_reliability_diagram()
    print("Figures written to", FIG)
