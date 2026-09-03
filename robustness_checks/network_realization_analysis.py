"""
network_realization_analysis.py — Referee 1 major comment 3 / Referee 2 comment 7

Analyzes robustness_checks/data/network_realization_robustness_summary.csv
(produced by network_realization_robustness.py) against the corresponding
within-network variability already reported in the main text (rho_std from
the main data/02-*.csv / data/03-*.csv sweeps at the same (b, theta) point),
to answer the question the robustness study was run for: is the spread
across independent network realizations small compared to the spread we
already report and account for (100 initial conditions on a single fixed
network)?

Produces:
  - robustness_checks/data/network_realization_robustness_analysis.csv:
    the summary table with within_network_std and the ratio
    between_network_std / within_network_std joined in.
  - figures/network_realization_robustness.pdf: a parity scatter of
    within-network std (x) vs. between-network std (y), one point per
    (topology, z, correlation, L, regime) combination, colored by regime,
    with a y=x reference line -- points below the line are the reassuring
    majority (network-realization variability smaller than the
    already-reported within-network variability).

NOTE: row['corr'] must be accessed by key, not attribute (row.corr) --
pandas Series/DataFrame has a built-in .corr() correlation method that
silently shadows a column literally named 'corr' when accessed as an
attribute, returning the bound method instead of the column value. This
bit us once during interactive analysis (silently pulled data/03-*.csv,
the wrong condition, for every 'corr' row) before being caught.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(REPO, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(OUT_DIR, "data", "network_realization_robustness_summary.csv")
ANALYSIS_CSV = os.path.join(OUT_DIR, "data", "network_realization_robustness_analysis.csv")
TABLE_CSV = os.path.join(OUT_DIR, "data", "network_realization_robustness_table.csv")
FIG_PATH = os.path.join(FIG_DIR, "network_realization_robustness.pdf")

REGIME_ORDER = ["coop", "defect", "transition", "bistable"]
REGIME_LABELS = {"coop": "Cooperative", "defect": "Defective",
                  "transition": "Transition", "bistable": "Bistable"}
REGIME_COLORS = {"coop": "#2ca02c", "defect": "#d62728",
                  "transition": "#ff7f0e", "bistable": "#9467bd"}
REGIME_MARKERS = {"coop": "o", "defect": "s", "transition": "^", "bistable": "D"}


def within_network_std(row):
    """rho_std from the main sweep at this exact (b, theta) point. Uses
    row['corr'], NOT row.corr (see module docstring)."""
    prefix = "02" if row["corr"] == "corr" else "03"
    path = os.path.join(DATA_DIR, f'{prefix}-{row["topo"]}_z{row["z"]}-L{row["L"]}.csv')
    df = pd.read_csv(path)
    m = df[(np.isclose(df.b, row["b"])) & (np.isclose(df.theta, row["theta"]))]
    return float(m.iloc[0]["rho_std"]) if len(m) else np.nan


def main():
    summ = pd.read_csv(SUMMARY_CSV)
    summ["within_network_std"] = summ.apply(within_network_std, axis=1)
    summ["ratio"] = summ["between_network_std"] / summ["within_network_std"].replace(0, np.nan)
    summ.to_csv(ANALYSIS_CSV, index=False)
    print(f"Saved {ANALYSIS_CSV}")

    # ── Summary table, by regime ────────────────────────────────────────
    table = summ.groupby("regime").agg(
        n_points=("ratio", "size"),
        within_std_mean=("within_network_std", "mean"),
        between_std_mean=("between_network_std", "mean"),
        ratio_mean=("ratio", "mean"),
        ratio_median=("ratio", "median"),
        ratio_max=("ratio", "max"),
    ).reindex(REGIME_ORDER)
    table.to_csv(TABLE_CSV)
    print(f"Saved {TABLE_CSV}")
    print(table.round(3).to_string())

    # ── Parity scatter: within-network std (x) vs. between-network std (y) ──
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    lim = max(summ["within_network_std"].max(), summ["between_network_std"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", lw=1, ls="--", zorder=1,
            label=r"$y=x$")
    for regime in REGIME_ORDER:
        sub = summ[summ.regime == regime]
        ax.scatter(sub["within_network_std"], sub["between_network_std"],
                   s=36, alpha=0.8, color=REGIME_COLORS[regime],
                   marker=REGIME_MARKERS[regime], label=REGIME_LABELS[regime],
                   zorder=2, edgecolors="white", linewidths=0.4)
    ax.set_xlabel(r"Within-network std (100 initial conditions, main sweep)")
    ax.set_ylabel(r"Between-network std (10 realizations)")
    pad = 0.015 * lim
    ax.set_xlim(-pad, lim)
    ax.set_ylim(-pad, lim)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax.set_title("Between-network vs. within-network variability")
    fig.savefig(FIG_PATH, bbox_inches="tight")
    print(f"Saved {FIG_PATH}")


if __name__ == "__main__":
    main()
