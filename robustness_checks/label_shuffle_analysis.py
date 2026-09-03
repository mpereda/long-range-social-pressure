"""
label_shuffle_analysis.py — Referee 2 #6 follow-up

Analyzes robustness_checks/data/label_shuffle_control.csv: does destroying
the residual inter-layer degree correlation r_k (by permuting G_vig's node
labels, on top of the existing 'uncorr' independent-realization condition)
further reduce BA's residual sensitivity to decorrelation, confirming r_k
(not some other latent effect of independent realizations) as the cause?

The z=16 'coop' point (b=1.0, theta=0.0) is degenerate: at this minimal
temptation the dynamics never leaves the initial state, so all three
conditions and both L give bit-for-bit identical rho -- it carries no
information about the mechanism and is flagged, not excluded from the
table (for transparency) but excluded from the summary statistics.

Produces:
  - a per-point table (corr, uncorr, uncorr_shuffled means + SEM, and the
    two differences) for the Supplemental table.
  - a categorical dot-and-line figure: one point per (topology, regime, L),
    three markers (corr/uncorr/shuffled) connected by a line, y = rho.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(REPO, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

IN_CSV = os.path.join(OUT_DIR, "data", "label_shuffle_control.csv")
TABLE_CSV = os.path.join(OUT_DIR, "data", "label_shuffle_control_table.csv")
FIG_PATH = os.path.join(FIG_DIR, "label_shuffle_control.pdf")

COND_COLORS = {"corr": "#1f77b4", "uncorr": "#2ca02c", "uncorr_shuffled": "#d62728"}
COND_MARKERS = {"corr": "o", "uncorr": "s", "uncorr_shuffled": "^"}
COND_LABELS = {"corr": "correlated", "uncorr": "uncorrelated",
               "uncorr_shuffled": "uncorrelated,\nlabel-shuffled"}
DEGENERATE = {(16, "coop")}  # b=1.0, theta=0.0: frozen at the initial state


def main():
    df = pd.read_csv(IN_CSV)
    df["sem"] = df["rho_std"] / 10.0  # 100 replications

    rows = []
    for (z, regime, L), g in df.groupby(["z", "regime", "L"]):
        g = g.set_index("condition")
        row = dict(z=z, regime=regime, L=L,
                   b=g["b"].iloc[0], theta=g["theta"].iloc[0],
                   rho_corr=g.loc["corr", "rho_mean"],
                   rho_uncorr=g.loc["uncorr", "rho_mean"],
                   rho_shuffled=g.loc["uncorr_shuffled", "rho_mean"],
                   corr_to_uncorr=g.loc["uncorr", "rho_mean"] - g.loc["corr", "rho_mean"],
                   uncorr_to_shuffled=g.loc["uncorr_shuffled", "rho_mean"] - g.loc["uncorr", "rho_mean"],
                   sem_uncorr=g.loc["uncorr", "sem"],
                   sem_shuffled=g.loc["uncorr_shuffled", "sem"],
                   degenerate=(z, regime) in DEGENERATE)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["z", "L", "regime"])
    table["combined_sem"] = np.sqrt(table.sem_uncorr**2 + table.sem_shuffled**2)
    table["z_score"] = table.uncorr_to_shuffled / table.combined_sem
    table.to_csv(TABLE_CSV, index=False)
    print(table.round(3).to_string(index=False))
    print(f"Saved {TABLE_CSV}")

    informative = table[~table.degenerate & (table.corr_to_uncorr.abs() > 0.02)]
    same_dir = (informative.corr_to_uncorr * informative.uncorr_to_shuffled) > 0
    print(f"\nInformative points: {len(informative)}; "
          f"same-direction (shuffle continues the corr->uncorr shift): {same_dir.sum()}")

    # ── Figure: one panel per z, x = (regime, L), y = rho, 3 conditions ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True, sharey=True)
    for ax, z in zip(axes, [4, 16]):
        sub = table[table.z == z].reset_index(drop=True)
        xlabels = [f"{r}\nL={L}" + ("*" if deg else "")
                   for r, L, deg in zip(sub.regime, sub.L, sub.degenerate)]
        x = np.arange(len(sub))
        for cond, col in [("rho_corr", "corr"), ("rho_uncorr", "uncorr"),
                           ("rho_shuffled", "uncorr_shuffled")]:
            yerr = sub["sem_uncorr"] if col == "uncorr" else (
                sub["sem_shuffled"] if col == "uncorr_shuffled" else None)
            ax.errorbar(x, sub[cond], yerr=yerr, fmt=COND_MARKERS[col], color=COND_COLORS[col],
                        label=COND_LABELS[col], ms=7, capsize=3, lw=1)
        for xi in x:
            ax.plot([xi, xi, xi], [sub.rho_corr[xi], sub.rho_uncorr[xi], sub.rho_shuffled[xi]],
                    color="grey", lw=0.7, zorder=0, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_title(f"BA $z={z}$")
        ax.set_ylim(-0.05, 1.05)
        if z == 4:
            ax.set_ylabel(r"$\langle\rho\rangle$")
    axes[1].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle("Correlated vs. uncorrelated vs. label-shuffled BA multiplex "
                  "(* = degenerate point, see text)")
    fig.savefig(FIG_PATH, bbox_inches="tight")
    print(f"Saved {FIG_PATH}")


if __name__ == "__main__":
    main()
