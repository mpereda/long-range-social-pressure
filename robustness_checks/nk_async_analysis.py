"""
nk_async_analysis.py — Referee 2 comment 8

Analyzes robustness_checks/data/nk_async_robustness.csv (produced by
nk_async_robustness.py) to answer the question the study was run for: does
the central finding -- that extending vigilance from L=1 to L=4 raises
cooperation -- hold qualitatively at population sizes N != 1000, Fermi
noise K != 0.1, and under asynchronous strategy updating?

For each (topology, z, regime) representative point, computes the L=1->4
gain [rho(L=4) - rho(L=1)] under the baseline condition (N=1000, K=0.1,
synchronous) and under every alternative condition (N in {250,500,2000},
K in {0.05,0.2}, async at N=1000/K=0.1), then compares.

Produces:
  - robustness_checks/data/nk_async_robustness_analysis.csv: gain per
    (topo, z, regime, condition), joined with the baseline gain and their
    difference.
  - robustness_checks/data/nk_async_robustness_table.csv: summary table,
    one row per non-baseline condition.
  - figures/nk_async_robustness.pdf: three-panel parity scatter (baseline
    gain on x, alternative-condition gain on y, y=x reference line) --
    panel (a) N sweep, (b) K sweep, (c) sync vs. async.
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

IN_CSV = os.path.join(OUT_DIR, "data", "nk_async_robustness.csv")
ANALYSIS_CSV = os.path.join(OUT_DIR, "data", "nk_async_robustness_analysis.csv")
TABLE_CSV = os.path.join(OUT_DIR, "data", "nk_async_robustness_table.csv")
FIG_PATH = os.path.join(FIG_DIR, "nk_async_robustness.pdf")

N_COLORS = {250: "#1f77b4", 500: "#2ca02c", 2000: "#d62728"}
K_COLORS = {0.05: "#1f77b4", 0.2: "#d62728"}
CONDITION_ORDER = [("N", 250), ("N", 500), ("N", 2000),
                    ("K", 0.05), ("K", 0.2), ("async", 1.0)]
CONDITION_LABELS = {("N", 250): "N=250", ("N", 500): "N=500", ("N", 2000): "N=2000",
                     ("K", 0.05): "K=0.05", ("K", 0.2): "K=0.20", ("async", 1.0): "Asynchronous"}


def gain_by_point(sub):
    """rho(L=4) - rho(L=1), indexed by (topo, z, regime)."""
    piv = sub.pivot_table(index=["topo", "z", "regime"], columns="L", values="rho_mean")
    return piv[4] - piv[1]


def main():
    df = pd.read_csv(IN_CSV)

    baseline = df[(df.sweep == "N") & (df.value == 1000)]
    base_gain = gain_by_point(baseline).rename("base_gain")

    rows = []
    for sweep, value in CONDITION_ORDER:
        sub = df[(df.sweep == sweep) & (np.isclose(df.value, value))]
        g = gain_by_point(sub).rename("cond_gain")
        joined = pd.concat([base_gain, g], axis=1).dropna().reset_index()
        joined["sweep"] = sweep
        joined["value"] = value
        joined["diff"] = joined["cond_gain"] - joined["base_gain"]
        rows.append(joined)
    analysis = pd.concat(rows, ignore_index=True)
    analysis.to_csv(ANALYSIS_CSV, index=False)
    print(f"Saved {ANALYSIS_CSV}")

    table = analysis.groupby(["sweep", "value"]).agg(
        n_points=("diff", "size"),
        mean_abs_diff=("diff", lambda x: x.abs().mean()),
        max_abs_diff=("diff", lambda x: x.abs().max()),
    ).reset_index()
    table["condition"] = table.apply(lambda r: CONDITION_LABELS[(r.sweep, r.value)], axis=1)
    table = table[["condition", "n_points", "mean_abs_diff", "max_abs_diff"]]
    table.to_csv(TABLE_CSV, index=False)
    print(table.round(3).to_string(index=False))
    print(f"Saved {TABLE_CSV}")

    # ── Six-panel parity scatter: rows = L, columns = sweep type ────────
    # Shows rho itself at each L separately (not the collapsed L=1->4
    # gain), per Maria's request -- don't discard the per-L information.
    baseline_raw = df[(df.sweep == "N") & (df.value == 1000)].set_index(
        ["topo", "z", "regime", "L"])["rho_mean"]

    lim = 1.03
    pad = 0.02
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2), constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.12)
    col_titles = ["(a) Population size $N$", "(b) Fermi noise $K$",
                  "(c) Synchronous vs. asynchronous"]

    for col, (sweep, colors, marker, title) in enumerate([
            ("N", N_COLORS, "o", col_titles[0]),
            ("K", K_COLORS, "s", col_titles[1]),
            ("async", {1.0: "#9467bd"}, "^", col_titles[2])]):
        for row, L in enumerate([1, 4]):
            ax = axes[row, col]
            ax.plot([0, lim], [0, lim], color="grey", lw=1, ls="--", zorder=1, label=r"$y=x$")
            sub = df[(df.sweep == sweep) & (df.L == L)]
            for value, color in colors.items():
                s = sub[np.isclose(sub.value, value)]
                x = [baseline_raw.get((t, z, r, L), np.nan)
                     for t, z, r in zip(s.topo, s.z, s.regime)]
                ax.scatter(x, s["rho_mean"], s=40, alpha=0.8, color=color,
                           marker=marker, label=CONDITION_LABELS[(sweep, value)],
                           zorder=2, edgecolors="white", linewidths=0.4)
            ax.set_xlim(-pad, lim)
            ax.set_ylim(-pad, lim)
            ax.set_aspect("equal")
            ax.set_xlabel(r"Baseline $\langle\rho\rangle$" + "\n(N=1000, K=0.1, sync)")
            ax.set_ylabel(r"Alternative-condition $\langle\rho\rangle$")
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.text(-0.22, 0.5, f"$L={L}$", transform=ax.transAxes, fontsize=13,
                        fontweight="bold", va="center", ha="center", rotation=90)
            ax.legend(loc="lower right", framealpha=0.9, fontsize=8)

    fig.suptitle(r"Robustness of $\langle\rho\rangle$ at $L=1$ and $L=4$ to $N$, $K$, and update scheme",
                 fontsize=12)
    fig.savefig(FIG_PATH, bbox_inches="tight")
    print(f"Saved {FIG_PATH}")


if __name__ == "__main__":
    main()
