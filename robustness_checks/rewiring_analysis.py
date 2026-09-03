"""
rewiring_analysis.py — Referee 1 major comment 4

Analyzes robustness_checks/data/rewiring_interpolation.csv: is the
correlated -> uncorrelated transition continuous or discontinuous with
respect to inter-layer edge overlap omega (parametrized by rewired
fraction r)?

Produces:
  - a transition-width table: for each (b, L), the r-range over which
    <rho> crosses from 0.9 down to 0.1 (linear interpolation between grid
    points), and the peak per-replication std reached along the sweep.
  - a 2x2 figure: top row <rho>(r) per b, bottom row std(rho)(r) per b,
    columns L=2, L=4.
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

IN_CSV = os.path.join(OUT_DIR, "data", "rewiring_interpolation.csv")
TABLE_CSV = os.path.join(OUT_DIR, "data", "rewiring_interpolation_table.csv")
FIG_PATH = os.path.join(FIG_DIR, "rewiring_interpolation.pdf")

B_COLORS = {1.3: "#7f7f7f", 1.5: "#1f77b4", 1.7: "#2ca02c", 1.9: "#ff7f0e", 2.0: "#d62728"}


def crossing_r(r, rho, level):
    """First r (linear interp) where rho crosses `level` going downward."""
    for i in range(len(r) - 1):
        if rho[i] >= level >= rho[i + 1] and rho[i] != rho[i + 1]:
            frac = (rho[i] - level) / (rho[i] - rho[i + 1])
            return r[i] + frac * (r[i + 1] - r[i])
    return np.nan


def main():
    df = pd.read_csv(IN_CSV)

    rows = []
    for L in [2, 4]:
        for b in sorted(df.b.unique()):
            sub = df[(df.L == L) & (df.b == b)].sort_values("r")
            r = sub.r.values
            rho = sub.rho_mean.values
            std = sub.rho_std.values
            r90 = crossing_r(r, rho, 0.9)
            r10 = crossing_r(r, rho, 0.1)
            width = r10 - r90 if (not np.isnan(r90) and not np.isnan(r10)) else np.nan
            rows.append(dict(L=L, b=b, r_at_rho0p9=r90, r_at_rho0p1=r10,
                              width=width, max_std=std.max(),
                              r_at_max_std=r[std.argmax()]))
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_CSV, index=False)
    print(table.round(3).to_string(index=False))
    print(f"Saved {TABLE_CSV}")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, constrained_layout=True)
    for col, L in enumerate([2, 4]):
        sub = df[df.L == L]
        ax_rho, ax_std = axes[0, col], axes[1, col]
        for b in sorted(sub.b.unique()):
            s = sub[sub.b == b].sort_values("r")
            ax_rho.plot(s.r, s.rho_mean, "o-", color=B_COLORS[b], label=f"$b={b}$", ms=4)
            ax_std.plot(s.r, s.rho_std, "o-", color=B_COLORS[b], ms=4)
        ax_rho.set_title(f"$L={L}$")
        ax_rho.set_ylim(-0.03, 1.03)
        ax_std.set_ylim(-0.02, 0.55)
        ax_std.set_xlabel("Rewired fraction $r$")
        if col == 0:
            ax_rho.set_ylabel(r"$\langle\rho\rangle$")
            ax_std.set_ylabel(r"std($\rho$) across replications")
    axes[0, 1].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(r"Correlated ($r=0$) $\to$ uncorrelated ($r=1$) interpolation, ER $z=4$, $\theta=0.2$")
    fig.savefig(FIG_PATH, bbox_inches="tight")
    print(f"Saved {FIG_PATH}")


if __name__ == "__main__":
    main()
