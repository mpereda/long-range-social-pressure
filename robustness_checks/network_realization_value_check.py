"""
network_realization_value_check.py — Referee 1 major comment 3 / Referee 2 comment 7,
follow-up prompted by Maria: does the *value* of <rho> reported throughout the paper
(single fixed network realization) agree with the value from other realizations, not
just its spread?

network_realization_analysis.py already compares between-network std to within-network
std (dispersion). It never checks whether the specific fixed realization used in the
main sweep is itself typical of the 10-realization distribution, or a systematic
outlier at some points. This script does that: for each of the 111 representative
(topology, z, correlation, L, regime) points, it compares the main-sweep <rho> (from
data/02-*.csv / data/03-*.csv, the actual fixed realization used everywhere else in the
paper) against the mean of the 10 independent realizations in
network_realization_robustness.csv (none of which reuse the main-sweep's seed -- see
that script's net_seed = 1000*rlz + 7 convention).

Produces:
  - robustness_checks/data/network_realization_value_check.csv: per-point comparison
    (orig, 10-realization mean, difference, z-score against the 10-realization spread).
  - prints the summary used in the writeup: overall correlation, per-regime mean/median
    |diff|, and the list of points where |diff| > 0.1.
"""
import os
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

IN_CSV = os.path.join(OUT_DIR, "data", "network_realization_robustness.csv")
OUT_CSV = os.path.join(OUT_DIR, "data", "network_realization_value_check.csv")


def main():
    df = pd.read_csv(IN_CSV)

    rows = []
    for (topo, z, corr, L, regime), g in df.groupby(["topo", "z", "corr", "L", "regime"]):
        b, theta = g["b"].iloc[0], g["theta"].iloc[0]
        new_mean = g["rho_mean"].mean()
        new_std = g["rho_mean"].std(ddof=1)
        prefix = "02" if corr == "corr" else "03"
        orig_df = pd.read_csv(os.path.join(DATA_DIR, f"{prefix}-{topo}_z{z}-L{L}.csv"))
        orig_row = orig_df[(orig_df.b == b) & (orig_df.theta == theta)]
        orig_val = float(orig_row["rho_mean"].iloc[0])
        zscore = (orig_val - new_mean) / new_std if new_std > 0 else np.nan
        rows.append(dict(topo=topo, z=z, corr=corr, L=L, regime=regime, b=b, theta=theta,
                          orig=orig_val, realization_mean=new_mean, realization_std=new_std,
                          diff=orig_val - new_mean, zscore=zscore))
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)

    print(f"N points: {len(res)}")
    print(f"Overall correlation(orig, 10-realization mean): {res['orig'].corr(res['realization_mean']):.4f}")
    print()
    print("Mean/median |diff| by regime:")
    g = res.groupby("regime")["diff"].agg(mean_abs=lambda x: x.abs().mean(),
                                           median_abs=lambda x: x.abs().median(),
                                           n="size")
    print(g.round(3).to_string())
    print()
    outliers = res[res["diff"].abs() > 0.1].sort_values("diff", key=lambda s: s.abs(), ascending=False)
    print(f"Points with |diff| > 0.1: {len(outliers)} (all in regime: "
          f"{sorted(outliers['regime'].unique())})")
    print(outliers.round(3).to_string(index=False))
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
