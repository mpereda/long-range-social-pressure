"""
network_realization_robustness.py — Referee 1 #3 + Referee 2 #7

Checks whether the paper's conclusions depend on the single fixed network
realization used per topology/degree in the main sweep (notebooks 02/03).

For each (topology, z, correlation, L) combination already swept in
data/02-*.csv (correlated) and data/03-*.csv (uncorrelated), picks up to
4 representative (b, theta) points from the *corrected* sweep (coop,
defect, transition, bistable) and re-runs them on N_NET_REALIZATIONS
independent network realizations (100 replications each, same as the main
text), reporting the between-network spread of rho_mean alongside the
within-network (between-initial-condition) spread already known from
rho_std.

Must be run AFTER the corrected 02/03 sweeps (needs their CSVs to pick
representative points). Checkpoints after every realization so a partial
run is never lost.
"""
import os, sys, time
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from model import build_network, game_csr, shells_csr, geometric_kernel, run_replications

DATA_DIR = os.path.join(REPO, "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "network_realization_robustness.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "network_realization_robustness_summary.csv")

NETWORKS = [("BA", 4), ("BA", 16), ("ER", 4), ("ER", 16)]
CORRELATIONS = ["corr", "uncorr"]  # corr -> data/02-*.csv, uncorr -> data/03-*.csv
L_VALUES = [1, 4]                  # extremes: direct-neighbor vs widest range studied
N_NET_REALIZATIONS = 10
N_REP = 100
LAM = 0.5
N_NODES = 1000
K_FERMI = 0.1


def pick_representative_points(df):
    """From a (b, theta, rho_mean, rho_std) sweep, pick up to 4 representative
    points: 'coop' (max rho_mean), 'defect' (min rho_mean), 'transition'
    (rho_mean closest to 0.5), 'bistable' (max rho_std). De-duplicated."""
    candidates = [
        ("coop", df.loc[df["rho_mean"].idxmax()]),
        ("defect", df.loc[df["rho_mean"].idxmin()]),
        ("transition", df.loc[(df["rho_mean"] - 0.5).abs().idxmin()]),
        ("bistable", df.loc[df["rho_std"].idxmax()]),
    ]
    seen, out = set(), []
    for label, row in candidates:
        key = (round(float(row["b"]), 6), round(float(row["theta"]), 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((label, float(row["b"]), float(row["theta"])))
    return out


def main():
    rows = []
    if os.path.exists(OUT_CSV):
        rows = pd.read_csv(OUT_CSV).to_dict("records")
        print(f"Resuming: {len(rows)} results already present in {OUT_CSV}")
    done = {(r["topo"], r["z"], r["corr"], r["L"], r["regime"], r["net_realization"]) for r in rows}

    t0 = time.time()
    for topo, z in NETWORKS:
        for corr in CORRELATIONS:
            for L in L_VALUES:
                prefix = "02" if corr == "corr" else "03"
                key = f"{topo}_z{z}"
                path = os.path.join(DATA_DIR, f"{prefix}-{key}-L{L}.csv")
                if not os.path.exists(path):
                    print(f"  MISSING: {path} -- skipping (was it swept?)")
                    continue
                df = pd.read_csv(path)
                points = pick_representative_points(df)
                for label, b, theta in points:
                    for rlz in range(N_NET_REALIZATIONS):
                        if (topo, z, corr, L, label, rlz) in done:
                            continue
                        net_seed = 1000 * rlz + 7
                        if corr == "corr":
                            G = build_network(topo, N_NODES, z, seed=net_seed)
                            Gg, Gv = G, G
                        else:
                            Gg = build_network(topo, N_NODES, z, seed=net_seed)
                            Gv = build_network(topo, N_NODES, z, seed=net_seed + 500_000)
                        gp, gd = game_csr(Gg)
                        sp, sd = shells_csr(Gv, L)
                        al = geometric_kernel(L, LAM)
                        t1 = time.time()
                        rhos = run_replications(gp, gd, sp, sd, al, b, theta,
                                                 K=K_FERMI, n_rep=N_REP, n_jobs=-1,
                                                 base_seed=rlz * 100_000)
                        dt = time.time() - t1
                        rows.append(dict(topo=topo, z=z, corr=corr, L=L, regime=label,
                                          b=b, theta=theta, net_realization=rlz,
                                          rho_mean=float(rhos.mean()), rho_std=float(rhos.std(ddof=1)),
                                          seconds=dt))
                        print(f"{key} {corr} L={L} {label:10s} b={b} th={theta} rlz={rlz}: "
                              f"rho={rhos.mean():.4f} ({dt:.1f}s)", flush=True)
                        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"TOTAL TIME: {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    summary = df.groupby(["topo", "z", "corr", "L", "regime", "b", "theta"]).agg(
        between_network_mean=("rho_mean", "mean"),
        between_network_std=("rho_mean", "std"),
        n_realizations=("rho_mean", "count"),
    ).reset_index()
    summary.to_csv(SUMMARY_CSV, index=False)
    print(summary.to_string(index=False))
    print("Saved:", OUT_CSV, SUMMARY_CSV)


if __name__ == "__main__":
    main()
