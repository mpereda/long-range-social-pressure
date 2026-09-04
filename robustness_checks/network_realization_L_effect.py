"""
network_realization_L_effect.py — Referee 1 major comment 3 / Referee 2 comment 7,
follow-up prompted by Maria: does the L=1->4 cooperation GAIN (the paper's central
claim), not just the absolute rho value at one L, hold up across independent network
realizations?

network_realization_robustness.py picks representative (b, theta) points SEPARATELY
for each L (coop/defect/transition/bistable are each redefined per L, since the
heatmap shifts with L), so its 10-realization data cannot be used to compute a
per-realization L=1->4 gain at a fixed point.

This script follows the same scope and point-selection convention as
nk_async_robustness.py (Referee 2 #8), which asked the analogous question for N, K,
and update scheme: correlated multiplex only, representative points taken once from
the L=4 sweep (data/02-*.csv) -- up to 4 per topology (coop/defect/transition/
bistable), 14 points total across the 4 topologies. For each of those 14 fixed
points, it runs L=1 AND L=4 on each of the 10 independent network realizations
already used in network_realization_robustness.py (identical net_seed = 1000*rlz + 7
convention -- realization 0 (seed=7) is therefore the same network nk_async_robustness
used as its own baseline, though this script does not depend on reusing that file),
so the gain rho(L=4)-rho(L=1) can be computed per realization and its spread compared
to the single-realization gain already reported in the main text.

Produces:
  - robustness_checks/data/network_realization_L_effect.csv: rho_mean, rho_std for
    L=1 and L=4 at each of the 14 (topo, z, regime, b, theta) x 10-realization
    combinations, correlated multiplex only.
"""
import os, sys, time
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from model import build_network, game_csr, shells_csr, geometric_kernel, run_replications

DATA_DIR = os.path.join(REPO, "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_CSV = os.path.join(OUT_DIR, "network_realization_L_effect.csv")

NETWORKS = [("BA", 4), ("BA", 16), ("ER", 4), ("ER", 16)]
L_VALUES = [1, 4]
N_NET_REALIZATIONS = 10
N_REP = 100
LAM = 0.5
N_NODES = 1000
K_FERMI = 0.1


def pick_representative_points(df):
    """Identical convention to network_realization_robustness.py and
    nk_async_robustness.py."""
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
    done = {(r["topo"], r["z"], r["regime"], r["L"], r["net_realization"]) for r in rows}

    t0 = time.time()
    for topo, z in NETWORKS:
        path = os.path.join(DATA_DIR, f"02-{topo}_z{z}-L4.csv")
        df = pd.read_csv(path)
        points = pick_representative_points(df)
        print(f"{topo}_z{z}: {len(points)} points -> {[p[0] for p in points]}")

        for label, b, theta in points:
            for rlz in range(N_NET_REALIZATIONS):
                net_seed = 1000 * rlz + 7
                G = build_network(topo, N_NODES, z, seed=net_seed)  # correlated: G_vig = G_game
                gp, gd = game_csr(G)
                for L in L_VALUES:
                    key = (topo, z, label, L, rlz)
                    if key in done:
                        continue
                    sp, sd = shells_csr(G, L)
                    al = geometric_kernel(L, LAM)
                    t1 = time.time()
                    rhos = run_replications(gp, gd, sp, sd, al, b, theta, K=K_FERMI,
                                             n_rep=N_REP, n_jobs=-1, base_seed=rlz * 100_000)
                    dt = time.time() - t1
                    rows.append(dict(topo=topo, z=z, regime=label, b=b, theta=theta, L=L,
                                      net_realization=rlz, rho_mean=float(rhos.mean()),
                                      rho_std=float(rhos.std(ddof=1)), seconds=dt))
                    print(f"{topo}_z{z} {label:10s} L={L} rlz={rlz}: "
                          f"rho={rhos.mean():.4f} ({dt:.1f}s)", flush=True)
                    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f"TOTAL TIME: {time.time()-t0:.1f}s")
    print("Saved:", OUT_CSV)


if __name__ == "__main__":
    main()
