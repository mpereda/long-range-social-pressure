"""
network_realization_Lgain_theta03.py — Referee 1 major comment 3 / Referee 2 comment 7,
second follow-up: does the cooperation gain from extending L hold up across independently
generated networks, at EVERY L (not only the L=1 vs. L=4 endpoint already checked in
network_realization_L_effect.py)?

Design, deliberately different from network_realization_L_effect.py's 14-point,
10-realization design:

  - That earlier check (and the underlying 111-point study, network_realization_robustness.py)
    is built for BROAD COVERAGE of the parameter space: many (topology, degree, correlation,
    L, regime) combinations, each with the same realization/replication budget used
    throughout the paper (10 realizations x 100 replications). It answers "is there
    disagreement between networks ANYWHERE in the space we explored".

  - This script instead asks a NARROWER but STATISTICALLY DEEPER question at a handful of
    points: does the L-dependence of cooperation hold up, with real statistical power on the
    realization axis specifically? Referee 1's major comment 3 raises exactly this trade-off:
    "I would suggest the authors consider at least 10 different realizations... If the
    computation costs are significantly higher, perhaps the number of initializations can be
    reduced." So here: MORE realizations (30, vs. 10 elsewhere).

    NOTE: an earlier pass at 20 replications/realization (archived as
    data/network_realization_Lgain_theta03_nrep20_PARTIAL.csv) found one point (BA z=4,
    b=1.8) where a single realization's L=1->4 gain came out negative against a typical
    +0.12; that realization's within-realization std was itself unusually high at both L=1
    and L=4 (an internally noisy/near-bistable network), and at 20 replications the
    estimated gain was not statistically distinguishable from zero (|gain|/SEM = 0.82). Kept
    replications at 100 (matching the budget used throughout the rest of the paper, not
    reduced) specifically to resolve whether that is genuine between-network variability or
    estimation noise.

Points: fixed at theta=0.3, the reference line already used for the critical-temptation
percentages quoted in the abstract and Sec. III B (the ``theta=0.3 crossing'' convention).
Two b values per topology, chosen from the corrected L=1/L=4 main-sweep data as the points
of largest and second-largest L=1->4 gain along that line (i.e., where the effect being
defended is strongest, so where a referee would most want to see it holds up):

  BA  z=4:  b=1.8 (gain 0.124), b=2.0 (gain 0.317)
  BA  z=16: b=1.5 (gain 0.551), b=1.7 (gain 0.414)
  ER  z=4:  b=1.4 (gain 0.831), b=1.6 (gain 0.997)
  ER  z=16: b=1.1 (gain 0.675), b=1.2 (gain 0.785)

All L in {1,2,3,4} (not just 1 and 4), correlated multiplex only, so the full incremental
L=1->2->3->4 pattern reported in Sec. III B can be checked, not only the two endpoints.

net_seed = 1000*rlz + 7 for rlz in 0..29 -- same convention as
network_realization_robustness.py and network_realization_L_effect.py, so realizations
0-9 here are identical networks to the ones already used in those studies.

Produces:
  - robustness_checks/data/network_realization_Lgain_theta03.csv: rho_mean, rho_std for each
    (topology, b, L, realization) combination.
"""
import os, sys, time
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from model import build_network, game_csr, shells_csr, geometric_kernel, run_replications

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_CSV = os.path.join(OUT_DIR, "network_realization_Lgain_theta03.csv")

THETA = 0.3
POINTS = [
    ("BA", 4, 1.8), ("BA", 4, 2.0),
    ("BA", 16, 1.5), ("BA", 16, 1.7),
    ("ER", 4, 1.4), ("ER", 4, 1.6),
    ("ER", 16, 1.1), ("ER", 16, 1.2),
]
L_VALUES = [1, 2, 3, 4]
N_NET_REALIZATIONS = 30
N_REP = 100
LAM = 0.5
N_NODES = 1000
K_FERMI = 0.1


def main():
    rows = []
    if os.path.exists(OUT_CSV):
        rows = pd.read_csv(OUT_CSV).to_dict("records")
        print(f"Resuming: {len(rows)} results already present in {OUT_CSV}")
    done = {(r["topo"], r["z"], r["b"], r["L"], r["net_realization"]) for r in rows}

    t0 = time.time()
    # Group by (topo, z) so each network realization is built once and reused across
    # both b points at that topology/degree (correlated: G_vig = G_game).
    by_topo = {}
    for topo, z, b in POINTS:
        by_topo.setdefault((topo, z), []).append(b)

    for (topo, z), bs in by_topo.items():
        for rlz in range(N_NET_REALIZATIONS):
            net_seed = 1000 * rlz + 7
            needed = any((topo, z, b, L, rlz) not in done for b in bs for L in L_VALUES)
            if not needed:
                continue
            G = build_network(topo, N_NODES, z, seed=net_seed)
            gp, gd = game_csr(G)
            for b in bs:
                for L in L_VALUES:
                    key = (topo, z, b, L, rlz)
                    if key in done:
                        continue
                    sp, sd = shells_csr(G, L)
                    al = geometric_kernel(L, LAM)
                    t1 = time.time()
                    rhos = run_replications(gp, gd, sp, sd, al, b, THETA, K=K_FERMI,
                                             n_rep=N_REP, n_jobs=-1, base_seed=rlz * 100_000)
                    dt = time.time() - t1
                    rows.append(dict(topo=topo, z=z, b=b, theta=THETA, L=L,
                                      net_realization=rlz, rho_mean=float(rhos.mean()),
                                      rho_std=float(rhos.std(ddof=1)), seconds=dt))
                    print(f"{topo}_z{z} b={b} L={L} rlz={rlz}: "
                          f"rho={rhos.mean():.4f} ({dt:.1f}s)", flush=True)
                    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f"TOTAL TIME: {time.time()-t0:.1f}s")
    print("Saved:", OUT_CSV)


if __name__ == "__main__":
    main()
