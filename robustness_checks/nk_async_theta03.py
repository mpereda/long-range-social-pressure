"""
nk_async_theta03.py — Referee 2 #8, redesign

Checks whether the paper's central observation -- that extending vigilance from L=1 to
L=4 substantially raises cooperation -- depends on the choice of population size N, Fermi
noise K, or synchronous vs. asynchronous strategy updating, all fixed at N=1000, K=0.1,
synchronous throughout the main text.

Replaces nk_async_robustness.py, which picked its 14 points by REGIME at L=4
(coop/defect/transition/bistable -- same convention as network_realization_robustness.py).
That selection is the wrong one for this specific question: a "coop" or "defect" point at
L=4 can have essentially zero L=1->4 gain if L=1 is already saturated there too (no room
to improve), so testing whether "the gain survives" at such a point is not a meaningful
test -- there is barely any gain to lose. This script instead reuses the 8 points from
network_realization_Lgain_theta03.py (Referee 1 major comment 3 / Referee 2 comment 7,
second follow-up): the two (b, theta) points per topology with the LARGEST L=1->4 gain
along the theta=0.3 reference line already used for the critical-temptation percentages,
so the N/K/update-rule sensitivity is tested exactly where there is a real gain to
potentially lose.

For each of the 8 points:
  (a) N sweep: reruns at N in {250, 500, 1000, 2000} (baseline K=0.1, sync),
      for both L=1 and L=4, so the L=1->4 gain itself (not just each L
      separately) can be checked for N-dependence.
  (b) K sweep: reruns at K in {0.05, 0.1, 0.2} (baseline N=1000, sync),
      same L=1 and L=4 pair.
  (c) async control: reruns at the baseline N=1000, K=0.1 but with the
      asynchronous Fermi update (update_rule='fermi_async'), same L pair.

All three sets use 100 replications per (point, L, condition), matching the main text and
the discontinued nk_async_robustness.py. A single fixed network realization per point
(net_seed=7, same as before) -- this script checks N/K/async robustness, not
network-realization robustness (that is network_realization_Lgain_theta03.py's job);
keeping them orthogonal avoids conflating the two questions.

Produces:
  - robustness_checks/data/nk_async_theta03.csv: rho_mean, rho_std for each
    (topo, z, b, L, sweep, value) combination.
"""
import os, sys, time
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from model import build_network, game_csr, shells_csr, geometric_kernel, run_replications

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "nk_async_theta03.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "nk_async_theta03_summary.csv")

THETA = 0.3
POINTS = [
    ("BA", 4, 1.8), ("BA", 4, 2.0),
    ("BA", 16, 1.5), ("BA", 16, 1.7),
    ("ER", 4, 1.4), ("ER", 4, 1.6),
    ("ER", 16, 1.1), ("ER", 16, 1.2),
]
L_VALUES = [1, 4]                   # the extremes whose gain is the paper's central claim
N_VALUES = [250, 500, 1000, 2000]   # baseline: 1000
K_VALUES = [0.05, 0.1, 0.2]         # baseline: 0.1
N_REP = 100
LAM = 0.5
N_BASELINE = 1000
K_BASELINE = 0.1
NET_SEED = 7  # single fixed network realization per point


def run_condition(topo, z, L, b, theta, N, K, update_rule, seed):
    G = build_network(topo, N, z, seed=seed)
    gp, gd = game_csr(G)
    sp, sd = shells_csr(G, L)
    al = geometric_kernel(L, LAM)
    rhos = run_replications(gp, gd, sp, sd, al, b, theta, K=K, n_rep=N_REP,
                             n_jobs=-1, base_seed=seed * 100_000,
                             update_rule=update_rule)
    return float(rhos.mean()), float(rhos.std(ddof=1))


def main():
    rows = []
    if os.path.exists(OUT_CSV):
        rows = pd.read_csv(OUT_CSV).to_dict("records")
        print(f"Resuming: {len(rows)} results already present in {OUT_CSV}")
    done = {(r["topo"], r["z"], r["b"], r["L"], r["sweep"], r["value"]) for r in rows}

    t0 = time.time()
    for topo, z, b in POINTS:
        for L in L_VALUES:
            for N in N_VALUES:
                key = (topo, z, b, L, "N", N)
                if key in done:
                    continue
                t1 = time.time()
                mean, std = run_condition(topo, z, L, b, THETA, N, K_BASELINE,
                                           "fermi", NET_SEED)
                dt = time.time() - t1
                rows.append(dict(topo=topo, z=z, b=b, theta=THETA, L=L, sweep="N",
                                  value=N, K=K_BASELINE, update_rule="fermi",
                                  rho_mean=mean, rho_std=std, seconds=dt))
                print(f"{topo}_z{z} b={b} L={L} N={N:5d} K={K_BASELINE}: "
                      f"rho={mean:.4f} ({dt:.1f}s)", flush=True)
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

            for K in K_VALUES:
                if K == K_BASELINE:
                    continue
                key = (topo, z, b, L, "K", K)
                if key in done:
                    continue
                t1 = time.time()
                mean, std = run_condition(topo, z, L, b, THETA, N_BASELINE, K,
                                           "fermi", NET_SEED)
                dt = time.time() - t1
                rows.append(dict(topo=topo, z=z, b=b, theta=THETA, L=L, sweep="K",
                                  value=K, K=K, update_rule="fermi",
                                  rho_mean=mean, rho_std=std, seconds=dt))
                print(f"{topo}_z{z} b={b} L={L} N={N_BASELINE} K={K}: "
                      f"rho={mean:.4f} ({dt:.1f}s)", flush=True)
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

            key = (topo, z, b, L, "async", 1)
            if key not in done:
                t1 = time.time()
                mean, std = run_condition(topo, z, L, b, THETA, N_BASELINE,
                                           K_BASELINE, "fermi_async", NET_SEED)
                dt = time.time() - t1
                rows.append(dict(topo=topo, z=z, b=b, theta=THETA, L=L, sweep="async",
                                  value=1, K=K_BASELINE, update_rule="fermi_async",
                                  rho_mean=mean, rho_std=std, seconds=dt))
                print(f"{topo}_z{z} b={b} L={L} ASYNC N={N_BASELINE} "
                      f"K={K_BASELINE}: rho={mean:.4f} ({dt:.1f}s)", flush=True)
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f"TOTAL TIME: {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    if len(df):
        summary = df.groupby(["topo", "z", "b", "L", "sweep", "value"]).agg(
            rho_mean=("rho_mean", "mean"),
            rho_std=("rho_std", "mean"),
        ).reset_index()
        summary.to_csv(SUMMARY_CSV, index=False)
        print(summary.to_string(index=False))
    print("Saved:", OUT_CSV, SUMMARY_CSV)


if __name__ == "__main__":
    main()
