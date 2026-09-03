"""
nk_async_robustness.py — Referee 2 #8

Checks whether the paper's central observation -- that extending vigilance
from L=1 to L=4 substantially raises cooperation -- depends on the choice
of population size N, Fermi noise K, or synchronous vs. asynchronous
strategy updating, all fixed at N=1000, K=0.1, synchronous throughout the
main text.

For each (topology, z) combination, picks up to 4 representative (b, theta)
points from the *corrected* correlated-multiplex sweep (data/02-*.csv) at
L=4 (coop, defect, transition, bistable -- same convention as
network_realization_robustness.py), then for each point:

  (a) N sweep: reruns at N in {250, 500, 1000, 2000} (baseline K=0.1, sync),
      for both L=1 and L=4, so the L=1->4 gain itself (not just each L
      separately) can be checked for N-dependence.
  (b) K sweep: reruns at K in {0.05, 0.1, 0.2} (baseline N=1000, sync),
      same L=1 and L=4 pair.
  (c) async control: reruns at the baseline N=1000, K=0.1 but with the
      asynchronous Fermi update (update_rule='fermi_async'), same L pair.

All three sets use 100 replications per (point, L, condition), matching the
main text. The N=1000, K=0.1, synchronous entries are shared between (a)
and (b) (computed once, reused) to avoid redundant work.

Must be run after the corrected 02 sweep (needs its CSVs to pick
representative points). Checkpoints after every (point, L, condition) so a
partial run is never lost.

NOTE (2026-09-03): written but not yet run -- scheduled to run after the
network-realization robustness study (comment 7) finishes and frees up
compute. See robustness_checks/network_realization_robustness.py for that
one; this script follows the same structure deliberately.
"""
import os, sys, time
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from model import build_network, game_csr, shells_csr, geometric_kernel, run_replications

DATA_DIR = os.path.join(REPO, "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "nk_async_robustness.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "nk_async_robustness_summary.csv")

NETWORKS = [("BA", 4), ("BA", 16), ("ER", 4), ("ER", 16)]
L_VALUES = [1, 4]                   # the extremes whose gain is the paper's central claim
N_VALUES = [250, 500, 1000, 2000]   # baseline: 1000
K_VALUES = [0.05, 0.1, 0.2]         # baseline: 0.1
N_REP = 100
LAM = 0.5
N_BASELINE = 1000
K_BASELINE = 0.1


def pick_representative_points(df):
    """Same convention as network_realization_robustness.py: up to 4
    representative points -- 'coop' (max rho_mean), 'defect' (min rho_mean),
    'transition' (rho_mean closest to 0.5), 'bistable' (max rho_std)."""
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


def run_condition(topo, z, L, b, theta, N, K, update_rule, seed):
    """Build a fresh correlated-multiplex network at size N and run N_REP
    replications at the given L, b, theta, K, update_rule."""
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
    done = {(r["topo"], r["z"], r["regime"], r["L"], r["sweep"], r["value"]) for r in rows}

    t0 = time.time()
    net_seed = 7  # single fixed network realization per (topo, z) -- this
                  # script checks N/K/async robustness, not network-realization
                  # robustness (that's comment 7's script); reusing one seed
                  # keeps the two robustness checks orthogonal.

    for topo, z in NETWORKS:
        path = os.path.join(DATA_DIR, f"02-{topo}_z{z}-L4.csv")
        if not os.path.exists(path):
            print(f"  MISSING: {path} -- skipping (was it swept?)")
            continue
        df = pd.read_csv(path)
        points = pick_representative_points(df)

        for label, b, theta in points:
            for L in L_VALUES:
                # (a) N sweep, baseline K, sync. N_BASELINE point is shared
                # with (b)'s baseline, computed once under sweep='N'.
                for N in N_VALUES:
                    key = (topo, z, label, L, "N", N)
                    if key in done:
                        continue
                    t1 = time.time()
                    mean, std = run_condition(topo, z, L, b, theta, N, K_BASELINE,
                                               "fermi", net_seed)
                    dt = time.time() - t1
                    rows.append(dict(topo=topo, z=z, regime=label, b=b, theta=theta,
                                      L=L, sweep="N", value=N, K=K_BASELINE,
                                      update_rule="fermi", rho_mean=mean, rho_std=std,
                                      seconds=dt))
                    print(f"{topo}_z{z} {label:10s} L={L} N={N:5d} K={K_BASELINE}: "
                          f"rho={mean:.4f} ({dt:.1f}s)", flush=True)
                    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

                # (b) K sweep, baseline N, sync. Skip K_BASELINE (already
                # have it from the N sweep above at N=N_BASELINE).
                for K in K_VALUES:
                    if K == K_BASELINE:
                        continue
                    key = (topo, z, label, L, "K", K)
                    if key in done:
                        continue
                    t1 = time.time()
                    mean, std = run_condition(topo, z, L, b, theta, N_BASELINE, K,
                                               "fermi", net_seed)
                    dt = time.time() - t1
                    rows.append(dict(topo=topo, z=z, regime=label, b=b, theta=theta,
                                      L=L, sweep="K", value=K, K=K,
                                      update_rule="fermi", rho_mean=mean, rho_std=std,
                                      seconds=dt))
                    print(f"{topo}_z{z} {label:10s} L={L} N={N_BASELINE} K={K}: "
                          f"rho={mean:.4f} ({dt:.1f}s)", flush=True)
                    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

                # (c) async control, baseline N and K.
                key = (topo, z, label, L, "async", 1)
                if key not in done:
                    t1 = time.time()
                    mean, std = run_condition(topo, z, L, b, theta, N_BASELINE,
                                               K_BASELINE, "fermi_async", net_seed)
                    dt = time.time() - t1
                    rows.append(dict(topo=topo, z=z, regime=label, b=b, theta=theta,
                                      L=L, sweep="async", value=1, K=K_BASELINE,
                                      update_rule="fermi_async", rho_mean=mean,
                                      rho_std=std, seconds=dt))
                    print(f"{topo}_z{z} {label:10s} L={L} ASYNC N={N_BASELINE} "
                          f"K={K_BASELINE}: rho={mean:.4f} ({dt:.1f}s)", flush=True)
                    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f"TOTAL TIME: {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    if len(df):
        summary = df.groupby(["topo", "z", "regime", "L", "sweep", "value"]).agg(
            rho_mean=("rho_mean", "mean"),
            rho_std=("rho_std", "mean"),
        ).reset_index()
        summary.to_csv(SUMMARY_CSV, index=False)
        print(summary.to_string(index=False))
    print("Saved:", OUT_CSV, SUMMARY_CSV)


if __name__ == "__main__":
    main()
