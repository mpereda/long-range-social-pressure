"""
label_shuffle_control.py — Referee 2 #6 follow-up

Disentangles edge overlap from inter-layer degree correlation as the source
of Barabasi-Albert's residual sensitivity to "uncorrelated" layers.

Sec. III.C reports that the nominally uncorrelated BA multiplex (G_vig and
G_game independent realizations of the same random-graph model) still has
substantial inter-layer degree correlation, r_k = corr(k_game, k_vig):
r_k=0.68 (z=4), r_k=0.86 (z=16). Both share zero edges by construction, so
this r_k is entirely a degree-correlation effect, not an edge-overlap
effect -- but the two layers were also generated with correlated node
*labels* (both indexed 0..N-1 in Barabasi-Albert arrival order), which is
what produces the r_k. This script adds a third condition, on top of the
'corr' (G_vig=G_game) and 'uncorr' (independent realizations, same labels)
conditions already in the main sweep:

  'uncorr_shuffled': independent realizations as in 'uncorr', but with
  G_vig's node labels additionally permuted uniformly at random. This
  keeps G_vig's own topology (degree sequence, structure) identical to the
  'uncorr' condition, and edge overlap with G_game remains zero in all
  three conditions -- but it destroys the shared-arrival-order alignment
  that produces r_k, driving it to ~0.

If BA's residual sensitivity (documented in Sec. III.C: mean
|rho_uncorr - rho_corr| rising from 0.013 to 0.021 over L=1->4, critical-
temptation gap widening to 0.10 in b by L=4) shrinks further under label
shuffling, that confirms the residual sensitivity is attributable to r_k
(degree correlation) specifically, as the arrival-order mechanism in
Sec. III.C claims -- not to some other latent effect of the independent-
realization construction.

For each z in {4, 16}, picks up to 4 representative (b, theta) points from
the corrected correlated sweep (data/02-BA_z{4,16}-L4.csv, same convention
as network_realization_robustness.py and nk_async_robustness.py), then for
L in {1, 4} runs all three conditions (corr, uncorr, uncorr_shuffled) at
each point, 100 replications each, and also reports r_k measured directly
on the (game, vig) pair used in each condition as a sanity check that the
shuffle actually drives r_k -> 0.

NOTE (2026-09-03): written but not yet run -- scheduled after comment 7's
network-realization study and comment 8's N/K/async study, whichever
finishes compute first.
"""
import os, sys, time
import numpy as np
import networkx as nx
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from model import build_network, game_csr, shells_csr, geometric_kernel, run_replications

DATA_DIR = os.path.join(REPO, "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "label_shuffle_control.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "label_shuffle_control_summary.csv")

Z_VALUES = [4, 16]
L_VALUES = [1, 4]
N_REP = 100
LAM = 0.5
N_NODES = 1000
K_FERMI = 0.1


def degree_correlation(Gg, Gv):
    """r_k = corr(k_i^game, k_i^vig), same definition used in Sec. III.C."""
    nodes = sorted(Gg.nodes())
    kg = np.array([Gg.degree(i) for i in nodes], dtype=float)
    kv = np.array([Gv.degree(i) for i in nodes], dtype=float)
    return float(np.corrcoef(kg, kv)[0, 1])


def shuffle_labels(G, seed):
    """Relabel G's nodes with a uniformly random permutation of 0..N-1.
    Preserves G's own topology (degree sequence, structure) exactly;
    destroys any positional alignment with another graph's labels."""
    rng = np.random.default_rng(seed)
    N = G.number_of_nodes()
    perm = rng.permutation(N)
    mapping = {old: int(new) for old, new in zip(sorted(G.nodes()), perm)}
    return nx.relabel_nodes(G, mapping, copy=True)


def pick_representative_points(df):
    """Same convention as the other robustness scripts."""
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


def run_condition(Gg, Gv, L, b, theta, seed):
    gp, gd = game_csr(Gg)
    sp, sd = shells_csr(Gv, L)
    al = geometric_kernel(L, LAM)
    rhos = run_replications(gp, gd, sp, sd, al, b, theta, K=K_FERMI, n_rep=N_REP,
                             n_jobs=-1, base_seed=seed * 100_000)
    return float(rhos.mean()), float(rhos.std(ddof=1))


def main():
    rows = []
    if os.path.exists(OUT_CSV):
        rows = pd.read_csv(OUT_CSV).to_dict("records")
        print(f"Resuming: {len(rows)} results already present in {OUT_CSV}")
    done = {(r["z"], r["regime"], r["L"], r["condition"]) for r in rows}

    t0 = time.time()
    net_seed = 7  # base seed; game/vig/shuffle each derive a distinct seed from it

    for z in Z_VALUES:
        path = os.path.join(DATA_DIR, f"02-BA_z{z}-L4.csv")
        if not os.path.exists(path):
            print(f"  MISSING: {path} -- skipping (was it swept?)")
            continue
        df = pd.read_csv(path)
        points = pick_representative_points(df)

        Gg = build_network("BA", N_NODES, z, seed=net_seed)
        Gv_indep = build_network("BA", N_NODES, z, seed=net_seed + 500_000)
        Gv_shuffled = shuffle_labels(Gv_indep, seed=net_seed + 999_000)

        rk_corr = degree_correlation(Gg, Gg)
        rk_uncorr = degree_correlation(Gg, Gv_indep)
        rk_shuffled = degree_correlation(Gg, Gv_shuffled)
        print(f"BA z={z}: r_k corr={rk_corr:.3f} uncorr={rk_uncorr:.3f} "
              f"shuffled={rk_shuffled:.3f}", flush=True)

        conditions = [("corr", Gg, Gg), ("uncorr", Gg, Gv_indep),
                      ("uncorr_shuffled", Gg, Gv_shuffled)]

        for label, b, theta in points:
            for L in L_VALUES:
                for cond, Gg_c, Gv_c in conditions:
                    key = (z, label, L, cond)
                    if key in done:
                        continue
                    t1 = time.time()
                    mean, std = run_condition(Gg_c, Gv_c, L, b, theta, net_seed)
                    dt = time.time() - t1
                    rk = {"corr": rk_corr, "uncorr": rk_uncorr,
                          "uncorr_shuffled": rk_shuffled}[cond]
                    rows.append(dict(z=z, regime=label, b=b, theta=theta, L=L,
                                      condition=cond, r_k=rk, rho_mean=mean,
                                      rho_std=std, seconds=dt))
                    print(f"BA_z{z} {label:10s} L={L} {cond:16s} r_k={rk:.3f}: "
                          f"rho={mean:.4f} ({dt:.1f}s)", flush=True)
                    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f"TOTAL TIME: {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    if len(df):
        summary = df.groupby(["z", "regime", "L", "condition"]).agg(
            r_k=("r_k", "mean"), rho_mean=("rho_mean", "mean"),
            rho_std=("rho_std", "mean"),
        ).reset_index()
        summary.to_csv(SUMMARY_CSV, index=False)
        print(summary.to_string(index=False))
    print("Saved:", OUT_CSV, SUMMARY_CSV)


if __name__ == "__main__":
    main()
