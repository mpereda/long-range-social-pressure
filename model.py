"""
model.py — long-range social pressure simulation engine
Pereda & Muller (in preparation)
"""

import math
import numpy as np
import networkx as nx
import pandas as pd
from joblib import Parallel, delayed
from numba import njit


# ─── Network construction ──────────────────────────────────────────────────


def build_network(topology, N, z, seed=None):
    """Connected random graph (BA or ER) with approximately mean degree z."""
    rng = np.random.default_rng(seed)
    seed_int = int(rng.integers(0, 2**31))
    if topology == "BA":
        G = nx.barabasi_albert_graph(N, m=max(1, z // 2), seed=seed_int)
    elif topology == "ER":
        G = nx.erdos_renyi_graph(N, p=z / (N - 1), seed=seed_int)
    else:
        raise ValueError(f"topology must be 'BA' or 'ER', got {topology!r}")
    Gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.convert_node_labels_to_integers(Gcc)


# ─── CSR data structures ───────────────────────────────────────────────────


def game_csr(G):
    """CSR adjacency list for the game layer (ptr shape: N+1, data shape: 2*E)."""
    N = G.number_of_nodes()
    ptr = np.zeros(N + 1, dtype=np.int64)
    for i in range(N):
        ptr[i + 1] = ptr[i] + G.degree(i)
    data = np.empty(int(ptr[-1]), dtype=np.int32)
    for i in range(N):
        nbrs = sorted(G.neighbors(i))
        data[ptr[i] : ptr[i + 1]] = nbrs
    return ptr, data


def shells_csr(G, L):
    """
    BFS shell membership for the vigilance layer, in CSR format.

    ptr[i, d]  = start in data of shell (d+1) of node i, for d = 0..L-1
    ptr[i, L]  = end sentinel for node i
    """
    N = G.number_of_nodes()
    raw = [[[] for _ in range(L)] for _ in range(N)]
    for i in range(N):
        for j, dist in nx.single_source_shortest_path_length(G, i, cutoff=L).items():
            if j != i and 1 <= dist <= L:
                raw[i][dist - 1].append(j)

    total = sum(len(raw[i][d]) for i in range(N) for d in range(L))
    ptr  = np.zeros((N, L + 1), dtype=np.int64)
    data = np.empty(total, dtype=np.int32)
    pos  = 0
    for i in range(N):
        for d in range(L):
            ptr[i, d] = pos
            s = raw[i][d]
            data[pos : pos + len(s)] = s
            pos += len(s)
        ptr[i, L] = pos
    return ptr, data


def geometric_kernel(L, lam=0.5):
    """
    Normalized geometric decay kernel.
    alpha[d] = lam^d / sum(lam^0 + ... + lam^{L-1}),  d = 0..L-1 (distance = d+1).
    L=1 → alpha=[1.0], recovering the PRE 2016 limit.
    """
    alpha = np.fromiter((lam ** d for d in range(L)), dtype=np.float64, count=L)
    return alpha / alpha.sum()


# ─── Numba kernels ─────────────────────────────────────────────────────────


@njit(cache=True)
def _influence(V, shell_ptr, shell_data, alpha, N, L):
    """
    I_i = Σ_{d=0}^{L-1} alpha[d] * (vigilant in shell d+1) / (size of shell d+1).
    Empty shells contribute 0.
    """
    I = np.zeros(N, dtype=np.float64)
    for i in range(N):
        acc = 0.0
        for d in range(L):
            start = shell_ptr[i, d]
            end   = shell_ptr[i, d + 1]
            size  = end - start
            if size == 0:
                continue
            cnt = 0
            for k in range(start, end):
                if V[shell_data[k]]:
                    cnt += 1
            acc += alpha[d] * cnt / size
        I[i] = acc
    return I


@njit(cache=True)
def _payoffs(C, T, game_ptr, game_data, N):
    """
    PD payoffs with R=1, S=0, T_i (individual temptation), P=0.
    pi[i] = (# cooperator game-neighbors) * R  if C[i],
           = (# cooperator game-neighbors) * T[i]  otherwise.
    """
    pi = np.zeros(N, dtype=np.float64)
    for i in range(N):
        c_nbr = 0
        for k in range(game_ptr[i], game_ptr[i + 1]):
            if C[game_data[k]]:
                c_nbr += 1
        pi[i] = float(c_nbr) if C[i] else T[i] * float(c_nbr)
    return pi


@njit(cache=True)
def _fermi(C, pi, game_ptr, game_data, K, r1, r2, N):
    """
    Synchronous Fermi update.
    Each node i picks a random game-neighbor j (drawn via r1) and copies j's
    strategy with probability 1 / (1 + exp(-(pi_j - pi_i) / K)).
    r2 provides the Bernoulli draw.
    """
    new_C = C.copy()
    for i in range(N):
        deg = game_ptr[i + 1] - game_ptr[i]
        if deg == 0:
            continue
        j_idx = min(int(r1[i] * deg), deg - 1)
        j     = game_data[game_ptr[i] + j_idx]
        diff  = (pi[j] - pi[i]) / K
        if diff > 500.0:
            prob = 1.0
        elif diff < -500.0:
            prob = 0.0
        else:
            prob = 1.0 / (1.0 + math.exp(-diff))
        if r2[i] < prob:
            new_C[i] = C[j]
    return new_C


# ─── Single replication ────────────────────────────────────────────────────

_MIN_GENS    = 500
_CHECK_EVERY = 100
_SLOPE_THR   = 1e-2
_MAX_GENS    = 500_000


def _run_one(game_ptr, game_data, shell_ptr, shell_data, alpha, b, theta, K, seed):
    """
    Single replication. Returns ⟨ρ⟩ averaged over 100 generations immediately
    after the adaptive slope criterion declares stationarity.
    """
    rng = np.random.default_rng(seed)
    N   = int(game_ptr.shape[0] - 1)
    L   = int(alpha.shape[0])

    C = rng.random(N) < 0.5          # True = cooperator
    V = C & (rng.random(N) < 0.5)   # vigilant cooperators only (25% of population)

    def _step():
        nonlocal C, V
        I  = _influence(V, shell_ptr, shell_data, alpha, N, L)
        Tv = 1.0 + (b - 1.0) * (1.0 - I)
        pi = _payoffs(C, Tv, game_ptr, game_data, N)
        C  = _fermi(C, pi, game_ptr, game_data, K, rng.random(N), rng.random(N), N)
        V  = C & (I >= theta)
        return float(C.mean())

    rho_hist = []
    for t in range(1, _MAX_GENS + 1):
        rho_hist.append(_step())
        if t >= _MIN_GENS and t % _CHECK_EVERY == 0:
            slope = abs(rho_hist[-1] - rho_hist[-1 - _CHECK_EVERY]) / _CHECK_EVERY
            if slope < _SLOPE_THR:
                return float(np.mean([_step() for _ in range(_CHECK_EVERY)]))

    return float(np.mean(rho_hist[-_CHECK_EVERY:]))


# ─── Public API ────────────────────────────────────────────────────────────


def run_replications(game_ptr, game_data, shell_ptr, shell_data, alpha,
                     b, theta, K=0.1, n_rep=100, n_jobs=-1, base_seed=0):
    """
    Run n_rep independent replications in parallel (threads; numba releases GIL).
    Returns float array of shape (n_rep,) with stationary ρ per replication.
    """
    return np.array(
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_run_one)(
                game_ptr, game_data, shell_ptr, shell_data,
                alpha, b, theta, K, base_seed + rep,
            )
            for rep in range(n_rep)
        )
    )


def run_sweep(game_ptr, game_data, shell_ptr, shell_data, alpha,
              b_values, theta_values,
              K=0.1, n_rep=100, n_jobs=-1, base_seed=0):
    """
    Sweep over all (b, θ) combinations.
    Returns pd.DataFrame with columns: b, theta, rho_mean, rho_std.
    """
    records = []
    for b in b_values:
        for theta in theta_values:
            rhos = run_replications(
                game_ptr, game_data, shell_ptr, shell_data, alpha,
                b, theta, K, n_rep, n_jobs, base_seed,
            )
            records.append({
                "b":        round(float(b), 6),
                "theta":    round(float(theta), 6),
                "rho_mean": float(rhos.mean()),
                "rho_std":  float(rhos.std(ddof=1)),
            })
    return pd.DataFrame(records)


def warm_up():
    """Pre-compile numba kernels on a small graph to avoid first-call JIT delay."""
    G  = nx.barabasi_albert_graph(20, 2, seed=0)
    gp, gd = game_csr(G)
    sp, sd = shells_csr(G, 2)
    al = geometric_kernel(2, 0.5)
    N  = G.number_of_nodes()
    V  = np.ones(N, dtype=np.bool_)
    C  = np.ones(N, dtype=np.bool_)
    T  = np.ones(N, dtype=np.float64)
    pi = _payoffs(C, T, gp, gd, N)
    _influence(V, sp, sd, al, N, 2)
    _fermi(C, pi, gp, gd, 0.1, np.random.random(N), np.random.random(N), N)
