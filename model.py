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
    Unnormalized geometric decay kernel: alpha[d] = lam^d, d = 0..L-1.
    L=1 → alpha=[1.0], recovering the PRE 2016 limit.
    Adding circles always adds influence (I_i capped at 1 in _influence).
    """
    return np.fromiter((lam ** d for d in range(L)), dtype=np.float64, count=L)


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
        I[i] = min(acc, 1.0)
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


@njit(cache=True)
def _payoff_one(node, C, T, game_ptr, game_data):
    """
    Payoff of a single node under the current population state C, evaluated
    on demand (used by the asynchronous update below, which needs the payoff
    of only two nodes -- the focal agent and its chosen neighbor -- at each
    elementary step, rather than the full array _payoffs() computes).
    """
    c_nbr = 0
    for k in range(game_ptr[node], game_ptr[node + 1]):
        if C[game_data[k]]:
            c_nbr += 1
    return float(c_nbr) if C[node] else T[node] * float(c_nbr)


@njit(cache=True)
def _fermi_async_generation(C, T, game_ptr, game_data, K, N, r_agent, r_nbr_frac, r_bernoulli):
    """
    Asynchronous Fermi update, R2#8 robustness control (PRE major revision).
    One generation = N sequential elementary updates, in the convention of
    Szabo & Fath (2007, Phys. Rep. 446, 97): at each elementary step t,
    an agent i = r_agent[t] and one of its game-neighbors j are drawn at
    random, their payoffs are computed from the CURRENT state of C (not a
    snapshot frozen at the start of the generation, unlike the synchronous
    _fermi() above), and i copies j's strategy with the usual Fermi
    probability. Because C is mutated in place as the loop proceeds, later
    elementary steps within the same generation already see the outcome of
    earlier ones -- this is what makes the update asynchronous.
    r_agent draws N agents uniformly with replacement (so some agents are
    updated more than once per generation and some not at all, matching the
    standard convention); r_nbr_frac and r_bernoulli play the same role as
    r1/r2 in _fermi(). T (the temptation array, itself derived from the
    vigilance state) is held fixed for the whole generation, exactly as in
    the synchronous version -- only the strategy-update mechanism differs,
    so any difference in outcome between the sync and async controls is
    attributable specifically to that, not to a coupled change in how
    vigilance is computed.
    Mutates and returns C.
    """
    for t in range(N):
        i = r_agent[t]
        deg = game_ptr[i + 1] - game_ptr[i]
        if deg == 0:
            continue
        j_idx = min(int(r_nbr_frac[t] * deg), deg - 1)
        j     = game_data[game_ptr[i] + j_idx]
        pi_i  = _payoff_one(i, C, T, game_ptr, game_data)
        pi_j  = _payoff_one(j, C, T, game_ptr, game_data)
        diff  = (pi_j - pi_i) / K
        if diff > 500.0:
            prob = 1.0
        elif diff < -500.0:
            prob = 0.0
        else:
            prob = 1.0 / (1.0 + math.exp(-diff))
        if r_bernoulli[t] < prob:
            C[i] = C[j]
    return C


@njit(cache=True)
def _rep(C, pi, game_ptr, game_data, b, r1, r2, N):
    """
    Synchronous replicator (proportional imitation) update — PRE 2016 rule
    (Pereda2016 Eq. 4; Roca2009 Eq. 35).
    Node i picks a random game-neighbor j (via r1); copies j if pi_j > pi_i
    with probability (pi_j - pi_i) / Phi_ij, Phi_ij = max(k_i, k_j) * b
    (per-pair normalization; b = max(1,T) - min(0,S) under our weak-PD
    payoffs R=1, S=P=0, T=b).
    NOTE: an earlier version used a single global Phi = k_max * b (k_max the
    largest degree in the whole network) instead of the per-pair max(k_i,k_j).
    This is a valid upper bound (still keeps the probability in [0,1]) but
    not the PRE2016/Roca2009 rule: it makes imitation uniformly slower for
    any pair not involving the network's single highest-degree hub, which in
    heterogeneous networks (BA) is nearly every pair. Fixed 2026-09-02.
    """
    new_C = C.copy()
    for i in range(N):
        deg_i = game_ptr[i + 1] - game_ptr[i]
        if deg_i == 0:
            continue
        j_idx = min(int(r1[i] * deg_i), deg_i - 1)
        j     = game_data[game_ptr[i] + j_idx]
        deg_j = game_ptr[j + 1] - game_ptr[j]
        phi   = max(deg_i, deg_j) * b
        diff  = pi[j] - pi[i]
        if diff > 0.0 and r2[i] < diff / phi:
            new_C[i] = C[j]
    return new_C


# ─── Single replication ────────────────────────────────────────────────────

_MIN_GENS    = 500
_CHECK_EVERY = 100
_SLOPE_THR   = 1e-2
_MAX_GENS    = 500_000


def _windowed_stationary_mean(step_fn):
    """
    Adaptive convergence detection, window-mean form.

    Runs step_fn() (one generation, returns the instantaneous cooperator
    fraction) and accumulates non-overlapping windows of _CHECK_EVERY
    generations. After a minimum transient of _MIN_GENS generations, the
    system is declared stationary once two *consecutive window means* agree
    to within _SLOPE_THR, i.e. |mean(window_k) - mean(window_{k-1})| <
    _SLOPE_THR. The mean of window_k is then returned as the stationary
    value. This corrects an earlier implementation that compared two single
    raw generations 100 apart divided by 100 (bounded by construction at
    1/100, hence trivially satisfied on the first check at t=_MIN_GENS
    regardless of whether the system had actually converged) instead of
    comparing two window means as intended and as described in
    Pereda (2016).
    """
    n_windows_min = _MIN_GENS // _CHECK_EVERY
    prev_window = None
    window_mean = None
    for k in range(1, _MAX_GENS // _CHECK_EVERY + 1):
        window_mean = sum(step_fn() for _ in range(_CHECK_EVERY)) / _CHECK_EVERY
        if k > n_windows_min and prev_window is not None:
            if abs(window_mean - prev_window) < _SLOPE_THR:
                return window_mean
        prev_window = window_mean
    return window_mean


def _run_one(game_ptr, game_data, shell_ptr, shell_data, alpha, b, theta, K, seed):
    """
    Single replication. Returns ⟨ρ⟩, the mean cooperator fraction of the
    100-generation window at which the adaptive slope criterion declares
    stationarity (see _windowed_stationary_mean).
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

    return _windowed_stationary_mean(_step)


def _run_one_rep(game_ptr, game_data, shell_ptr, shell_data, alpha, b, theta, seed):
    """Single replication with replicator dynamics (PRE 2016 rule, per-pair Φ=max(k_i,k_j)·b)."""
    rng   = np.random.default_rng(seed)
    N     = int(game_ptr.shape[0] - 1)
    L     = int(alpha.shape[0])

    C = rng.random(N) < 0.5
    V = C & (rng.random(N) < 0.5)

    def _step():
        nonlocal C, V
        I  = _influence(V, shell_ptr, shell_data, alpha, N, L)
        Tv = 1.0 + (b - 1.0) * (1.0 - I)
        pi = _payoffs(C, Tv, game_ptr, game_data, N)
        C  = _rep(C, pi, game_ptr, game_data, b, rng.random(N), rng.random(N), N)
        V  = C & (I >= theta)
        return float(C.mean())

    return _windowed_stationary_mean(_step)


def _run_one_async(game_ptr, game_data, shell_ptr, shell_data, alpha, b, theta, K, seed):
    """
    Single replication, ASYNCHRONOUS Fermi update (R2#8 robustness control).
    Identical to _run_one() except the strategy update within each generation
    calls _fermi_async_generation() (N sequential elementary updates) instead
    of the synchronous _fermi() (one simultaneous sweep). Vigilance/influence
    are still recomputed once per generation at generation boundaries in
    both cases -- see _fermi_async_generation()'s docstring for why.
    """
    rng = np.random.default_rng(seed)
    N   = int(game_ptr.shape[0] - 1)
    L   = int(alpha.shape[0])

    C = rng.random(N) < 0.5
    V = C & (rng.random(N) < 0.5)

    def _step():
        nonlocal C, V
        I  = _influence(V, shell_ptr, shell_data, alpha, N, L)
        Tv = 1.0 + (b - 1.0) * (1.0 - I)
        C_new = C.copy()
        r_agent    = rng.integers(0, N, size=N)
        r_nbr_frac = rng.random(N)
        r_bernoulli = rng.random(N)
        C  = _fermi_async_generation(C_new, Tv, game_ptr, game_data, K, N,
                                      r_agent, r_nbr_frac, r_bernoulli)
        V  = C & (I >= theta)
        return float(C.mean())

    return _windowed_stationary_mean(_step)


# ─── Public API ────────────────────────────────────────────────────────────


def run_replications(game_ptr, game_data, shell_ptr, shell_data, alpha,
                     b, theta, K=0.1, n_rep=100, n_jobs=-1, base_seed=0,
                     update_rule="fermi"):
    """
    Run n_rep independent replications in parallel (threads; numba releases GIL).
    Returns float array of shape (n_rep,) with stationary ρ per replication.

    update_rule: 'fermi' (default, K=0.1), 'rep' (replicator, PRE 2016), or
    'fermi_async' (asynchronous Fermi control, R2#8 robustness check --
    N sequential elementary updates per generation instead of one
    synchronous sweep; see _fermi_async_generation()'s docstring).
    """
    if update_rule == "fermi":
        worker = delayed(_run_one)
        args   = lambda rep: (game_ptr, game_data, shell_ptr, shell_data,
                               alpha, b, theta, K, base_seed + rep)
    elif update_rule == "rep":
        worker = delayed(_run_one_rep)
        args   = lambda rep: (game_ptr, game_data, shell_ptr, shell_data,
                               alpha, b, theta, base_seed + rep)
    elif update_rule == "fermi_async":
        worker = delayed(_run_one_async)
        args   = lambda rep: (game_ptr, game_data, shell_ptr, shell_data,
                               alpha, b, theta, K, base_seed + rep)
    else:
        raise ValueError(
            f"update_rule must be 'fermi', 'rep', or 'fermi_async', got {update_rule!r}")

    return np.array(
        Parallel(n_jobs=n_jobs, prefer="threads")(
            worker(*args(rep)) for rep in range(n_rep)
        )
    )


def run_sweep(game_ptr, game_data, shell_ptr, shell_data, alpha,
              b_values, theta_values,
              K=0.1, n_rep=100, n_jobs=-1, base_seed=0, update_rule="fermi"):
    """
    Sweep over all (b, θ) combinations.
    Returns pd.DataFrame with columns: b, theta, rho_mean, rho_std.
    update_rule: 'fermi' or 'rep'.
    """
    records = []
    for b in b_values:
        for theta in theta_values:
            rhos = run_replications(
                game_ptr, game_data, shell_ptr, shell_data, alpha,
                b, theta, K, n_rep, n_jobs, base_seed, update_rule,
            )
            records.append({
                "b":        round(float(b), 6),
                "theta":    round(float(theta), 6),
                "rho_mean": float(rhos.mean()),
                "rho_std":  float(rhos.std(ddof=1)),
            })
    return pd.DataFrame(records)


def warm_up():
    """Pre-compile all numba kernels on a small graph to avoid first-call JIT delay."""
    G  = nx.barabasi_albert_graph(20, 2, seed=0)
    gp, gd = game_csr(G)
    sp, sd = shells_csr(G, 2)
    al = geometric_kernel(2, 0.5)
    N  = G.number_of_nodes()
    V  = np.ones(N, dtype=np.bool_)
    C  = np.ones(N, dtype=np.bool_)
    T  = np.ones(N, dtype=np.float64)
    r1 = np.random.random(N)
    r2 = np.random.random(N)
    _influence(V, sp, sd, al, N, 2)
    pi = _payoffs(C, T, gp, gd, N)
    _fermi(C, pi, gp, gd, 0.1, r1, r2, N)
    _rep(C, pi, gp, gd, 1.5, r1, r2, N)
