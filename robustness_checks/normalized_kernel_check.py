"""
Normalized vs. unnormalized kernel check (R2 comment 4).

R2 pointed out that our geometric kernel alpha_d = lambda^(d-1) is NOT normalized, so
increasing L increases both the *range* of vigilance and its *total accumulated magnitude*.
They suggested comparing against a normalized kernel, alpha_d = lambda^(d-1) / sum(lambda^k),
to disentangle range from magnitude.

We identified this exact issue during early model development (2026-06-27, see project
memory decision-kernel-normalization.md) and already derived the result analytically: under
a homogeneous-vigilance approximation (m_i^d/k_i^d ~ v for all d), the normalized kernel
gives I_i = v * sum(alpha_d) = v, EXACTLY independent of L and lambda. Extending the
vigilance range would then have no effect at all. We verified this with an early
(now-archived) simulation run, but that used the model.py with both the stopping-criterion
bug and the global-Phi bug not yet fixed. This script re-verifies the same comparison with
the corrected model.py, using representative (b, theta) points per topology rather than the
full grid, since only the qualitative collapse (L=1 ~ L=4) needs confirming, not a precise
sweep.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import model

N        = 1000
n_rep    = 50
LAM      = 0.5
K_fermi  = 0.1
NET_SEED = 0        # same network realization as the main sweep
SIM_SEED = 0
N_JOBS   = 4         # keep modest: main resweep is running concurrently

NETWORKS = [
    ('BA_z4',  'BA', 4),
    ('BA_z16', 'BA', 16),
    ('ER_z4',  'ER', 4),
    ('ER_z16', 'ER', 16),
]

# Representative (b, theta) points: cooperative, defective, and two transition-ish points.
POINTS = [(1.1, 0.8), (1.9, 0.2), (1.4, 0.5), (1.6, 0.3)]

def normalized_kernel(L, lam):
    alpha = model.geometric_kernel(L, lam)
    return alpha / alpha.sum()

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(out_dir, 'data'), exist_ok=True)

model.warm_up()

rows = []
for key, topo, z in NETWORKS:
    G = model.build_network(topo, N, z, seed=NET_SEED)
    gp, gd = model.game_csr(G)
    for L in (1, 4):
        alpha = normalized_kernel(L, LAM)
        sp, sd = model.shells_csr(G, L)
        for b, theta in POINTS:
            rhos = model.run_replications(
                gp, gd, sp, sd, alpha, b, theta, K_fermi, n_rep, N_JOBS, SIM_SEED,
                update_rule='fermi',
            )
            rows.append({
                'topology': key, 'L': L, 'b': b, 'theta': theta,
                'rho_mean': float(rhos.mean()), 'rho_std': float(rhos.std(ddof=1)),
            })
            print(f'{key} L={L} b={b} theta={theta}: rho_mean={rhos.mean():.4f}', flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(out_dir, 'data', 'normalized_kernel_check.csv'), index=False)

# Summary: L=1 vs L=4 difference per topology
print('\n=== Summary: |rho(L=1) - rho(L=4)| under NORMALIZED kernel ===')
summary = []
for key, _, _ in NETWORKS:
    sub = df[df.topology == key]
    l1 = sub[sub.L == 1].set_index(['b', 'theta']).rho_mean
    l4 = sub[sub.L == 4].set_index(['b', 'theta']).rho_mean
    diff = (l1 - l4).abs()
    summary.append({'topology': key, 'mean_abs_diff': diff.mean(), 'max_abs_diff': diff.max()})
    print(f'{key}: mean|diff|={diff.mean():.4f}  max|diff|={diff.max():.4f}')

pd.DataFrame(summary).to_csv(os.path.join(out_dir, 'data', 'normalized_kernel_check_summary.csv'), index=False)
print('\nDone.')
