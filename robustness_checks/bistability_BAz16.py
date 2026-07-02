"""
Bistability check for BA z=16 at L=1.

The main simulations (notebook 01) report standard deviations reaching sigma≈0.41
near the transition region of BA z=16. This script verifies that this high variance
reflects genuine bistability (a bimodal distribution of final states) rather than
a broad unimodal distribution.

We run 200 independent replications at a representative parameter point
(b=1.6, theta=0, L=1, Fermi K=0.1) and plot the histogram of stationary
cooperation fractions.

Result: the distribution is bimodal, with ~92/200 replications converging to
the defection attractor (rho < 0.2) and ~66/200 converging to the cooperation
attractor (rho > 0.8), consistent with the bistable interpretation stated in
Sec. III.A of the paper.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import model

N_REP  = 200
B      = 1.6
THETA  = 0.0
K      = 0.1
L      = 1
NET_SEED = 0   # same as notebook 01

# Build network (same realization used in all main simulations)
G  = model.build_network('BA', 1000, 16, seed=NET_SEED)
gp, gd = model.game_csr(G)
sp, sd = model.shells_csr(G, L)
al     = model.geometric_kernel(L, 0.5)

model.warm_up()

rhos = np.array([model._run_one(gp, gd, sp, sd, al, B, THETA, K, seed)
                 for seed in range(N_REP)])

print(f'mean = {rhos.mean():.3f},  std = {rhos.std():.3f}')
print(f'rho < 0.2 (defection attractor):   {(rhos < 0.2).sum()}/{N_REP}')
print(f'rho > 0.8 (cooperation attractor):  {(rhos > 0.8).sum()}/{N_REP}')
print(f'middle (0.2 <= rho <= 0.8):         {((rhos >= 0.2) & (rhos <= 0.8)).sum()}/{N_REP}')

# Plot histogram
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16,
                     'xtick.labelsize': 12, 'ytick.labelsize': 12, 'figure.dpi': 120})
fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
ax.hist(rhos, bins=20, color=mpl.colormaps['Dark2'](1), edgecolor='white', lw=0.5)
ax.set_xlabel(r'$\langle\rho\rangle$ (stationary cooperation fraction)')
ax.set_ylabel(f'Count (out of {N_REP})')
ax.set_title(rf'BA $z=16$, $b={B}$, $\theta={THETA}$, $L={L}$')
ax.set_xlim(0, 1)
out = os.path.join(os.path.dirname(__file__), 'bistability_histogram_BAz16.pdf')
fig.savefig(out, bbox_inches='tight')
print(f'Saved {out}')
plt.show()
