"""
rewiring_main_fig.py — compact main-text panel for Referee 1 major comment 4.

Single-panel version of the rewiring-interpolation figure (L=4 only,
<rho> vs rewired fraction r, one line per b), for the main text. The full
2x2 comparison (both L, mean and std) is Fig. S9 in the Supplemental
Material.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_PATH = os.path.join(REPO, "figures", "rewiring_interpolation_main.pdf")

B_COLORS = {1.3: "#7f7f7f", 1.5: "#1f77b4", 1.7: "#2ca02c", 1.9: "#ff7f0e", 2.0: "#d62728"}

df = pd.read_csv(os.path.join(REPO, "robustness_checks", "data", "rewiring_interpolation.csv"))

fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
sub = df[df.L == 4]
for b in sorted(sub.b.unique()):
    s = sub[sub.b == b].sort_values("r")
    ax.plot(s.r, s.rho_mean, "o-", color=B_COLORS[b], label=f"$b={b}$", ms=5)
ax.set_xlabel(r"Rewired fraction $r$ (edge overlap $\omega\approx1-r$)")
ax.set_ylabel(r"$\langle\rho\rangle$")
ax.set_ylim(-0.03, 1.03)
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
fig.savefig(FIG_PATH, bbox_inches="tight")
print("Saved", FIG_PATH)
