"""
Relative gain metrics (R1 minor comment 6).

R1 pointed out that the absolute cooperation gain Delta<rho> = <rho>(L) - <rho>(L=1) makes
most sense when <rho>(L) ~ 0, and suggested a relative gain metric instead, to support the
claim that "L=1 -> 2 already captures most of the achievable gain."

We report two complementary relative quantities, at the (b, theta) point of largest total
gain for each topology (to avoid noisy fractions where the total L=1->4 gain is small):

  (a) fraction of total achievable gain captured at each L:
        [<rho>(L) - <rho>(1)] / [<rho>(4) - <rho>(1)]
  (b) gain relative to the remaining headroom (how much of the possible increase toward
      full cooperation is captured):
        [<rho>(L) - <rho>(1)] / [1 - <rho>(1)]

Requires data/02-{topology}-L{1,2,3,4}.csv (notebook 02 output) to exist.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

TOPOS = [
    ('BA_z4',  'BA, $z=4$'),
    ('BA_z16', 'BA, $z=16$'),
    ('ER_z4',  'ER, $z=4$'),
    ('ER_z16', 'ER, $z=16$'),
]

# ── Style (edit here to restyle the figure) ─────────────────────────────────
COLORMAP    = 'Dark2'   # matplotlib colormap used for the 4 topology colors
LINEWIDTH   = 2.0
MARKERSIZE  = 6
FIGSIZE     = (9, 3.8)
FONT_SIZE   = 11

plt.rcParams.update({
    'font.size': FONT_SIZE, 'axes.labelsize': FONT_SIZE + 1,
    'xtick.labelsize': FONT_SIZE - 1, 'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': FONT_SIZE - 2, 'figure.dpi': 130,
})

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(root, 'data')
fig_dir = os.path.join(root, 'figures')
out_dir = os.path.dirname(os.path.abspath(__file__))

_cmap = mpl.colormaps[COLORMAP]
COLORS = {key: _cmap(i) for i, (key, _) in enumerate(TOPOS)}

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)

rows = []
for key, label in TOPOS:
    data = {L: pd.read_csv(os.path.join(data_dir, f'02-{key}-L{L}.csv'))
              .set_index(['b', 'theta']).rho_mean
            for L in [1, 2, 3, 4]}
    total_gain = data[4] - data[1]
    idx = total_gain.abs().idxmax()   # (b, theta) of largest |total L=1->4 gain|
    b_star, th_star = idx
    rho = np.array([data[L].loc[idx] for L in [1, 2, 3, 4]])
    L_vals = np.array([1, 2, 3, 4])

    frac_of_total = (rho - rho[0]) / (rho[-1] - rho[0])
    headroom_frac = (rho - rho[0]) / (1 - rho[0])

    axes[0].plot(L_vals, frac_of_total, 'o-', color=COLORS[key], label=label,
                 lw=LINEWIDTH, ms=MARKERSIZE)
    axes[1].plot(L_vals, headroom_frac, 'o-', color=COLORS[key], label=label,
                 lw=LINEWIDTH, ms=MARKERSIZE)

    for L, r, fo, hf in zip(L_vals, rho, frac_of_total, headroom_frac):
        rows.append({'topology': key, 'b': b_star, 'theta': th_star, 'L': L,
                      'rho_mean': r, 'frac_of_total_gain': fo, 'frac_of_headroom': hf})

axes[0].set_xlabel('$L$')
axes[0].set_ylabel(r'$[\langle\rho\rangle(L)-\langle\rho\rangle(1)]\,/\,'
                    r'[\langle\rho\rangle(4)-\langle\rho\rangle(1)]$')
axes[0].set_title('(a) Fraction of total achievable gain')
axes[0].set_xticks([1, 2, 3, 4])
axes[0].axhline(1, color='grey', lw=0.5, ls=':')
axes[0].axhline(0, color='grey', lw=0.5, ls=':')
axes[0].legend(loc='lower right', framealpha=0.9)

axes[1].set_xlabel('$L$')
axes[1].set_ylabel(r'$[\langle\rho\rangle(L)-\langle\rho\rangle(1)]\,/\,'
                    r'[1-\langle\rho\rangle(1)]$')
axes[1].set_title('(b) Gain relative to remaining headroom')
axes[1].set_xticks([1, 2, 3, 4])
axes[1].axhline(1, color='grey', lw=0.5, ls=':')
axes[1].axhline(0, color='grey', lw=0.5, ls=':')

os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, 'relative_gain_metrics.pdf')
fig.savefig(fig_path, bbox_inches='tight')
print(f'Saved {fig_path}')

df = pd.DataFrame(rows)
csv_path = os.path.join(out_dir, 'data', 'relative_gain_metrics.csv')
df.to_csv(csv_path, index=False)
print(f'Saved {csv_path}')
print(df.to_string(index=False))
