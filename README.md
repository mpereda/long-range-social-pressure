# long-range-social-pressure

Code accompanying the paper **"Long-range social pressure and the evolution of cooperation in complex networks"** (M. Pereda and G. Muller, in preparation).

## Overview

We extend the model of Pereda (2016) by allowing the social pressure exerted by vigilant individuals to propagate beyond direct neighbors in a duplex network. In the base model, a player's temptation to defect decreases with the fraction of vigilant direct neighbors. Here, vigilance influence reaches nodes at network distance up to L, weighted by a geometric decay kernel (α_l ∝ λ^(l−1)).

The two coupled layers are:
- **Game layer** — Prisoner's Dilemma played between direct neighbors
- **Vigilance layer** — influence I_i computed via BFS up to distance L

When L = 1 both layers coincide and the model reduces to Pereda (2016). For L > 1 the vigilance layer has longer reach than the game layer.

## Repository structure

```
model.py                         simulation engine (numba-accelerated)

01-REP-vs-Fermi.ipynb            validation: replicator vs Fermi update rule at L = 1
02-long-range-correlated.ipynb   main results: L = 1–4, maximally correlated duplex
03-long-range-uncorrelated.ipynb L = 1–4, uncorrelated duplex
04-lambda-sensitivity.ipynb      sensitivity to the geometric decay parameter λ

data/                            simulation outputs (not tracked by git)
figures/                         paper figures
```

## Dependencies

```
pip install -r requirements.txt
```

Requires Python ≥ 3.10.

## Model summary

| Parameter | Value |
|-----------|-------|
| N | 1000 |
| Replications | 100 |
| b | 1.0 – 2.0 (step 0.1) |
| θ | 0.0 – 1.0 (step 0.1) |
| L | 1, 2, 3, 4 |
| λ | 0.5 (main); 0.1–0.9 (sensitivity) |
| Update rule | Fermi, K = 0.1 |
| Networks | BA and ER, z = 4 and z = 16 |
| Stopping condition | Adaptive: every 100 gens (min 500), stationary if \|⟨ρ⟩(t) − ⟨ρ⟩(t−100)\| / 100 < 10⁻²; then average ⟨ρ⟩ over next 100 gens |

## References

- Pereda, M. (2016). Evolution of cooperation under social pressure in multiplex networks. *Phys. Rev. E* **94**, 032314.
- Pereda, M. & Vilone, D. (2017). Social pressure and environmental effects on networks: A path to cooperation. *Games* **8**, 7.
