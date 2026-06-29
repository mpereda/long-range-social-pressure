# Archive v1 — Normalized geometric kernel

## What this is

Complete simulation results using the **normalized** geometric kernel:

    alpha_d = lambda^(d-1) / sum(lambda^k, k=0..L-1)    [sum = 1]

## Why archived (not the final model)

Mathematically, when vigilance is spatially homogeneous (m_i^d/k_i^d = rho for all d):

    I_i = sum(alpha_d * rho) = rho * sum(alpha_d) = rho

I_i is independent of L and lambda. The stationary states in this model are nearly
homogeneous (full cooperation or full defection), so L and lambda have no visible
effect on stationary cooperation. Confirmed by simulations: notebooks 01-04 show
identical heatmaps across all L and lambda values.

## Scientific value

These results are NOT wrong — they show that the normalized kernel produces no
long-range effect. This is itself a result worth a paragraph in the paper:
"A normalized kernel renders I_i independent of L when vigilance is spatially
homogeneous, motivating the adoption of an unnormalized formulation."

## Contents

- figures/: all paper figures from notebooks 01-04 (normalized kernel)
- model_v1_normalized.py: model.py at this version (git commit 5fc95a0)

## Notebooks (in main repo)

The executed notebooks 01-04 with their outputs are in the main repo at git
commit 5fc95a0. The data/ CSVs are gitignored (local only).

## Next version

v2 uses unnormalized kernel with cap:

    alpha_d = lambda^(d-1)    [NOT normalized]
    I_i = min(1, sum(alpha_d * m_i^d / k_i^d))

Quick test confirmed strong effect of L near the phase boundary (2026-06-27).
