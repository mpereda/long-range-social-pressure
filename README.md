![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-Under_Review-yellow)
![Target](https://img.shields.io/badge/Target-Physical_Review_E-red)

# long-range-social-pressure

Code accompanying the paper **"Long-range social pressure and the evolution of cooperation in multiplex networks"** (M. Pereda and G. Muller, in preparation).

## Overview

We extend the model of Pereda (2016) by allowing the social pressure exerted by vigilant individuals to propagate beyond direct neighbors in a multiplex network. In the base model, a player's temptation to defect decreases with the fraction of vigilant direct neighbors. Here, vigilance influence reaches nodes at network distance up to L, weighted by a geometric decay kernel (α_l ∝ λ^(l−1)).

The two coupled layers are:
- **Game layer** — Prisoner's Dilemma played between direct neighbors
- **Vigilance layer** — influence I_i computed via BFS up to distance L

When L = 1 both layers coincide and the model reduces to Pereda (2016). For L > 1 the vigilance layer has longer reach than the game layer.

Besides synthetic Barabási–Albert and Erdős–Rényi networks, we validate the model on a real social network: the CKM physician dataset (Coleman, Katz & Menzel, 1966), using friendship ties as the game layer and advice-seeking ties as the vigilance layer.

## Repository structure

- [`model.py`](model.py): simulation engine ([Numba](https://numba.readthedocs.io/en/stable/)-accelerated)
- [`01-REP-vs-Fermi.ipynb`](01-REP-vs-Fermi.ipynb): validation — replicator vs Fermi update rule at L = 1
- [`02-long-range-correlated.ipynb`](02-long-range-correlated.ipynb): main results — L = 1–4, correlated multiplex
- [`03-long-range-uncorrelated.ipynb`](03-long-range-uncorrelated.ipynb): L = 1–4, uncorrelated multiplex
- [`04-lambda-sensitivity.ipynb`](04-lambda-sensitivity.ipynb): sensitivity to the geometric decay parameter λ
- [`05-real-network-CKM.ipynb`](05-real-network-CKM.ipynb): validation on the CKM real physician network (4 towns)
- [`real_networks/export_CKM.R`](real_networks/export_CKM.R): R script that exports the CKM dataset (via the [spatialprobit](https://cran.r-project.org/package=spatialprobit) package) to CSV for notebook 05; exported CSVs themselves are not tracked by git — run `Rscript export_CKM.R` from `real_networks/` to regenerate them
- [`data/`](data/): simulation outputs (not tracked by git)
- [`figures/`](figures/): paper figures
- [`robustness_checks/`](robustness_checks/): additional robustness studies (see below)
- [`archive_v1_normalized_kernel/`](archive_v1_normalized_kernel/): archived v1 results (normalized kernel, superseded)

## Runtime

Notebooks 02/03 (N = 1000, full L = 1–4 sweep) take ~7.3h each; notebook 04 (λ-sensitivity,
N = 1000) ~19.6h; notebook 05 (CKM, N ≤ 110) ~18min. Benchmarked on an Apple M5 Pro (24 GB
RAM, 15 cores), `n_jobs=-1`. Numba JIT-compiles on first call, so the first sweep in any
session is slower than subsequent ones.

## Robustness checks

[`robustness_checks/`](robustness_checks/) holds standalone scripts (reusing `model.py`) for
additional checks beyond the main notebooks. Each produces a CSV under
`robustness_checks/data/` (not tracked by git); scripts named `*_analysis.py` read back the
CSV from the matching `*.py` script to summarize it.

- **Network realization** — does a single fixed network realization per topology/degree
  drive the results?
  - `network_realization_robustness.py` / `network_realization_analysis.py`: reruns
    representative points on 10 independent realizations per topology, degree, correlation,
    and L; compares between- vs. within-network dispersion.
  - `network_realization_value_check.py`: checks whether the fixed realization's ⟨ρ⟩ is
    itself typical of that spread, not just low-variance relative to it.
  - `network_realization_Lgain_theta03.py`: checks whether the L = 1→2→3→4 cooperation gain
    itself (not just ⟨ρ⟩ at one L) holds up across 30 independent realizations, at points
    chosen for the largest gain along a fixed θ reference line. Supersedes
    `network_realization_L_effect.py` (kept for reference), which anchored point selection at
    L = 4 by regime instead of by gain size.

- **Population size, Fermi noise, and update scheme** — does N = 1000, K = 0.1, synchronous
  updating drive the results?
  - `nk_async_theta03.py`: reruns the same points as `network_realization_Lgain_theta03.py`
    at several N and K values and under an asynchronous Fermi update. Supersedes
    `nk_async_robustness.py` / `nk_async_analysis.py` (kept for reference).

- **Inter-layer structure**
  - `label_shuffle_control.py` / `label_shuffle_analysis.py`: disentangles edge overlap from
    inter-layer degree correlation as the source of Barabási–Albert's residual sensitivity to
    "uncorrelated" layers, by additionally shuffling node labels.
  - `rewiring_interpolation.py` / `rewiring_analysis.py` / `rewiring_main_fig.py`:
    interpolates continuously between the correlated and uncorrelated multiplex via
    degree-preserving edge rewiring, to check whether the transition between them is
    continuous or discontinuous.

- **Model design**
  - `normalized_kernel_check.py`: compares the unnormalized geometric kernel used throughout
    against a normalized alternative.
  - `relative_gain_metrics.py`: a relative (as well as absolute) cooperation-gain metric.
  - `bistability_BAz16.py`: checks that the high variance observed near the transition region
    of BA z = 16 reflects genuine bistability (individual replications settling into
    different attractors) rather than non-convergence.

## Dependencies

Install with:

```
pip install -r requirements.txt
```

Requires Python ≥ 3.10. Main dependencies:

- [NumPy](https://numpy.org/doc/) ≥ 1.24
- [Numba](https://numba.readthedocs.io/en/stable/) ≥ 0.58 — JIT compilation for the simulation engine
- [NetworkX](https://networkx.org/documentation/stable/) ≥ 3.0 — network generation and BFS
- [pandas](https://pandas.pydata.org/docs/) ≥ 2.0
- [Matplotlib](https://matplotlib.org/stable/index.html) ≥ 3.7
- [SciPy](https://docs.scipy.org/doc/scipy/) ≥ 1.10
- [Jupyter](https://jupyter.org/) ≥ 1.0

## Model summary

| Parameter | Value |
|-----------|-------|
| N | 1000 |
| Replications | 100 |
| b | 1.0 – 2.0 (step 0.1) |
| θ | 0.0 – 1.0 (step 0.1) |
| L | 1, 2, 3, 4 |
| λ | 0.5 (main, synthetic); 0.1–0.9 (sensitivity); 0.65 (CKM real network) |
| Update rule | Fermi, K = 0.1 |
| Networks | BA and ER (z = 4, z = 16); CKM real physician network (4 towns, N = 34–110) |
| Stopping condition | Adaptive: non-overlapping 100-gen windows (min 500 gens), stationary once two *consecutive window means* agree within 10⁻² (\|mean(window_k) − mean(window_{k−1})\| < 10⁻²); window mean returned as the stationary ⟨ρ⟩ |

## References

- Pereda, M. (2016). Evolution of cooperation under social pressure in multiplex networks. *Phys. Rev. E* **94**, 032314.
- Pereda, M. & Vilone, D. (2017). Social pressure and environmental effects on networks: A path to cooperation. *Games* **8**, 7.
- Coleman, J., Katz, E., & Menzel, H. (1966). *Medical Innovation: A Diffusion Study*. Bobbs Merrill. (source of the CKM physician network, via the [`spatialprobit`](https://cran.r-project.org/package=spatialprobit) R package)
