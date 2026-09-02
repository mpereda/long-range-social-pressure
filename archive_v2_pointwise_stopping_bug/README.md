# Archive: v2 results, point-wise stopping-criterion bug

Production sweep results (notebooks 01-05) generated before 2026-09-02, using
the pre-fix stopping criterion in `model.py` (`_run_one` / `_run_one_rep`):
a raw point-to-point difference 100 generations apart, divided by 100. Since
rho in [0,1], this bound is <=0.01 by construction and the criterion fires
trivially at the first eligible check (t=500) regardless of whether the
system had actually converged -- confirmed empirically: in a diagnostic
across 9 representative (topology, z, L, b, theta) points, the old criterion
stopped at generation 500 in every single one of 72 test runs, with no
variance at all.

Flagged by PRE Referee 2 ("Clarification of the convergence criterion in
Eq. (5)"). Fixed by switching to a window-mean criterion (compare two
consecutive 100-generation window means directly, no extra /100), matching
PRE2016's original description and the design intent. See
`_windowed_stationary_mean` in model.py and the project memory
`decision-stopping-criterion.md` / `plan-PRE-major-revision.md` for full
detail.

Kept for provenance / possible supplementary comparison figure, not for use
in the paper's main results.
