# Archive: notebook 01 results, global-Phi replicator bug

The replicator/proportional-imitation update rule (`_rep` in model.py) used a single
GLOBAL normalization Phi = k_max * b (k_max = the largest degree anywhere in the network)
instead of the PER-PAIR Phi_ij = max(k_i, k_j) * b used in Pereda2016 and Roca2009 (Eq. 35).
Traced to an early model-exploration document (`IDEAS TFM Gabrielle.docx`) that explicitly
considered three options -- (A) global Phi "easy", (B) per-pair Phi, (C) mean payoff -- and
noted even then that (A) makes imitation "muy lenta" and unevenly so in heterogeneous
networks; (A) is what ended up in model.py, mislabeled as "the PRE2016 rule".

Confirmed empirically (diagnostic script, 2026-09-02): negligible difference on ER networks
(homogeneous degree, so max(k_i,k_j) ~ k_max for most pairs), but a LARGE difference on BA
networks -- up to Δrho ~ 0.46 on BA z=16, enough to flip cooperative vs largely-defective
regimes. This directly affects the REP-vs-Fermi validation in Sec. III.A (L=1).

Fixed 2026-09-02: `_rep` now computes phi = max(deg_i, deg_j) * b per pair. These CSVs
(notebook 01, generated with the already-corrected window-mean stopping criterion but the
WRONG global Phi) are archived here for provenance, not for use in the paper.
