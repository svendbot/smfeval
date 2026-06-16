Metrics and how to read them
============================

``smfeval`` scores the **translation** marginal of a filter's predictive pose
belief and reports a handful of numbers, each measuring a different property of
that belief. This page explains what every number means, the equation behind
it, its units, and which way is "good", so you can act on the value rather than
trust it.

What is scored
--------------

A *belief* is the filter's one-step-ahead predictive pose distribution: a mean
and a covariance for a Gaussian filter, or weighted particles for an ensemble.
``smfeval`` scores the translation block of that belief — the estimated minus
the reference position, :math:`e = \hat t - t_\mathrm{ref} \in \mathbb{R}^3`,
against the :math:`3\times 3` translation covariance :math:`\Sigma` (rotation
integrated out).

Orientation is **not** scored. A proper score on :math:`SO(3)` needs a belief
density whose normaliser is intractable for the natural rotation families, and
the first-order tangent approximation has not been audited, so rotation scoring
is left to future work (paper, §II.b / §V.d).

How the numbers are produced
----------------------------

Each score is computed at every matched timestep against the realised
ground-truth position (*prequential* evaluation, Dawid 1984) and then
aggregated. **Smaller is better unless noted.** Because the per-step series is
autocorrelated, the aggregate carries a stationary-bootstrap interval rather
than an iid one; the :doc:`diagnostics` page explains how to read it.

The four groups below separate the *scores* (do the numbers match reality?)
from the *calibration* checks that diagnose them, the *alignment* step that
decides whether they are meaningful at all, and the *diagnostics* that flag
confounders.

.. toctree::
   :maxdepth: 1

   proper_scores
   calibration
   alignment
   diagnostics

The format and conventions are in ``SQUARE_spec.md``; the theory is in the
paper (see :doc:`index`).
