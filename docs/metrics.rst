Metrics and how to read them
============================

``smfeval`` reports several proper scoring rules and calibration
diagnostics side by side. Each touches a different part of the predictive
distribution — bulk shape, tails, joint structure, a chosen coverage
level — so no single number is sufficient. This page gives a high-level
interpretation of each output; the format, taxonomy, and theory are in
``SQUARE_spec.md`` and the paper (see :doc:`index`).

Throughout, a *belief* is the filter's one-step-ahead predictive pose
distribution (mean + covariance for a Gaussian filter, weighted particles
for an ensemble). A score is computed at every matched timestep and
aggregated; smaller is better unless noted.

Calibration verdict: NEES and ANEES
-----------------------------------

The headline ``smfeval nees`` check. The **NEES** (normalized estimation
error squared) is the squared error measured in the belief's own
covariance, :math:`e^\top \Sigma^{-1} e` — read it as "how many sigma the
error is, squared." Under a truthful translation covariance it follows a
:math:`\chi^2_3` law, whose median is **2.366**.

- **median NEES** much larger than 2.37 ⇒ over-confident (covariance too
  tight); much smaller ⇒ conservative (too loose).
- **scale gap** :math:`k = \text{median NEES} / 2.366` is the factor by
  which the covariance is too tight; per axis that is :math:`\sqrt{k}`.
- **ANEES** is the mean NEES, tested against a two-sided :math:`\chi^2`
  consistency interval to yield an ``optimistic`` / ``consistent`` /
  ``conservative`` verdict. The mean is outlier-dominated, so a median far
  below the ANEES points to a heavy-dynamics tail rather than a uniformly
  bad bulk.
- **coverage** is the fraction of poses whose truth lands inside the
  nominal (e.g. 90%) credible ellipsoid; a calibrated belief hits the
  nominal level.

No-reference pairwise NEES
--------------------------

``smfeval pair`` scores two filters against each other with the reference
never consulted: filter A is aligned to filter B and their difference is
scored under the summed covariances, giving a pairwise NEES that is again
:math:`\chi^2_3` under truthful, independent beliefs. Common-mode error
and an understated reference covariance both push the statistic *down*, so
an elevated value is a **lower bound** on the miscalibration — when it
fires, it fires honestly.

CRPS (translation and rotation)
-------------------------------

The continuous ranked probability score is a strictly proper score on a
scalar marginal (here the translation magnitude and the rotation angle).
It rewards a sharp belief centred on the truth. Note it **saturates**
toward the raw error once the belief is badly over-confident, so on its
own it understates gross miscalibration — read it together with NEES.

Relative-pose CRPS (short windows)
----------------------------------

``--rpe-window`` scores short-horizon pose *increments* instead of
absolute poses. Over a short window the normalized error stays
:math:`O(1)`, where CRPS is most sensitive — this restores the
calibration signal that absolute-pose CRPS loses once a filter is
over-confident. Reported per window length.

Energy score
------------

The multivariate generalization of CRPS (the kernel/energy form of
Gneiting & Raftery, 2007). It scores the full joint predictive
distribution, not a single marginal, so it catches mis-shaped
cross-covariance that the per-axis scores miss.

Gaussian log score and its calibration/sharpness split
------------------------------------------------------

For a Gaussian belief the negative log density splits **exactly** into

.. math::

   -\log p = \underbrace{\tfrac12\, e^\top\Sigma^{-1}e}_{\text{calibration}}
     + \underbrace{\tfrac12\bigl(\log\det\Sigma + d\log 2\pi\bigr)}_{\text{sharpness}}.

The calibration term is half the NEES (does the spread match the error?);
the sharpness term rewards a confident belief (small covariance). A filter
can win on sharpness while failing calibration — that is exactly the
over-confidence smfeval is built to surface. The joint SE(3) score is
reported alongside its translation- and rotation-marginal components,
since the 6×6 covariance can hide a pathology in just one block.

Interval score
--------------

A proper score for a single chosen central interval (e.g. 90%): it adds
the interval width to a penalty for the truth falling outside it. Read it
as "how tight *and* honest is the belief at this one coverage level."

PIT and coverage calibration
----------------------------

The probability integral transform maps each realised error through the
belief's CDF; under a calibrated belief the PIT values are uniform, tested
with a Kolmogorov–Smirnov statistic. Combined with the empirical coverage
of the credible region, this is the distribution-free check that the
stated uncertainty matches reality — independent of any single scoring
rule.

Track-frame bias vs variance
----------------------------

Windowed translation error is decomposed into a systematic **bias** and a
random **variance** along track-frame axes (along / cross / vertical). The
``bias_fraction`` (bias² / MSE) says whether the error is a fixable offset
(extrinsic, time-offset, scale, gravity) or irreducible noise, and the
dominant axis localizes the channel. This drives the report's recommended
actions.

Ensemble diagnostics
--------------------

For particle/ensemble filters the report adds the effective sample size
:math:`N_\text{eff}` per step and the fraction of steps in degeneracy
(:math:`N_\text{eff} < N/10`) — a low :math:`N_\text{eff}` means the
weighted belief is effectively a handful of particles, so its scores are
unreliable regardless of the numbers.

Reading the confidence intervals
--------------------------------

Scores are evaluated prequentially, so the per-step series is
autocorrelated (drift accumulates; loop closure and degeneracy persist for
many frames). ``smfeval`` reports each aggregate with a **stationary
bootstrap** interval (Politis & Romano, 1994) and prints the mean **block
length** (Politis & White, 2004) next to it as a temporal-dependence
diagnostic — a large block length means few effectively independent
samples, so treat the interval, not the point estimate, as the result.
