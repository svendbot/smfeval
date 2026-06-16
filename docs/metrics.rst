Metrics and how to read them
============================

``smfeval`` reports several proper scoring rules and calibration
diagnostics side by side. Each touches a different part of the predictive
**translation** distribution (bulk shape, tails, a chosen coverage level),
so no single number is sufficient. This page gives a high-level
interpretation of each output. The format and conventions are in
``SQUARE_spec.md``; the theory is in the paper (see :doc:`index`).

Throughout, a *belief* is the filter's one-step-ahead predictive pose
distribution (mean and covariance for a Gaussian filter, weighted particles
for an ensemble). All proper scores act on the **translation** marginal of
that belief; orientation is not scored, because a proper score on SO(3)
needs a belief density whose normaliser is intractable for the natural
rotation families (paper §II.b / §V.d). A score is computed at every
matched timestep and aggregated. Smaller is better unless noted.

Calibration verdict (NEES and ANEES)
------------------------------------

The headline ``smfeval nees`` check. The **NEES** (normalized estimation
error squared) is the squared error measured in the belief's own
covariance, :math:`e^\top \Sigma^{-1} e`. Read it as how many sigma the
error is, squared. Under a truthful translation covariance it follows a
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
  nominal (e.g. 90%) credible ellipsoid. A calibrated belief hits the
  nominal level.

No-reference pairwise NEES
--------------------------

``smfeval pair`` scores two filters against each other with the reference
never consulted. Filter A is aligned to filter B and their difference is
scored under the summed covariances, giving a pairwise NEES that is again
:math:`\chi^2_3` under truthful, independent beliefs. Common-mode error
and an understated reference covariance both push the statistic *down*, so
an elevated value is a **lower bound** on the miscalibration.

CRPS (translation)
------------------

The continuous ranked probability score is a strictly proper score on a
scalar marginal (here the per-axis translation error). It rewards a sharp
belief centred on the truth. It **saturates** toward the raw error once the
belief is badly over-confident, so on its own it understates gross
miscalibration. Read it together with NEES.

Relative-pose CRPS (short windows)
----------------------------------

``--rpe-window`` scores short-horizon pose *increments* instead of
absolute poses. Over a short window the normalized error stays
:math:`O(1)`, where CRPS is most sensitive. This restores the calibration
signal that absolute-pose CRPS loses once a filter is over-confident.
Reported per window length.

Energy score
------------

The multivariate generalization of CRPS (the kernel/energy form of
Gneiting & Raftery, 2007), applied to the 3-D translation vector. It
scores the joint translation distribution, so it catches mis-shaped
translation cross-covariance that the per-axis CRPS misses. At the
reported covariance it reduces to the per-pose error norm that APE
aggregates, so it reproduces existing point accuracy as a special case.

Gaussian log score and its calibration/sharpness split
------------------------------------------------------

For a Gaussian belief the negative log density of the translation residual
(:math:`d=3`) splits **exactly** into

.. math::

   -\log p = \underbrace{\tfrac12\, e^\top\Sigma^{-1}e}_{\text{calibration}}
     + \underbrace{\tfrac12\bigl(\log\det\Sigma + d\log 2\pi\bigr)}_{\text{sharpness}}.

The calibration term is half the NEES (does the spread match the error?);
the sharpness term rewards a confident belief (small covariance). A filter
can win on sharpness while failing calibration, which is exactly the
over-confidence smfeval is built to surface. The score acts on the
translation block of the pose belief (rotation integrated out).

Interval score
--------------

A proper score for a single chosen central interval (e.g. 90%). It adds
the interval width to a penalty for the truth falling outside it. It
captures how tight and honest the belief is at one coverage level.

PIT and coverage calibration
----------------------------

The probability integral transform maps each realised error through the
belief's CDF. Under a calibrated belief the PIT values are uniform, tested
with a Kolmogorov–Smirnov statistic. Combined with the empirical coverage
of the credible region, this is a distribution-free check that the stated
uncertainty matches reality, independent of any single scoring rule.

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
(:math:`N_\text{eff} < N/10`). A low :math:`N_\text{eff}` means the
weighted belief is effectively a handful of particles, so its scores are
unreliable regardless of the numbers.

Reading the confidence intervals
--------------------------------

Scores are evaluated prequentially, so the per-step series is
autocorrelated (drift accumulates; loop closure and degeneracy persist for
many frames). ``smfeval`` reports each aggregate with a **stationary
bootstrap** interval (Politis & Romano, 1994) and prints the mean **block
length** (Politis & White, 2004) next to it as a temporal-dependence
diagnostic. A large block length means few effectively independent samples,
so treat the interval, not the point estimate, as the result.

Sync and alignment caveats
--------------------------

Two upstream steps decide whether the scores are even meaningful.

- **Synchronization risk.** Nearest-neighbour timestamp matching can leave
  slop between an estimate pose and its ground-truth partner. smfeval reports
  a sync-risk score :math:`r = \lVert v_\text{gt}\rVert\,|\Delta t| /
  \sigma_\text{trans}` and flags pairs above ~0.3. When many fire, cross-check
  with ``--sync=interpolate_gt`` before trusting a calibration finding.
- **Alignment bias.** The alignment transform is fit on the same poses it is
  scored against, so post-alignment residuals are biased low. This is worst on
  short trajectories and high-DoF gauges (``se3``, ``sim3``). ``--n_to_align``
  fits on a prefix and scores the remainder to remove that bias.
