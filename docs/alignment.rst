Alignment
=========

Before any score is computed the estimate is brought into the reference frame by
a fitted gauge transform. This step decides whether the scores mean anything, so
it is reported in full and carries its own propriety caveat.

Gauge fitting
-------------

The estimate and reference live in frames that differ by an unobservable gauge
(global pose, and sometimes scale). ``smfeval`` fits that transform by Umeyama
alignment on the matched translations and reports the mode, the fitted
translation and rotation, the scale (for ``sim3``), the degrees of freedom
removed, and the post-alignment fit residual. The gauge is set by the declared
``--gauge`` / ``--align``:

- ``none`` — score in the given frame, 0 DoF removed.
- ``gravity_yaw`` — fix the unobservable yaw and planar offset (4 DoF).
- ``se3`` — full rigid alignment (6 DoF).
- ``sim3`` — rigid alignment plus a scale (7 DoF).

The fixed-gauge propriety caveat
--------------------------------

The scores are proper **only at a fixed gauge**. Full-trajectory alignment
estimates the gauge from the very poses it is then scored against, which is not
itself a proper operation: the fit absorbs part of the error, so post-alignment
residuals are **biased low**. The effect is worst on short trajectories and
high-DoF gauges (``se3``, ``sim3``), where the alignment has the most freedom
to flatter the estimate.

Removing the bias
-----------------

``--n_to_align N`` fits the gauge on a prefix of :math:`N` poses and scores the
remainder, so the scored poses never enter the fit. Use it whenever the report
warns that many DoF were removed over a short trajectory; the warning fires when
the trajectory length is small relative to the degrees of freedom removed.
