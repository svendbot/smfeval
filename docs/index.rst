smfeval
=======

Probabilistic SLAM trajectory format and scoring tool.

This package implements proper scoring rules and calibration diagnostics
for the **translation** marginal of probabilistic pose-trajectory
predictions on SE(3). The scoring side is built on the kernel-score /
energy-form characterisation of strictly proper rules (Gneiting &
Raftery, 2007); the SE(3) machinery (alignment, residuals) follows Solà,
Deray & Atchuthan (2018) and Barfoot (2017).

Orientation is not scored. A proper score on SO(3) needs a belief density
whose normaliser is intractable for the natural rotation families, and the
first-order tangent alternative awaits a rotation-frame audit, so rotation
scoring is left to future work (the full argument is in a forthcoming
paper).

Scoring rules are evaluated *prequentially* (Dawid, 1984): at each
matched timestep the one-step-ahead predictive belief is scored against
the realised reference pose, and the resulting score series is
aggregated with a percentile interval. Because that series is
autocorrelated — drift accumulates, regimes (loop closure, degeneracy
exit) persist for many frames — the iid percentile bootstrap (Efron,
1979) under-covers. ``smfeval`` therefore replaces it with the
stationary bootstrap of Politis & Romano (1994), with a mean block
length selected automatically by the Politis & White (2004) flat-top
lag-window estimator. The estimated block length is reported next to
every score CI as a temporal-dependence diagnostic.

For Gaussian beliefs the translation-marginal log score is reported,
obtained by selecting the translation 3-vector / 3×3 sub-block of the
SE(3) residual and its covariance (rotation integrated out). It splits
exactly into a calibration term (½·NEES) and a sharpness term, so a
filter that sharpens without earning it fails calibration while winning
on sharpness — the over-confidence smfeval is built to surface.

Per-trajectory point summaries (count / mean / median / std / min /
max) follow the TUM RGB-D benchmark convention (Sturm et al., 2012).

.. toctree::
   :maxdepth: 2
   :caption: Metrics

   metrics

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
