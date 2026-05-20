smfeval
=======

Probabilistic SLAM trajectory format and scoring tool.

This package implements proper scoring rules and calibration diagnostics
for probabilistic pose-trajectory predictions on SE(3). The scoring side
is built on the kernel-score / energy-form characterisation of strictly
proper rules (Gneiting & Raftery, 2007); the SE(3) machinery follows
Solà, Deray & Atchuthan (2018) and Barfoot (2017).

Scoring rules are evaluated *prequentially* (Dawid, 1984): at each
matched timestep the one-step-ahead predictive belief is scored against
the realised ground-truth pose, and the resulting score series is
aggregated with a percentile interval. Because that series is
autocorrelated — drift accumulates, regimes (loop closure, degeneracy
exit) persist for many frames — the iid percentile bootstrap (Efron,
1979) under-covers. ``smfeval`` therefore replaces it with the
stationary bootstrap of Politis & Romano (1994), with a mean block
length selected automatically by the Politis & White (2004) flat-top
lag-window estimator. The estimated block length is reported next to
every score CI as a temporal-dependence diagnostic.

For Gaussian beliefs the joint SE(3) log score is reported alongside
its translation-marginal and rotation-marginal components: the 6×6
covariance mixes translation and rotation, so a single scalar hides
calibration pathologies that target only one block (e.g. a LiDAR
filter well-calibrated in translation but overconfident in yaw under
geometric degeneracy along the motion direction). The marginals are
obtained by selecting the corresponding 3-vector / 3×3 sub-block of
the SE(3) residual and its covariance; together with the joint they
decompose the scalar without losing the cross-covariance information.

Per-trajectory point summaries (count / mean / median / std / min /
max) follow the TUM RGB-D benchmark convention (Sturm et al., 2012).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
