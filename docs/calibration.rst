Calibration
===========

Calibration checks ask a different question from the proper scores: not "is the
belief sharp and accurate?" but "is the stated uncertainty the right size?" A
sharp, over-confident filter scores well on CRPS yet fails here. These are the
headline diagnostics — NEES and coverage — plus the distribution-free PIT check
and the no-reference pairwise route.

NEES and the calibration verdict
--------------------------------

The normalised estimation error squared is the error measured in the belief's
own covariance,

.. math::

   \mathrm{NEES} = e^\top \Sigma^{-1} e,

i.e. how many sigma the error is, squared. Under a honest translation
covariance it follows a :math:`\chi^2_3` law, whose median is **2.366**. The
``smfeval nees`` verb summarises the per-pose series into three lines:

- **median NEES** — much larger than 2.37 means over-confident (covariance too
  tight); much smaller means conservative (too loose).
- **scale gap** :math:`k = \operatorname{median}\mathrm{NEES} / 2.366` — the
  factor by which the covariance is too tight; per axis that is
  :math:`\sqrt{k}` (variance versus standard deviation).
- **coverage** — the fraction of poses whose reference lands inside the nominal
  (e.g. 90%) credible ellipsoid. A calibrated belief hits the nominal level.

The qualitative verdict comes from **ANEES**, the mean NEES, tested against a
two-sided :math:`\chi^2` consistency interval to print ``optimistic`` /
``consistent`` / ``conservative``. The mean is outlier-dominated, so a median
far below the ANEES points to a heavy-dynamics *tail* rather than a uniformly
miscalibrated *bulk* — the distinction that routes the fix (a robust likelihood
for a tail, a covariance widening for the bulk).

No-reference pairwise NEES
--------------------------

``smfeval pair`` scores two filters against each other with no reference
consulted. Filter A is aligned to filter B and their difference
:math:`d = \hat t_A - \hat t_B` is scored under the summed covariances,

.. math::

   \mathrm{NEES}_\mathrm{pair} = d^\top (\Sigma_A + \Sigma_B)^{-1} d,

again :math:`\chi^2_3` under honest, independent beliefs. The form drops the
cross-covariance :math:`C_{AB}`; common-mode error and an understated reference
covariance both push the statistic *down*, so an elevated value is a **lower
bound** on the miscalibration, never an over-statement.

PIT and coverage
----------------

The probability integral transform maps each realised error through the
belief's predictive CDF, :math:`p = F_\mathrm{pred}(y_\mathrm{obs})`, computed
empirically from samples of the translation magnitude. Under a calibrated
belief the PIT values are uniform on :math:`[0,1]`, tested with a
Kolmogorov–Smirnov statistic (a small p-value flags miscalibration). Combined
with the empirical coverage of the credible ellipsoid, this is a
distribution-free check that the stated uncertainty matches reality,
independent of any single scoring rule. The standardised translation z-score
(mean and std of :math:`\lVert L^{-1} e\rVert / \sqrt{3}`, with
:math:`\Sigma = LL^\top`) is reported alongside; std :math:`>1` reads
over-confident, :math:`<1` conservative.

Ensemble diagnostics
--------------------

For particle/ensemble filters the report adds the effective sample size per
step, :math:`N_\mathrm{eff} = 1 / \sum_i w_i^2`, and the fraction of steps in
degeneracy (:math:`N_\mathrm{eff} < N/10`). A low :math:`N_\mathrm{eff}` means
the weighted belief is effectively a handful of particles, so its scores are
unreliable regardless of the numbers — read this before trusting an ensemble
filter's scores.
