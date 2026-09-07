Calibration
===========

Calibration checks ask a different question from the proper scores: not "is the
belief sharp and accurate?" but "is the stated uncertainty the right size?" A
sharp, over-confident filter scores well on CRPS yet fails here. These are the
headline diagnostics — NEES and coverage — plus the no-reference pairwise
route.

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

Coverage and standardised residuals
-----------------------------------

Coverage is the fraction of poses whose reference lands inside the nominal
:math:`1-\alpha` credible ellipsoid, :math:`e^\top \Sigma_t^{-1} e \le
\chi^2_{3,\,1-\alpha}`. Under a calibrated belief each pose is an independent
Bernoulli trial with success probability :math:`1-\alpha`, so the hit count is
Binomial and the report tests the realised rate with an **exact two-sided
binomial p-value** (``coverage_p``). A small p-value with coverage *below*
nominal reads over-confident; below nominal but not significant usually just
means the trajectory is short.

The standardised translation z-score (mean and std of
:math:`\lVert L^{-1} e\rVert / \sqrt{3}`, with :math:`\Sigma_t = LL^\top`) is
reported alongside; std :math:`>1` reads over-confident, :math:`<1`
conservative.

.. note::

   Coverage and the log score are computed on the residual in the
   perturbation convention the header declares — :math:`\log(T_\mathrm{est}^{-1}
   T_\mathrm{ref})` for ``right_perturbation``, :math:`\log(T_\mathrm{ref}
   T_\mathrm{est}^{-1})` for ``left_perturbation``. A covariance is the
   covariance *of a perturbation*, so scoring the wrong one against it inflates
   the NEES by the adjoint mismatch and reads as over-confidence in a filter
   that is honest.

Ensemble diagnostics
--------------------

For particle/ensemble filters the report adds the effective sample size per
step, :math:`N_\mathrm{eff} = 1 / \sum_i w_i^2`, and the fraction of steps in
degeneracy (:math:`N_\mathrm{eff} < N/10`). A low :math:`N_\mathrm{eff}` means
the weighted belief is effectively a handful of particles, so its scores are
unreliable regardless of the numbers — read this before trusting an ensemble
filter's scores.
