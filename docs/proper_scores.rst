Proper scores
=============

A scoring rule assigns the belief a penalty :math:`S(F, y)` at the realised
position :math:`y`. It is *strictly proper* when the expected penalty is
minimised only by reporting the true error distribution, so a filter cannot
improve its score by hedging or sharpening its covariance dishonestly. All the
rules below are strictly proper on the translation marginal. Smaller is better.

Translation CRPS
----------------

The continuous ranked probability score on each translation axis. For a
Gaussian belief :math:`\mathcal N(\mu, \sigma^2)` it is closed form,

.. math::

   \mathrm{CRPS}\bigl(\mathcal N(\mu,\sigma^2), y\bigr)
   = \sigma\Bigl[\omega\bigl(2\Phi(\omega) - 1\bigr)
       + 2\varphi(\omega) - \tfrac{1}{\sqrt{\pi}}\Bigr],
   \qquad \omega = \frac{y - \mu}{\sigma},

with :math:`\Phi, \varphi` the standard normal CDF and PDF (Gneiting &
Raftery 2007). For an ensemble it is the usual sample U-statistic. The reported
value is the mean over the three axes.

- **Units:** metres. **Direction:** smaller is sharper-and-accurate.
- **How to read it:** CRPS rewards a sharp belief centred on the reference. When
  the belief is badly over-confident (:math:`|\omega| \gg 1`) it **saturates**
  toward the raw error :math:`|y-\mu|` and stops responding to :math:`\sigma`,
  so on its own it understates gross miscalibration. Read it next to NEES
  (:doc:`calibration`).

Relative-pose CRPS
------------------

The same CRPS applied to short-horizon position *increments* rather than
absolute positions (``--rpe-window``, given a list of window lengths in
seconds). Over a short window the normalised increment error stays
:math:`O(1)`, the regime where CRPS is most sensitive, so this recovers the
local calibration signal that absolute CRPS loses once a filter is
over-confident. Reported per window length, with the per-window RPE RMSE and
mean :math:`z^2` alongside.

Energy score
------------

The multivariate generalisation of CRPS (Gneiting & Raftery 2007, §4.2),
applied to the :math:`3`-D translation vector. For independent draws
:math:`X, X' \sim F` and observation :math:`y`,

.. math::

   \mathrm{ES}(F, y) = \mathbb{E}\,\lVert X - y\rVert
       - \tfrac12\,\mathbb{E}\,\lVert X - X'\rVert.

- **Units:** metres. **Direction:** smaller is better.
- **How to read it:** the first term rewards accuracy, the second rewards
  sharpness. At the reported covariance the energy score reduces to the
  per-pose error norm that absolute pose error (APE) aggregates, so it
  reproduces existing point-accuracy practice as a special case rather than
  replacing it.

Gaussian log score
------------------

The negative log density of the reference under the predictive Gaussian (the
*ignorance* score, Good 1952), on the translation residual (:math:`d = 3`):

.. math::

   -\log p(e)
   = \underbrace{\tfrac12\, e^\top \Sigma^{-1} e}_{\text{calibration}}
   + \underbrace{\tfrac12\bigl(\log\det\Sigma + d\log 2\pi\bigr)}_{\text{sharpness}}.

The split is exact and ``smfeval`` reports both terms (under ``--calibration``).

- **Units:** nats. The sharpness term carries the units of :math:`\Sigma`, so
  the absolute log score is **not** scale-invariant — compare values only under
  a fixed unit convention.
- **How to read it:** the **calibration** term is half the NEES — it asks
  whether the spread matches the error, and keeps growing under over-confidence
  where CRPS saturates. The **sharpness** term rewards a tight covariance. A
  filter can win on sharpness while failing calibration; that combination is
  the over-confidence the tool exists to surface. See the calibration/sharpness
  verdict in :doc:`calibration`.

Interval score
--------------

A proper score for a single central credible interval at level :math:`1-\alpha`
(e.g. 90%), evaluated on the translation-magnitude marginal with bounds
:math:`l, u` from the predictive samples:

.. math::

   \mathrm{IS}_\alpha(l, u; y) = (u - l)
     + \tfrac{2}{\alpha}(l - y)\,\mathbf 1\{y < l\}
     + \tfrac{2}{\alpha}(y - u)\,\mathbf 1\{y > u\}.

- **Units:** metres. **Direction:** smaller is better.
- **How to read it:** it pays the interval width plus a penalty, scaled by
  :math:`2/\alpha`, when the reference falls outside. It captures, at one chosen
  coverage level, whether the belief is both tight and honest: a narrow
  interval that misses is punished, a wide interval that always contains is
  penalised on width.
