Diagnostics and caveats
=======================

These outputs do not score the belief; they attribute a failure to a cause, flag
confounders that can fake one, and tell you how much to trust the aggregates.

Track-frame bias vs variance
----------------------------

Windowed translation error is split into a systematic **bias** and a random
**variance** along track-frame axes (along / cross / vertical). The reported

.. math::

   \text{bias\_fraction} = \frac{\lVert \text{bias} \rVert^2}{\mathrm{MSE}}

says whether the error is a fixable offset (close to 1) or irreducible noise
(close to 0), and the dominant axis localises the channel:

- **vertical** — gravity alignment or initial attitude;
- **along** — scale or time-offset (grows with speed);
- **cross** — a lateral extrinsic or heading error.

This is what separates "recalibrate the rig" from "tune the noise model", and it
drives the report's recommended actions.

Synchronization risk
--------------------

Nearest-neighbour timestamp matching can leave slop between an estimate pose and
its reference partner. The sync-risk score

.. math::

   r = \frac{\lVert v_\mathrm{gt} \rVert \, |\Delta t|}{\sigma_\mathrm{trans}}

measures the position error that slop induces, relative to the belief's own
translation sigma; pairs above :math:`\sim 0.3` are flagged. Matching error
shrinks the short-window covariance the same way genuine local over-confidence
does, so when many pairs fire, re-score with ``--sync=interpolate_ref`` (a GP on
:math:`SE(3)` queried at each estimate timestamp) before trusting a
short-horizon calibration finding.

Reading the confidence intervals
--------------------------------

Scores are evaluated prequentially, so the per-step series is autocorrelated:
drift accumulates, and regimes (loop closure, degeneracy exit) persist for many
frames. An iid bootstrap therefore under-covers. ``smfeval`` reports each
aggregate with a **stationary bootstrap** interval (Politis & Romano 1994) and
prints the mean **block length** (Politis & White 2004) next to it. The block
length is a temporal-dependence diagnostic: a large value means few effectively
independent samples, so treat the interval — not the point estimate — as the
result.
