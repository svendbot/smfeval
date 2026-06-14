API reference
=============

Scoring rules and calibration
-----------------------------

.. autosummary::
   :toctree: _autosummary

   smfeval.scoring.crps
   smfeval.scoring.energy
   smfeval.scoring.interval
   smfeval.scoring.logscore
   smfeval.scoring.calibration
   smfeval.scoring.pairwise
   smfeval.scoring.relative
   smfeval.scoring.bias_variance
   smfeval.scoring.ensemble_diag
   smfeval.scoring.gaussian_diag
   smfeval.scoring.summary

SE(3) geometry
--------------

.. autosummary::
   :toctree: _autosummary

   smfeval.se3.lie
   smfeval.se3.quat

Alignment
---------

.. autosummary::
   :toctree: _autosummary

   smfeval.align.fit
   smfeval.align.propagate

Synchronisation
---------------

.. autosummary::
   :toctree: _autosummary

   smfeval.sync.match
   smfeval.sync.interpolate
   smfeval.sync.risk

I/O and types
-------------

.. autosummary::
   :toctree: _autosummary

   smfeval.io.header
   smfeval.io.load
   smfeval.io.reader
   smfeval.io.writer
   smfeval.steps
   smfeval.format

Reporting and CLI
-----------------

.. autosummary::
   :toctree: _autosummary

   smfeval.report.builder
   smfeval.report.verdict
   smfeval.report.diagnostics
   smfeval.report.text
   smfeval.report.recommendations
   smfeval.cli.main
