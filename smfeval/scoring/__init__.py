from smfeval.scoring.bias_variance import (
  BiasVarianceResult,
  bias_variance,
)
from smfeval.scoring.calibration import (
  CalibrationResult,
  calibrate,
)
from smfeval.scoring.crps import translation_crps
from smfeval.scoring.energy import energy_score
from smfeval.scoring.ensemble_diag import (
  EnsembleDiagnostic,
  ensemble_diagnostics,
)
from smfeval.scoring.interval import (
  interval_score,
  translation_magnitude_interval_score,
)
from smfeval.scoring.logscore import (
  AneesResult,
  DecomposedLogScore,
  GaussianLogScore,
  ScoreComponents,
  anees_consistency,
  batched_score_components,
  gaussian_log_score,
  gaussian_log_score_components,
  student_t_logscore_sweep,
  student_t_neg_log_density,
  translation_components,
)
from smfeval.scoring.pairwise import (
  PROPRIETY_CAVEAT,
  PairInputError,
  PairResult,
  pair_translation_nees,
)
from smfeval.scoring.relative import (
  RelativeCalibrationResult,
  RelativeCrpsResult,
  relative_calibration,
  relative_translation_crps,
)
from smfeval.scoring.summary import (
  ScoreSummary,
  politis_white_block_length,
  summarize,
)

# Human-readable (label, unit) for each score key in a report's scores dict.
# Renderers iterate the dict and look labels up here, so a newly added score
# cannot silently vanish from the text report.
SCORE_LABELS: dict[str, tuple[str, str]] = {
  "translation_crps": ("Translation CRPS", "m"),
  "energy_score": ("Energy score", "m"),
  "log_score_translation": ("Log score (translation)", ""),
  "interval_score": ("Interval score", ""),
}

__all__ = [
  "PROPRIETY_CAVEAT",
  "AneesResult",
  "BiasVarianceResult",
  "CalibrationResult",
  "DecomposedLogScore",
  "EnsembleDiagnostic",
  "GaussianLogScore",
  "PairInputError",
  "PairResult",
  "RelativeCalibrationResult",
  "RelativeCrpsResult",
  "ScoreComponents",
  "ScoreSummary",
  "anees_consistency",
  "batched_score_components",
  "bias_variance",
  "calibrate",
  "relative_calibration",
  "relative_translation_crps",
  "energy_score",
  "ensemble_diagnostics",
  "gaussian_log_score",
  "gaussian_log_score_components",
  "student_t_logscore_sweep",
  "student_t_neg_log_density",
  "interval_score",
  "pair_translation_nees",
  "politis_white_block_length",
  "summarize",
  "translation_components",
  "translation_crps",
  "translation_magnitude_interval_score",
]
