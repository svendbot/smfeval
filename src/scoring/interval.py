"""Interval score at coverage 1-α for a univariate predictive (Gneiting & Raftery 2007)."""

import numpy as np

from src.scoring._predictive import translation_samples
from src.steps import Step
from src.types import TangentOrder


def interval_score(
    lower: float, upper: float, observation: float, alpha: float = 0.1
) -> float:
    """IS_α(l, u; y) = (u - l) + (2/α)(l - y)·1{y<l} + (2/α)(y - u)·1{y>u}."""
    width = upper - lower
    pen = 0.0
    if observation < lower:
        pen += (2.0 / alpha) * (lower - observation)
    elif observation > upper:
        pen += (2.0 / alpha) * (observation - upper)
    return float(width + pen)


def interval_from_samples(samples: np.ndarray, alpha: float = 0.1) -> tuple[float, float]:
    """Equal-tailed (α/2, 1-α/2) interval from a sample vector."""
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lo, hi


def translation_magnitude_interval_score(
    pred_step: Step,
    gt_translation: np.ndarray,
    alpha: float = 0.1,
    n_samples: int = 128,
    rng: np.random.Generator | None = None,
    tangent_order: TangentOrder = TangentOrder.TRANS_ROT,
) -> float:
    """Interval score on translation magnitude ‖t - μ_t‖.

    Predictive samples from the step's belief give the equal-tailed interval;
    the observation is the GT translation's distance from the predictive mean.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    samples, mu = translation_samples(pred_step, n_samples, rng, tangent_order)
    mags = np.linalg.norm(samples - mu, axis=1)
    lo, hi = interval_from_samples(mags, alpha)
    obs = float(np.linalg.norm(gt_translation - mu))
    return interval_score(lo, hi, obs, alpha)
