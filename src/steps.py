from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianStep:
    timestamp: float
    translation: np.ndarray
    quat_xyzw: np.ndarray
    covariance: np.ndarray


@dataclass
class EnsembleStep:
    timestamp: float
    particles: np.ndarray
    weights: np.ndarray | None


@dataclass
class DeterministicStep:
    timestamp: float
    translation: np.ndarray
    quat_xyzw: np.ndarray


Step = GaussianStep | EnsembleStep | DeterministicStep
