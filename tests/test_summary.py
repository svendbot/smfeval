import numpy as np

from src.scoring.summary import (
    politis_white_block_length,
    summarize,
)


def test_summary_shape_and_values():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=500) + 0.5
    s = summarize(arr, n_bootstrap=1000, ci_level=0.95, rng=rng)
    assert s.n == 500
    assert abs(s.mean - 0.5) < 0.1
    assert s.ci_low <= s.mean <= s.ci_high
    assert s.min <= s.median <= s.max
    assert s.std > 0.0


def test_summary_drops_nonfinite():
    s = summarize(np.array([1.0, 2.0, np.nan, np.inf, 3.0]))
    assert s.n == 3
    assert s.mean == 2.0


def test_summary_singleton_ci_collapses():
    s = summarize([3.14])
    assert s.n == 1
    assert s.mean == 3.14
    assert s.ci_low == s.ci_high == 3.14
    assert s.std == 0.0


def test_summary_empty_returns_nans():
    s = summarize([])
    assert s.n == 0
    for v in (s.mean, s.median, s.std, s.min, s.max, s.ci_low, s.ci_high, s.block_length):
        assert np.isnan(v)


def test_summary_ci_narrows_with_more_data():
    rng = np.random.default_rng(1)
    small = summarize(rng.normal(size=30), n_bootstrap=2000, rng=rng)
    big = summarize(rng.normal(size=3000), n_bootstrap=2000, rng=rng)
    small_w = small.ci_high - small.ci_low
    big_w = big.ci_high - big.ci_low
    assert big_w < small_w


def test_summary_to_dict_round_trip():
    s = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    d = s.to_dict()
    assert d["n"] == 5
    assert d["mean"] == 3.0
    assert "ci_low" in d and "ci_high" in d
    assert "block_length" in d


def _ar1(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1): x_t = ρ x_{t-1} + ε_t, ε_t ~ N(0, 1-ρ²)."""
    x = np.empty(n)
    x[0] = rng.normal() / np.sqrt(max(1.0 - rho**2, 1e-12))
    noise = rng.normal(scale=np.sqrt(max(1.0 - rho**2, 1e-12)), size=n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + noise[t]
    return x


def test_pw_block_length_near_one_for_white_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    b = politis_white_block_length(x)
    assert 1.0 <= b < 3.0, f"expected ~1 for white noise, got {b}"


def test_pw_block_length_grows_with_autocorrelation():
    rng = np.random.default_rng(0)
    low = politis_white_block_length(_ar1(2000, 0.0, rng))
    high = politis_white_block_length(_ar1(2000, 0.9, rng))
    assert high > 3.0 * low, f"high ρ should give a larger block: low={low}, high={high}"


def test_pw_block_length_handles_constant_series():
    assert politis_white_block_length(np.full(100, 3.14)) == 1.0


def test_block_bootstrap_ci_widens_with_autocorrelation():
    """Stationary bootstrap on an AR(1) series must produce a wider CI on the
    mean than the iid bootstrap on the same series, because the iid bootstrap
    underestimates the long-run variance."""
    rng = np.random.default_rng(7)
    x = _ar1(1000, 0.85, rng)
    iid = summarize(x, n_bootstrap=4000, block_length=1.0, rng=np.random.default_rng(0))
    blk = summarize(x, n_bootstrap=4000, rng=np.random.default_rng(0))
    iid_w = iid.ci_high - iid.ci_low
    blk_w = blk.ci_high - blk.ci_low
    assert blk_w > 1.5 * iid_w, f"iid CI={iid_w:.4f}, block CI={blk_w:.4f}"
    assert blk.block_length > 2.0


def test_block_bootstrap_matches_iid_on_white_noise():
    """When PW estimates ℓ ≈ 1, the stationary bootstrap should produce a CI
    similar to the iid bootstrap (no autocorrelation to account for)."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=1000)
    iid = summarize(x, n_bootstrap=4000, block_length=1.0, rng=np.random.default_rng(0))
    blk = summarize(x, n_bootstrap=4000, rng=np.random.default_rng(0))
    iid_w = iid.ci_high - iid.ci_low
    blk_w = blk.ci_high - blk.ci_low
    assert 0.7 * iid_w < blk_w < 1.4 * iid_w


def test_summary_records_block_length_used():
    rng = np.random.default_rng(0)
    s = summarize(rng.normal(size=200), block_length=12.0, rng=rng)
    assert s.block_length == 12.0
