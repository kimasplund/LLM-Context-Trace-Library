"""LCTL Statistical Tests - Statistical significance testing for A/B comparisons."""

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StatisticalResult:
    """Result of a statistical significance test."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size_a: int = 0
    sample_size_b: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "significant": self.significant,
            "alpha": self.alpha,
            "sample_size_a": self.sample_size_a,
            "sample_size_b": self.sample_size_b,
        }
        if self.effect_size is not None:
            result["effect_size"] = self.effect_size
        if self.confidence_interval is not None:
            result["confidence_interval"] = {
                "lower": self.confidence_interval[0],
                "upper": self.confidence_interval[1],
            }
        return result


def _mean(data: List[float]) -> float:
    """Calculate mean of data."""
    if not data:
        return 0.0
    return sum(data) / len(data)


def _variance(data: List[float], ddof: int = 1) -> float:
    """Calculate variance of data with degrees of freedom adjustment."""
    if len(data) <= ddof:
        return 0.0
    m = _mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - ddof)


def _std(data: List[float], ddof: int = 1) -> float:
    """Calculate standard deviation of data."""
    return math.sqrt(_variance(data, ddof))


def _t_cdf(t: float, df: int) -> float:
    """Approximate Student's t CDF using numerical integration.

    Uses the regularized incomplete beta function approximation.
    """
    if df <= 0:
        return 0.5

    x = df / (df + t * t)

    if t < 0:
        return 0.5 * _incomplete_beta(df / 2, 0.5, x)
    else:
        return 1 - 0.5 * _incomplete_beta(df / 2, 0.5, x)


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Approximation of the regularized incomplete beta function.

    Uses continued fraction expansion for numerical stability.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1 - x)
    )

    if x < (a + 1) / (a + b + 2):
        return bt * _beta_cf(a, b, x) / a
    else:
        return 1 - bt * _beta_cf(b, a, 1 - x) / b


def _beta_cf(a: float, b: float, x: float) -> float:
    """Continued fraction for incomplete beta function."""
    max_iter = 100
    eps = 1e-10
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1 - qab * x / qap

    if abs(d) < eps:
        d = eps
    d = 1 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1 / d
        delta = d * c
        h *= delta

        if abs(delta - 1) < eps:
            break

    return h


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using error function approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def welch_t_test(
    sample_a: List[float],
    sample_b: List[float],
    alpha: float = 0.05,
) -> StatisticalResult:
    """Perform Welch's t-test for independent samples with unequal variances.

    This is more robust than Student's t-test when variances are unequal.

    Args:
        sample_a: First sample data.
        sample_b: Second sample data.
        alpha: Significance level (default 0.05).

    Returns:
        StatisticalResult with test statistic and p-value.

    Raises:
        ValueError: If either sample has fewer than 2 observations.
    """
    n_a = len(sample_a)
    n_b = len(sample_b)

    if n_a < 2:
        raise ValueError(f"Sample A must have at least 2 observations, got {n_a}")
    if n_b < 2:
        raise ValueError(f"Sample B must have at least 2 observations, got {n_b}")

    mean_a = _mean(sample_a)
    mean_b = _mean(sample_b)
    var_a = _variance(sample_a)
    var_b = _variance(sample_b)

    se_a = var_a / n_a
    se_b = var_b / n_b
    se_diff = math.sqrt(se_a + se_b)

    if se_diff == 0:
        t_stat = 0.0
        df = n_a + n_b - 2
    else:
        t_stat = (mean_a - mean_b) / se_diff

        df_num = (se_a + se_b) ** 2
        df_denom = (se_a ** 2) / (n_a - 1) + (se_b ** 2) / (n_b - 1)
        df = df_num / df_denom if df_denom > 0 else n_a + n_b - 2

    p_value = 2 * (1 - _t_cdf(abs(t_stat), int(df)))
    p_value = max(0.0, min(1.0, p_value))

    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

    return StatisticalResult(
        test_name="welch_t_test",
        statistic=t_stat,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=effect_size,
        sample_size_a=n_a,
        sample_size_b=n_b,
    )


def mann_whitney_u(
    sample_a: List[float],
    sample_b: List[float],
    alpha: float = 0.05,
) -> StatisticalResult:
    """Perform Mann-Whitney U test (non-parametric alternative to t-test).

    This test does not assume normal distribution and is robust to outliers.

    Args:
        sample_a: First sample data.
        sample_b: Second sample data.
        alpha: Significance level (default 0.05).

    Returns:
        StatisticalResult with U statistic and p-value.

    Raises:
        ValueError: If either sample is empty.
    """
    n_a = len(sample_a)
    n_b = len(sample_b)

    if n_a == 0:
        raise ValueError("Sample A must not be empty")
    if n_b == 0:
        raise ValueError("Sample B must not be empty")

    combined = [(v, "a", i) for i, v in enumerate(sample_a)]
    combined.extend((v, "b", i) for i, v in enumerate(sample_b))
    combined.sort(key=lambda x: x[0])

    ranks: Dict[Tuple[str, int], float] = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1

        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            _, group, idx = combined[k]
            ranks[(group, idx)] = avg_rank
        i = j

    r_a = sum(ranks[("a", i)] for i in range(n_a))

    u_a = r_a - n_a * (n_a + 1) / 2
    u_b = n_a * n_b - u_a
    u_stat = min(u_a, u_b)

    mean_u = n_a * n_b / 2
    std_u = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)

    if std_u > 0:
        z_stat = (u_stat - mean_u) / std_u
        p_value = 2 * _normal_cdf(z_stat)
    else:
        z_stat = 0.0
        p_value = 1.0

    p_value = max(0.0, min(1.0, p_value))

    effect_size = 1 - (2 * u_stat) / (n_a * n_b)

    return StatisticalResult(
        test_name="mann_whitney_u",
        statistic=u_stat,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=effect_size,
        sample_size_a=n_a,
        sample_size_b=n_b,
    )


def bootstrap_confidence_interval(
    sample_a: List[float],
    sample_b: List[float],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> StatisticalResult:
    """Compute bootstrap confidence interval for difference in means.

    Uses resampling to estimate the sampling distribution of the difference.

    Args:
        sample_a: First sample data.
        sample_b: Second sample data.
        n_bootstrap: Number of bootstrap iterations (default 10000).
        confidence_level: Confidence level (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        StatisticalResult with confidence interval.

    Raises:
        ValueError: If either sample is empty.
    """
    if not sample_a:
        raise ValueError("Sample A must not be empty")
    if not sample_b:
        raise ValueError("Sample B must not be empty")

    rng = random.Random(seed)

    observed_diff = _mean(sample_a) - _mean(sample_b)

    bootstrap_diffs: List[float] = []
    for _ in range(n_bootstrap):
        boot_a = [rng.choice(sample_a) for _ in range(len(sample_a))]
        boot_b = [rng.choice(sample_b) for _ in range(len(sample_b))]
        bootstrap_diffs.append(_mean(boot_a) - _mean(boot_b))

    bootstrap_diffs.sort()

    alpha = 1 - confidence_level
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    lower_idx = max(0, min(lower_idx, n_bootstrap - 1))
    upper_idx = max(0, min(upper_idx, n_bootstrap - 1))

    ci_lower = bootstrap_diffs[lower_idx]
    ci_upper = bootstrap_diffs[upper_idx]

    significant = not (ci_lower <= 0 <= ci_upper)

    boot_std = _std(bootstrap_diffs, ddof=0)
    z_stat = observed_diff / boot_std if boot_std > 0 else 0.0
    p_value = 2 * (1 - _normal_cdf(abs(z_stat)))
    p_value = max(0.0, min(1.0, p_value))

    return StatisticalResult(
        test_name="bootstrap_ci",
        statistic=observed_diff,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        confidence_interval=(ci_lower, ci_upper),
        sample_size_a=len(sample_a),
        sample_size_b=len(sample_b),
    )


def run_all_tests(
    sample_a: List[float],
    sample_b: List[float],
    alpha: float = 0.05,
    bootstrap_seed: Optional[int] = None,
) -> Dict[str, StatisticalResult]:
    """Run all statistical tests on two samples.

    Args:
        sample_a: First sample data.
        sample_b: Second sample data.
        alpha: Significance level (default 0.05).
        bootstrap_seed: Random seed for bootstrap (optional).

    Returns:
        Dictionary mapping test names to StatisticalResult instances.
    """
    results: Dict[str, StatisticalResult] = {}

    if len(sample_a) >= 2 and len(sample_b) >= 2:
        results["t_test"] = welch_t_test(sample_a, sample_b, alpha)

    if sample_a and sample_b:
        results["mann_whitney_u"] = mann_whitney_u(sample_a, sample_b, alpha)
        results["bootstrap_ci"] = bootstrap_confidence_interval(
            sample_a, sample_b, seed=bootstrap_seed
        )

    return results


def is_significant(results: Dict[str, StatisticalResult]) -> bool:
    """Check if any test shows statistical significance.

    Args:
        results: Dictionary of test results from run_all_tests.

    Returns:
        True if at least one test shows significance.
    """
    return any(r.significant for r in results.values())


def consensus_significant(
    results: Dict[str, StatisticalResult], threshold: float = 0.5
) -> bool:
    """Check if a threshold of tests agree on significance.

    Args:
        results: Dictionary of test results from run_all_tests.
        threshold: Fraction of tests that must agree (default 0.5).

    Returns:
        True if threshold fraction of tests show significance.
    """
    if not results:
        return False

    significant_count = sum(1 for r in results.values() if r.significant)
    return significant_count / len(results) >= threshold
