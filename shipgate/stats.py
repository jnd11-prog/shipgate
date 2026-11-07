"""Small, dependency-free statistics for eval results.

Eval suites are small (tens to hundreds of cases), so point estimates like "92% pass rate"
hide a lot of uncertainty. These helpers quantify it:

- wilson_interval: confidence interval for a pass rate (well-behaved near 0 and 1 and at small n,
  unlike the normal approximation).
- mcnemar_exact: paired significance test for "did version B change outcomes vs version A on the
  same cases?" Only discordant cases carry information.
- cases_needed: how many all-passing cases a suite needs before it can certify a pass-rate bar.
- cohen_kappa: chance-corrected agreement between an LLM judge and human labels.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

# Two-sided z-scores for the confidence levels we expose.
_Z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


def z_for(confidence: float) -> float:
    if confidence not in _Z:
        raise ValueError(f"confidence must be one of {sorted(_Z)}")
    return _Z[confidence]


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    if not 0 <= successes <= n:
        raise ValueError("successes must be in [0, n]")
    z = z_for(confidence)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # At the boundaries the exact bound is 0 or 1; pin it so rounding can't push it past p.
    lo = 0.0 if successes == 0 else max(0.0, center - half)
    hi = 1.0 if successes == n else min(1.0, center + half)
    return (lo, hi)


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value.

    b = cases that passed before and fail now (regressions)
    c = cases that failed before and pass now (fixes)
    Under H0 (no real change), each discordant case is a fair coin flip.
    """
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * _binom_cdf(min(b, c), n))


def cases_needed(target: float, confidence: float = 0.95, limit: int = 100_000) -> int:
    """Smallest n such that n/n passing cases puts the Wilson lower bound at or above `target`."""
    if not 0 < target < 1:
        raise ValueError("target must be in (0, 1)")
    lo, hi = 1, 1
    while wilson_interval(hi, hi, confidence)[0] < target:
        lo, hi = hi, hi * 2
        if hi > limit:
            raise ValueError("target not reachable within limit")
    while lo < hi:  # lower bound is monotone in n when all cases pass
        mid = (lo + hi) // 2
        if wilson_interval(mid, mid, confidence)[0] >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def cohen_kappa(pairs: Sequence[Tuple[bool, bool]]) -> Dict[str, float]:
    """Agreement stats for (human, judge) boolean label pairs."""
    n = len(pairs)
    if n == 0:
        raise ValueError("no labels")
    tp = sum(1 for h, j in pairs if h and j)
    tn = sum(1 for h, j in pairs if not h and not j)
    fp = sum(1 for h, j in pairs if not h and j)  # judge passed something a human failed
    fn = sum(1 for h, j in pairs if h and not j)
    po = (tp + tn) / n
    h_yes, j_yes = (tp + fn) / n, (tp + fp) / n
    pe = h_yes * j_yes + (1 - h_yes) * (1 - j_yes)
    kappa = 1.0 if pe == 1 else (po - pe) / (1 - pe)
    return {
        "n": n,
        "agreement": po,
        "kappa": kappa,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        # Of the replies a human would fail, how many does the judge catch? This is the number that
        # matters for a release gate: a lenient judge silently ships bad outputs.
        "fail_recall": tn / (tn + fp) if (tn + fp) else 1.0,
    }
