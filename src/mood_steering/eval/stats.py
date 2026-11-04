"""Agrégation par bras : moyenne et IC95 par bootstrap percentile."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Percentile ``q`` dans [0, 1], interpolation linéaire."""
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> Interval:
    """Moyenne et intervalle percentile du bootstrap des moyennes (déterministe)."""
    if not values:
        raise ValueError("aucune valeur à agréger")
    if n_boot < 1:
        raise ValueError("n_boot doit être >= 1")
    n = len(values)
    mean = sum(values) / n
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    alpha = (1.0 - level) / 2.0
    # La moyenne observée borne l'intervalle : il la contient toujours.
    low = min(_percentile(means, alpha), mean)
    high = max(_percentile(means, 1.0 - alpha), mean)
    return Interval(mean=mean, low=low, high=high, n=n)


def aggregate(
    scores_by_arm: Mapping[str, Sequence[float]], *, n_boot: int = 2000, seed: int = 0
) -> dict[str, Interval]:
    """Moyenne et IC95 bootstrap par bras ; même graine pour chaque bras."""
    return {
        arm: bootstrap_ci(list(scores), n_boot=n_boot, seed=seed)
        for arm, scores in scores_by_arm.items()
    }
