"""Agrégation par bras : moyenne, IC95 par bootstrap percentile, bootstrap apparié.

Un seul générateur (``random.Random(seed)``) et un seul percentile pour tout le dépôt. Le
critère de « ton déplacé » (dose utile) est unique : l'IC95 du bootstrap apparié de la
différence (bras - neutre) exclut 0 (``excludes_zero``, ``paired_shift``).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Percentile ``q`` dans [0, 1], interpolation linéaire."""
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def interval_from_draws(draws: Sequence[float], point: float, *, level: float = 0.95) -> Interval:
    """IC percentile des tirages valides (nan écartés) autour d'une estimation ponctuelle."""
    v = sorted(x for x in draws if not math.isnan(x))
    if not v:
        return Interval(point, math.nan, math.nan, 0)
    alpha = (1.0 - level) / 2.0
    return Interval(point, percentile(v, alpha), percentile(v, 1.0 - alpha), len(v))


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
    means = [sum(rng.choices(values, k=n)) / n for _ in range(n_boot)]
    iv = interval_from_draws(means, mean, level=level)
    return Interval(mean, iv.low, iv.high, n)


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], *, n_boot: int = 2000, seed: int = 0
) -> Interval:
    """Moyenne des écarts par item ``a_i - b_i`` (séquences alignées) et IC95 percentile."""
    if not a or len(a) != len(b):
        raise ValueError("séquences appariées vides ou de longueurs différentes")
    return bootstrap_ci([x - y for x, y in zip(a, b, strict=True)], n_boot=n_boot, seed=seed)


def paired_bootstrap_stat(
    columns: Mapping[str, Sequence[float]],
    stat: Callable[[Mapping[str, Sequence[float]]], float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> Interval:
    """Valeur de ``stat`` et IC95 percentile ; les items sont rééchantillonnés ensemble.

    ``columns`` : colonnes alignées par item (mêmes indices tirés pour toutes). ``stat`` reçoit
    les colonnes rééchantillonnées (listes) ; ses nan sont écartés de l'IC.
    """
    n = len(next(iter(columns.values())))
    if n == 0 or any(len(c) != n for c in columns.values()):
        raise ValueError("colonnes vides ou de longueurs différentes")
    if n_boot < 1:
        raise ValueError("n_boot doit être >= 1")
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.choices(range(n), k=n)
        draws.append(stat({k: [v[i] for i in idx] for k, v in columns.items()}))
    return interval_from_draws(draws, stat(columns))


def excludes_zero(iv: Interval | None, direction: int = 0) -> bool:
    """L'IC95 exclut 0 : ``direction`` +1 (côté positif), -1 (côté négatif) ou 0 (l'un ou l'autre).

    ``None`` (pas de mesure) et un IC nan ne l'excluent pas.
    """
    if iv is None or math.isnan(iv.low) or math.isnan(iv.high):
        return False
    if direction > 0:
        return iv.low > 0
    if direction < 0:
        return iv.high < 0
    return iv.low > 0 or iv.high < 0


def paired_shift(
    arm: Sequence[float],
    neutral: Sequence[float],
    *,
    direction: int,
    n_boot: int = 2000,
    seed: int = 0,
) -> bool:
    """Critère unique de « ton déplacé » (dose utile) : IC95 apparié de ``bras - neutre``.

    Vrai si l'IC95 du bootstrap apparié de la différence moyenne exclut 0 du côté de
    ``direction`` (signe de la dose).
    """
    diff = paired_bootstrap_diff(arm, neutral, n_boot=n_boot, seed=seed)
    return excludes_zero(diff, direction)


def aggregate(
    scores_by_arm: Mapping[str, Sequence[float]], *, n_boot: int = 2000, seed: int = 0
) -> dict[str, Interval]:
    """Moyenne et IC95 bootstrap par bras ; même graine pour chaque bras."""
    return {
        arm: bootstrap_ci(list(scores), n_boot=n_boot, seed=seed)
        for arm, scores in scores_by_arm.items()
    }


# Quantiles t de Student à 97,5 % (IC95 bilatéral), ddl 1 à 30.
_T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


def t_quantile_975(df: int) -> float:
    """Quantile 97,5 % de la loi de Student à ``df`` ddl (table, Cornish-Fisher au-delà de 30)."""
    if df < 1:
        raise ValueError("ddl >= 1")
    if df <= len(_T975):
        return _T975[df - 1]
    z = 1.959964
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def student_ci(values: Sequence[float]) -> Interval:
    """Moyenne et IC95 de Student (t à n - 1 ddl) : pour quelques répétitions (n petit).

    Le bootstrap percentile sous-couvre sur n < 10 ; ici ``low`` et ``high`` valent nan si
    n < 2 (pas d'intervalle sur une seule valeur).
    """
    if not values:
        raise ValueError("aucune valeur à agréger")
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return Interval(mean, math.nan, math.nan, n)
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    half = t_quantile_975(n - 1) * sd / math.sqrt(n)
    return Interval(mean, mean - half, mean + half, n)
