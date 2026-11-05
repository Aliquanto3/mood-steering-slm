"""Profils de dose par token (CAP-3) : la dose du vecteur varie au fil de la génération.

Un profil est un objet appelable ``dose_at(t) -> float`` (``t`` = index du token généré,
à partir de 0), borné dans [-1, 1]. Le module ne dépend ni du moteur ni de numpy.

Calendrier (``Steerer.open_stream``) : la dose ``schedule(t)`` est posée juste avant
l'évaluation dont les logits donnent le token ``t`` ; elle agit donc sur ce token même, et
la dose du token 0 s'applique aussi à l'évaluation du prompt (un profil constant équivaut
à ``--control-vector`` en CLI). Le pic 1.2 (``docs/pics/cvec-entre-tokens.md``) évaluait le
prompt sans vecteur et indexait la dose par le token suivant (« posée avant le token N,
agit sur N+1 ») : l'indice d'un profil ici est décalé d'un cran par rapport à ce protocole.
Les doses sont validées dans [-1, 1] sans tolérance (avant l'arrondi à 10^-3 du Steerer).
Génération gloutonne : la graine (``seed``) est sans effet sur le texte.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

# Un profil se manipule comme un appelable ; ``DoseSchedule`` en fixe la signature.


class DoseSchedule(Protocol):
    """Profil de dose : ``schedule(t)`` rend la dose du token ``t`` (t >= 0), dans [-1, 1]."""

    def __call__(self, t: int) -> float: ...


def _check_dose(name: str, value: float) -> None:
    if not -1.0 <= value <= 1.0:  # rejette aussi NaN
        raise ValueError(f"{name} hors de [-1, 1] : {value!r}")


def constant(dose: float) -> DoseSchedule:
    """Profil constant : la dose vaut ``dose`` à tout token."""
    _check_dose("dose", dose)

    def schedule(t: int) -> float:
        return dose

    return schedule


def pulse(rise: float, peak: float, half_life: float, *, start: float = 0.0) -> DoseSchedule:
    """Impulsion : montée linéaire de ``start`` à ``peak``, puis décroissance exponentielle.

    - tokens ``t < rise`` : ``start + (peak - start) * t / rise`` (linéaire) ;
    - ``t = rise`` : ``peak`` (sommet atteint exactement) ;
    - ``t > rise`` : ``peak * 0.5 ** ((t - rise) / half_life)`` (demi-vie en tokens, vers 0).

    ``rise = 0`` : le sommet est atteint dès le token 0. Exige ``rise >= 0``,
    ``half_life > 0``, ``peak`` et ``start`` dans [-1, 1].
    """
    if not (math.isfinite(rise) and rise >= 0):
        raise ValueError(f"rise doit être fini et >= 0 : {rise!r}")
    if not (math.isfinite(half_life) and half_life > 0):
        raise ValueError(f"half_life doit être fini et > 0 : {half_life!r}")
    _check_dose("peak", peak)
    _check_dose("start", start)

    def schedule(t: int) -> float:
        if t < 0:
            raise ValueError(f"index de token négatif : {t!r}")
        if t < rise:
            return start + (peak - start) * t / rise
        return peak * 0.5 ** ((t - rise) / half_life)

    return schedule


def sample(schedule: Callable[[int], float], n: int) -> list[float]:
    """Les doses des ``n`` premiers tokens (``t = 0..n-1``), pour l'affichage."""
    if n < 0:
        raise ValueError(f"n doit être >= 0 : {n!r}")
    return [float(schedule(t)) for t in range(n)]
