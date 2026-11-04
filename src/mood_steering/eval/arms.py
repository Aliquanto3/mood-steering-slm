"""Bras du banc : neutre, prompt joyeux, steering à plusieurs doses."""

from __future__ import annotations

import re
from dataclasses import dataclass

KIND_NEUTRAL = "neutral"
KIND_PROMPT = "prompt"
KIND_STEERING = "steering"
KINDS = (KIND_NEUTRAL, KIND_PROMPT, KIND_STEERING)

DEFAULT_DOSES = (0.25, 0.5, 0.75, 1.0, -0.5)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.]+$")


@dataclass(frozen=True)
class Arm:
    """Un bras du banc. ``name`` sert de nom de fichier : lettres, chiffres, ``_`` et ``.``."""

    name: str
    kind: str
    dose: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind inconnu : {self.kind!r} (attendu {KINDS})")
        if not _SAFE_NAME.fullmatch(self.name) or self.name.endswith("."):
            raise ValueError(f"nom de bras non sûr pour un fichier : {self.name!r}")
        if self.kind != KIND_STEERING and self.dose != 0:
            raise ValueError(f"dose non nulle hors steering : {self.dose!r}")
        if self.kind == KIND_STEERING and not -1.0 <= self.dose <= 1.0:
            raise ValueError(f"dose hors de [-1, 1] : {self.dose!r}")


def steering_arm_name(dose: float) -> str:
    """``steering_p0.50`` / ``steering_m0.50`` (``p``/``m`` plutôt que ``+``/``-``)."""
    return f"steering_{'m' if dose < 0 else 'p'}{abs(dose):.2f}"


def default_arms(doses: tuple[float, ...] = DEFAULT_DOSES) -> list[Arm]:
    """Neutre, prompt joyeux, puis un bras steering par dose (au moins une négative)."""
    if not any(d < 0 for d in doses) or not any(d > 0 for d in doses):
        raise ValueError("il faut au moins une dose positive et une dose négative")
    arms = [Arm("neutral", KIND_NEUTRAL), Arm("prompt_joyful", KIND_PROMPT)]
    arms += [Arm(steering_arm_name(d), KIND_STEERING, float(d)) for d in doses if d != 0]
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"doses en double : {doses!r}")
    return arms
