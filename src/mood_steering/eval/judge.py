"""Juge en aveugle : Claude Opus 5.5 note le ton exprimé par une sortie, bras masqué.

Le juge ne voit que le texte de la sortie (une requête par sortie, indépendantes) et rend
un score d'expression d'humeur dans [-1, 1] (-1 triste, 0 neutre, +1 joyeux), pas un
ressenti. Le bras est masqué par un identifiant opaque ; la table de correspondance reste
côté appelant (``JudgeBatch.mapping``).

Taille du sous-échantillon (question ouverte du SPEC), fixée à ``DEFAULT_SAMPLE_SIZE`` = 200
sorties au total (tous bras confondus, tirage à graine fixe) : ~100 items x ~7 bras
(neutre, prompt joyeux, doses de steering) = ~700 sorties, soit ~30 % ; assez pour comparer
les bras (~25-30 sorties par bras) pour un coût borné. Estimation (heuristique ci-dessous,
sorties de 128 tokens max, ~400 caractères) : ~270 tokens en entrée et 25 en sortie par
requête, soit ~0,0016 $ par sortie et ~0,32 $ pour 200 sorties au prix Opus 5.5
(4 $ / Mtok en entrée, 20 $ / Mtok en sortie). Un jugement des ~700 sorties coûterait
~1,1 $ : écarté, le gain de précision ne justifie pas le coût.
Plafond par défaut de ``run_judge`` : ``DEFAULT_MAX_COST_USD`` = 1 $.

Heuristique de tokens : ``ceil(caractères / 3)`` (français, volontairement pessimiste),
plus ``OVERHEAD_TOKENS`` par requête ; sortie estimée à ``OUTPUT_TOKENS`` par requête.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "claude-opus-5-5"
PRICE_PER_MTOK_USD = {"claude-opus-5-5": (4.0, 20.0)}  # (entrée, sortie)
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_MAX_COST_USD = 1.0
DEFAULT_ENV_PATH = Path("C:/Users/Utilisateur/Documents/GitHub/mood-steering-slm/.env")
OVERHEAD_TOKENS = 120  # consigne système + gabarit de la requête
OUTPUT_TOKENS = 25  # {"score": -0.35} avec marge
MAX_TOKENS = 64

JUDGE_SYSTEM = (
    "Tu évalues le ton exprimé par un texte (expression d'humeur, pas un ressenti). "
    "Donne un score entre -1 (ton très triste) et 1 (ton très joyeux), 0 si neutre. "
    'Réponds uniquement par un objet JSON : {"score": <nombre>}.'
)


class JudgeParseError(ValueError):
    pass


@dataclass(frozen=True)
class JudgeItem:
    blind_id: str
    text: str


@dataclass(frozen=True)
class JudgeBatch:
    items: tuple[JudgeItem, ...]
    # blind_id -> (bras, item_id) ; ne jamais l'envoyer au juge.
    mapping: dict[str, tuple[str, str]] = field(repr=False)


@dataclass(frozen=True)
class JudgeResult:
    scores: dict[str, float]
    failures: dict[str, str]  # blind_id -> raison (réponse invalide)


def prepare_judge_batch(
    records: Sequence[Mapping[str, Any]], sample_size: int, seed: int
) -> JudgeBatch:
    """Tire ``sample_size`` sorties, masque le bras et mélange l'ordre (graine fixe)."""
    if sample_size < 1:
        raise ValueError("sample_size doit être >= 1")
    rng = random.Random(seed)
    chosen = (
        list(records) if sample_size >= len(records) else rng.sample(list(records), sample_size)
    )
    rng.shuffle(chosen)
    items: list[JudgeItem] = []
    mapping: dict[str, tuple[str, str]] = {}
    for rec in chosen:
        blind = f"j{rng.getrandbits(48):012x}"
        while blind in mapping:
            blind = f"j{rng.getrandbits(48):012x}"
        mapping[blind] = (str(rec["arm"]), str(rec["item_id"]))
        items.append(JudgeItem(blind, str(rec["output"])))
    return JudgeBatch(tuple(items), mapping)


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 3)


def estimate_cost(batch: JudgeBatch, model: str = DEFAULT_MODEL) -> float:
    """Coût estimé en dollars, avant tout appel."""
    if model not in PRICE_PER_MTOK_USD:
        raise ValueError(f"prix inconnu pour le modèle {model!r}")
    p_in, p_out = PRICE_PER_MTOK_USD[model]
    tokens_in = sum(estimate_tokens(i.text) + OVERHEAD_TOKENS for i in batch.items)
    tokens_out = OUTPUT_TOKENS * len(batch.items)
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


_JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_judge_response(text: str) -> float:
    """Extrait ``score`` d'un JSON, même entouré de texte ou de balises de code."""
    for match in _JSON_OBJECT.finditer(text):
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or "score" not in data:
            continue
        score = data["score"]
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise JudgeParseError(f"score non numérique : {score!r}")
        if not math.isfinite(score) or not -1.0 <= score <= 1.0:
            raise JudgeParseError(f"score hors de [-1, 1] : {score!r}")
        return float(score)
    raise JudgeParseError("aucun JSON avec « score » dans la réponse")


def resolve_api_key(env_path: Path | None = None) -> str:
    """``ANTHROPIC_API_KEY`` de l'environnement, sinon du ``.env`` ; jamais journalisée."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    if path.is_file():
        from dotenv import dotenv_values  # extra ``eval``

        key = dotenv_values(path).get("ANTHROPIC_API_KEY")
        if key:
            return key
    raise RuntimeError("ANTHROPIC_API_KEY introuvable (environnement ou fichier .env)")


def make_client(env_path: Path | None = None) -> Any:
    from anthropic import Anthropic  # extra ``eval``

    return Anthropic(api_key=resolve_api_key(env_path))


def _response_text(response: Any) -> str:
    return "".join(getattr(b, "text", "") for b in response.content)


def run_judge(
    batch: JudgeBatch,
    client: Any = None,
    model: str = DEFAULT_MODEL,
    max_cost_usd: float = DEFAULT_MAX_COST_USD,
    env_path: Path | None = None,
) -> JudgeResult:
    """Interroge le juge, une requête par sortie ; refuse si le coût estimé dépasse le plafond.

    ``client`` : objet avec ``messages.create(model=, max_tokens=, system=, messages=)``
    (interface du SDK Anthropic). Réponse invalide : consignée dans ``failures``.
    """
    cost = estimate_cost(batch, model)
    if cost > max_cost_usd:
        raise RuntimeError(
            f"coût estimé {cost:.4f} $ > plafond {max_cost_usd:.4f} $ : appel refusé"
        )
    if client is None:
        client = make_client(env_path)
    scores: dict[str, float] = {}
    failures: dict[str, str] = {}
    for item in batch.items:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": f"Texte :\n{item.text}"}],
        )
        try:
            scores[item.blind_id] = parse_judge_response(_response_text(response))
        except JudgeParseError as exc:
            failures[item.blind_id] = str(exc)
    return JudgeResult(scores, failures)


def scores_by_arm(batch: JudgeBatch, scores: Mapping[str, float]) -> dict[str, list[float]]:
    """Rattache les scores du juge à leur bras via la table de correspondance."""
    out: dict[str, list[float]] = defaultdict(list)
    for blind_id, score in scores.items():
        out[batch.mapping[blind_id][0]].append(score)
    return dict(out)
