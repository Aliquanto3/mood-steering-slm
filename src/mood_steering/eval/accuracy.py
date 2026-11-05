"""Exactitude sur QCM : item, prompt, extraction de la lettre et score.

Un item dont la lettre n'est pas extraite compte faux et porte ``not_extracted``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mood_steering.eval.runner import Item

LETTERS = ("A", "B", "C", "D")

ANSWER_INSTRUCTION = (
    "Réponds uniquement par la lettre de la bonne réponse (A, B, C ou D) : "
    "une seule lettre, sans explication."
)


@dataclass(frozen=True)
class McqItem:
    id: str
    question: str
    choices: dict[str, str]
    answer: str
    subject: str
    lang: str

    def __post_init__(self) -> None:
        if tuple(self.choices) != LETTERS:
            raise ValueError(f"choix A à D attendus, dans l'ordre : {tuple(self.choices)!r}")
        if self.answer not in LETTERS:
            raise ValueError(f"réponse attendue parmi {LETTERS} : {self.answer!r}")


def format_prompt(item: McqItem) -> str:
    """Prompt français : question, quatre choix, consigne « une seule lettre »."""
    lines = [f"Question ({item.subject}) :", item.question.strip(), ""]
    lines += [f"{letter}. {item.choices[letter].strip()}" for letter in LETTERS]
    lines += ["", ANSWER_INSTRUCTION]
    return "\n".join(lines)


def mcq_to_item(item: McqItem) -> Item:
    """Conversion vers ``runner.Item`` ; la bonne lettre voyage dans ``meta``."""
    return Item(
        id=item.id,
        prompt=format_prompt(item),
        meta={"answer": item.answer, "subject": item.subject, "lang": item.lang},
    )


# --- Extraction -------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_MARKDOWN = re.compile(r"[*_`#>]+")
# « Réponse : B », « la bonne réponse est la lettre B », « answer is (B) »…
_EXPLICIT = re.compile(
    r"(?i:(?:bonne\s+|juste\s+|vraie\s+)?r[ée]ponse(?:\s+(?:finale|correcte|juste|choisie))?"
    r"|answer|lettre|option|choix)"
    r"[\s:=\-]*(?:(?i:est|serait|is|sera)[\s:=\-]*)?"
    r"(?:(?i:la\s+lettre|l'option|le\s+choix|option|lettre)\s*)?"
    r"\(?([A-D])(?![A-Za-zÀ-ÿ0-9])"
    # « option A ou B » : alternative, pas une réponse.
    r"(?!\)?\s*(?i:ou|or|et|and|/)\s*\(?[A-D](?![A-Za-zÀ-ÿ0-9]))"
)
_PAREN_LOWER = re.compile(r"^\s*\(([a-d])\)\s*[.!]?\s*$")
_ALONE = re.compile(r"^\s*\(?([A-Da-d])\)?\s*[.!:)]*\s*$")
# Lettre en tête suivie d'un séparateur. « A » suivi d'un mot en minuscule est la
# préposition « à » (« A partir de… », « A mon avis »), sauf « A est… » ; B à D non.
_LEADING = re.compile(
    r"^\(?(?:(A)(?:[.):,\]]|\s*$|\s+(?![a-zà-ÿ])|\s+(?=(?:est|serait|sera|is)\b))"
    r"|([B-D])(?:[.):,\]]|\s*$|\s))"
)
# « A ou B », « (B) et (C) » : alternative, ambiguë.
_ALTERNATIVE = re.compile(
    r"(?<![A-Za-zÀ-ÿ0-9])\(?[A-D]\)?\s*(?i:ou|or|et|and|/)\s*\(?[A-D](?![A-Za-zÀ-ÿ0-9])"
)
# Lettre isolée ; « A » suivi d'un mot en minuscule est la préposition « à » sans accent.
_STANDALONE = re.compile(
    r"(?<![A-Za-zÀ-ÿ0-9])\(?(?:(A)(?!\s+[a-zà-ÿ])|([B-D]))\)?(?![A-Za-zÀ-ÿ0-9])"
)
_SHORT = 40


def extract_letter(output: str) -> str | None:
    """Lettre A-D de la réponse, ou ``None`` si absente ou ambiguë.

    Étapes : retrait des blocs ``<think>`` et du gras Markdown ; réponse nue (« B »,
    « (b) ») ; formules explicites (« Réponse : B ») ; sortie courte à une seule lettre
    majuscule candidate ; lettre en tête (« B. Paris »). Plusieurs lettres distinctes
    candidates au même niveau, ou « A ou B » en tête : ambigu, ``None``. Un bloc
    ``<think>`` jamais refermé (raisonnement tronqué) donne ``None`` ; une ``</think>``
    orpheline (``<think>`` pré-ouvert par le gabarit) : seul le texte qui suit compte.
    Une majuscule suivie d'un mot (« A partir de… ») ou une alternative (« option A ou
    B ») n'est pas une réponse.
    """
    if not output:
        return None
    text = _THINK_BLOCK.sub(" ", output)
    # Gabarit qui pré-ouvre ``<think>`` (Qwen3.5) : seule la fin ``</think>`` est émise.
    closing = text.lower().rfind("</think>")
    if closing >= 0:
        text = text[closing + len("</think>") :]
    if re.search(r"<think>", text, re.IGNORECASE):
        return None
    text = _MARKDOWN.sub("", text).strip()
    if not text:
        return None
    m = _ALONE.match(text) or _PAREN_LOWER.match(text)
    if m:
        return m.group(1).upper()
    explicit = {m.group(1) for m in _EXPLICIT.finditer(text)}
    if len(explicit) > 1:
        return None
    if explicit:
        return next(iter(explicit))
    if _ALTERNATIVE.search(text):
        return None
    if len(text) <= _SHORT:
        found = {m.group(1) or m.group(2) for m in _STANDALONE.finditer(text)}
        return next(iter(found)) if len(found) == 1 else None
    lead = _LEADING.match(text)
    return (lead.group(1) or lead.group(2)) if lead else None


# --- Score ------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemScore:
    item_id: str
    letter: str | None
    correct: bool
    not_extracted: bool


@dataclass(frozen=True)
class AccuracyResult:
    accuracy: float
    n: int
    n_correct: int
    n_not_extracted: int
    items: tuple[ItemScore, ...]

    @property
    def scores(self) -> list[float]:
        """1.0 / 0.0 par item, pour ``stats.aggregate`` (IC95)."""
        return [1.0 if s.correct else 0.0 for s in self.items]


def _index_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for rec in records:
        item_id = rec["item_id"]
        if item_id in by_id:
            raise ValueError(f"item en double dans les enregistrements : {item_id!r}")
        by_id[item_id] = rec
    return by_id


def score_accuracy(
    records: Sequence[Mapping[str, Any]], items: Sequence[McqItem]
) -> AccuracyResult:
    """Exactitude d'un bras : ``records`` = lignes du runner (``item_id``, ``output``).

    Un item sans lettre extraite compte faux (``not_extracted``). Un enregistrement
    manquant ou inconnu lève ``ValueError`` : un trou ne doit pas fausser le taux.
    """
    by_id = _index_records(records)
    known = {i.id for i in items}
    unknown = sorted(set(by_id) - known)
    if unknown:
        raise ValueError(f"enregistrements sans item : {unknown!r}")
    scores: list[ItemScore] = []
    for item in items:
        if item.id not in by_id:
            raise ValueError(f"enregistrement manquant pour l'item {item.id!r}")
        letter = extract_letter(by_id[item.id]["output"])
        scores.append(ItemScore(item.id, letter, letter == item.answer, letter is None))
    if not scores:
        raise ValueError("aucun item à évaluer")
    n_correct = sum(s.correct for s in scores)
    return AccuracyResult(
        accuracy=n_correct / len(scores),
        n=len(scores),
        n_correct=n_correct,
        n_not_extracted=sum(s.not_extracted for s in scores),
        items=tuple(scores),
    )
