"""Mesure du ton d'une sortie : expression d'humeur, jamais ressenti.

Deux scores signés dans [-1, 1], calculés par un classifieur injectable (une fonction
``textes -> [{libellé: probabilité}]``) ; le classifieur réel (``transformers``,
extra ``eval``) n'est chargé que par ``load_hf_classifier``.

* Score principal : ``P(joy) - P(sadness)`` de
  ``tabularisai/multilingual-emotion-classification``.
* Contrôle : polarité de ``cmarkea/distilcamembert-base-sentiment`` =
  ``(E[étoiles] - 3) / 2`` avec ``E[étoiles] = sum(k * P(k étoiles))``, k de 1 à 5.

Libellés vérifiés le 2026-11-05 sur le ``config.json`` (``id2label``) de chaque modèle,
sur le Hub Hugging Face.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

EMOTION_MODEL = "tabularisai/multilingual-emotion-classification"
# id2label : anger, contempt, disgust, fear, frustration, gratitude, joy, love,
# neutral, sadness, surprise.
EMOTION_LABELS = (
    "anger",
    "contempt",
    "disgust",
    "fear",
    "frustration",
    "gratitude",
    "joy",
    "love",
    "neutral",
    "sadness",
    "surprise",
)
JOY_LABEL = "joy"
SADNESS_LABEL = "sadness"

SENTIMENT_MODEL = "cmarkea/distilcamembert-base-sentiment"
# id2label : "1 star", "2 stars", ..., "5 stars".
SENTIMENT_LABELS = {"1 star": 1, "2 stars": 2, "3 stars": 3, "4 stars": 4, "5 stars": 5}

Classifier = Callable[[Sequence[str]], Sequence[Mapping[str, float]]]


def _require(probs: Mapping[str, float], labels: Sequence[str]) -> None:
    missing = [label for label in labels if label not in probs]
    if missing:
        raise ValueError(
            f"libellés attendus absents du classifieur : {missing!r} (reçus : {sorted(probs)!r})"
        )


def joy_minus_sadness(probs: Mapping[str, float]) -> float:
    _require(probs, (JOY_LABEL, SADNESS_LABEL))
    return float(probs[JOY_LABEL]) - float(probs[SADNESS_LABEL])


def stars_polarity(probs: Mapping[str, float]) -> float:
    """Espérance des étoiles ramenée à [-1, 1] : ``(E - 3) / 2``."""
    _require(probs, tuple(SENTIMENT_LABELS))
    total = sum(float(probs[label]) for label in SENTIMENT_LABELS)
    if total <= 0:
        raise ValueError("probabilités de sentiment nulles")
    expected = sum(n * float(probs[label]) for label, n in SENTIMENT_LABELS.items()) / total
    return (expected - 3.0) / 2.0


class ToneScorer:
    """Score joie - tristesse et polarité de contrôle ; classifieurs injectables."""

    def __init__(self, emotion: Classifier, sentiment: Classifier | None = None) -> None:
        self._emotion = emotion
        self._sentiment = sentiment

    def score(self, texts: Sequence[str]) -> list[float]:
        return [joy_minus_sadness(p) for p in self._call(self._emotion, texts)]

    def control(self, texts: Sequence[str]) -> list[float]:
        if self._sentiment is None:
            raise ValueError("pas de classifieur de contrôle (sentiment) fourni")
        return [stars_polarity(p) for p in self._call(self._sentiment, texts)]

    @staticmethod
    def _call(clf: Classifier, texts: Sequence[str]) -> Sequence[Mapping[str, float]]:
        out = clf(list(texts))
        if len(out) != len(texts):
            raise ValueError(f"{len(out)} résultats pour {len(texts)} textes")
        return out

    @classmethod
    def from_huggingface(cls) -> ToneScorer:
        return cls(load_hf_classifier(EMOTION_MODEL), load_hf_classifier(SENTIMENT_MODEL))


def load_hf_classifier(model_id: str, *, batch_size: int = 16) -> Classifier:
    """Classifieur ``transformers`` (CPU) ; import paresseux (extra ``eval``)."""
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - dépend de l'extra
        raise ImportError("extra requis : uv sync --extra eval") from exc
    pipe = pipeline("text-classification", model=model_id, top_k=None, truncation=True)

    def classify(texts: Sequence[str]) -> list[dict[str, float]]:
        results = pipe(list(texts), batch_size=batch_size)
        return [{r["label"]: float(r["score"]) for r in row} for row in results]

    return classify
