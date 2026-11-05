"""Chargeur des QCM français (extra optionnel ``eval`` : huggingface_hub, pyarrow).

Source : ``CohereLabs/Global-MMLU-Lite`` (config ``fr``, Apache 2.0), colonne
``cultural_sensitivity_label`` : seules les questions ``CA`` (culturellement neutres,
« culturally agnostic ») sont gardées. Repli : ``openai/MMMLU`` (``FR_FR``, MIT), sans
étiquette de neutralité culturelle. Fichiers téléchargés sous ``cache_dir`` (``data/``,
jamais versionné). Imports lourds paresseux ; aucun test ne télécharge.
"""

from __future__ import annotations

import csv
import random
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mood_steering.eval.accuracy import LETTERS, McqItem

GLOBAL_MMLU_REPO = "CohereLabs/Global-MMLU-Lite"
GLOBAL_MMLU_FILE = "fr/test-00000-of-00001.parquet"
# Révisions épinglées (commits du Hub, 2026-11-05) : un jeu qui change ne change pas le banc.
GLOBAL_MMLU_REVISION = "36c2fd756f19ccf13a9a96c8e53ccecc02192b8b"
CULTURE_COLUMN = "cultural_sensitivity_label"
CULTURE_NEUTRAL = "CA"
MMMLU_REPO = "openai/MMMLU"
MMMLU_FILE = "test/mmlu_FR-FR.csv"
MMMLU_REVISION = "325a01dc3e173cac1578df94120499aaca2e2504"
DEFAULT_CACHE_DIR = Path("data") / "datasets"


def global_mmlu_rows_to_items(rows: Sequence[Mapping[str, Any]]) -> list[McqItem]:
    """Lignes Global-MMLU-Lite -> items, questions culturellement neutres seulement."""
    return [
        McqItem(
            id=f"gmmlu-fr:{row['sample_id']}",
            question=row["question"],
            choices={letter: row[f"option_{letter.lower()}"] for letter in LETTERS},
            answer=str(row["answer"]).strip().upper(),
            subject=row["subject"],
            lang="fr",
        )
        for row in rows
        if row[CULTURE_COLUMN] == CULTURE_NEUTRAL
    ]


def mmmlu_rows_to_items(rows: Sequence[Mapping[str, Any]]) -> list[McqItem]:
    """Lignes MMMLU FR_FR -> items (aucune étiquette de neutralité culturelle)."""
    return [
        McqItem(
            id=f"mmmlu-fr:{row['Unnamed: 0']}",
            question=row["Question"],
            choices={letter: row[letter] for letter in LETTERS},
            answer=str(row["Answer"]).strip().upper(),
            subject=row["Subject"],
            lang="fr",
        )
        for row in rows
    ]


def draw_items(pool: Sequence[McqItem], n: int, seed: int) -> list[McqItem]:
    """Tire ``n`` items sans remise, à graine fixe ; ordre trié par id (stable)."""
    if n > len(pool):
        raise ValueError(f"{n} items demandés, {len(pool)} disponibles")
    ordered = sorted(pool, key=lambda i: i.id)
    return sorted(random.Random(seed).sample(ordered, n), key=lambda i: i.id)


def _fetch_global_mmlu(cache_dir: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download  # extra ``eval``

    path = hf_hub_download(
        GLOBAL_MMLU_REPO,
        GLOBAL_MMLU_FILE,
        repo_type="dataset",
        revision=GLOBAL_MMLU_REVISION,
        local_dir=cache_dir / "gmmlu",
    )
    return pq.read_table(path).to_pylist()


def _fetch_mmmlu(cache_dir: Path) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download  # extra ``eval``

    path = hf_hub_download(
        MMMLU_REPO,
        MMMLU_FILE,
        repo_type="dataset",
        revision=MMMLU_REVISION,
        local_dir=cache_dir / "mmmlu",
    )
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_global_mmlu_lite(
    n: int = 100,
    seed: int = 0,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    fetch_primary: Callable[[Path], list[dict[str, Any]]] = _fetch_global_mmlu,
    fetch_fallback: Callable[[Path], list[dict[str, Any]]] = _fetch_mmmlu,
) -> list[McqItem]:
    """``n`` items tirés à graine fixe de Global-MMLU-Lite fr (neutres), repli MMMLU FR_FR.

    Le repli joue si le téléchargement ou la lecture du jeu principal échoue (réseau,
    dépendance absente, colonne manquante) ou s'il reste moins de ``n`` items ; il émet
    un ``UserWarning``. La source se lit dans le préfixe des ids (``gmmlu-fr:`` ou
    ``mmmlu-fr:``). Un jeu principal lu mais sans aucune question ``CA`` lève
    ``RuntimeError`` : c'est un changement de schéma, pas un incident à masquer.
    """
    cache_dir = Path(cache_dir)
    try:
        rows = fetch_primary(cache_dir)
        pool = global_mmlu_rows_to_items(rows)
    except Exception as exc:  # noqa: BLE001 - tout échec du jeu principal déclenche le repli
        return _fallback(n, seed, cache_dir, fetch_fallback, exc)
    if rows and not pool:
        raise RuntimeError(
            f"{GLOBAL_MMLU_REPO} : aucune ligne {CULTURE_COLUMN} == {CULTURE_NEUTRAL!r}"
        )
    if len(pool) < n:
        exc = ValueError(f"{len(pool)} questions culturellement neutres, {n} demandées")
        return _fallback(n, seed, cache_dir, fetch_fallback, exc)
    return draw_items(pool, n, seed)


def _fallback(
    n: int,
    seed: int,
    cache_dir: Path,
    fetch_fallback: Callable[[Path], list[dict[str, Any]]],
    exc: Exception,
) -> list[McqItem]:
    warnings.warn(
        f"{GLOBAL_MMLU_REPO} indisponible ({exc!r}) : repli sur {MMMLU_REPO}", stacklevel=3
    )
    try:
        pool = mmmlu_rows_to_items(fetch_fallback(cache_dir))
    except Exception as exc2:
        raise RuntimeError(
            f"Global-MMLU-Lite ({exc!r}) et MMMLU ({exc2!r}) indisponibles"
        ) from exc2
    return draw_items(pool, n, seed)
