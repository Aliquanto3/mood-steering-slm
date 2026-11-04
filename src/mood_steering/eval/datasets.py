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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from mood_steering.eval.accuracy import LETTERS, McqItem

GLOBAL_MMLU_REPO = "CohereLabs/Global-MMLU-Lite"
GLOBAL_MMLU_FILE = "fr/test-00000-of-00001.parquet"
CULTURE_COLUMN = "cultural_sensitivity_label"
CULTURE_NEUTRAL = "CA"
MMMLU_REPO = "openai/MMMLU"
MMMLU_FILE = "test/mmlu_FR-FR.csv"
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
        GLOBAL_MMLU_REPO, GLOBAL_MMLU_FILE, repo_type="dataset", local_dir=cache_dir / "gmmlu"
    )
    return pq.read_table(path).to_pylist()


def _fetch_mmmlu(cache_dir: Path) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download  # extra ``eval``

    path = hf_hub_download(
        MMMLU_REPO, MMMLU_FILE, repo_type="dataset", local_dir=cache_dir / "mmmlu"
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
    dépendance absente, colonne manquante) ou s'il reste moins de ``n`` items.
    """
    cache_dir = Path(cache_dir)
    try:
        pool = global_mmlu_rows_to_items(fetch_primary(cache_dir))
        if len(pool) < n:
            raise ValueError("pas assez de questions culturellement neutres")
    except Exception as exc:  # noqa: BLE001 - tout échec du jeu principal déclenche le repli
        try:
            pool = mmmlu_rows_to_items(fetch_fallback(cache_dir))
        except Exception as exc2:
            raise RuntimeError(
                f"Global-MMLU-Lite ({exc!r}) et MMMLU ({exc2!r}) indisponibles"
            ) from exc2
    return draw_items(pool, n, seed)
