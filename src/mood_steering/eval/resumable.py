"""JSONL reprenable : lecture tolérante à un arrêt brutal, réécriture atomique.

Une sortie de modèle peut contenir U+2028, U+2029 ou U+0085, que ``json.dumps(ensure_ascii=
False)`` n'échappe pas : on coupe donc sur le saut de ligne seul, jamais avec
``splitlines()``. Un arrêt
au milieu d'une ligne laisse une dernière ligne tronquée (voire un caractère UTF-8 coupé) : elle
est ignorée, et la lecture s'arrête à la première ligne illisible.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

Row = dict[str, Any]


def read_rows(path: Path, *, strict: bool = False) -> list[Row]:
    """Lignes complètes d'un JSONL ; ``[]`` si le fichier n'existe pas.

    La première ligne illisible (tronquée par un arrêt brutal) et les suivantes sont ignorées ;
    avec ``strict`` (fichier terminé, écrit d'un bloc), elle lève ``ValueError``.
    """
    path = Path(path)
    if not path.is_file():
        return []
    rows: list[Row] = []
    for line in path.read_bytes().split(b"\n"):
        if not line:
            continue
        try:
            rows.append(json.loads(line.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if strict:
                raise ValueError(f"{path} : ligne JSONL illisible ({exc})") from exc
            break
    return rows


def part_path(path: Path) -> Path:
    """Fichier temporaire ``{nom}.part`` à côté de ``path``."""
    return path.with_name(path.name + ".part")


def dump_row(row: Mapping[str, Any]) -> str:
    """Une ligne JSONL (sans accents échappés), terminée par un saut de ligne."""
    return json.dumps(row, ensure_ascii=False) + "\n"


def write_rows_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Écrit ``rows`` dans ``path`` via ``.part`` puis ``os.replace`` (jamais à moitié écrit)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = part_path(path)
    with part.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(dump_row(row))
    os.replace(part, path)


def clean_for_append(path: Path) -> list[Row]:
    """Lit ``path`` et le réécrit sans sa ligne tronquée éventuelle ; rend les lignes valides.

    Prépare une reprise en ajout (``open("a")``) : sans cela, l'ajout se collerait à la ligne
    tronquée. Un fichier absent n'est pas créé.
    """
    path = Path(path)
    if not path.is_file():
        return []
    rows = read_rows(path)
    write_rows_atomic(path, rows)
    return rows
