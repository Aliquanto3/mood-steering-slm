"""Calibration : vecteur de contrôle d'une humeur, généré une fois puis mis en cache.

Le vecteur est produit par `llama-cvector-generator` (GGUF `controlvector`, tenseurs
`direction.N` avec N = 1..layer_count, F32 1D [n_embd]) et rangé sous
`data/moods/{model_id}/{axis}.gguf`. `load_vector` le relit en tableau (layer_count, n_embd),
ligne 0 = couche 1, couches absentes à zéro : la forme attendue par `Steerer` (vecteur
complet, quelle que soit `il_start`).
"""

from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from importlib import resources
from pathlib import Path

import numpy as np
from gguf import GGUFReader

DEFAULT_ROOT = Path("data/moods")
GENERATOR_ENV = "MOOD_CVECTOR_GENERATOR"
_DIRECTION = re.compile(r"^direction\.(\d+)$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

Generator = Callable[[Sequence[str]], object]


def _check_name(label: str, value: str) -> str:
    if not isinstance(value, str) or ".." in value or not _SAFE_NAME.match(value):
        raise ValueError(
            f"{label} invalide : {value!r} (lettres, chiffres, '.', '_' et '-' ; "
            "ni séparateur de chemin ni '..')"
        )
    return value


def vector_path(model_id: str, axis: str, root: Path = DEFAULT_ROOT) -> Path:
    """Chemin du vecteur en cache : ``root/{model_id}/{axis}.gguf``."""
    return Path(root) / _check_name("model_id", model_id) / f"{_check_name('axis', axis)}.gguf"


def _field(reader: GGUFReader, key: str):
    field = reader.fields.get(key)
    return None if field is None else field.contents()


def load_vector(path: str | Path, *, n_embd: int | None = None) -> np.ndarray:
    """Relit un GGUF `controlvector` en float32 de forme ``(layer_count, n_embd)``.

    Ligne 0 = couche 1 ; les couches absentes du fichier restent à zéro. Sans
    ``controlvector.layer_count``, la plus haute couche présente fixe le nombre de lignes.
    """
    reader = GGUFReader(str(path))
    directions: dict[int, np.ndarray] = {}
    dim: int | None = None
    for t in reader.tensors:
        m = _DIRECTION.match(t.name)
        if m is None or int(m.group(1)) == 0:
            raise ValueError(f"nom de tenseur inattendu : {t.name!r}")
        if t.tensor_type.name != "F32":
            raise ValueError(f"type non F32 : {t.name} ({t.tensor_type.name})")
        if len(t.shape) != 1:
            raise ValueError(f"forme non 1D : {t.name} {[int(s) for s in t.shape]}")
        size = int(t.shape[0])
        if dim is None:
            dim = size
        elif size != dim:
            raise ValueError(f"dimensions incohérentes : {t.name} a {size}, attendu {dim}")
        directions[int(m.group(1))] = np.array(t.data, dtype=np.float32).reshape(-1)
    if dim is None:
        raise ValueError(f"aucune direction dans {path}")
    if n_embd is not None and dim != n_embd:
        raise ValueError(f"dimension {dim} différente de n_embd={n_embd}")
    declared = _field(reader, "controlvector.layer_count")
    top = max(directions)
    layer_count = int(declared) if declared is not None else top
    if top > layer_count:
        raise ValueError(f"direction.{top} au-delà de layer_count={layer_count}")
    out = np.zeros((layer_count, dim), dtype=np.float32)
    for il, vec in directions.items():
        out[il - 1] = vec
    return out


def default_pairs_path() -> Path:
    """Chemin du JSONL de paires joie/tristesse fourni avec le paquet."""
    return Path(str(resources.files("mood_steering").joinpath("pairs", "joy_sadness.jsonl")))


def _read_pairs(path: Path) -> tuple[list[str], list[str]]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"fichier de paires introuvable : {path}") from exc
    pos: list[str] = []
    neg: list[str] = []
    # Découpe sur "\n" seulement : str.splitlines couperait aussi sur U+2028, U+0085, etc.
    for n, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            p, q = row["positive"], row["negative"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{n} : paire invalide ({exc})") from exc
        for label, value in (("positive", p), ("negative", q)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{path}:{n} : {label} doit être un texte non vide")
        pos.append(p)
        neg.append(q)
    if not pos:
        raise ValueError(f"aucune paire dans {path}")
    return pos, neg


def _escape(text: str) -> str:
    """Une invite par ligne : le générateur lit ligne à ligne puis traite les échappements."""
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


def _run(cmd: Sequence[str]) -> None:
    subprocess.run(list(cmd), check=True)


def calibrate(
    model_path: str | Path,
    model_id: str,
    axis: str = "joy_sadness",
    *,
    pairs_path: str | Path | None = None,
    force: bool = False,
    method: str = "mean",
    generator: Generator | None = None,
    generator_exe: str | Path | None = None,
    root: Path = DEFAULT_ROOT,
) -> Path:
    """Rend le chemin du vecteur de ``model_id`` / ``axis``, en le générant si absent.

    Le cache est relu tel quel (le générateur n'est pas lancé) sauf si ``force``.
    ``generator`` reçoit la ligne de commande complète (liste de chaînes) ; par défaut
    `subprocess.run(check=True)`. Le binaire vient de ``generator_exe`` ou de la variable
    d'environnement ``MOOD_CVECTOR_GENERATOR``.

    La clé du cache n'est que ``model_id`` / ``axis`` : changer ``method`` ou les paires
    demande ``force=True``. La sortie est relue par `load_vector` avant d'être installée ;
    en cas d'échec, l'ancien cache reste en place.
    """
    target = vector_path(model_id, axis, root)
    if target.exists() and not force:
        return target
    exe = generator_exe or os.environ.get(GENERATOR_ENV)
    if generator is None and not exe:
        raise RuntimeError(f"binaire llama-cvector-generator introuvable : définir {GENERATOR_ENV}")
    pos, neg = _read_pairs(Path(pairs_path) if pairs_path else default_pairs_path())
    run = generator or _run
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pos_file, neg_file = tmp_dir / "pos.txt", tmp_dir / "neg.txt"
        pos_file.write_text("".join(_escape(t) + "\n" for t in pos), "utf-8", newline="\n")
        neg_file.write_text("".join(_escape(t) + "\n" for t in neg), "utf-8", newline="\n")
        # Sortie dans le dossier cible : os.replace y reste atomique (même volume).
        fd, out_name = tempfile.mkstemp(prefix=f".{axis}.", suffix=".tmp", dir=target.parent)
        os.close(fd)
        out = Path(out_name)
        out.unlink()
        try:
            cmd = [
                str(exe or "llama-cvector-generator"), "-m", str(model_path),
                "--positive-file", str(pos_file), "--negative-file", str(neg_file),
                "-o", str(out), "-ngl", "0", "--method", method,
            ]  # fmt: skip
            run(cmd)
            if not out.exists():
                raise RuntimeError(f"le générateur n'a pas produit {out.name}")
            error = None
            try:
                load_vector(out)
            except Exception as exc:  # toute sortie illisible est refusée
                # Sans chaînage : la trace garderait le memmap du lecteur ouvert, et Windows
                # refuserait alors de supprimer le fichier temporaire.
                error = f"{type(exc).__name__}: {exc}"
            if error is not None:
                gc.collect()
                raise RuntimeError(f"sortie du générateur illisible : {error}")
            os.replace(out, target)
        finally:
            out.unlink(missing_ok=True)
    return target
