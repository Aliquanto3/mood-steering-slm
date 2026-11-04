"""Runner du banc : mêmes items, même ordre, même graine pour chaque bras.

Graine d'un item : ``seed + index`` (index de l'item dans la liste), identique pour
tous les bras. Sortie : ``{out_dir}/{arm.name}.jsonl``, une ligne JSON par item,
écrite d'abord dans ``{arm.name}.jsonl.part`` puis renommée à la fin du bras.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mood_steering.eval.arms import KIND_PROMPT, KIND_STEERING, Arm, check_safe_name
from mood_steering.steering import (
    JOYFUL_SYSTEM_PROMPT,
    SAD_SYSTEM_PROMPT,
    Steerer,
    compose_system,
)

NEUTRAL_INSTRUCTION = "Réponds de façon neutre, sans exprimer d'émotion particulière."
DEFAULT_RUNS_DIR = Path("data") / "runs"


@dataclass(frozen=True)
class Item:
    id: str
    prompt: str
    meta: dict[str, Any] = field(default_factory=dict)


def item_seed(seed: int, index: int) -> int:
    return seed + index


def _system_for(arm: Arm, steerer: Steerer, neutral_instruction: bool) -> tuple[str | None, str]:
    """(``system`` passé au Steerer, ``system`` effectif enregistré)."""
    extra = NEUTRAL_INSTRUCTION if neutral_instruction else None
    if arm.kind == KIND_PROMPT:
        full = compose_system(JOYFUL_SYSTEM_PROMPT, extra)
        return full, full
    if arm.kind == KIND_STEERING and arm.dose != 0 and not steerer.has_vector:
        # Repli du Steerer : il compose lui-même le prompt d'humeur avec la consigne.
        base = JOYFUL_SYSTEM_PROMPT if arm.dose > 0 else SAD_SYSTEM_PROMPT
        return extra, compose_system(base, extra)
    return extra, extra or ""


def run_arm(
    arm: Arm,
    items: Sequence[Item],
    steerer: Steerer,
    *,
    run_id: str,
    model_id: str,
    out_dir: Path,
    seed: int = 0,
    max_tokens: int = 128,
    neutral_instruction: bool = False,
) -> Path:
    system, recorded = _system_for(arm, steerer, neutral_instruction)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{arm.name}.jsonl"
    part = out_dir / f"{arm.name}.jsonl.part"
    with part.open("w", encoding="utf-8", newline="\n") as fh:
        for index, item in enumerate(items):
            s = item_seed(seed, index)
            t0 = time.perf_counter()
            res = steerer.generate(
                item.prompt, arm.dose, system=system, max_tokens=max_tokens, seed=s
            )
            latency = time.perf_counter() - t0
            row = {
                "run_id": run_id,
                "model_id": model_id,
                "arm": arm.name,
                "kind": arm.kind,
                "dose": arm.dose,
                "item_id": item.id,
                "seed": s,
                "system": recorded,
                "output": res.text,
                "fallback": res.fallback,
                "neutral_instruction": neutral_instruction,
                "latency_s": latency,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(part, path)
    return path


def run_bench(
    arms: Sequence[Arm],
    items: Sequence[Item],
    steerer: Steerer,
    *,
    run_id: str,
    model_id: str,
    out_dir: Path | None = None,
    seed: int = 0,
    max_tokens: int = 128,
    neutral_instruction: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Passe les mêmes items à chaque bras ; rend ``{nom du bras: fichier JSONL}``.

    Refuse un ``out_dir`` qui contient déjà des ``.jsonl`` ou ``.jsonl.part``, sauf
    ``overwrite=True`` qui les supprime d'abord : ``load_run`` ne relit ainsi jamais
    le bras d'un run précédent.
    """
    check_safe_name(run_id, "run_id")
    names = [a.name for a in arms]
    # casefold : deux noms qui ne diffèrent que par la casse visent le même fichier
    # sous Windows et macOS.
    if len({n.casefold() for n in names}) != len(names):
        raise ValueError(f"noms de bras en double : {names!r}")
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError("identifiants d'items en double")
    target = Path(out_dir) if out_dir is not None else DEFAULT_RUNS_DIR / run_id
    stale = sorted([*target.glob("*.jsonl"), *target.glob("*.jsonl.part")])
    if stale and not overwrite:
        raise FileExistsError(f"run déjà présent dans {target} (overwrite=True pour l'écraser)")
    for old in stale:
        old.unlink()
    return {
        arm.name: run_arm(
            arm,
            items,
            steerer,
            run_id=run_id,
            model_id=model_id,
            out_dir=target,
            seed=seed,
            max_tokens=max_tokens,
            neutral_instruction=neutral_instruction,
        )
        for arm in arms
    }


def load_run(out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Relit un run : ``{nom du bras: lignes}``, dans l'ordre des items.

    Les ``.jsonl.part`` (bras interrompu) ne répondent pas au motif : ignorés.
    """
    result: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(Path(out_dir).glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            result[path.stem] = [json.loads(line) for line in fh if line.strip()]
    return result
