"""Mesure des trois bras anti-boucle (story bonus 6.2, SPEC CAP-9).

Bras, même protocole (mêmes items et même ordre, génération gloutonne, même plafond de tokens,
même gabarit de chat, transformers) :

- ``origin`` : Qwen3.5-0.8B d'origine ;
- ``surgery`` : checkpoint édité par le MIT (poids modifiés, permanent) ;
- ``ablation@<alpha>`` : modèle d'origine + ``DirectionAblator`` (directions par couche, dosable).

Mesures par bras : taux de réponses tronquées (le plafond est atteint sans token de fin), exactitude
QCM (non-extraits et tronquées comptés faux), longueur moyenne, avec IC95 bootstrap. Aucun poids
ni direction n'est versionné : sorties sous ``data/runs/antiloop/<run_id>/`` (ignoré).

    uv run --extra ablation python scripts/run_antiloop.py --run-id r1 --origin <base> \\
        --surgery <wsp> --directions-dir <dossier layer15.npy / layer19.npy>

``--origin`` et ``--surgery`` sont obligatoires, sauf si ``ANTILOOP_ORIGIN`` / ``ANTILOOP_SURGERY``
sont définies. La configuration du run est figée dans ``config.json`` : une reprise sous une autre
configuration est refusée.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mood_steering.eval.accuracy import McqItem, extract_letter, format_prompt
from mood_steering.eval.stats import bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "data" / "runs" / "antiloop"
DIRECTION_LAYERS = (15, 19)
END_OF_TURN = "<|im_end|>"  # fin de tour ChatML (Qwen) : absente du eos_token_id de config.json

# (génère la réponse d'un prompt) -> (texte, nombre de tokens générés, fin naturelle atteinte)
Generate = Callable[[str], tuple[str, int, bool]]


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str  # "origin" | "surgery" | "ablation"
    alpha: float | None = None


def build_arms(alphas: Sequence[float], only: Sequence[str] | None = None) -> list[Arm]:
    # Ordre : origine puis ablations (même modèle chargé), chirurgie en dernier.
    arms = [Arm("origin", "origin")]
    arms += [Arm(f"ablation@{a:g}", "ablation", a) for a in alphas]
    arms.append(Arm("surgery", "surgery"))
    if only:
        unknown = sorted(set(only) - {a.name for a in arms})
        if unknown:
            raise ValueError(f"bras inconnus : {unknown}")
        arms = [a for a in arms if a.name in only]
    return arms


def is_truncated(generated_ids: Sequence[int], max_new_tokens: int, eos_ids: Sequence[int]) -> bool:
    """Tronquée : le plafond est atteint et aucun token de fin n'a été émis."""
    eos = set(eos_ids)
    return len(generated_ids) >= max_new_tokens and not any(t in eos for t in generated_ids)


def collect_eos_ids(tok: Any, model: Any) -> list[int]:
    """Union des tokens de fin : ``generation_config``, tokenizer et ``<|im_end|>`` s'il est connu.

    Qwen3.5 met ``<|endoftext|>`` dans ``config.json`` mais termine ses tours par ``<|im_end|>`` :
    sans l'union, une réponse finie serait comptée tronquée (et la génération continuerait).
    """
    ids: set[int] = set()
    cfg = getattr(model, "generation_config", None)
    for eos in (getattr(cfg, "eos_token_id", None), getattr(tok, "eos_token_id", None)):
        if isinstance(eos, int):
            ids.add(eos)
        elif isinstance(eos, (list, tuple)):
            ids.update(int(e) for e in eos)
    end = tok.convert_tokens_to_ids(END_OF_TURN)
    if isinstance(end, int) and end >= 0 and end != getattr(tok, "unk_token_id", None):
        ids.add(end)
    if not ids:
        raise ValueError("aucun token de fin connu : troncature non mesurable")
    return sorted(ids)


def check_run_config(run_dir: Path, config: Mapping[str, Any]) -> None:
    """Écrit ``config.json`` au premier lancement ; refuse une reprise sous une autre config."""
    path = run_dir / "config.json"
    current = json.loads(json.dumps(config))  # normalise (tuples en listes)
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        keys = previous.keys() | current.keys()
        diff = sorted(k for k in keys if previous.get(k) != current.get(k))
        if diff:
            raise ValueError(f"{path} : configuration différente du run existant ({diff})")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")


def item_sources(items: Sequence[McqItem]) -> list[str]:
    """Source des items, lue dans le préfixe des ids (``gmmlu-fr`` ou ``mmmlu-fr``)."""
    return sorted({i.id.split(":", 1)[0] for i in items})


def arm_file(run_dir: Path, arm: Arm) -> Path:
    return run_dir / f"{arm.name.replace('@', '_a')}.jsonl"


def read_records(path: Path) -> list[dict[str, Any]]:
    """Lit le JSONL ; ignore une dernière ligne tronquée par un arrêt brutal."""
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return out


def run_arm(
    arm: Arm,
    items: Sequence[McqItem],
    generate: Generate,
    run_dir: Path,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Génère les items dans l'ordre, en ajout au JSONL ; saute ceux déjà faits (reprise)."""
    path = arm_file(run_dir, arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = read_records(path)
    ids = [r["item_id"] for r in done]
    if ids != [i.id for i in items][: len(ids)]:
        raise ValueError(f"{path.name} : reprise incohérente avec les items demandés")
    with path.open("w", encoding="utf-8") as fh:  # réécrit les lignes valides puis ajoute
        for rec in done:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for item in items[len(done) :]:
            text, n_tokens, finished = generate(format_prompt(item))
            rec = {
                "item_id": item.id,
                "arm": arm.name,
                "output": text,
                "n_tokens": n_tokens,
                "truncated": (not finished) and n_tokens >= max_new_tokens,
                "max_new_tokens": max_new_tokens,
            }
            done.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
    return done


def _ci(values: Sequence[float]) -> dict[str, float]:
    iv = bootstrap_ci(list(values))
    return {"mean": iv.mean, "low": iv.low, "high": iv.high}


def summarize_arm(records: Sequence[Mapping[str, Any]], items: Sequence[McqItem]) -> dict[str, Any]:
    """Taux de troncature, exactitude (non-extraits faux) et longueur moyenne, IC95 inclus."""
    by_id = {r["item_id"]: r for r in records}
    missing = [i.id for i in items if i.id not in by_id]
    if missing:
        raise ValueError(f"enregistrements manquants : {missing[:3]}")
    recs = [by_id[i.id] for i in items]
    trunc = [1.0 if r["truncated"] else 0.0 for r in recs]
    letters = [extract_letter(r["output"]) for r in recs]
    # Une réponse tronquée (raisonnement inachevé) compte fausse, même si une lettre en sort.
    correct = [
        1.0 if letter == i.answer and not r["truncated"] else 0.0
        for letter, i, r in zip(letters, items, recs, strict=True)
    ]
    return {
        "n": len(recs),
        "truncation_rate": _ci(trunc),
        "accuracy": _ci(correct),
        "n_not_extracted": sum(letter is None for letter in letters),
        "mean_tokens": sum(r["n_tokens"] for r in recs) / len(recs),
    }


# --- Modèles (transformers, importés à la demande) ---------------------------------


def _load(path: Path, device: str, dtype: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype))
    return tok, model.to(device).eval()


def make_generate(tok: Any, model: Any, max_new_tokens: int, thinking: bool = True) -> Generate:
    import torch

    eos_ids = collect_eos_ids(tok, model)

    def generate(prompt: str) -> tuple[str, int, bool]:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        enc = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_ids,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        new = out[0, enc["input_ids"].shape[1] :].tolist()
        finished = not is_truncated(new, max_new_tokens, eos_ids)
        return tok.decode(new, skip_special_tokens=True), len(new), finished

    return generate


def run_arms(
    arms: Sequence[Arm],
    items: Sequence[McqItem],
    load: Callable[[str], tuple[Any, Any]],
    make_gen: Callable[[Any, Any], Generate],
    directions: Mapping[int, Any] | None,
    run_dir: Path,
    max_new_tokens: int,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Joue les bras dans l'ordre ; ``DirectionAblator`` n'est posé que sur les bras ``ablation``
    et retiré avant le bras suivant. ``summary.json`` garde les bras d'un lancement précédent."""
    from mood_steering.ablation import DirectionAblator

    path = run_dir / "summary.json"
    previous: dict[str, Any] = {}
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
    for key in ("arms", "effective_dtype", "eos_token_ids"):  # par bras ou par modèle
        summary[key] = {**previous.get(key, {}), **summary.get(key, {})}
    loaded: dict[str, tuple[Any, Any]] = {}
    for arm in arms:
        key = "surgery" if arm.kind == "surgery" else "origin"
        if key not in loaded:
            loaded.clear()  # un seul modèle en mémoire
            loaded[key] = load(key)
        tok, model = loaded[key]
        ablator = None
        if arm.kind == "ablation":
            if directions is None:
                raise ValueError("bras ablation sans directions")
            ablator = DirectionAblator(model, directions, arm.alpha or 0.0)
        try:
            records = run_arm(arm, items, make_gen(tok, model), run_dir, max_new_tokens)
        finally:
            if ablator:
                ablator.remove()
        summary["arms"][arm.name] = summarize_arm(records, items)
        print(arm.name, json.dumps(summary["arms"][arm.name]), flush=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--arms", nargs="+", help="sous-ensemble de bras (ex. origin ablation@1)")
    origin, surgery = _env_path("ANTILOOP_ORIGIN"), _env_path("ANTILOOP_SURGERY")
    ap.add_argument("--origin", type=Path, default=origin, required=origin is None)
    ap.add_argument("--surgery", type=Path, default=surgery, required=surgery is None)
    ap.add_argument("--directions-dir", type=Path, default=ROOT / "data" / "directions")
    ap.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="gabarit avec <think> ouvert (défaut) ; sans, le gabarit Qwen3.5 le referme",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    import transformers

    from mood_steering.eval.datasets import load_global_mmlu_lite

    if args.threads:
        torch.set_num_threads(args.threads)
    arms = build_arms(args.alphas, args.arms)
    items = load_global_mmlu_lite(n=args.n)
    run_dir = RUNS_DIR / args.run_id
    paths = {"origin": args.origin, "surgery": args.surgery}
    config = {
        "origin": str(args.origin.resolve()),
        "surgery": str(args.surgery.resolve()),
        "directions_dir": str(args.directions_dir.resolve()),
        "n": len(items),
        "item_sources": item_sources(items),
        "max_new_tokens": args.max_new_tokens,
        "thinking": args.thinking,
        "device": args.device,
        "dtype": args.dtype,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    check_run_config(run_dir, config)
    summary: dict[str, Any] = {"run_id": args.run_id, **config, "arms": {}}

    def load(key: str) -> tuple[Any, Any]:
        tok, model = _load(paths[key], args.device, args.dtype)
        summary.setdefault("effective_dtype", {})[key] = str(next(model.parameters()).dtype)
        summary.setdefault("eos_token_ids", {})[key] = collect_eos_ids(tok, model)
        return tok, model

    directions = None
    if any(a.kind == "ablation" for a in arms):
        directions = {i: np.load(args.directions_dir / f"layer{i}.npy") for i in DIRECTION_LAYERS}
    run_arms(
        arms,
        items,
        load,
        lambda tok, model: make_generate(tok, model, args.max_new_tokens, args.thinking),
        directions,
        run_dir,
        args.max_new_tokens,
        summary,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
