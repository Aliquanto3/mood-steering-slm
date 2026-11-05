"""Mesure des trois bras anti-boucle (story bonus 6.2, SPEC CAP-9).

Bras, même protocole (mêmes items et même ordre, génération gloutonne, même plafond de tokens,
même gabarit de chat, transformers) :

- ``origin`` : Qwen3.5-0.8B d'origine ;
- ``surgery`` : checkpoint édité par le MIT (poids modifiés, permanent) ;
- ``ablation@<alpha>`` : modèle d'origine + ``DirectionAblator`` (directions par couche, dosable).

Mesures par bras : taux de réponses tronquées (le plafond est atteint sans token de fin), exactitude
QCM (non-extraits comptés faux), longueur moyenne, avec IC95 bootstrap. Aucun poids ni direction
n'est versionné : sorties sous ``data/runs/antiloop/<run_id>/`` (ignoré).

    uv run python scripts/run_antiloop.py --run-id r1 --n 100 --max-new-tokens 512
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mood_steering.eval.accuracy import McqItem, extract_letter, format_prompt
from mood_steering.eval.stats import bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "data" / "runs" / "antiloop"
SIBLING = Path.home() / "Documents" / "GitHub" / "slm-weight-editing" / "models" / "hf"
DEFAULT_ORIGIN = SIBLING / "qwen3.5-base"
DEFAULT_SURGERY = SIBLING / "qwen3.5-wsp"

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
    return len(generated_ids) >= max_new_tokens and not any(t in eos_ids for t in generated_ids)


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
    correct = [1.0 if letter == i.answer else 0.0 for letter, i in zip(letters, items, strict=True)]
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

    cfg = model.generation_config
    eos = cfg.eos_token_id if cfg.eos_token_id is not None else tok.eos_token_id
    eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]

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
                temperature=None,
                top_p=None,
                top_k=None,
            )
        new = out[0, enc["input_ids"].shape[1] :].tolist()
        finished = not is_truncated(new, max_new_tokens, eos_ids)
        return tok.decode(new, skip_special_tokens=True), len(new), finished

    return generate


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--arms", nargs="+", help="sous-ensemble de bras (ex. origin ablation@1)")
    ap.add_argument("--origin", type=Path, default=DEFAULT_ORIGIN)
    ap.add_argument("--surgery", type=Path, default=DEFAULT_SURGERY)
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

    from mood_steering.ablation import DirectionAblator
    from mood_steering.eval.datasets import load_global_mmlu_lite

    if args.threads:
        import torch

        torch.set_num_threads(args.threads)
    arms = build_arms(args.alphas, args.arms)
    items = load_global_mmlu_lite(n=args.n)
    run_dir = RUNS_DIR / args.run_id
    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "n": len(items),
        "max_new_tokens": args.max_new_tokens,
        "thinking": args.thinking,
        "device": args.device,
        "dtype": args.dtype,
        "arms": {},
    }

    loaded: dict[str, tuple[Any, Any]] = {}
    for arm in arms:
        key = "surgery" if arm.kind == "surgery" else "origin"
        if key not in loaded:
            loaded = {
                key: _load(
                    args.surgery if key == "surgery" else args.origin, args.device, args.dtype
                )
            }  # un seul modèle en mémoire
        tok, model = loaded[key]
        ablator = None
        if arm.kind == "ablation":
            dirs = {i: np.load(args.directions_dir / f"layer{i}.npy") for i in (15, 19)}
            ablator = DirectionAblator(model, dirs, arm.alpha or 0.0)
        try:
            gen = make_generate(tok, model, args.max_new_tokens, args.thinking)
            records = run_arm(arm, items, gen, run_dir, args.max_new_tokens)
        finally:
            if ablator:
                ablator.remove()
        summary["arms"][arm.name] = summarize_arm(records, items)
        print(arm.name, json.dumps(summary["arms"][arm.name]), flush=True)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
