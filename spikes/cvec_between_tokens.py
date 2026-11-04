"""Pic 1.2 : un vecteur de contrôle réappliqué entre deux tokens modifie-t-il la suite ?

Génération gloutonne token par token avec llama-cpp-python (extra optionnel `llama`).
Quatre exécutions, même prompt, même graine :

- base : aucun vecteur ;
- full : vecteur appliqué dès le début ;
- late : vecteur appliqué après N tokens générés, cache KV conservé ;
- off  : vecteur dès le début, retiré après N tokens (dose 0), cache KV conservé.

Usage : uv run python spikes/cvec_between_tokens.py models/x.gguf --n-switch 20 --scale 8
"""

from __future__ import annotations

import argparse
import ctypes
import math
import random
from collections.abc import Sequence

CVEC_FIRST_LAYER = 1  # le tampon du binding commence à la couche 1 (la couche 0 est exclue)


def make_synthetic_vector(n_embd: int, n_layers: int, seed: int, scale: float) -> list[float]:
    """Direction aléatoire unitaire par couche, multipliée par `scale`.

    Tampon plat de taille n_embd * n_layers, couche 1 en premier (`n_layers` = nombre de
    couches ciblées, de `il_start` à `il_end` inclus).
    """
    if n_embd <= 0 or n_layers <= 0:
        raise ValueError("n_embd et n_layers doivent être positifs")
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(n_layers):
        raw = [rng.gauss(0.0, 1.0) for _ in range(n_embd)]
        norm = math.sqrt(sum(x * x for x in raw))
        out.extend(scale * x / norm for x in raw)
    return out


def first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    """Indice du premier token différent, ou None si les suites sont identiques.

    Si l'une est un préfixe strict de l'autre, la divergence est à la longueur de la plus courte.
    """
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def summarize(runs: dict[str, list[int]], n_switch: int) -> dict[str, int | None]:
    """Divergences utiles pour la note (indices dans la suite générée, 0 = 1er token généré)."""
    return {
        "late_vs_base": first_divergence(runs["late"], runs["base"]),
        "full_vs_base": first_divergence(runs["full"], runs["base"]),
        "off_vs_full": first_divergence(runs["off"], runs["full"]),
        "off_vs_base": first_divergence(runs["off"], runs["base"]),
        "late_vs_full": first_divergence(runs["late"], runs["full"]),
        "n_switch": n_switch,
    }


def _set_cvec(llama_cpp, ctx, data: list[float] | None, n_embd: int, il_start: int, il_end: int):
    if data is None:
        rc = llama_cpp.llama_set_adapter_cvec(ctx, None, 0, n_embd, 0, 0)
    else:
        arr = (ctypes.c_float * len(data))(*data)
        rc = llama_cpp.llama_set_adapter_cvec(ctx, arr, len(data), n_embd, il_start, il_end)
    if rc != 0:
        raise RuntimeError(f"llama_set_adapter_cvec a renvoyé {rc}")


def generate(
    llm,
    llama_cpp,
    prompt: str,
    n_tokens: int,
    schedule: dict[int, list[float] | None],
    n_embd: int,
    il_start: int,
    il_end: int,
) -> list[int]:
    """Génère `n_tokens` tokens gloutons ; `schedule[i]` est appliqué avant le token d'indice i."""
    llm.reset()
    ctx = llm._ctx.ctx
    _set_cvec(llama_cpp, ctx, None, n_embd, il_start, il_end)
    llm.eval(llm.tokenize(prompt.encode(), add_bos=True))
    out: list[int] = []
    for i in range(n_tokens):
        if i in schedule:
            _set_cvec(llama_cpp, ctx, schedule[i], n_embd, il_start, il_end)
        tok = llm.sample(top_k=1, temp=0.0)
        out.append(tok)
        llm.eval([tok])
    _set_cvec(llama_cpp, ctx, None, n_embd, il_start, il_end)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--prompt", default="The weather this morning was")
    p.add_argument("--n-tokens", type=int, default=60)
    p.add_argument("--n-switch", type=int, default=20)
    p.add_argument("--scale", type=float, default=8.0)
    p.add_argument("--il-start", type=int, default=CVEC_FIRST_LAYER)
    p.add_argument("--il-end", type=int, default=None, help="inclus ; défaut : dernière couche")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-gpu-layers", type=int, default=0)
    args = p.parse_args()

    import llama_cpp  # import paresseux : l'extra `llama` est optionnel

    llm = llama_cpp.Llama(
        args.model, n_ctx=512, n_gpu_layers=args.n_gpu_layers, seed=args.seed, verbose=False
    )
    n_embd = llm.n_embd()
    n_layer = llama_cpp.llama_model_n_layer(llm._model.model)
    il_end = args.il_end if args.il_end is not None else n_layer - 1
    n_target = il_end - args.il_start + 1
    vec = make_synthetic_vector(n_embd, n_target, args.seed, args.scale)
    print(
        f"llama-cpp-python {llama_cpp.__version__} n_embd={n_embd} n_layer={n_layer} "
        f"couches {args.il_start}..{il_end} tampon={len(vec)} floats scale={args.scale}"
    )

    def run(sched):
        return generate(
            llm, llama_cpp, args.prompt, args.n_tokens, sched, n_embd, args.il_start, il_end
        )

    n = args.n_switch
    runs = {
        "base": run({}),
        "full": run({0: vec}),
        "late": run({n: vec}),
        "off": run({0: vec, n: None}),
    }
    runs["base2"] = run({})  # contrôle de déterminisme
    for name, toks in runs.items():
        print(f"--- {name}\n{llm.detokenize(toks).decode(errors='replace')!r}")
    print("determinisme base == base2 :", runs["base"] == runs["base2"])
    print(summarize(runs, n))


if __name__ == "__main__":
    main()
