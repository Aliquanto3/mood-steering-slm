"""Pic 1.4 : calibration joie / tristesse, choix des modèles.

Sous-commandes :
  vector MODELE SORTIE [--method mean|pca]   vecteur par llama-cvector-generator (paires du JSONL)
  scan MODELE VECTEUR SORTIE.jsonl           balayage couches x doses x prompts (binding, CPU)
  summary SORTIE.jsonl                       table par couches et dose, plage utile, critères
  speed MODELE VECTEUR [--streams 3]          tokens/s CPU de flux entrelacés (GPU désactivé)

La logique pure (paires, score de ton, cohérence, plage utile) n'a besoin que de la bibliothèque
standard ; `scan` demande l'extra `llama` et `vector` le binaire LLAMA_CVECTOR_GENERATOR.
Le score de ton est une heuristique lexicale : il mesure l'expression d'une humeur dans le texte,
jamais un ressenti du modèle.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

PAIRS_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "mood_steering" / "pairs" / "joy_sadness.jsonl"
)

# Questions neutres en tour de chat : le modèle choisit le ton, le prompt n'en impose aucun.
PROMPTS = [
    "Raconte ta matinée en trois phrases.",
    "Écris un court message à un collègue pour reporter notre réunion de demain.",
    "Décris en quelques phrases le trajet entre la gare et le centre-ville.",
]

# Gabarits de chat écrits à la main (le BOS est ajouté par le tokeniseur). Qwen3.5 : bloc de
# réflexion vide, pour que la réponse commence tout de suite.
CHAT_TEMPLATES = {
    "qwen35": "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "llama": (
        "<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
}


def chat_prompt(arch: str, question: str) -> str:
    """Prompt de chat pour l'architecture GGUF (`general.architecture`)."""
    if arch not in CHAT_TEMPLATES:
        raise ValueError(f"architecture sans gabarit : {arch!r}")
    return CHAT_TEMPLATES[arch].format(q=question)


N_CTX = 512
SEED = 42

# Plages de couches en fractions du nombre de couches (bornes incluses, couche 0 exclue).
LAYER_FRACTIONS = {"all": (0.0, 1.0), "mid": (0.25, 0.75), "high": (0.5, 0.85)}

# Lexique (expressions régulières en minuscules). Français d'abord, anglais pour repérer une
# bascule de langue qui garde le ton. Limite : pas de négation (« pas heureux » compte joie).
JOY = [
    r"heureu\w*", r"bonheur\w*", r"joie\w*", r"joyeu\w*", r"ravi\w*", r"souri\w*", r"rire",
    r"ri", r"rit", r"riant\w*", r"rieu\w*", r"merveill\w*", r"formidable\w*", r"génia\w*",
    r"magnifique\w*", r"content\w*", r"enthousias\w*", r"gai", r"gaie\w*", r"gaiement",
    r"radieu\w*", r"enchant\w*", r"ador\w*", r"plaisir\w*", r"fantastique\w*", r"excellent\w*",
    r"belle", r"beau", r"réjoui\w*", r"allégresse", r"euphori\w*", r"fête\w*", r"célébr\w*",
    r"chance\w*", r"espoir\w*", r"optimis\w*", r"émerveill\w*", r"super", r"génial\w*",
    r"délicieu\w*", r"chaleureu\w*", r"aux anges", r"bonne humeur", r"sourire\w*",
    r"happy", r"happiness", r"joy\w*", r"smil\w*", r"laugh\w*", r"wonderful", r"great",
    r"magique\w*", r"vibrant\w*", r"dynamique\w*", r"agréable\w*", r"ensoleill\w*",
    r"lumineu\w*", r"rayonn\w*", r"festi\w*", r"amus\w*", r"joli\w*", r"splendide\w*",
    r"superbe\w*", r"incroyable\w*", r"passionn\w*", r"bravo", r"youpi", r"hourra",
    r"amazing", r"fantastic", r"beautiful", r"excit\w*", r"delight\w*", r"cheer\w*",
    r"fun", r"enjoy\w*", r"celebrat\w*", r"awesome", r"glad", r"love\w*",
]  # fmt: skip
SAD = [
    r"trist\w*", r"chagrin\w*", r"pleur\w*", r"larme\w*", r"malheur\w*", r"déprim\w*",
    r"mélancoli\w*", r"abattu\w*", r"seul", r"seule", r"seuls", r"solitude", r"peine\w*",
    r"douleur\w*", r"désol\w*", r"sombre\w*", r"gris", r"grise", r"morne\w*", r"accabl\w*",
    r"désesp\w*", r"regret\w*", r"lourd\w*", r"vide", r"sanglot\w*", r"effondr\w*",
    r"navr\w*", r"lugubre\w*", r"lassitude", r"déçu\w*", r"décep\w*", r"soupir\w*",
    r"angoiss\w*", r"misère\w*", r"souffr\w*", r"deuil\w*", r"funèbre\w*", r"hélas",
    r"sad\w*", r"cry\w*", r"cried", r"tears?", r"lonely", r"grief", r"sorrow\w*",
    r"nostalgi\w*", r"fatigu\w*", r"épuis\w*", r"morose\w*", r"terne\w*", r"inquiet\w*",
    r"ennui\w*", r"ennuy\w*", r"abandon\w*", r"isolé\w*", r"pénible\w*", r"perte\w*",
    r"miser\w*", r"depress\w*", r"gloom\w*", r"painful", r"alone", r"unhappy", r"tired",
]  # fmt: skip
# Émojis d'humeur, comptés à part (\b ne s'applique pas à eux).
JOY_EMOJI = set("🎉🥳😊😄😃😁🌟✨🌞🎊💖😍🙂🤩❤🥰😀🎈")
SAD_EMOJI = set("😢😭😞😔💔☹🙁🥀😿😟😩")
_JOY_RE = re.compile(r"\b(?:" + "|".join(JOY) + r")\b")
_SAD_RE = re.compile(r"\b(?:" + "|".join(SAD) + r")\b")
_FR = set("le la les de des du et je il elle est un une que qui dans pour pas en au ce mais "
          "avec sur me ma mon nous sont était a ai été très".split())  # fmt: skip
_EN = set("the and is a to of in it i was my that for with but this we are have".split())

# Seuils de la plage utile (consignés dans docs/pics/choix-des-modeles.md).
MIN_SHIFT = 0.25  # écart de ton minimal avec la dose 0
MIN_COHERENT = 2 / 3  # part minimale de sorties cohérentes et en français


def load_pairs(path: str | Path = PAIRS_PATH) -> list[tuple[str, str]]:
    """Lit le JSONL de paires : une ligne par paire, clés `positive` et `negative` seules."""
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != {"positive", "negative"}:
            raise ValueError(f"ligne {n} : clés attendues positive et negative")
        pos, neg = row["positive"], row["negative"]
        if not all(isinstance(s, str) and s.strip() and "\n" not in s for s in (pos, neg)):
            raise ValueError(f"ligne {n} : textes vides ou multilignes")
        if pos == neg:
            raise ValueError(f"ligne {n} : positif et négatif identiques")
        for s in (pos, neg):
            if s in seen:
                raise ValueError(f"ligne {n} : texte en double : {s!r}")
            seen.add(s)
        pairs.append((pos, neg))
    return pairs


def tone_score(text: str) -> float:
    """(joie − tristesse) / (joie + tristesse + 1) : 0 sans mot d'humeur, borné dans ]−1, 1[."""
    low = text.lower()
    j = len(_JOY_RE.findall(low)) + sum(c in JOY_EMOJI for c in text)
    s = len(_SAD_RE.findall(low)) + sum(c in SAD_EMOJI for c in text)
    return (j - s) / (j + s + 1)


def is_french(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    fr = sum(w in _FR for w in words)
    en = sum(w in _EN for w in words)
    return fr >= 2 and fr > en


def is_coherent(text: str) -> bool:
    """Texte lisible : assez de mots, peu de répétition, surtout des lettres."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < 8:
        return False
    if len(set(words)) / len(words) < 0.45:
        return False
    chars = [c for c in text if not c.isspace()]
    if sum(c.isalpha() for c in chars) / len(chars) < 0.7:
        return False
    trigrams = Counter(zip(words, words[1:], words[2:], strict=False))
    return not trigrams or max(trigrams.values()) < 3


def layer_range(n_layer: int, frac: tuple[float, float]) -> tuple[int, int]:
    """Couches ciblées (incluses) pour une fraction : 1 <= début <= fin <= n_layer − 1."""
    last = n_layer - 1
    start = min(last, max(1, round(frac[0] * n_layer)))
    end = min(last, max(start, round(frac[1] * n_layer)))
    return start, end


@dataclass(frozen=True)
class DoseStat:
    dose: float
    tone: float  # moyenne sur les prompts
    coherent: float  # part cohérente et en français
    n: int


def aggregate(rows: list[dict]) -> dict[str, list[DoseStat]]:
    """Regroupe les sorties par plage de couches puis par dose, doses triées."""
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["layers"], float(r["dose"]))].append(r)
    out: dict[str, list[DoseStat]] = defaultdict(list)
    for (layers, dose), rs in sorted(groups.items()):
        tone = sum(r["tone"] for r in rs) / len(rs)
        coh = sum(bool(r["coherent"] and r["french"]) for r in rs) / len(rs)
        out[layers].append(DoseStat(dose, tone, coh, len(rs)))
    return dict(out)


def useful_range(
    stats: list[DoseStat], sign: int = 1, min_shift: float = MIN_SHIFT
) -> tuple[float, float] | None:
    """Plus longue suite contiguë de doses du signe donné, cohérentes et qui déplacent le ton.

    Base = dose 0 (ton 0 si absente). Une dose est utile si sa part cohérente atteint
    MIN_COHERENT et si son ton s'écarte de la base d'au moins `min_shift` dans le sens du signe.
    Rend (dose la plus faible, dose la plus forte) en valeur absolue, ou None.
    """
    base = next((s.tone for s in stats if s.dose == 0), 0.0)
    side = sorted((s for s in stats if s.dose * sign > 0), key=lambda s: abs(s.dose))
    best: list[DoseStat] = []
    run: list[DoseStat] = []
    for s in side:
        if s.coherent >= MIN_COHERENT and (s.tone - base) * sign >= min_shift:
            run.append(s)
            if (len(run), _ratio(run)) > (len(best), _ratio(best)):
                best = list(run)
        else:
            run = []
    if not best:
        return None
    return abs(best[0].dose), abs(best[-1].dose)


def _ratio(run: list[DoseStat]) -> float:
    return abs(run[-1].dose) / abs(run[0].dose) if run else 0.0


def meets_ratio(rng: tuple[float, float] | None, factor: float = 1.5) -> bool:
    """Critère de l'epic : dose max >= 1,5 x dose min."""
    # Tolérance : 1,5 x 0,1 vaut 0,15000000000000002 en flottant.
    return rng is not None and rng[1] >= factor * rng[0] - 1e-9


# --- génération du vecteur -----------------------------------------------------------------


def generate_vector(model: str, out: str, method: str = "mean", threads: int = 6) -> None:
    exe = os.environ.get("LLAMA_CVECTOR_GENERATOR")
    if not exe:
        sys.exit("définir LLAMA_CVECTOR_GENERATOR")
    pairs = load_pairs()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        pos, neg = Path(tmp, "pos.txt"), Path(tmp, "neg.txt")
        pos.write_text("\n".join(p for p, _ in pairs) + "\n", encoding="utf-8")
        neg.write_text("\n".join(n for _, n in pairs) + "\n", encoding="utf-8")
        cmd = [
            exe, "-m", model, "--positive-file", str(pos), "--negative-file", str(neg),
            "-o", out, "-ngl", "0", "-t", str(threads), "--method", method,
        ]  # fmt: skip
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


# --- balayage -------------------------------------------------------------------------------


def _done_keys(path: Path) -> set[tuple[str, float, int]]:
    if not path.exists():
        return set()
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    return {(r["layers"], float(r["dose"]), int(r["prompt"])) for r in rows}


def _generate(llm, prompt: str, n_predict: int) -> tuple[str, int]:
    """Génération gloutonne token par token, contexte vidé avant (`reset()`).

    Boucle explicite plutôt que `create_completion`, qui échouait (« failed to find a memory
    slot ») sur Qwen3.5 hybride à forte dose ; c'est aussi la boucle du futur calendrier de dose.
    """
    import llama_cpp

    vocab = llama_cpp.llama_model_get_vocab(llm.model)
    toks = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
    # Contexte vidé : la réutilisation du préfixe KV fausserait la comparaison des doses.
    llm.reset()
    llm.eval(toks)
    out: list[int] = []
    for _ in range(n_predict):
        t = llm.sample(top_k=1, temp=0.0, repeat_penalty=1.1)
        if llama_cpp.llama_vocab_is_eog(vocab, t):
            break
        out.append(t)
        llm.eval([t])
    return llm.detokenize(out).decode("utf-8", errors="replace"), len(out)


def scan(
    model: str,
    vector: str,
    out: str,
    doses: list[float],
    layers: list[str],
    n_predict: int = 60,
    threads: int = 6,
) -> None:
    """Une génération gloutonne par (couches, dose, prompt), `reset()` avant chacune.

    Le fichier de sortie est complété ligne à ligne : une reprise saute ce qui est déjà fait.
    """
    import ctypes

    import llama_cpp
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from read_cvector_gguf import read_cvector

    cv = read_cvector(vector)
    llm = llama_cpp.Llama(
        model, n_ctx=N_CTX, seed=SEED, n_threads=threads, n_gpu_layers=0, verbose=False
    )
    n_layer = llama_cpp.llama_model_n_layer(llm.model)
    flat = cv.flat(n_layer)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out_path)
    model_id = Path(model).stem
    arch = llm.metadata["general.architecture"]
    with out_path.open("a", encoding="utf-8") as f:
        for name in layers:
            il_start, il_end = layer_range(n_layer, LAYER_FRACTIONS[name])
            for dose in doses:
                for ip, prompt in enumerate(PROMPTS):
                    if (name, float(dose), ip) in done:
                        continue
                    if dose == 0:
                        rc = llama_cpp.llama_set_adapter_cvec(llm.ctx, None, 0, cv.n_embd, 0, 0)
                        buf = None
                    else:
                        buf = flat * np.float32(dose)  # vivant pendant l'appel
                        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                        rc = llama_cpp.llama_set_adapter_cvec(
                            llm.ctx, ptr, buf.size, cv.n_embd, il_start, il_end
                        )
                    if rc != 0:
                        raise RuntimeError(f"llama_set_adapter_cvec a échoué ({rc})")
                    t0 = time.perf_counter()
                    try:
                        text, n_tok = _generate(llm, chat_prompt(arch, prompt), n_predict)
                        error = None
                    except RuntimeError as e:  # échec de llama_decode : dose inutilisable
                        text, n_tok, error = "", 0, str(e)
                    row = {
                        "model": model_id,
                        "layers": name,
                        "il": [il_start, il_end],
                        "dose": dose,
                        "prompt": ip,
                        "text": text,
                        "tone": round(tone_score(text), 4),
                        "coherent": is_coherent(text),
                        "french": is_french(text),
                        "tokens": n_tok,
                        "seconds": round(time.perf_counter() - t0, 2),
                        "error": error,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"{name} {il_start}-{il_end} x{dose:+.3f} p{ip} : {text[:90]!r}")


def speed(
    model: str,
    vector: str,
    streams: int = 3,
    threads: int = 4,
    n_predict: int = 64,
    dose: float = 0.1,
    layers: str = "mid",
) -> dict:
    """Vitesse CPU de `streams` flux entrelacés token par token (GPU désactivé).

    Un contexte par flux (poids partagés par mmap), le dernier sous vecteur : c'est le schéma de
    la démo trois colonnes, où chaque colonne a sa dose et donc son propre contexte.
    """
    import ctypes

    import llama_cpp
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from read_cvector_gguf import read_cvector

    cv = read_cvector(vector)
    llms = [
        llama_cpp.Llama(
            model, n_ctx=N_CTX, seed=SEED, n_threads=threads, n_gpu_layers=0, verbose=False
        )
        for _ in range(streams)
    ]
    arch = llms[0].metadata["general.architecture"]
    n_layer = llama_cpp.llama_model_n_layer(llms[0].model)
    il_start, il_end = layer_range(n_layer, LAYER_FRACTIONS[layers])
    buf = cv.flat(n_layer) * np.float32(dose)
    ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    rc = llama_cpp.llama_set_adapter_cvec(llms[-1].ctx, ptr, buf.size, cv.n_embd, il_start, il_end)
    if rc != 0:
        raise RuntimeError(f"llama_set_adapter_cvec a échoué ({rc})")
    toks = llms[0].tokenize(chat_prompt(arch, PROMPTS[0]).encode("utf-8"), special=True)
    t0 = time.perf_counter()
    for llm in llms:
        llm.reset()
        llm.eval(toks)
    t_prompt = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(n_predict):  # pas d'arrêt sur EOS : même nombre de tokens par flux
        for llm in llms:
            llm.eval([llm.sample(top_k=1, temp=0.0)])
    t_gen = time.perf_counter() - t0
    res = {
        "model": Path(model).stem,
        "streams": streams,
        "threads": threads,
        "prompt_tokens": len(toks),
        "prompt_s": round(t_prompt, 2),
        "tokens_per_stream": n_predict,
        "per_stream_tok_s": round(n_predict / t_gen, 2),
        "total_tok_s": round(streams * n_predict / t_gen, 2),
    }
    print(json.dumps(res, ensure_ascii=False))
    return res


def summary(path: str) -> None:
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]
    for layers, stats in aggregate(rows).items():
        il = next(r["il"] for r in rows if r["layers"] == layers)
        print(f"\n== {rows[0]['model']} couches {layers} {il[0]}-{il[1]}")
        print("dose\tton\tcohérent")
        for s in stats:
            print(f"{s.dose:+.3f}\t{s.tone:+.2f}\t{s.coherent:.2f}\t(n={s.n})")
        for sign, label in [(1, "joie"), (-1, "tristesse")]:
            rng = useful_range(stats, sign)
            print(f"plage utile {label} : {rng} ; max >= 1,5 x min : {meets_ratio(rng)}")


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vector")
    v.add_argument("model")
    v.add_argument("out")
    v.add_argument("--method", choices=["mean", "pca"], default="mean")
    v.add_argument("--threads", type=int, default=6)
    s = sub.add_parser("scan")
    s.add_argument("model")
    s.add_argument("vector")
    s.add_argument("out")
    s.add_argument("--doses", type=_floats, default=[0.0, 0.05, 0.1, 0.2, 0.4, -0.1, -0.2])
    s.add_argument("--layers", default="all,mid,high")
    s.add_argument("--tokens", type=int, default=60)
    s.add_argument("--threads", type=int, default=6)
    sub.add_parser("summary").add_argument("path")
    b = sub.add_parser("speed")
    b.add_argument("model")
    b.add_argument("vector")
    b.add_argument("--streams", type=int, default=3)
    b.add_argument("--threads", type=int, default=4)
    b.add_argument("--tokens", type=int, default=64)
    b.add_argument("--dose", type=float, default=0.1)
    b.add_argument("--layers", default="mid")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console Windows en cp1252
    if a.cmd == "vector":
        generate_vector(a.model, a.out, a.method, a.threads)
    elif a.cmd == "scan":
        scan(a.model, a.vector, a.out, a.doses, a.layers.split(","), a.tokens, a.threads)
    elif a.cmd == "speed":
        speed(a.model, a.vector, a.streams, a.threads, a.tokens, a.dose, a.layers)
    else:
        summary(a.path)


if __name__ == "__main__":
    main()
