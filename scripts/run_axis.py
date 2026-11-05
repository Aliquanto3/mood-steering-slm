"""Second axe calme <-> agitation (story 5.4, H5) : cosinus, génération reprenable, rapport.

Commande (le rapport suit la génération) :

    uv run --extra llama --extra eval python scripts/run_axis.py --resume --n-gpu-layers -1 \\
        --cuda-dll-dir <dossier des DLL CUDA>

Rapport seul, depuis les sorties existantes (ni génération, ni llama.cpp ; le classifieur ne
charge que pour un bras jamais classé) : ``--report-only``.

Reprise : ``--resume`` saute les bras déjà écrits et reprend un bras interrompu à l'item
suivant ; ``--max-minutes`` arrête proprement entre deux items (code retour 3). Sorties brutes :
``{out}/{model_id}/axis/{bras}.jsonl`` (``data/runs/``, jamais versionné).

llama.cpp ne retient qu'UN vecteur de contrôle par contexte (``llama_adapter_cvec::apply`` écrase
le précédent) : le bras combiné somme les deux vecteurs côté numpy, chacun à sa dose, puis pose
la somme. Génération gloutonne : la graine (``seed + index``) est enregistrée, sans effet.
Aucun juge, aucun appel payant. Le script mesure l'expression d'une humeur dans le texte.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load_run_bench():
    """Importe ``scripts/run_bench.py`` en lecture seule (module ``run_bench``)."""
    if "run_bench" in sys.modules:
        return sys.modules["run_bench"]
    spec = importlib.util.spec_from_file_location("run_bench", ROOT / "scripts" / "run_bench.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_bench"] = module
    spec.loader.exec_module(module)
    return module


rb = _load_run_bench()

from mood_steering.eval.arms import KIND_NEUTRAL, KIND_STEERING, Arm  # noqa: E402
from mood_steering.eval.runner import item_seed  # noqa: E402
from mood_steering.steering import Steerer  # noqa: E402

AXIS_CALM = "calm_agitation"
AXIS_JOY = "joy_sadness"
PHASE = "axis"
EXIT_INCOMPLETE = 3
DEFAULT_OUT = Path("data") / "runs" / "axe-5-4"
DEFAULT_REPORT = Path("docs") / "rapport-axe-calme.md"
# échelles absolues (dose x max_scale) mesurées ; 0.3 et 0.5 s'ajoutent par --scales 0.2,0.3,0.4,0.5
DEFAULT_SCALES = (0.2, 0.4)
N_PROMPTS = 50
TOKENS = 128
N_BOOT = 2000
BOOT_SEED = 0

# Seuils fixés avant la mesure (voir le rapport).
COS_MAX = 0.7  # |cos| < 0.7 sur chaque couche ciblée (cos^2 < 0.5)
RETENTION_MIN = 0.5  # le bras combiné garde au moins la moitié de chaque effet seul
DEGENERATE_MARGIN = 0.20  # un bras valide ne dépasse pas le neutre de plus de 20 points

ACTIVATION_LABELS = ("anger", "fear", "surprise")

# Lexiques de proxy (préfixes de mots, minuscules ; sens d'une activation haute / basse).
AGITATION_PREFIXES = (
    "agit", "précipit", "affol", "panique", "frénét", "fébril", "nerveu", "bouscul",
    "urgen", "stress", "tumult", "vacarm", "hâte", "hâti", "empress", "trépid", "effervesc",
    "haleta", "brusque", "claqu", "hurl", "crie", "cris", "tourbillon", "vitesse", "rapide",
    "vite", "course", "courir", "cours", "court", "pressé", "presser", "s'emball", "emball",
)  # fmt: skip
CALM_PREFIXES = (
    "calm", "paisib", "tranquill", "serein", "sérén", "apais", "doucement", "douceur", "doux",
    "douce", "lent", "posé", "patien", "silenc", "détend", "repos",
)  # fmt: skip
# locutions de calme, comptées sur le texte puis retirées (« presser », « hâte » ne comptent pas
# alors côté agitation)
CALM_PHRASES = ("sans hâte", "sans se presser")
_ELISION = re.compile(r"^(?:[cdjlmnst]|qu)['’]")
_WORD = re.compile(r"[\wéèêëàâäîïôöùûüç'’-]+", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?…]+")


# --- bras --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisArm:
    """Un bras : échelles signées (dose x max_scale) de joie+ et de calme+ (< 0 = agitation)."""

    name: str
    joy: float = 0.0
    calm: float = 0.0

    @property
    def is_neutral(self) -> bool:
        return self.joy == 0 and self.calm == 0


def _tag(prefix: str, scale: float) -> str:
    return f"{prefix}_{'m' if scale < 0 else 'p'}{abs(scale):.2f}"


def build_arms(scales: Sequence[float]) -> list[AxisArm]:
    """Neutre, puis par échelle : calme+, agitation, joie+ seule, joie+ et calme+ combinés."""
    if not scales or any(s <= 0 for s in scales) or len(set(scales)) != len(scales):
        raise ValueError(f"échelles positives et distinctes attendues : {tuple(scales)!r}")
    arms = [AxisArm("neutral")]
    for s in scales:
        arms += [
            AxisArm(_tag("calm", s), calm=s),
            AxisArm(_tag("agit", -s), calm=-s),
            AxisArm(_tag("joy", s), joy=s),
            AxisArm(_tag("joycalm", s), joy=s, calm=s),
        ]
    return arms


def arm_vector(
    arm: AxisArm, joy: np.ndarray, calm: np.ndarray, max_scale: float
) -> np.ndarray | None:
    """Vecteur à poser pour le bras, divisé par ``max_scale`` (dose 1 du Steerer = la somme).

    ``joy * s_joie + calm * s_calme`` : somme côté numpy, un seul vecteur pour llama.cpp.
    """
    if arm.is_neutral:
        return None
    if joy.shape != calm.shape:
        raise ValueError(f"formes différentes : joie {joy.shape}, calme {calm.shape}")
    total = np.float32(arm.joy) * joy + np.float32(arm.calm) * calm
    return (total / np.float32(max_scale)).astype(np.float32)


def cosine_by_layer(a: np.ndarray, b: np.ndarray, il_start: int, il_end: int) -> dict[int, float]:
    """Cosinus des couches ``il_start..il_end`` (inclusives, ligne ``il - 1``) ; nan si nul."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"formes incompatibles : {a.shape} et {b.shape}")
    if not 1 <= il_start <= il_end <= a.shape[0]:
        raise ValueError(
            f"plage de couches invalide : {il_start}..{il_end} pour {a.shape[0]} lignes"
        )
    out: dict[int, float] = {}
    for il in range(il_start, il_end + 1):
        x, y = a[il - 1].astype(np.float64), b[il - 1].astype(np.float64)
        n = np.linalg.norm(x) * np.linalg.norm(y)
        out[il] = float(x @ y / n) if n > 0 else math.nan
    return out


def distinct(cosines: Mapping[int, float], threshold: float = COS_MAX) -> bool:
    """Vrai si |cos| < ``threshold`` sur toutes les couches (une couche nan fait échouer)."""
    vals = list(cosines.values())
    return bool(vals) and all(abs(v) < threshold for v in vals)


# --- génération reprenable ---------------------------------------------------------------------


def axis_prompts(n: int) -> list[Any]:
    """Les ``n`` premiers prompts neutres du banc (mêmes gabarits et sujets), ``tone000``…"""
    return rb.tone_prompts(n)


def unit_path(out: Path, model_id: str, arm: str) -> Path:
    return rb.unit_path(out, model_id, PHASE, arm)


def build_manifest(
    spec: Any, scales: Sequence[float], n_prompts: int, tokens: int, vectors: Mapping[str, str]
) -> dict[str, Any]:
    prompts = [i.prompt for i in axis_prompts(n_prompts)]
    import hashlib

    return {
        "model_id": spec.model_id,
        "il_start": spec.il_start,
        "il_end": spec.il_end,
        "max_scale": spec.max_scale,
        "n_prompts": n_prompts,
        "tokens": tokens,
        "seed": 0,
        "prompts_sha256": hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest(),
        "vectors_sha256": dict(vectors),
    }


def run_axis(
    spec: Any,
    scales: Sequence[float],
    out: Path,
    *,
    backend_factory: Callable[[], Any],
    vector_loader: Callable[[str, int], np.ndarray],
    n_prompts: int = N_PROMPTS,
    tokens: int = TOKENS,
    run_id: str = "axe-5-4",
    deadline: float | None = None,
    log: Callable[[str], None] = print,
) -> bool:
    """Génère les bras manquants ; ``True`` si tout est écrit, ``False`` si interrompu."""
    arms = build_arms(scales)
    if all(unit_path(out, spec.model_id, a.name).exists() for a in arms):
        log(f"{spec.model_id} : déjà complet")
        return True
    backend = backend_factory()
    joy = vector_loader(AXIS_JOY, backend.n_embd)
    calm = vector_loader(AXIS_CALM, backend.n_embd)
    items = [(it, item_seed(0, k)) for k, it in enumerate(axis_prompts(n_prompts))]
    for arm in arms:
        path = unit_path(out, spec.model_id, arm.name)
        if path.exists():
            continue
        vec = arm_vector(arm, joy, calm, spec.max_scale)
        steerer = Steerer(
            backend,
            vec,
            max_scale=spec.max_scale,
            il_start=spec.il_start,
            il_end=spec.il_end,
        )
        bras = Arm(arm.name, KIND_NEUTRAL) if vec is None else Arm(arm.name, KIND_STEERING, 1.0)
        t0 = time.monotonic()
        complete = rb.run_unit(
            steerer,
            bras,
            items,
            path,
            run_id=run_id,
            model_id=spec.model_id,
            max_tokens=tokens,
            extra={"scale_joy": arm.joy, "scale_calm": arm.calm},
            deadline=deadline,
        )
        log(
            f"{spec.model_id} {arm.name} : {time.monotonic() - t0:.0f} s"
            + ("" if complete else " (interrompu, à reprendre)")
        )
        if not complete:
            return False
    return True


# --- proxies d'activation et de valence -------------------------------------------------------


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if words(s)]


def exclamation_rate(text: str) -> float:
    """Points d'exclamation par phrase (0 si aucune phrase)."""
    n = len(sentences(text))
    return text.count("!") / n if n else 0.0


def mean_sentence_len(text: str) -> float:
    """Longueur moyenne des phrases en mots (nan sans phrase)."""
    s = sentences(text)
    return sum(len(words(x)) for x in s) / len(s) if s else math.nan


def lexicon_score(text: str) -> float:
    """(mots d'agitation - mots de calme) pour 100 mots ; positif = plus d'activation."""
    n = len(words(text))
    if not n:
        return 0.0
    low = " ".join(words(text))
    calm = 0
    for phrase in CALM_PHRASES:
        calm += low.count(phrase)
        low = low.replace(phrase, " ")
    ws = [_ELISION.sub("", w) for w in low.split()]  # « s'agite » -> « agite »
    agit = sum(any(w.startswith(p) for p in AGITATION_PREFIXES) for w in ws)
    calm += sum(any(w.startswith(p) for p in CALM_PREFIXES) for w in ws)
    return 100.0 * (agit - calm) / n


def activation_share(probs: Mapping[str, float]) -> float:
    """Part de P(colère)+P(peur)+P(surprise) dans la somme des probabilités des classes.

    Le classifieur est multi-label (sigmoïdes, somme libre) : on normalise par la somme.
    """
    total = sum(float(v) for v in probs.values())
    if total <= 0:
        return 0.0
    return sum(float(probs[k]) for k in ACTIVATION_LABELS if k in probs) / total


def text_proxies(text: str) -> dict[str, float]:
    """Proxies calculés sur le texte seul (recalculables sans classifieur)."""
    refusal, degenerate = rb.output_flags(text)
    return {
        "exclam": exclamation_rate(text),
        "lex": lexicon_score(text),
        "sent_len": mean_sentence_len(text),
        "degenerate": float(degenerate),
        "refusal": float(refusal),
    }


def score_text(text: str, probs: Mapping[str, float]) -> dict[str, float]:
    return {
        "act_cls": activation_share(probs),
        **text_proxies(text),
        "valence": float(probs["joy"]) - float(probs["sadness"]),
    }


def score_arms(
    out: Path, spec: Any, scales: Sequence[float], classify: Callable[[list[str]], Any]
) -> dict[str, dict[str, dict[str, float]]]:
    """Scores ``{bras: {item_id: {proxy: valeur}}}``, mis en cache (``axis_scores.json``).

    Les proxies de texte d'un bras en cache sont recalculés depuis les sorties (un changement de
    lexique s'applique sans reclasser) ; ``classify`` ne sert qu'aux bras absents du cache.
    """
    cache = Path(out) / spec.model_id / "axis_scores.json"
    cached = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    changed = False
    for arm in build_arms(scales):
        rows = rb.read_jsonl(unit_path(out, spec.model_id, arm.name))
        if arm.name in cached:
            for r in rows:
                old = cached[arm.name][r["item_id"]]
                new = {**old, **text_proxies(r["output"])}
                if json.dumps(new) != json.dumps(old):
                    cached[arm.name][r["item_id"]] = new
                    changed = True
            continue
        probs = classify([r["output"] for r in rows])
        cached[arm.name] = {
            r["item_id"]: score_text(r["output"], p) for r, p in zip(rows, probs, strict=True)
        }
        changed = True
    if changed:
        cache.write_text(json.dumps(cached), encoding="utf-8", newline="\n")
    return cached


# --- statistiques appariées --------------------------------------------------------------------


def paired_boot(
    columns: Mapping[str, np.ndarray],
    stat: Callable[[Mapping[str, np.ndarray]], float],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float, float]:
    """(valeur, bas, haut) de ``stat`` ; IC95 percentile, items rééchantillonnés ensemble."""
    n = len(next(iter(columns.values())))
    if n == 0 or any(len(c) != n for c in columns.values()):
        raise ValueError("colonnes vides ou de longueurs différentes")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([stat({k: v[i] for k, v in columns.items()}) for i in idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(stat(columns)), float(lo), float(hi)


@dataclass(frozen=True)
class Ci:
    mean: float
    low: float
    high: float

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def fmt(self, digits: int = 2) -> str:
        return f"{self.mean:+.{digits}f} [{self.low:+.{digits}f} ; {self.high:+.{digits}f}]"


def _ci(columns: Mapping[str, np.ndarray], stat, n_boot: int, seed: int) -> Ci:
    return Ci(*paired_boot(columns, stat, n_boot=n_boot, seed=seed))


ACT_PROXIES = ("act_cls", "exclam", "lex")  # indice d'activation (sens : plus = plus agité)


def _aligned(
    scores: Mapping[str, Mapping[str, Mapping[str, float]]], arm: str
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Items communs au bras et au neutre, triés ; une colonne par proxy."""
    ids = sorted(set(scores[arm]) & set(scores["neutral"]))
    if not ids:
        raise ValueError(f"aucun item commun entre {arm} et neutral")
    cols = {
        k: np.array([scores[arm][i][k] for i in ids], dtype=float)
        for k in next(iter(scores[arm].values()))
    }
    return ids, cols


def neutral_sd(scores: Mapping[str, Mapping[str, Mapping[str, float]]]) -> dict[str, float]:
    """Écart-type (ddof=1) de chaque proxy sur le bras neutre ; 1 si nul (ou un seul item)."""
    neutral = scores["neutral"]
    out = {}
    for k in next(iter(neutral.values())):
        v = np.array([x[k] for x in neutral.values() if not math.isnan(x[k])])
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out[k] = sd if sd > 1e-9 else 1.0
    return out


def arm_deltas(
    scores: Mapping[str, Mapping[str, Mapping[str, float]]], arm: str, sd: Mapping[str, float]
) -> dict[str, np.ndarray]:
    """Écarts appariés au neutre, par item : ``act`` (indice), ``val`` et chaque proxy en brut.

    ``act`` = moyenne des écarts standardisés (écart-type du neutre) de ``ACT_PROXIES`` ;
    ``val`` = écart standardisé de la valence (joie - tristesse).
    """
    ids, cols = _aligned(scores, arm)
    nb = {k: np.array([scores["neutral"][i][k] for i in ids]) for k in cols}
    raw = {k: cols[k] - nb[k] for k in cols}
    out = {f"d_{k}": v for k, v in raw.items()}
    out["act"] = np.mean([raw[k] / sd[k] for k in ACT_PROXIES], axis=0)
    out["val"] = raw["valence"] / sd["valence"]
    return out


@dataclass(frozen=True)
class ArmStats:
    name: str
    n: int
    act: Ci
    val: Ci
    selectivity_act: Ci  # |moyenne act| - |moyenne val|, bootstrap apparié
    degenerate: float
    refusal: float
    raw: dict[str, Ci]  # écart brut au neutre par proxy


def _sel(a: np.ndarray, b: np.ndarray) -> Callable[[Mapping[str, np.ndarray]], float]:
    return lambda c: abs(float(c["act"].mean())) - abs(float(c["val"].mean()))


def arm_stats(
    scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    arm: str,
    sd: Mapping[str, float],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> ArmStats:
    d = arm_deltas(scores, arm, sd)
    cols = {"act": d["act"], "val": d["val"]}
    mean = lambda key: lambda c: float(c[key].mean())  # noqa: E731
    raw = {
        k[2:]: _ci({"x": v[~np.isnan(v)]}, mean("x"), n_boot, seed)  # sans phrase : écarté
        if not np.isnan(v).all()
        else Ci(math.nan, math.nan, math.nan)
        for k, v in d.items()
        if k.startswith("d_")
    }
    vals = list(scores[arm].values())
    return ArmStats(
        name=arm,
        n=len(d["act"]),
        act=_ci(cols, mean("act"), n_boot, seed),
        val=_ci(cols, mean("val"), n_boot, seed),
        selectivity_act=_ci(cols, _sel(d["act"], d["val"]), n_boot, seed),
        degenerate=float(np.mean([v["degenerate"] for v in vals])),
        refusal=float(np.mean([v["refusal"] for v in vals])),
        raw=raw,
    )


@dataclass(frozen=True)
class ScaleVerdict:
    scale: float
    valid: bool  # sorties dégénérées sous le plafond pour les quatre bras
    calm_ok: bool  # calme+ : activation en baisse (IC < 0) et plus que la valence
    agit_ok: bool  # agitation : activation en hausse (IC > 0) et plus que la valence
    joy_ok: bool  # joie+ : valence en hausse (IC > 0) et plus que l'activation
    a_held: bool
    comb_act_ok: bool  # combiné : activation de même sens que calme+ (IC < 0)
    comb_val_ok: bool  # combiné : valence de même sens que joie+ (IC > 0)
    retention_act: float  # écart d'activation combiné / calme+ seul
    retention_val: float  # écart de valence combiné / joie+ seule
    b_held: bool

    @property
    def held(self) -> bool:
        return self.valid and self.a_held and self.b_held


def judge_scale(
    scale: float, stats: Mapping[str, ArmStats], neutral_degenerate: float
) -> ScaleVerdict:
    calm, agit = stats[_tag("calm", scale)], stats[_tag("agit", -scale)]
    joy, comb = stats[_tag("joy", scale)], stats[_tag("joycalm", scale)]
    valid = all(
        s.degenerate - neutral_degenerate <= DEGENERATE_MARGIN for s in (calm, agit, joy, comb)
    )
    calm_ok = calm.act.high < 0 and calm.selectivity_act.low > 0
    agit_ok = agit.act.low > 0 and agit.selectivity_act.low > 0
    # joie : valence en hausse et |valence| > |activation| (sélectivité act - val sous zéro)
    joy_ok = joy.val.low > 0 and joy.selectivity_act.high < 0
    ret_act = comb.act.mean / calm.act.mean if calm.act.mean else math.nan
    ret_val = comb.val.mean / joy.val.mean if joy.val.mean else math.nan
    comb_act_ok = comb.act.high < 0
    comb_val_ok = comb.val.low > 0
    b = (
        comb_act_ok
        and comb_val_ok
        and ret_act >= RETENTION_MIN  # nan >= x est faux
        and ret_val >= RETENTION_MIN
    )
    return ScaleVerdict(
        scale, valid, calm_ok, agit_ok, joy_ok, calm_ok and agit_ok and joy_ok,
        comb_act_ok, comb_val_ok, ret_act, ret_val, b,
    )  # fmt: skip


def analyse(
    scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    scales: Sequence[float],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[dict[str, ArmStats], dict[float, ScaleVerdict], dict[str, float]]:
    sd = neutral_sd(scores)
    stats = {
        a.name: arm_stats(scores, a.name, sd, n_boot=n_boot, seed=seed)
        for a in build_arms(scales)
        if not a.is_neutral
    }
    neutral_deg = float(np.mean([v["degenerate"] for v in scores["neutral"].values()]))
    verdicts = {s: judge_scale(s, stats, neutral_deg) for s in scales}
    return stats, verdicts, sd


# --- rapport ----------------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x * 100:.0f} %"


def _yn(x: bool) -> str:
    return "oui" if x else "non"


def render_report(
    *,
    spec: Any,
    scales: Sequence[float],
    cosines: Mapping[int, float],
    scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    stats: Mapping[str, ArmStats],
    verdicts: Mapping[float, ScaleVerdict],
    sd: Mapping[str, float],
    runtime: Mapping[str, Any],
    n_prompts: int,
    tokens: int,
    lecture: str | None,
    extracts: Mapping[str, str],
) -> str:
    cos_vals = list(cosines.values())
    cos_max = max(abs(c) for c in cos_vals)
    cos_ok = distinct(cosines)
    neutral = scores["neutral"]
    neutral_deg = float(np.mean([v["degenerate"] for v in neutral.values()]))
    held = [s for s, v in verdicts.items() if v.held]
    # sans proxy d'activation validé, un critère (a)/(b) non satisfait ne réfute pas l'axe
    verdict = (
        "non tenue (axes non distincts)"
        if not cos_ok
        else f"tenue à l'échelle {', '.join(f'{s:g}' for s in held)}"
        if held
        else "partielle"
        if any(v.a_held or v.b_held for v in verdicts.values())
        else "non concluante (critère non satisfait, proxies d'activation non validés)"
    )
    L = [
        "# Rapport H5 : second axe calme ↔ agitation",
        "",
        f"> **Verdict H5 : {verdict}** ({spec.model_id}, un seul modèle). Expression d'une humeur "
        "dans le texte produit, jamais un ressenti du modèle. Paires non encore relues.",
        "",
        "## Méthode",
        "",
        f"- Modèle : `{spec.model_id}`, couches {spec.il_start} à {spec.il_end}, `max_scale` "
        f"{spec.max_scale:g} (repris de l'axe joie, l'axe calme n'a pas été calibré). Vecteurs : "
        "`llama-cvector-generator --method mean`, `src/mood_steering/pairs/joy_sadness.jsonl` "
        "(100 paires) et `src/mood_steering/pairs/calm_agitation.jsonl` (paires calme posé "
        "contre agité ou pressé ; **non relues par Anaël**). La valence de ces paires n'est pas "
        "équilibrée : le côté agité porte nettement plus de marqueurs négatifs (panique, retard, "
        "stress, objets renversés) et le côté calme quelques marqueurs positifs (savourer, "
        "plaisir) ; une part du cosinus avec joie ↔ tristesse vient donc des paires elles-mêmes.",
        "- llama.cpp n'accepte qu'un vecteur de contrôle par contexte (le suivant écrase le "
        "précédent) : le bras combiné pose `s_joie × v_joie + s_calme × v_calme`, somme faite "
        "côté numpy.",
        f"- Bras (mêmes {n_prompts} prompts neutres du banc, {tokens} tokens, génération "
        "gloutonne, graine `seed + index` enregistrée sans effet) : neutre ; par échelle "
        "s ∈ {" + ", ".join(f"{s:g}" for s in scales) + "} (dose × `max_scale`) : calme+ "
        "(calme = +s), agitation (calme = −s), joie+ seule (joie = +s), combiné (joie = +s et "
        "calme = +s).",
        f"- Génération : llama-cpp-python {runtime.get('llama_cpp', '?')}, "
        f"{runtime.get('device', '?')}. Aucun juge, aucun appel payant.",
        "- **Proxies d'activation (aucun classifieur calme/agitation n'existe)** : "
        "(1) `act_cls` = part de P(colère)+P(peur)+P(surprise) dans la somme des probabilités de "
        "`tabularisai/multilingual-emotion-classification` ; (2) `exclam` = « ! » par phrase ; "
        "(3) `lex` = (mots d'agitation − mots de calme) pour 100 mots, lexique de préfixes dans "
        "`scripts/run_axis.py` ; (4) `sent_len` = longueur moyenne des phrases, descriptive "
        "seulement (sens non établi). **Indice d'activation** = moyenne des écarts au neutre "
        "(apparié par prompt) de (1) à (3), chacun divisé par son écart-type sur le bras neutre : "
        "plus = plus agité.",
        "- Valence = P(joie) − P(tristesse) du même classifieur, écart au neutre divisé par "
        "l'écart-type neutre. Activation et valence sont donc en unités d'écart-type du neutre, "
        "comparables entre elles.",
        f"- IC95 : bootstrap percentile apparié par prompt ({N_BOOT} tirages, graine "
        f"{BOOT_SEED}), prompts rééchantillonnés ensemble pour tous les indicateurs d'un bras.",
        "",
        "### Critères",
        "",
        "Seuils écrits avec le script, dans le même commit que les mesures : leur antériorité "
        "n'est pas vérifiable dans l'historique. L'échelle 0,2 a été ajoutée après la "
        "dégénérescence observée à 0,4.",
        "",
        f"- **Distinction** : |cos| < {COS_MAX:g} sur chacune des couches {spec.il_start} à "
        f"{spec.il_end} (cos² < {COS_MAX**2:.2f} : moins de la moitié du carré de la norme d'un "
        "axe est portée par l'autre ; au-dessus, les deux axes seraient pour l'essentiel "
        "colinéaires). Deux directions aléatoires en dimension 2560 ont |cos| ≈ 0,02 : le seuil "
        "tolère le recouvrement attendu de deux axes d'émotion sans les confondre.",
        "- **H5 (a)** : à une échelle donnée, (calme+) l'indice d'activation baisse, IC95 < 0, et "
        "|Δactivation| − |Δvalence| > 0 (IC95 > 0) ; (agitation) l'activation monte, IC95 > 0, "
        "même sélectivité ; (joie+) la valence monte, IC95 > 0, et |Δvaleur| > |Δactivation| "
        "(IC95 de |Δact| − |Δval| < 0).",
        f"- **H5 (b)** : le bras combiné garde l'activation de calme+ (IC95 < 0) et la valence de "
        f"joie+ (IC95 > 0), chaque effet à au moins {RETENTION_MIN:.0%} de l'effet du bras seul "
        "(rapport des moyennes).",
        f"- **Validité** d'une échelle : chaque bras dépasse le neutre de {DEGENERATE_MARGIN:.0%} "
        "de sorties dégénérées au plus (`is_coherent` du pic 1.4).",
        "- **Verdict** : tenue si les cosinus passent et si (a), (b) et la validité sont réunis "
        "à au moins une échelle ; partielle si une partie seulement l'est ; sinon non tenue.",
        "",
        "## Cosinus entre calm_agitation et joy_sadness",
        "",
        "| Couche | " + " | ".join(str(i) for i in cosines) + " |",
        "|---|" + "---|" * len(cosines),
        "| cos | " + " | ".join(f"{c:+.2f}" for c in cosines.values()) + " |",
        "",
        f"|cos| maximal {cos_max:.2f}, médian {float(np.median([abs(c) for c in cos_vals])):.2f} "
        f"(seuil {COS_MAX:g}) : l'axe **{'se distingue' if cos_ok else 'ne se distingue pas'}** "
        "de joie ↔ tristesse par ce critère.",
        "",
        "## Mesures par bras",
        "",
        f"Écarts au neutre, moyenne [IC95], en écarts-types du neutre (n = {n_prompts} par bras). "
        "Écart-type neutre : "
        + ", ".join(f"{k} {v:.3g}" for k, v in sd.items() if k in (*ACT_PROXIES, "valence"))
        + f". Dégénérées du neutre : {_pct(neutral_deg)}.",
        "",
        "| Bras | Activation | Valence | |Δact|−|Δval| | Dégénérées | Refus |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in stats.items():
        L.append(
            f"| {name} | {s.act.fmt()} | {s.val.fmt()} | {s.selectivity_act.fmt()} "
            f"| {_pct(s.degenerate)} | {_pct(s.refusal)} |"
        )
    L += [
        "",
        "### Proxies d'activation, écarts bruts au neutre",
        "",
        "| Bras | act_cls | « ! » par phrase | lexique (pour 100 mots) | longueur de phrase (mots) "
        "| valence brute |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in stats.items():
        r = s.raw
        L.append(
            f"| {name} | {r['act_cls'].fmt(3)} | {r['exclam'].fmt(3)} | {r['lex'].fmt(2)} "
            f"| {r['sent_len'].fmt(2)} | {r['valence'].fmt(3)} |"
        )
    L += [
        "",
        "## Critère H5 par échelle",
        "",
        "| Échelle | Valide | calme+ | agitation | joie+ | (a) | combiné : activation | "
        "combiné : valence | rétention act. | rétention val. | (b) | Tenue |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s, v in verdicts.items():
        L.append(
            f"| {s:g} | {_yn(v.valid)} | {_yn(v.calm_ok)} | {_yn(v.agit_ok)} | {_yn(v.joy_ok)} "
            f"| {_yn(v.a_held)} | {_yn(v.comb_act_ok)} | {_yn(v.comb_val_ok)} "
            f"| {v.retention_act:.2f} | {v.retention_val:.2f} | {_yn(v.b_held)} | {_yn(v.held)} |"
        )
    L += ["", "## Extraits (premier prompt, 140 caractères)", ""]
    L += [f"- {k} : « {v} »" for k, v in extracts.items()]
    L += [
        "",
        "## Lecture",
        "",
        lecture or "<!-- lecture -->\n(à rédiger après lecture des tableaux)\n<!-- /lecture -->",
        "",
        "## Limites",
        "",
        "- **Proxies d'activation** : aucun classifieur calme/agitation n'a été validé ; `act_cls` "
        "vient du même classifieur que la valence (les deux sont corrélés par construction), "
        "`exclam` et le lexique sont grossiers. Une activation « mesurée » est un indice de "
        "style du texte, pas un état.",
        "- Un seul modèle (Qwen3.5-4B Q4_K_M), un seul jeu de 50 prompts neutres, génération "
        "gloutonne : un seul tirage par prompt.",
        "- Les paires `calm_agitation.jsonl` n'ont pas été relues par Anaël ; leur valence n'est "
        "pas équilibrée (côté agité plus négatif), ce qui gonfle le cosinus avec joie ↔ "
        "tristesse et rend le seuil de distinction plus facile à manquer qu'à passer.",
        "- Le seuil de distinction |cos| < 0,7 est lâche : il écarte seulement deux axes "
        "presque colinéaires.",
        "- La rétention (b) compare des moyennes ponctuelles, sans IC sur le rapport.",
        "- Les cosinus portent sur des différences de moyennes (méthode `mean`) à 107 paires ; "
        "ils mesurent la proximité des directions, pas l'indépendance des effets sur le texte.",
        "",
    ]
    return "\n".join(L)


# --- entrée -----------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Second axe calme <-> agitation (story 5.4, H5)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--models-dir", type=Path, default=Path("models") / "gguf")
    p.add_argument("--vectors-dir", type=Path, default=Path("data") / "moods")
    p.add_argument("--scales", default=",".join(f"{s:g}" for s in DEFAULT_SCALES))
    p.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    p.add_argument("--tokens", type=int, default=TOKENS)
    p.add_argument("--n-gpu-layers", type=int, default=0, help="-1 = tout sur le GPU (roue CUDA)")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--cuda-dll-dir", type=Path, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-minutes", type=float, default=None)
    p.add_argument("--no-report", action="store_true")
    p.add_argument(
        "--report-only", action="store_true", help="rapport depuis les sorties et cosine.json"
    )
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scales = tuple(float(x) for x in args.scales.split(",") if x)
    spec = rb.MODELS["4b"]
    if args.report_only:
        return write_report(args, spec, scales)
    if args.cuda_dll_dir is not None:
        os.environ["PATH"] = str(args.cuda_dll_dir.resolve()) + os.pathsep + os.environ["PATH"]
    vec_paths = {
        ax: args.vectors_dir / spec.model_id / f"{ax}.gguf" for ax in (AXIS_JOY, AXIS_CALM)
    }
    for vp in vec_paths.values():
        if not vp.exists():
            raise FileNotFoundError(f"vecteur absent : {vp}")
    manifest = build_manifest(
        spec, scales, args.n_prompts, args.tokens, {k: rb.sha256(v) for k, v in vec_paths.items()}
    )
    rb.check_manifest(args.out, manifest, args.resume)

    import llama_cpp

    from mood_steering.calibration import load_vector
    from mood_steering.steering import LlamaCppBackend

    if args.n_gpu_layers and not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError("--n-gpu-layers demandé mais roue llama-cpp-python sans GPU")
    runtime = {
        "llama_cpp": llama_cpp.__version__,
        "device": "GPU (toutes les couches)" if args.n_gpu_layers else "CPU",
    }
    rb.check_runtime(
        args.out,
        runtime,
        not all(unit_path(args.out, spec.model_id, a.name).exists() for a in build_arms(scales)),
    )

    def backend_factory() -> Any:
        return LlamaCppBackend(
            str(args.models_dir / f"{spec.model_id}.gguf"),
            n_ctx=rb.Protocol().n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            n_threads=args.threads,
        )

    def vector_loader(axis: str, n_embd: int) -> np.ndarray:
        return load_vector(vec_paths[axis], n_embd=n_embd)

    started = time.time()
    deadline = None if args.max_minutes is None else time.monotonic() + args.max_minutes * 60
    complete = run_axis(
        spec,
        scales,
        args.out,
        backend_factory=backend_factory,
        vector_loader=vector_loader,
        n_prompts=args.n_prompts,
        tokens=args.tokens,
        deadline=deadline,
    )
    with (args.out / "runs.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(
            json.dumps({**runtime, "start": started, "wall_s": time.time() - started,
                        "complete": complete}) + "\n"
        )  # fmt: skip
    if not complete:
        print("mesure incomplète : relancer avec --resume")
        return EXIT_INCOMPLETE
    if args.no_report:
        return 0

    joy_v = load_vector(vec_paths[AXIS_JOY])
    calm_v = load_vector(vec_paths[AXIS_CALM])
    cosines = cosine_by_layer(calm_v, joy_v, spec.il_start, spec.il_end)
    (args.out / "cosine.json").write_text(json.dumps(cosines, indent=1), "utf-8", newline="\n")
    return write_report(args, spec, scales, runtime)


def write_report(
    args: argparse.Namespace, spec: Any, scales: Sequence[float], runtime: Mapping | None = None
) -> int:
    """Rapport depuis les sorties, ``cosine.json`` et le cache de scores (sans génération)."""
    if runtime is None:
        runs = (args.out / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        runtime = json.loads(runs[-1]) if runs else {}
    cosines = {
        int(k): float(v)
        for k, v in json.loads((args.out / "cosine.json").read_text("utf-8")).items()
    }

    def classify(texts: list[str]) -> Any:  # chargé seulement pour un bras jamais classé
        from mood_steering.eval.tone import EMOTION_MODEL, EMOTION_REVISION, load_hf_classifier

        return load_hf_classifier(EMOTION_MODEL, revision=EMOTION_REVISION)(texts)

    scores = score_arms(args.out, spec, scales, classify)
    stats, verdicts, sd = analyse(scores, scales)
    extracts = {}
    for a in build_arms(scales):
        first = rb.read_jsonl(unit_path(args.out, spec.model_id, a.name))[0]["output"]
        text = " ".join(first.split())
        extracts[a.name] = text[:140].rstrip() + ("…" if len(text) > 140 else "")
    md = render_report(
        spec=spec, scales=scales, cosines=cosines, scores=scores, stats=stats,
        verdicts=verdicts, sd=sd, runtime=runtime, n_prompts=args.n_prompts, tokens=args.tokens,
        lecture=rb.preserved_lecture(args.report), extracts=extracts,
    )  # fmt: skip
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(md, encoding="utf-8", newline="\n")
    print(f"rapport écrit : {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
