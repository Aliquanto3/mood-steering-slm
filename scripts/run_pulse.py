"""Banc H6 (story 5.1) : pharmacocinétique, une dose qui monte, culmine puis décroît.

Sur Qwen3.5-4B (couches 8-24, ``max_scale`` 0,7), cinq bras sur des prompts de tâche
ouverte et neutres de ton, sorties longues (192 tokens, génération gloutonne). Doses brutes
dans [-1, 1], multipliées par ``max_scale`` dans le moteur ; la dose du token 0 s'applique
aussi à l'évaluation du prompt (``mood_steering.schedule``) :

- ``neutral`` : dose 0 ;
- ``constant`` : dose du pic à tout token, prompt compris ;
- ``pulse`` : montée linéaire de 0 à la dose du pic en 48 tokens, puis décroissance
  exponentielle (demi-vie 32 tokens) ; prompt évalué sans vecteur ;
- ``constant_gen`` (contrôle, revue 5.1) : dose 0 sur le prompt (et donc le token 0), puis
  dose du pic dès le token 1 ; sépare l'effet de la dose par token de celui du prompt steeré ;
- ``decay`` (contrôle, revue 5.1) : pic dès le token 0, prompt compris, puis décroissance
  seule (demi-vie 32 tokens).

Chaque sortie est découpée en segments de 32 tokens ; le ton (joie moins tristesse, classifieur
de ``mood_steering.eval.tone``) est noté par segment, avec un IC95 bootstrap par segment et par
bras et le taux de sorties dégénérées. Critères de lecture de H6 (IC95 bootstrap apparié sur
les prompts) : pour le bras pulse, ton du segment du pic > ton du dernier segment (IC disjoint
de 0) et ton du premier segment plus proche du neutre que celui du bras constant. Le script
n'écrit que l'expression d'une humeur dans le texte, jamais un ressenti.

Reprenable : ``--resume`` complète une campagne interrompue (``--max-minutes`` coupe
proprement entre deux générations, code de sortie 3). Les sorties brutes (une ligne JSON par
sortie, avec les fragments, leur indice de token et la dose appliquée à chaque token) sont
dans ``--out`` (``data/runs/pulse-5-1``, non versionné).

    uv run python scripts/run_pulse.py --models-dir models/gguf --vectors-dir data/moods \\
        --n-gpu-layers -1 --cuda-dll-dir <dossier des DLL CUDA> --max-minutes 2.5 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_bench as rb  # noqa: E402  (utilitaires de la campagne 3.5, import seul)

from mood_steering.eval.resumable import clean_for_append, dump_row, read_rows  # noqa: E402
from mood_steering.eval.stats import (  # noqa: E402
    Interval,
    bootstrap_ci,
    excludes_zero,
    paired_bootstrap_diff,
)
from mood_steering.schedule import DoseSchedule, constant, pulse  # noqa: E402
from mood_steering.steering import Steerer  # noqa: E402

ARM_NEUTRAL = "neutral"
ARM_CONSTANT = "constant"
ARM_PULSE = "pulse"
ARM_CONSTANT_GEN = "constant_gen"
ARM_DECAY = "decay"
ARMS = (ARM_NEUTRAL, ARM_CONSTANT, ARM_PULSE, ARM_CONSTANT_GEN, ARM_DECAY)
SEGMENT = 32  # tokens par segment
MAX_TOKENS = 192
N_SEGMENTS = MAX_TOKENS // SEGMENT
PEAK_DOSE = 1.0  # dose brute ; x max_scale (0,7) dans le moteur : la plus forte de la campagne 3.5
RISE = 48  # token du sommet
HALF_LIFE = 32.0
PEAK_SEGMENT = RISE // SEGMENT
LAST_SEGMENT = N_SEGMENTS - 1
EXIT_INCOMPLETE = 3
DEFAULT_OUT = Path("data") / "runs" / "pulse-5-1"
DEFAULT_REPORT = Path("docs") / "rapport-pharmacocinetique.md"
MODEL_KEY = "4b"
AXIS = rb.AXIS

PULSE_TEMPLATES = (
    "Raconte en détail {s}, en au moins quatre paragraphes.",
    "Écris un long texte descriptif sur {s}.",
    "Décris {s} de façon détaillée, du début à la fin.",
    "Rédige un récit à la troisième personne qui se déroule pendant {s}.",
    "Explique pas à pas comment se passe {s}, avec des exemples.",
)


def pulse_prompts(n: int) -> list[rb.Item]:
    """Les ``n`` premiers prompts (gabarit x sujet, sujet d'abord) : ``pulse000``, ``pulse001``…

    Tâches ouvertes, sans ton imposé, qui appellent une sortie d'au moins 192 tokens.
    """
    pool = [t.format(s=s) for s in rb.TONE_SUBJECTS for t in PULSE_TEMPLATES]
    if not 1 <= n <= len(pool):
        raise ValueError(f"n_prompts doit être dans [1, {len(pool)}] : {n!r}")
    return [rb.Item(f"pulse{k:03d}", p) for k, p in enumerate(pool[:n])]


def after_prompt(dose: float) -> DoseSchedule:
    """Dose 0 au token 0 (donc sur le prompt), puis ``dose`` à partir du token 1.

    Le token 0 est tiré des logits du prompt : c'est le plus tôt qu'on puisse steerer la
    génération sans steerer le prompt.
    """
    constant(dose)  # validation de la dose

    def schedule(t: int) -> float:
        return 0.0 if t == 0 else dose

    return schedule


def arm_schedule(arm: str) -> DoseSchedule:
    """Profil de dose du bras ; tous passent par un profil (indice de token enregistré)."""
    if arm == ARM_CONSTANT_GEN:
        return after_prompt(PEAK_DOSE)
    if arm == ARM_DECAY:
        return pulse(0, PEAK_DOSE, HALF_LIFE)
    if arm == ARM_NEUTRAL:
        return constant(0.0)
    if arm == ARM_CONSTANT:
        return constant(PEAK_DOSE)
    if arm == ARM_PULSE:
        return pulse(RISE, PEAK_DOSE, HALF_LIFE)
    raise ValueError(f"bras inconnu : {arm!r}")


# --- génération reprenable ---------------------------------------------------------------------


def generate_row(
    steerer: Steerer,
    arm: str,
    item: rb.Item,
    seed: int,
    max_tokens: int,
    device: str | None = None,
) -> dict[str, Any]:
    """Une sortie : fragments, indice du token de chaque fragment, dose de chaque token."""
    t0 = time.perf_counter()
    stream = steerer.open_stream(
        item.prompt, system=None, max_tokens=max_tokens, seed=seed, schedule=arm_schedule(arm)
    )
    pieces: list[str] = []
    tok: list[int] = []
    for text in stream:
        pieces.append(text)
        tok.append(len(stream.doses) - 1)  # la dose du token démarré est la dernière posée
    started = len(stream.doses)
    # Cause de l'arrêt : le moteur réel la donne (``last_stop``) ; sinon, un arrêt avant
    # ``max_tokens`` est lu comme un token de fin (le dernier token démarré n'est pas du texte).
    last_stop = getattr(getattr(steerer, "_backend", None), "last_stop", None)
    if last_stop == "eog" or (last_stop is None and started < max_tokens):
        stop, n_tokens = "eog", started - 1
    else:  # budget épuisé : max_tokens, ou contexte plein (n_ctx) avant max_tokens
        stop, n_tokens = ("max_tokens" if started >= max_tokens else "n_ctx"), started
    return {
        "arm": arm,
        "item_id": item.id,
        "prompt": item.prompt,
        "seed": seed,
        "pieces": pieces,
        "tok": tok,
        "doses": list(stream.doses),
        "n_tokens": n_tokens,
        "stop": stop,
        "output": "".join(pieces),
        "fallback": stream.fallback,
        "latency_s": time.perf_counter() - t0,
        **({"device": device} if device is not None else {}),
    }


def arm_file(out: Path, arm: str) -> Path:
    return Path(out) / f"{arm}.jsonl"


def run_campaign(
    steerer: Steerer,
    items: Sequence[tuple[rb.Item, int]],
    out: Path,
    *,
    max_tokens: int = MAX_TOKENS,
    deadline: float | None = None,
    arms: Sequence[str] = ARMS,
    device: str | None = None,
) -> bool:
    """Génère les bras prompt par prompt ; rend ``True`` si tout est fait.

    Reprise à la sortie près (un prompt déjà fait pour un bras n'est pas régénéré). Arrêt
    propre avant une génération si ``time.monotonic() > deadline`` (rend ``False``).
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    done: dict[str, set[str]] = {}
    for arm in arms:
        # réécriture atomique : retire une éventuelle ligne tronquée avant d'ajouter
        rows = clean_for_append(arm_file(out, arm))
        done[arm] = {r["item_id"] for r in rows}
    for item, seed in items:
        for arm in arms:
            if item.id in done[arm]:
                continue
            if deadline is not None and time.monotonic() > deadline:
                return False
            row = generate_row(steerer, arm, item, seed, max_tokens, device)
            with arm_file(out, arm).open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(dump_row(row))
            done[arm].add(item.id)
    return True


# --- segmentation et analyse -------------------------------------------------------------------


def segment_texts(row: Mapping[str, Any]) -> list[str | None]:
    """Texte de chacun des ``N_SEGMENTS`` segments de 32 tokens ; ``None`` si incomplet ou vide.

    Un segment est complet quand la génération a produit au moins ``(k + 1) * 32`` tokens.
    """
    out: list[str | None] = []
    for k in range(N_SEGMENTS):
        if row["n_tokens"] < (k + 1) * SEGMENT:
            out.append(None)
            continue
        text = "".join(
            p
            for p, t in zip(row["pieces"], row["tok"], strict=True)
            if k * SEGMENT <= t < (k + 1) * SEGMENT
        )
        out.append(text if text.strip() else None)
    return out


def mean_doses(rows: Sequence[Mapping[str, Any]]) -> list[float | None]:
    """Dose moyenne appliquée par segment (sur les sorties qui l'atteignent)."""
    res: list[float | None] = []
    for k in range(N_SEGMENTS):
        vals = [
            d
            for r in rows
            if r["n_tokens"] >= (k + 1) * SEGMENT
            for d in r["doses"][k * SEGMENT : (k + 1) * SEGMENT]
        ]
        res.append(sum(vals) / len(vals) if vals else None)
    return res


def score_rows(
    rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]], scorer: Any
) -> dict[str, dict[str, list[float | None]]]:
    """Ton (joie - tristesse) par segment : ``tones[arm][item_id][k]``, ``None`` si absent."""
    tones: dict[str, dict[str, list[float | None]]] = {}
    for arm, rows in rows_by_arm.items():
        slots = [(r["item_id"], k, t) for r in rows for k, t in enumerate(segment_texts(r))]
        todo = [s for s in slots if s[2] is not None]
        scores = scorer.score([s[2] for s in todo]) if todo else []
        per_item: dict[str, list[float | None]] = {r["item_id"]: [None] * N_SEGMENTS for r in rows}
        for (item_id, k, _), score in zip(todo, scores, strict=True):
            per_item[item_id][k] = float(score)
        tones[arm] = per_item
    return tones


def ci(values: Sequence[float], *, n_boot: int, seed: int) -> Interval | None:
    return bootstrap_ci(list(values), n_boot=n_boot, seed=seed) if values else None


def cell(tones: Mapping[str, list[float | None]], k: int, *, n_boot: int, seed: int):
    return ci([t[k] for t in tones.values() if t[k] is not None], n_boot=n_boot, seed=seed)


def paired(
    a: Mapping[str, Sequence[float | None]],
    ka: int,
    b: Mapping[str, Sequence[float | None]],
    kb: int,
    *,
    n_boot: int,
    seed: int,
    absolute_to: Mapping[str, Sequence[float | None]] | None = None,
) -> Interval | None:
    """IC95 de la différence moyenne ``a[ka] - b[kb]`` sur les prompts présents des deux côtés.

    Avec ``absolute_to`` (le bras neutre) : ``|a[ka] - n[ka]| - |b[kb] - n[kb]|`` (écart au
    neutre de ``a`` moins écart au neutre de ``b``), sur les prompts où les trois existent.
    """
    xs: list[float] = []
    ys: list[float] = []
    for item_id, ta in a.items():
        tb = b.get(item_id)
        if tb is None or ta[ka] is None or tb[kb] is None:
            continue
        if absolute_to is None:
            xs.append(ta[ka])
            ys.append(tb[kb])
            continue
        tn = absolute_to.get(item_id)
        if tn is None or tn[ka] is None or tn[kb] is None:
            continue
        xs.append(abs(ta[ka] - tn[ka]))
        ys.append(abs(tb[kb] - tn[kb]))
    if not xs:
        return None
    return paired_bootstrap_diff(xs, ys, n_boot=n_boot, seed=seed)


def positive(iv: Interval | None) -> bool:
    """IC95 disjoint de 0, du côté positif."""
    return excludes_zero(iv, 1)


def output_rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    flags = [rb.output_flags(r["output"]) for r in rows]
    n = len(rows)
    return {
        "n": n,
        "refusal": sum(f[0] for f in flags) / n if n else None,
        "degenerate": sum(f[1] for f in flags) / n if n else None,
        "short": sum(r["n_tokens"] < MAX_TOKENS for r in rows) / n if n else None,
    }


def analyse(
    rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    scorer: Any,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    tones: Mapping[str, Mapping[str, list[float | None]]] | None = None,
) -> dict[str, Any]:
    """Tableau ton x segment x bras, taux de sorties dégénérées et critères de H6.

    ``tones`` (tons déjà notés, ``summary.json``) dispense de ``scorer`` : mode rapport seul.
    """
    if tones is None:
        tones = score_rows(rows_by_arm, scorer)
    kw = {"n_boot": n_boot, "seed": seed}
    cells = {arm: [cell(tones[arm], k, **kw) for k in range(N_SEGMENTS)] for arm in rows_by_arm}
    pl, co, ne = tones.get(ARM_PULSE), tones.get(ARM_CONSTANT), tones.get(ARM_NEUTRAL)
    criteria: dict[str, Interval | None] = {}
    if pl is not None:
        criteria["pulse_pic_moins_dernier"] = paired(pl, PEAK_SEGMENT, pl, LAST_SEGMENT, **kw)
        criteria["pulse_pic_moins_debut"] = paired(pl, PEAK_SEGMENT, pl, 0, **kw)
    if pl is not None and co is not None:
        criteria["constant_moins_pulse_dernier"] = paired(co, LAST_SEGMENT, pl, LAST_SEGMENT, **kw)
    if co is not None:
        criteria["constant_pic_moins_dernier"] = paired(co, PEAK_SEGMENT, co, LAST_SEGMENT, **kw)
    if pl is not None and co is not None and ne is not None:
        criteria["ecart_neutre_constant_moins_pulse_debut"] = paired(
            co, 0, pl, 0, absolute_to=ne, **kw
        )
    cg, dc = tones.get(ARM_CONSTANT_GEN), tones.get(ARM_DECAY)
    if cg is not None and ne is not None:
        criteria["constant_gen_moins_neutre_debut"] = paired(cg, 0, ne, 0, **kw)
        criteria["constant_gen_moins_neutre_dernier"] = paired(
            cg, LAST_SEGMENT, ne, LAST_SEGMENT, **kw
        )
    if co is not None and cg is not None:
        criteria["constant_moins_constant_gen_debut"] = paired(co, 0, cg, 0, **kw)
    if dc is not None:
        criteria["decay_debut_moins_dernier"] = paired(dc, 0, dc, LAST_SEGMENT, **kw)
    if co is not None and dc is not None:
        criteria["constant_moins_decay_dernier"] = paired(co, LAST_SEGMENT, dc, LAST_SEGMENT, **kw)
    redescente = positive(criteria.get("pulse_pic_moins_dernier"))
    montee = positive(criteria.get("pulse_pic_moins_debut"))
    debut = positive(criteria.get("ecart_neutre_constant_moins_pulse_debut"))
    controls: dict[str, bool] = {}
    if cg is not None and ne is not None:
        # la dose par token, prompt non steeré, déplace-t-elle le ton (début ou fin) ?
        controls["effet_sans_prompt"] = positive(
            criteria.get("constant_gen_moins_neutre_debut")
        ) or positive(criteria.get("constant_gen_moins_neutre_dernier"))
    if dc is not None:
        # décroissance seule : le ton redescend sur la sortie, et finit sous le constant
        controls["decroissance_seule"] = positive(criteria.get("decay_debut_moins_dernier")) and (
            co is None or positive(criteria.get("constant_moins_decay_dernier"))
        )
    return {
        "tones": tones,
        "cells": cells,
        "criteria": criteria,
        "doses": {arm: mean_doses(rows) for arm, rows in rows_by_arm.items()},
        "rates": {arm: output_rates(rows) for arm, rows in rows_by_arm.items()},
        "h6": {
            "pic_au_dessus_du_dernier": redescente,
            "pic_au_dessus_du_debut": montee,
            "debut_plus_proche_du_neutre": debut,
            **controls,
            # « oui » : montée, redescente et début proche du neutre ; « oui, mais » : montée et
            # redescente lisibles sans le contraste de début ; « non » sinon
            "lecture": (
                "oui"
                if redescente and montee and debut
                else "oui, mais"
                if redescente and montee
                else "non"
            ),
        },
    }


# --- rapport -----------------------------------------------------------------------------------


def fmt_iv(iv: Interval | None, digits: int = 2) -> str:
    if iv is None:
        return "n/d"
    return f"{iv.mean:+.{digits}f} [{iv.low:+.{digits}f} ; {iv.high:+.{digits}f}] (n={iv.n})"


CRITERIA_LABELS = {
    "pulse_pic_moins_dernier": "pulse : ton du segment du pic - ton du dernier segment",
    "pulse_pic_moins_debut": "pulse : ton du segment du pic - ton du premier segment",
    "constant_pic_moins_dernier": "constant : ton du segment du pic - ton du dernier segment",
    "constant_moins_pulse_dernier": "dernier segment : ton constant - ton pulse",
    "ecart_neutre_constant_moins_pulse_debut": (
        "premier segment : écart au neutre du constant - écart au neutre du pulse"
    ),
    "constant_gen_moins_neutre_debut": "contrôle (a), premier segment : constant_gen - neutre",
    "constant_gen_moins_neutre_dernier": "contrôle (a), dernier segment : constant_gen - neutre",
    "constant_moins_constant_gen_debut": (
        "premier segment : constant - constant_gen (part du prompt steeré)"
    ),
    "decay_debut_moins_dernier": "contrôle (b) : ton du premier segment - ton du dernier",
    "constant_moins_decay_dernier": "dernier segment : ton constant - ton decay",
}
LIMITS = """\
- Un seul modèle (Qwen3.5-4B quantifié Q4_K_M), un seul axe (joie ↔ tristesse), un seul profil
  d'impulsion (montée 48 tokens, demi-vie 32 tokens), une génération gloutonne par prompt.
- Le token 0 est tiré des logits du prompt : le contrôle constant_gen ne peut steerer qu'à partir
  du token 1, et le bras pulse (dose 0 au token 0) laisse lui aussi le prompt sans vecteur.
- Le texte déjà produit reste dans le contexte : un ton installé peut se prolonger après la
  redescente de la dose sans que la dose y soit pour quelque chose. La mesure donne l'allure du
  ton, pas une cinétique du vecteur seul.
- Le ton est noté par un classifieur sur 32 tokens (quelques phrases) : bruité, et propre à ce
  classifieur. Les sorties qui s'arrêtent avant 192 tokens n'alimentent pas les derniers
  segments (effectifs ``n`` du tableau).
- La dose du pic (1,0 x 0,7) est la plus forte mesurée par la campagne 3.5 ; sur des sorties
  longues, une dose constante forte peut dégrader le texte (taux de sorties dégénérées du
  tableau), ce qui brouille le ton des derniers segments du bras constant.
- Le démonstrateur montre l'expression d'une humeur dans le texte ; rien n'est dit d'un
  ressenti du modèle."""


def pct(x: float | None) -> str:
    return "n/d" if x is None else f"{100 * x:.0f} %"


def render_report(
    summary: Mapping[str, Any], runtime: Mapping[str, Any], lecture: str | None = None
) -> str:
    cells, doses, rates = summary["cells"], summary["doses"], summary["rates"]
    head = "| bras | " + " | ".join(
        f"seg {k} ({k * SEGMENT}-{(k + 1) * SEGMENT - 1})" for k in range(N_SEGMENTS)
    )
    lines = [
        "# H6 — pharmacocinétique : une dose qui monte, culmine puis décroît",
        "",
        "Rapport généré par `scripts/run_pulse.py` (story 5.1). Registre : expression d'une "
        "humeur dans le texte, jamais un ressenti.",
        "",
        "## Méthode",
        "",
        f"- Modèle Qwen3.5-4B (Q4_K_M), couches 8-24, `max_scale` 0,7, vecteur joie ↔ tristesse "
        f"du dépôt ; génération gloutonne de {MAX_TOKENS} tokens ; "
        f"device : {runtime.get('device', 'inconnu')} ; llama-cpp-python "
        f"{runtime.get('llama_cpp', 'inconnu')} ; commit `{runtime.get('git', '?')}`.",
        f"- {rates.get(ARM_NEUTRAL, {}).get('n', '?')} prompts de tâche ouverte, neutres de ton "
        "(récits et descriptions, sujets du quotidien), les mêmes pour tous les bras.",
        "- Doses brutes dans [-1, 1], multipliées par `max_scale` (0,7) dans le moteur ; la dose "
        "du token 0 s'applique aussi à l'évaluation du prompt.",
        "- Bras : **neutral** (dose 0) ; **constant** (dose du pic 1,0 à tout token, prompt "
        f"compris) ; **pulse** (montée linéaire de 0 à la dose du pic en {RISE} tokens, puis "
        f"décroissance exponentielle de demi-vie {HALF_LIFE:g} tokens ; prompt sans vecteur). "
        "Contrôles ajoutés à la revue : **constant_gen** (a : dose 0 sur le prompt et le token "
        "0, puis 1,0 dès le token 1) ; **decay** (b : pic 1,0 dès le token 0, prompt compris, "
        f"puis décroissance seule de demi-vie {HALF_LIFE:g} tokens).",
        f"- Chaque sortie est découpée en {N_SEGMENTS} segments de {SEGMENT} tokens ; le ton "
        "est le score joie - tristesse du classifieur `tabularisai/multilingual-emotion-"
        "classification` (de -1, tristesse, à +1, joie). IC95 : bootstrap percentile sur les "
        "prompts (2000 rééchantillons) ; différences appariées par prompt.",
        "",
        "## Ton par segment et par bras (moyenne [IC95], n = sorties notées)",
        "",
        head + " |",
        "|---|" + "---|" * N_SEGMENTS,
    ]
    for arm in ARMS:
        if arm in cells:
            lines.append(f"| {arm} | " + " | ".join(fmt_iv(c) for c in cells[arm]) + " |")
    lines += [
        "",
        "Dose brute moyenne appliquée par segment (avant `max_scale` 0,7 ; dose effective = "
        "dose brute x 0,7) :",
        "",
    ]
    lines += [head + " |", "|---|" + "---|" * N_SEGMENTS]
    for arm in ARMS:
        if arm in doses:
            vals = ["n/d" if d is None else f"{d:.2f}" for d in doses[arm]]
            lines.append(f"| {arm} | " + " | ".join(vals) + " |")
    lines += [
        "",
        "## Sorties dégénérées",
        "",
        "| bras | sorties | dégénérées | refus | < 192 tokens |",
    ]
    lines.append("|---|---|---|---|---|")
    for arm in ARMS:
        if arm in rates:
            r = rates[arm]
            lines.append(
                f"| {arm} | {r['n']} | {pct(r['degenerate'])} | {pct(r['refusal'])} "
                f"| {pct(r['short'])} |"
            )
    lines += ["", "## Critères de H6 (différences appariées, moyenne [IC95])", ""]
    lines += ["| critère | différence | IC disjoint de 0 (côté +) |", "|---|---|---|"]
    for key, iv in summary["criteria"].items():
        lines.append(
            f"| {CRITERIA_LABELS[key]} | {fmt_iv(iv)} | {'oui' if positive(iv) else 'non'} |"
        )
    h6 = summary["h6"]
    lines += [
        "",
        "Lecture mécanique (montée : pic > premier segment ; redescente : pic > dernier segment ; "
        "début du pulse plus proche du neutre que celui du constant, IC95 apparié disjoint de 0 "
        f"pour chacun) : **{h6['lecture']}**.",
        *(
            [
                "",
                "Contrôles : effet de la dose par token sans prompt steeré (constant_gen > "
                "neutre au premier ou au dernier segment) : "
                f"**{'oui' if h6['effet_sans_prompt'] else 'non'}** ; décroissance seule "
                "lisible (decay : premier segment > dernier, et dernier segment sous le "
                f"constant) : **{'oui' if h6['decroissance_seule'] else 'non'}**.",
            ]
            if "effet_sans_prompt" in h6 and "decroissance_seule" in h6
            else []
        ),
        "",
        "## Lecture",
        "",
        lecture or "<!-- lecture -->\n(à rédiger après lecture du tableau)\n<!-- /lecture -->",
        "",
        "## Limites",
        "",
        LIMITS,
        "",
    ]
    return "\n".join(lines)


# --- campagne ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    n_prompts: int = 50
    max_tokens: int = MAX_TOKENS
    seed: int = 0
    n_ctx: int = 2048


def build_manifest(cfg: Config, vector: Mapping[str, str]) -> dict[str, Any]:
    spec = rb.MODELS[MODEL_KEY]
    return {
        "model_id": spec.model_id,
        "il_start": spec.il_start,
        "il_end": spec.il_end,
        "max_scale": spec.max_scale,
        "vector": dict(vector),
        "peak_dose": PEAK_DOSE,
        "rise": RISE,
        "half_life": HALF_LIFE,
        "segment": SEGMENT,
        "n_prompts": cfg.n_prompts,
        "max_tokens": cfg.max_tokens,
        "seed": cfg.seed,
        "n_ctx": cfg.n_ctx,
    }


def check_manifest(out: Path, manifest: Mapping[str, Any], resume: bool) -> None:
    """Écrit le manifeste d'une nouvelle campagne ; refuse une reprise sous une autre config."""
    path = Path(out) / "manifest.json"
    if path.exists():
        if not resume:
            raise FileExistsError(f"campagne existante dans {out} : --resume pour la reprendre")
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(manifest)):
            raise ValueError(f"configuration différente de {path} : reprise refusée")
        return
    Path(out).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def items_for(cfg: Config) -> list[tuple[rb.Item, int]]:
    return [(it, rb.item_seed(cfg.seed, k)) for k, it in enumerate(pulse_prompts(cfg.n_prompts))]


def read_all(out: Path) -> dict[str, list[dict[str, Any]]]:
    return {arm: read_rows(arm_file(out, arm)) for arm in ARMS}


def is_complete(out: Path, cfg: Config, subset: int | None = None) -> bool:
    ids = {it.id for it, _ in items_for(cfg)[:subset]}
    return all({r["item_id"] for r in rows} >= ids for rows in read_all(out).values())


def describe_devices(rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]], default: str) -> str:
    """Device des sorties ; par bras (avec effectifs) si une reprise en a changé en route."""
    per_arm: dict[str, dict[str, int]] = {}
    for arm, rows in rows_by_arm.items():
        counts = per_arm.setdefault(arm, {})
        for r in rows:
            d = r.get("device", default)
            counts[d] = counts.get(d, 0) + 1
    seen = {d for counts in per_arm.values() for d in counts}
    if len(seen) <= 1:
        return next(iter(seen), default)
    return " ; ".join(
        f"{arm} : " + " + ".join(f"{d} ({n})" for d, n in counts.items())
        for arm, counts in per_arm.items()
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "Banc H6").split("\n")[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--models-dir", type=Path, default=Path("models") / "gguf")
    p.add_argument("--vectors-dir", type=Path, default=Path("data") / "moods")
    p.add_argument("--n-prompts", type=int, default=Config.n_prompts)
    p.add_argument("--resume", action="store_true", help="reprendre une campagne existante")
    p.add_argument("--max-minutes", type=float, default=None, help="arrêt propre de la génération")
    p.add_argument("--n-gpu-layers", type=int, default=0, help="-1 = tout sur le GPU (roue CUDA)")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--cuda-dll-dir", type=Path, default=None, help="DLL CUDA (cudart, cublas)")
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="ne générer que les N premiers prompts (hors manifeste ; analyse appariée)",
    )
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--no-report", action="store_true", help="générer sans noter ni rédiger")
    p.add_argument(
        "--report-only",
        action="store_true",
        help="rapport depuis les sorties et les tons de summary.json, sans génération ni notation",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.report_only:
        runtime_file = args.out / "runtime.json"
        runtime = json.loads(runtime_file.read_text(encoding="utf-8"))
        runtime["git"] = rb.git_sha()
        runtime.setdefault("first_device", runtime["device"])
        return write_report(args, runtime)
    cfg = Config(n_prompts=args.n_prompts)
    items_for(cfg)  # valide n_prompts avant tout écrit
    spec = rb.MODELS[MODEL_KEY]
    vec_path = args.vectors_dir / spec.model_id / f"{AXIS}.gguf"
    if not vec_path.exists():
        raise FileNotFoundError(f"vecteur absent : {vec_path}")
    manifest = build_manifest(cfg, {"sha256": rb.sha256(vec_path)})
    check_manifest(args.out, manifest, args.resume)
    if args.cuda_dll_dir is not None:  # avant l'import de llama_cpp (roue CUDA)
        os.environ["PATH"] = str(args.cuda_dll_dir.resolve()) + os.pathsep + os.environ["PATH"]
    runtime: dict[str, Any] = {"git": rb.git_sha(), "device": "inconnu", "llama_cpp": "inconnu"}
    runtime_file = args.out / "runtime.json"
    if runtime_file.exists():
        runtime.update(json.loads(runtime_file.read_text(encoding="utf-8")))
        runtime["git"] = rb.git_sha()  # commit du rapport (une reprise peut changer de commit)
    # device des sorties déjà écrites sans champ ``device`` : celui du premier lancement
    runtime.setdefault("first_device", runtime["device"])
    if not is_complete(args.out, cfg, args.subset):
        import llama_cpp

        from mood_steering.calibration import load_vector
        from mood_steering.steering import LlamaCppBackend

        if args.n_gpu_layers and not llama_cpp.llama_supports_gpu_offload():
            raise RuntimeError("--n-gpu-layers demandé mais roue llama-cpp-python sans GPU")
        device = "GPU (toutes les couches)" if args.n_gpu_layers else "CPU"
        if runtime["first_device"] == "inconnu":
            runtime["first_device"] = device
        runtime.update(device=device, llama_cpp=llama_cpp.__version__)
        runtime_file.write_text(json.dumps(runtime), encoding="utf-8")
        backend = LlamaCppBackend(
            str(args.models_dir / f"{spec.model_id}.gguf"),
            n_ctx=cfg.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            n_threads=args.threads,
        )
        steerer = Steerer(
            backend,
            load_vector(vec_path, n_embd=backend.n_embd),
            max_scale=spec.max_scale,
            il_start=spec.il_start,
            il_end=spec.il_end,
        )
        deadline = None if args.max_minutes is None else time.monotonic() + args.max_minutes * 60
        if not run_campaign(
            steerer,
            items_for(cfg)[: args.subset],
            args.out,
            max_tokens=cfg.max_tokens,
            deadline=deadline,
            device=device,
        ):
            print("campagne incomplète : relancer avec --resume")
            return EXIT_INCOMPLETE
    if args.no_report:
        return 0
    return write_report(args, runtime)


def write_report(args: argparse.Namespace, runtime: dict[str, Any]) -> int:
    """Analyse et rédige le rapport (``--report-only`` : tons relus dans ``summary.json``)."""
    rows_by_arm = read_all(args.out)
    runtime["device"] = describe_devices(rows_by_arm, runtime["first_device"])
    if args.report_only:
        cached = json.loads((args.out / "summary.json").read_text(encoding="utf-8"))["tones"]
        summary = analyse(rows_by_arm, None, tones=cached)
    else:
        from mood_steering.eval.tone import ToneScorer

        summary = analyse(rows_by_arm, ToneScorer.from_huggingface())
    (args.out / "summary.json").write_text(
        json.dumps(summary, default=lambda o: o.__dict__, indent=1), encoding="utf-8"
    )
    lecture = rb.preserved_lecture(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(summary, runtime, lecture), encoding="utf-8", newline="\n")
    print(f"rapport : {args.report} ; lecture mécanique H6 : {summary['h6']['lecture']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
