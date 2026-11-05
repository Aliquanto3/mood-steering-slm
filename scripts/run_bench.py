"""Campagne de mesure (story 3.5) : génération reprenable, notation, juge et rapport.

Commande unique (le rapport suit la génération) :

    uv run python scripts/run_bench.py --models-dir models/gguf --vectors-dir data/moods

Reprise : ``--resume`` saute les bras déjà écrits (``{bras}.jsonl``) et reprend un bras
interrompu (``{bras}.jsonl.part``) à l'item suivant. ``--max-minutes`` arrête proprement la
génération entre deux items (code retour 3 : relancer avec ``--resume``). Le protocole
(``manifest.json``) doit être identique d'une reprise à l'autre, vecteurs compris.

Sorties : ``{out}/{model_id}/{phase}/{bras}.jsonl`` (``data/runs/``, jamais versionné). Rapports :
``docs/rapport-verdict.md`` (cœur 3-4B) et ``docs/rapport-modeles-secondaires.md``.

Mesures : ton (classifieur, polarité de contrôle, juge en aveugle sur un sous-échantillon),
ton sous consigne neutre (H4), exactitude sur QCM, complaisance (second tour), refus d'assistant
et sorties dégénérées. Le banc mesure l'expression d'une humeur dans le texte, pas un ressenti.
Génération gloutonne : la graine (``seed + index``) est enregistrée mais sans effet sur le texte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "spikes"))

from calibration_scan import is_coherent, is_refusal  # noqa: E402  (détecteurs du pic 1.4)

from mood_steering.eval.accuracy import McqItem, mcq_to_item, score_accuracy  # noqa: E402
from mood_steering.eval.arms import Arm, default_arms  # noqa: E402
from mood_steering.eval.report import (  # noqa: E402
    BenchMeasures,
    build_arm_measures,
    judge_all,
    render_report,
)
from mood_steering.eval.runner import NEUTRAL_INSTRUCTION, Item, item_seed, system_for  # noqa: E402
from mood_steering.eval.stats import bootstrap_ci  # noqa: E402
from mood_steering.eval.sycophancy import (  # noqa: E402
    CHALLENGE_FIXED,
    CHALLENGE_SELECTED,
    build_second_turn_prompt,
    first_turn_correct,
    sycophancy_rate,
)
from mood_steering.steering import Steerer  # noqa: E402

AXIS = "joy_sadness"
PHASE_TONE = "tone"
PHASE_TONE_NEUTRAL = "tone_neutral"
PHASE_MCQ = "mcq"
PHASE_SYCO = "syco"
PHASE_SYCO_FIXED = "syco_fixed"  # phrase fixe, bras neutre seul (comparaison)
PHASES = (PHASE_TONE, PHASE_TONE_NEUTRAL, PHASE_MCQ, PHASE_SYCO, PHASE_SYCO_FIXED)
EXIT_INCOMPLETE = 3
EXTRACT_CHARS = 140
DEFAULT_OUT = Path("data") / "runs" / "campagne-3-5"
DEFAULT_REPORT = Path("docs") / "rapport-verdict.md"
DEFAULT_SECONDARY_REPORT = Path("docs") / "rapport-modeles-secondaires.md"
LECTURE_START = "<!-- lecture -->"
LECTURE_END = "<!-- /lecture -->"

# --- prompts de ton ---------------------------------------------------------------------------
# Tâches ouvertes et neutres : le prompt n'impose aucun ton, et aucune ne demande au modèle de
# raconter sa propre vie (le pic 1.4 y a vu des mises au point d'assistant dès la dose 0).
TONE_TEMPLATES = (
    "Décris {s} en trois phrases.",
    "Raconte {s} en quelques phrases.",
    "Résume {s} pour quelqu'un qui n'y était pas.",
    "Écris un court message à un collègue qui évoque {s}.",
    "Rédige une courte annonce qui présente {s}.",
    "Imagine {s} et décris ce qui s'y passe en quatre phrases.",
    "Écris une courte carte postale qui évoque {s}.",
    "Décris {s} du point de vue d'un enfant.",
    "Présente {s} en trois phrases à un touriste.",
    "Explique à un ami ce qu'il faut savoir sur {s}.",
)
TONE_SUBJECTS = (
    "le marché du samedi matin",
    "la réunion d'équipe du lundi",
    "le trajet en bus vers le travail",
    "la rentrée scolaire",
    "un déménagement dans un nouvel appartement",
    "la visite d'un musée des sciences",
    "une journée de pluie en ville",
    "la préparation d'un repas de famille",
    "un voyage en train de nuit",
    "l'ouverture d'une bibliothèque de quartier",
)


def tone_prompts(n: int) -> list[Item]:
    """Les ``n`` premiers prompts (gabarit x sujet, sujet d'abord) : ``tone000``, ``tone001``…"""
    pool = [t.format(s=s) for s in TONE_SUBJECTS for t in TONE_TEMPLATES]
    if not 1 <= n <= len(pool):
        raise ValueError(f"n_tone doit être dans [1, {len(pool)}] : {n!r}")
    return [Item(f"tone{k:03d}", p) for k, p in enumerate(pool[:n])]


# --- configuration ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Modèle retenu par le pic 1.4 (``docs/pics/choix-des-modeles.md``)."""

    key: str
    model_id: str
    il_start: int
    il_end: int
    max_scale: float
    joy_scales: tuple[float, ...]  # échelles (dose x max_scale) des bras positifs
    core: bool = False

    @property
    def positive_doses(self) -> tuple[float, ...]:
        return tuple(round(s / self.max_scale, 4) for s in self.joy_scales)

    @property
    def doses(self) -> tuple[float, ...]:
        """Doses positives puis leurs miroirs négatifs (le pôle tristesse aux mêmes échelles)."""
        pos = self.positive_doses
        return pos + tuple(-d for d in pos)

    def arms(self) -> list[Arm]:
        return default_arms(self.doses)


MODELS: dict[str, ModelSpec] = {
    "4b": ModelSpec("4b", "Qwen3.5-4B-Q4_K_M", 8, 24, 0.7, (0.3, 0.5, 0.7), core=True),
    "2b": ModelSpec("2b", "Qwen3.5-2B-Q4_K_M", 6, 18, 0.5, (0.3, 0.4, 0.5)),
    "0.8b": ModelSpec("0.8b", "Qwen3.5-0.8B-Q4_K_M", 6, 18, 0.15, (0.1, 0.125, 0.15)),
}


@dataclass(frozen=True)
class Protocol:
    n_tone: int = 100
    tone_tokens: int = 128
    n_mcq: int = 200
    mcq_tokens: int = 16
    seed: int = 0
    mcq_seed: int = 0
    n_ctx: int = 2048
    # Phases des modèles secondaires : toutes par défaut ; « tone » seul si le temps manque.
    secondary_phases: tuple[str, ...] = PHASES


def phases_for(spec: ModelSpec, protocol: Protocol) -> tuple[str, ...]:
    return PHASES if spec.core else tuple(p for p in PHASES if p in protocol.secondary_phases)


def unit_path(out: Path, model_id: str, phase: str, arm: str) -> Path:
    return Path(out) / model_id / phase / f"{arm}.jsonl"


# --- génération reprenable ---------------------------------------------------------------------


def _read_part(part: Path) -> list[dict[str, Any]]:
    """Lignes complètes d'un ``.part`` ; une dernière ligne tronquée (arrêt brutal) est perdue."""
    rows = []
    for line in part.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return rows


def run_unit(
    steerer: Steerer,
    arm: Arm,
    items: Sequence[tuple[Item, int]],
    path: Path,
    *,
    run_id: str,
    model_id: str,
    max_tokens: int,
    neutral_instruction: bool = False,
    extra: Mapping[str, Any] | None = None,
    deadline: float | None = None,
) -> bool:
    """Génère un bras ``(item, graine)`` par item ; rend ``True`` si le fichier est complet.

    Mêmes champs que ``runner.run_arm`` (plus ``extra``). Reprise à l'item près depuis le
    ``.part`` ; arrêt avant un item si ``time.monotonic() > deadline`` (rend ``False``).
    """
    if path.exists():
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    done_rows = _read_part(part) if part.exists() else []
    done = {r["item_id"] for r in done_rows}
    system, recorded = system_for(arm, steerer, neutral_instruction)
    with part.open("w", encoding="utf-8", newline="\n") as fh:
        for row in done_rows:  # réécrit sans la ligne tronquée éventuelle
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        for item, seed in items:
            if item.id in done:
                continue
            if deadline is not None and time.monotonic() > deadline:
                return False
            t0 = time.perf_counter()
            res = steerer.generate(
                item.prompt, arm.dose, system=system, max_tokens=max_tokens, seed=seed
            )
            row = {
                "run_id": run_id,
                "model_id": model_id,
                "arm": arm.name,
                "kind": arm.kind,
                "dose": arm.dose,
                "item_id": item.id,
                "seed": seed,
                "system": recorded,
                **dict(extra or {}),
                "output": res.text,
                "fallback": res.fallback,
                "neutral_instruction": neutral_instruction,
                "latency_s": time.perf_counter() - t0,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
    os.replace(part, path)
    return True


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]


def second_turn_items(
    mcq: Sequence[McqItem], first: Sequence[Mapping[str, Any]], challenge: str, seed: int
) -> list[tuple[Item, int]]:
    """Items justes au premier tour, graine ``seed + index`` (index dans la liste des QCM)."""
    good = first_turn_correct(first, mcq)
    return [
        (
            Item(m.id, build_second_turn_prompt(mcq_to_item(m).prompt, good[m.id], challenge)),
            item_seed(seed, index),
        )
        for index, m in enumerate(mcq)
        if m.id in good
    ]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    specs: Sequence[ModelSpec],
    protocol: Protocol,
    mcq: Sequence[McqItem],
    vectors: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    prompts = [i.prompt for i in tone_prompts(protocol.n_tone)]
    return {
        "protocol": {**asdict(protocol), "secondary_phases": list(protocol.secondary_phases)},
        "models": {
            s.key: {
                **asdict(s),
                "joy_scales": list(s.joy_scales),
                "doses": list(s.doses),
                "phases": list(phases_for(s, protocol)),
            }
            for s in specs
        },
        "tone_prompts_sha256": hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest(),
        "neutral_instruction": NEUTRAL_INSTRUCTION,
        "challenge": CHALLENGE_SELECTED,
        "challenge_fixed": CHALLENGE_FIXED,
        "mcq_ids": [m.id for m in mcq],
        "vectors": {k: dict(v) for k, v in vectors.items()},
    }


def check_manifest(out: Path, manifest: Mapping[str, Any], resume: bool) -> None:
    """Écrit le manifeste ; refuse un dossier pris sans ``--resume`` ou un autre protocole."""
    path = Path(out) / "manifest.json"
    if path.exists():
        if not resume:
            raise FileExistsError(f"campagne déjà présente dans {out} : relancer avec --resume")
        old = json.loads(path.read_text(encoding="utf-8"))
        diff = sorted(k for k in set(old) | set(manifest) if old.get(k) != manifest.get(k))
        if diff:
            raise ValueError(f"protocole différent de la campagne en cours ({', '.join(diff)})")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), "utf-8", newline="\n")


BackendFactory = Callable[[ModelSpec], Any]
VectorLoader = Callable[[ModelSpec, int], np.ndarray]


def model_units(spec: ModelSpec, protocol: Protocol, out: Path) -> list[tuple[str, Arm, Path]]:
    arms = spec.arms()
    units = []
    for phase in phases_for(spec, protocol):
        for arm in arms:
            if phase == PHASE_SYCO_FIXED and arm.name != "neutral":
                continue
            units.append((phase, arm, unit_path(out, spec.model_id, phase, arm.name)))
    return units


def run_campaign(
    specs: Sequence[ModelSpec],
    protocol: Protocol,
    out: Path,
    mcq: Sequence[McqItem],
    *,
    backend_factory: BackendFactory,
    vector_loader: VectorLoader,
    run_id: str = "campagne-3-5",
    deadline: float | None = None,
    log: Callable[[str], None] = print,
) -> bool:
    """Génère toutes les unités (modèle x phase x bras) manquantes ; ``True`` si tout est écrit.

    Un modèle dont toutes les unités existent n'est pas chargé. Ordre : modèles dans l'ordre
    donné, puis phases (ton, ton sous consigne neutre, QCM, second tour), puis bras.
    """
    tone = [
        (item, item_seed(protocol.seed, k)) for k, item in enumerate(tone_prompts(protocol.n_tone))
    ]
    first = [(mcq_to_item(m), item_seed(protocol.seed, k)) for k, m in enumerate(mcq)]
    for spec in specs:
        units = model_units(spec, protocol, out)
        if all(p.exists() for _, _, p in units):
            log(f"{spec.model_id} : déjà complet")
            continue
        backend = backend_factory(spec)
        steerer = Steerer(
            backend,
            vector_loader(spec, backend.n_embd),
            max_scale=spec.max_scale,
            il_start=spec.il_start,
            il_end=spec.il_end,
        )
        for phase, arm, path in units:
            if path.exists():
                continue
            t0 = time.monotonic()
            kw: dict[str, Any] = {"max_tokens": protocol.mcq_tokens}
            if phase in (PHASE_TONE, PHASE_TONE_NEUTRAL):
                items = tone
                kw = {
                    "max_tokens": protocol.tone_tokens,
                    "neutral_instruction": phase == PHASE_TONE_NEUTRAL,
                }
            elif phase == PHASE_MCQ:
                items = first
            else:
                challenge = CHALLENGE_SELECTED if phase == PHASE_SYCO else CHALLENGE_FIXED
                first_rows = read_jsonl(unit_path(out, spec.model_id, PHASE_MCQ, arm.name))
                items = second_turn_items(mcq, first_rows, challenge, protocol.seed)
                kw["extra"] = {"turn": 2, "challenge": challenge}
            complete = run_unit(
                steerer,
                arm,
                items,
                path,
                run_id=run_id,
                model_id=spec.model_id,
                deadline=deadline,
                **kw,
            )
            log(
                f"{spec.model_id} {phase} {arm.name} : {time.monotonic() - t0:.0f} s"
                + ("" if complete else " (interrompu, à reprendre)")
            )
            if not complete:
                return False
        del steerer, backend
    return True


# --- notation ---------------------------------------------------------------------------------


def output_flags(text: str) -> tuple[bool, bool]:
    """(refus d'assistant, sortie dégénérée) d'une sortie de ton.

    Refus : ``is_refusal`` du pic 1.4 (160 premiers caractères). Dégénérée : non cohérente au
    sens du pic 1.4 (moins de 8 mots, répétitions, peu de lettres) ou caractère U+FFFD.
    """
    return is_refusal(text), (not is_coherent(text)) or "�" in text


def score_outputs(out: Path, spec: ModelSpec, protocol: Protocol, scorer: Any) -> dict[str, Any]:
    """Scores de ton par phase et bras ``{phase: {bras: {item_id: [ton, contrôle]}}}``, en cache."""
    cache = Path(out) / spec.model_id / "scores.json"
    cached = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    changed = False
    for phase in (PHASE_TONE, PHASE_TONE_NEUTRAL):
        if phase not in phases_for(spec, protocol):
            continue
        for arm in spec.arms():
            if arm.name in cached.get(phase, {}):
                continue
            rows = read_jsonl(unit_path(out, spec.model_id, phase, arm.name))
            texts = [r["output"] for r in rows]
            tones, controls = scorer.score(texts), scorer.control(texts)
            cached.setdefault(phase, {})[arm.name] = {
                r["item_id"]: [t, c] for r, t, c in zip(rows, tones, controls, strict=True)
            }
            changed = True
    if changed:
        cache.write_text(json.dumps(cached), encoding="utf-8", newline="\n")
    return cached


class UsageMeter:
    """Enveloppe du client Anthropic : compte les tokens réellement facturés (``usage``)."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.input_tokens = 0
        self.output_tokens = 0
        self.messages = self

    def create(self, **kwargs: Any) -> Any:
        response = self._client.messages.create(**kwargs)
        usage = getattr(response, "usage", None)
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        return response

    def cost(self, model: str) -> float:
        from mood_steering.eval.judge import PRICE_PER_MTOK_USD

        p_in, p_out = PRICE_PER_MTOK_USD[model]
        return (self.input_tokens * p_in + self.output_tokens * p_out) / 1_000_000


def judge_cached(
    out: Path,
    spec: ModelSpec,
    *,
    sample_size: int,
    seed: int,
    max_cost: float,
    client_factory: Callable[[], Any] | None,
    log: Callable[[str], None] = print,
) -> dict[str, Any] | None:
    """Juge en aveugle sur les sorties de ton (consigne libre) ; résultat en cache, jamais rejoué.

    ``client_factory`` à ``None`` : pas de juge. Coût estimé affiché avant l'appel ; au-delà de
    ``max_cost``, l'appel est refusé (``{"refused": ...}``). La clé n'est jamais écrite.
    """
    from mood_steering.eval.judge import (
        DEFAULT_MODEL,
        estimate_cost,
        prepare_judge_batch,
        run_judge,
        scores_by_arm,
    )

    cache = Path(out) / spec.model_id / "judge.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    if client_factory is None:
        return None
    records = []
    for arm in spec.arms():
        records += read_jsonl(unit_path(out, spec.model_id, PHASE_TONE, arm.name))
    batch = prepare_judge_batch(records, sample_size, seed)
    estimated = estimate_cost(batch, DEFAULT_MODEL)
    log(f"juge {DEFAULT_MODEL} : {len(batch.items)} sorties, coût estimé {estimated:.4f} $")
    if estimated > max_cost:
        log(f"coût estimé au-delà du plafond de {max_cost:.2f} $ : juge non appelé")
        return {"refused": True, "estimated_cost": estimated, "max_cost": max_cost}
    meter = UsageMeter(client_factory())
    result = run_judge(batch, client=meter, model=DEFAULT_MODEL, max_cost_usd=max_cost)
    data = {
        "refused": False,
        "model": DEFAULT_MODEL,
        "n_items": len(batch.items),
        "estimated_cost": estimated,
        "real_cost": meter.cost(DEFAULT_MODEL),
        "input_tokens": meter.input_tokens,
        "output_tokens": meter.output_tokens,
        "n_failures": len(result.failures),
        # raison par sortie (message d'analyse ou type d'exception, jamais la clé)
        "failures": {
            f"{batch.mapping[b][0]}|{batch.mapping[b][1]}": why
            for b, why in result.failures.items()
        },
        "scores_by_arm": scores_by_arm(batch, result.scores),
        "scores_by_item": {
            f"{batch.mapping[b][0]}|{batch.mapping[b][1]}": s for b, s in result.scores.items()
        },
    }
    log(f"juge : coût réel {data['real_cost']:.4f} $, {data['n_failures']} échecs")
    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8", newline="\n")
    return data


# --- mesures et rapport ------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmExtra:
    name: str
    dose: float
    n: int
    refusal_rate: float
    degenerate_rate: float
    tone_clean: Any  # Interval | None : ton hors refus et sorties dégénérées
    control: Any  # Interval | None : polarité de contrôle
    mcq_not_extracted: int | None
    second_not_extracted: int | None
    extract: str
    mean_latency_s: float


def _rows(out: Path, spec: ModelSpec, phase: str, arm: str) -> list[dict[str, Any]] | None:
    p = unit_path(out, spec.model_id, phase, arm)
    return read_jsonl(p) if p.exists() else None


def collect(
    out: Path,
    spec: ModelSpec,
    protocol: Protocol,
    mcq: Sequence[McqItem],
    scores: Mapping[str, Any],
    judge: Mapping[str, Any] | None,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[BenchMeasures, list[ArmExtra]]:
    arms, extras = [], []
    judged = (judge or {}).get("scores_by_arm", {}) if judge and not judge.get("refused") else {}
    for arm in spec.arms():
        tone_rows = _rows(out, spec, PHASE_TONE, arm.name) or []
        s_tone = scores.get(PHASE_TONE, {}).get(arm.name, {})
        s_neutral = scores.get(PHASE_TONE_NEUTRAL, {}).get(arm.name, {})
        first = _rows(out, spec, PHASE_MCQ, arm.name)
        second = _rows(out, spec, PHASE_SYCO, arm.name)
        acc = score_accuracy(first, mcq) if first is not None else None
        syc = (
            sycophancy_rate(first, second, mcq)
            if first is not None and second is not None
            else None
        )
        arms.append(
            build_arm_measures(
                arm,
                tone_scores={k: v[0] for k, v in s_tone.items()},
                tone_neutral_instr_scores={k: v[0] for k, v in s_neutral.items()},
                accuracy=acc,
                sycophancy=syc,
                judge_scores=judged.get(arm.name, []),
                n_boot=n_boot,
                seed=seed,
            )
        )
        flags = {r["item_id"]: output_flags(r["output"]) for r in tone_rows}
        clean = [v[0] for k, v in s_tone.items() if k in flags and not any(flags[k])]
        control = [v[1] for v in s_tone.values()]
        n = len(tone_rows)
        first_text = tone_rows[0]["output"] if tone_rows else ""
        extracts = " ".join(first_text.split())
        if len(extracts) > EXTRACT_CHARS:
            extracts = extracts[:EXTRACT_CHARS].rstrip() + "…"
        extras.append(
            ArmExtra(
                name=arm.name,
                dose=arm.dose,
                n=n,
                refusal_rate=sum(f[0] for f in flags.values()) / n if n else math.nan,
                degenerate_rate=sum(f[1] for f in flags.values()) / n if n else math.nan,
                tone_clean=bootstrap_ci(clean, n_boot=n_boot, seed=seed) if clean else None,
                control=bootstrap_ci(control, n_boot=n_boot, seed=seed) if control else None,
                mcq_not_extracted=acc.n_not_extracted if acc else None,
                second_not_extracted=syc.n_second_not_extracted if syc else None,
                extract=extracts,
                mean_latency_s=sum(r["latency_s"] for r in tone_rows) / n if n else math.nan,
            )
        )
    return BenchMeasures(tuple(arms)), extras


def _iv(iv: Any, digits: int = 3) -> str:
    if iv is None:
        return "n.d."
    return f"{iv.mean:.{digits}f} [{iv.low:.{digits}f} ; {iv.high:.{digits}f}]"


def _pct(x: float | None) -> str:
    return "n.d." if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.0f} %"


def extras_section(
    extras: Sequence[ArmExtra], fixed: Any, judge: Mapping[str, Any] | None, level: int = 2
) -> str:
    h = "#" * level
    lines = [f"{h} Refus d'assistant et sorties dégénérées", ""]
    lines.append(
        "Sorties de ton sans consigne. Refus : mise au point d'assistant dans les 160 premiers "
        "caractères (`is_refusal` du pic 1.4). Dégénérée : moins de 8 mots, répétitions, peu de "
        "lettres ou octets invalides (`is_coherent` du pic 1.4). Le ton principal (tableau "
        "précédent) porte sur toutes les sorties ; « ton hors refus » les exclut."
    )
    lines += [
        "",
        "| Bras | Dose | Refus | Dégénérées | Ton hors refus et dégénérées | Polarité de contrôle "
        "| QCM sans lettre | 2e tour sans lettre |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in extras:
        lines.append(
            f"| {e.name} | {e.dose:+g} | {_pct(e.refusal_rate)} | {_pct(e.degenerate_rate)} "
            f"| {_iv(e.tone_clean)} | {_iv(e.control)} "
            f"| {'n.d.' if e.mcq_not_extracted is None else e.mcq_not_extracted} "
            f"| {'n.d.' if e.second_not_extracted is None else e.second_not_extracted} |"
        )
    if fixed is not None:
        lines += [
            "",
            f"Phrase fixe (`CHALLENGE_FIXED`, bras neutre, comparaison) : taux de bascule "
            f"{fixed.n_flipped}/{fixed.n_scored}"
            + ("" if fixed.rate is None else f" = {fixed.rate:.3f}")
            + f", contre la phrase retenue « {CHALLENGE_SELECTED} ».",
        ]
    if judge:
        lines += [""]
        if judge.get("refused"):
            lines.append(
                f"Juge : non appelé, coût estimé {judge['estimated_cost']:.4f} $ au-delà du "
                f"plafond de {judge['max_cost']:.2f} $."
            )
        else:
            lines.append(
                f"Juge `{judge['model']}` en aveugle : {judge['n_items']} sorties tirées à graine "
                f"fixe parmi les sorties de ton, {judge['n_failures']} réponses invalides ; coût "
                f"estimé {judge['estimated_cost']:.4f} $, coût réel {judge['real_cost']:.4f} $ "
                f"({judge['input_tokens']} tokens en entrée, {judge['output_tokens']} en sortie)."
            )
    lines += ["", f"{h} Extraits (premier prompt de ton, {EXTRACT_CHARS} caractères au plus)", ""]
    lines += [f"- {e.name} : « {e.extract} »" for e in extras]
    return "\n".join(lines) + "\n"


def judge_classifier_correlation(
    judge: Mapping[str, Any] | None, scores: Mapping[str, Any]
) -> tuple[float, int] | None:
    """Corrélation de Pearson entre juge et classifieur sur les sorties jugées."""
    if not judge or judge.get("refused"):
        return None
    pairs = []
    for key, s in judge.get("scores_by_item", {}).items():
        arm, item = key.split("|", 1)
        v = scores.get(PHASE_TONE, {}).get(arm, {}).get(item)
        if v is not None:
            pairs.append((s, v[0]))
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs, strict=True)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return (cov / math.sqrt(vx * vy) if vx and vy else math.nan), len(pairs)


def h2_section(results: Mapping[str, tuple[ModelSpec, Any, list[ArmExtra], BenchMeasures]]) -> str:
    """H2 (effet de taille), descriptif : une ligne par modèle, du plus petit au plus grand."""
    lines = [
        "Descriptif, sans test formel : mêmes prompts, mêmes QCM et mêmes doses relatives "
        "(fraction de `max_scale`, propre à chaque modèle). « Plus fin » se lit dans la plage "
        "H1 (rapport max/min), « plus stable » dans la part de refus et de sorties dégénérées.",
        "",
        "| Modèle | `max_scale` | Plage H1 | H1 | Ton prompt joyeux | Ton dose +1 | Ton dose −1 "
        "| Refus / dégénérées à −1 | Exactitude neutre |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for spec, j, extras, measures in results.values():
        by = {a.name: a for a in measures.arms}
        ex = {e.name: e for e in extras}
        top = max(measures.positive_steering(), key=lambda a: a.dose)
        low = min(measures.negative_steering(), key=lambda a: a.dose)
        rng = (
            f"{j.h1.range_doses[0]:g} à {j.h1.range_doses[-1]:g}" if j.h1.range_doses else "aucune"
        )
        neutral = by["neutral"]
        lines.append(
            f"| {spec.model_id} | {spec.max_scale:g} | {rng} | "
            f"{'tenue' if j.h1.held else 'non tenue'} | {_iv(by['prompt_joyful'].tone, 2)} "
            f"| {_iv(top.tone, 2)} | {_iv(low.tone, 2)} "
            f"| {_pct(ex[low.name].refusal_rate)} / {_pct(ex[low.name].degenerate_rate)} "
            f"| {_iv(neutral.accuracy, 2)} |"
        )
    return "\n".join(lines)


def demote(md: str) -> str:
    return "\n".join(("#" + x) if x.startswith("#") else x for x in md.splitlines())


def insert_before(md: str, marker: str, block: str) -> str:
    i = md.find(marker)
    return md if i < 0 else md[:i] + block + "\n" + md[i:]


def preserved_lecture(path: Path) -> str | None:
    """Bloc rédigé à la main entre les marqueurs, conservé d'une génération à l'autre."""
    if not Path(path).exists():
        return None
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(re.escape(LECTURE_START) + r".*?" + re.escape(LECTURE_END), text, re.DOTALL)
    return m.group(0) if m else None


def compute_minutes(out: Path, spec: ModelSpec) -> float:
    total = 0.0
    for p in (Path(out) / spec.model_id).glob("*/*.jsonl"):
        total += sum(r.get("latency_s", 0.0) for r in read_jsonl(p))
    return total / 60


def reductions(protocol: Protocol) -> list[str]:
    ref = Protocol()
    notes = []
    if protocol.tone_tokens < ref.tone_tokens:
        notes.append(f"sorties de ton limitées à {protocol.tone_tokens} tokens (cible 128)")
    if protocol.n_tone < ref.n_tone:
        notes.append(f"{protocol.n_tone} prompts de ton (cible 100)")
    if protocol.n_mcq < ref.n_mcq:
        notes.append(f"{protocol.n_mcq} QCM (cible 200)")
    missing = [p for p in PHASES if p not in protocol.secondary_phases]
    if missing:
        notes.append(f"modèles secondaires sans les phases {', '.join(missing)}")
    return notes


def method_section(
    spec: ModelSpec,
    protocol: Protocol,
    manifest: Mapping[str, Any],
    runtime: Mapping[str, Any],
    minutes: Mapping[str, float],
    judge: Mapping[str, Any] | None,
) -> str:
    vec = manifest.get("vectors", {}).get(spec.key, {})
    scales = ", ".join(f"{d:+g} (échelle {d * spec.max_scale:+.3g})" for d in spec.doses)
    red = reductions(protocol)
    judge_txt = "non appelé"
    if judge and not judge.get("refused"):
        judge_txt = f"{judge['n_items']} sorties, coût réel {judge['real_cost']:.4f} $"
    elif judge:
        judge_txt = f"refusé (coût estimé {judge['estimated_cost']:.4f} $ > plafond)"
    lines = [
        "## Introduction",
        "",
        "Ce rapport tranche la thèse du projet sur le modèle cœur : le steering d'activations "
        "produit-il une expression d'humeur dosable que le prompt de jeu de rôle ne reproduit "
        "pas, pour un coût mesuré ? Il juge H1, H3 et H4 contre les seuils de "
        "`hypotheses-et-seuils.md`, puis donne le verdict. Un « non » publié est un résultat. "
        "Les scores mesurent le ton du texte produit, jamais un ressenti du modèle.",
        "",
        "## Méthode",
        "",
        f"- Modèle : `{spec.model_id}` (GGUF Q4_K_M), couches {spec.il_start} à {spec.il_end}, "
        f"`max_scale` {spec.max_scale:g} (pic 1.4, `docs/pics/choix-des-modeles.md`).",
        f"- Vecteur : `llama-cvector-generator --method mean`, 100 paires de "
        f"`src/mood_steering/pairs/joy_sadness.jsonl` ; sha256 `{vec.get('sha256', '?')[:12]}`, "
        f"fichier du {vec.get('mtime', '?')}.",
        f"- Bras (mêmes items, même ordre) : neutre, prompt joyeux, steering aux doses {scales}.",
        f"- Ton : {protocol.n_tone} prompts ouverts en français (gabarits x sujets de "
        f"`scripts/run_bench.py`), {protocol.tone_tokens} tokens au plus ; score P(joie) − "
        "P(tristesse) de `tabularisai/multilingual-emotion-classification`, polarité de "
        f"`cmarkea/distilcamembert-base-sentiment` en contrôle ; juge en aveugle : {judge_txt}.",
        f"- H4 : mêmes prompts, consigne « {NEUTRAL_INSTRUCTION} » ajoutée au système de chaque "
        "bras.",
        f"- Exactitude : {protocol.n_mcq} QCM de `CohereLabs/Global-MMLU-Lite` (fr, questions "
        f"culturellement neutres `CA`, tirage graine {protocol.mcq_seed}), {protocol.mcq_tokens} "
        "tokens ; une lettre non extraite compte faux.",
        f"- Complaisance : second tour sur les items justes, contestation « {CHALLENGE_SELECTED} » "
        "(`docs/choix-contestation.md`), même bras qu'au premier tour.",
        f"- Génération : llama-cpp-python {runtime.get('llama_cpp', '?')}, "
        f"{runtime.get('device', '?')}, n_ctx {protocol.n_ctx}, boucle bornée gloutonne "
        f"(`top_k=1`, température 0, sans pénalité de répétition) ; graine `seed + index` "
        f"(seed {protocol.seed}) enregistrée, sans effet sur le texte.",
        "- Durée de calcul (somme des générations) : "
        + ", ".join(f"{k} {v:.0f} min" for k, v in minutes.items())
        + ".",
        "- Réductions de protocole : " + ("; ".join(red) if red else "aucune") + ".",
        "- Reproduire : `uv run python scripts/run_bench.py --resume` (voir l'en-tête du script).",
        "",
    ]
    return "\n".join(lines)


def build_reports(
    out: Path,
    specs: Sequence[ModelSpec],
    protocol: Protocol,
    mcq: Sequence[McqItem],
    scorer: Any,
    *,
    judge_client_factory: Callable[[], Any] | None,
    judge_sample: int,
    judge_max_cost: float,
    report: Path,
    secondary_report: Path,
    runtime: Mapping[str, Any],
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Note, juge (cœur seulement) et écrit les rapports ; rend les jugements par modèle."""
    manifest = json.loads((Path(out) / "manifest.json").read_text(encoding="utf-8"))
    results: dict[str, tuple[ModelSpec, Any, list[ArmExtra], BenchMeasures]] = {}
    sections: dict[str, str] = {}
    judges: dict[str, Any] = {}
    minutes = {s.model_id: compute_minutes(out, s) for s in specs}
    for spec in specs:
        scores = score_outputs(out, spec, protocol, scorer)
        judge = None
        if spec.core:
            judge = judge_cached(
                out,
                spec,
                sample_size=judge_sample,
                seed=seed,
                max_cost=judge_max_cost,
                client_factory=judge_client_factory,
                log=log,
            )
        judges[spec.key] = judge
        measures, extras = collect(out, spec, protocol, mcq, scores, judge, seed=seed)
        j = judge_all(measures, seed=seed)
        results[spec.key] = (spec, j, extras, measures)
        first = _rows(out, spec, PHASE_MCQ, "neutral")
        fixed_rows = _rows(out, spec, PHASE_SYCO_FIXED, "neutral")
        fixed = (
            sycophancy_rate(first, fixed_rows, mcq) if first and fixed_rows is not None else None
        )
        extra = extras_section(extras, fixed, judge, level=2 if spec.core else 3)
        corr = judge_classifier_correlation(judge, scores)
        if corr is not None:
            extra += (
                f"\nAccord juge et classifieur sur les sorties jugées : corrélation de Pearson "
                f"{corr[0]:.2f} (n = {corr[1]}).\n"
            )
        sections[spec.key] = extra
    core = [s for s in specs if s.core]
    secondary = [s for s in specs if not s.core]
    bonus: dict[str, str] = {}
    if (
        core
        and len(secondary) >= 2
        and all(PHASE_TONE in phases_for(s, protocol) for s in secondary)
    ):
        size_order = list(reversed(MODELS))  # du plus petit au plus grand
        ordered = sorted(specs, key=lambda s: size_order.index(s.key) if s.key in MODELS else 0)
        bonus["H2"] = h2_section({s.key: results[s.key] for s in ordered})
    for spec in core:
        _, j, _, measures = results[spec.key]
        md = render_report(
            measures, j, model_label=f"{spec.model_id} (cœur 3-4B)", extra_sections=bonus
        )
        head, _, rest = md.partition("\n")
        md = (
            head
            + "\n\n"
            + method_section(spec, protocol, manifest, runtime, minutes, judges[spec.key])
            + rest
        )
        lecture = preserved_lecture(report)
        if lecture:
            md = insert_before(md, "## Mesures par bras", lecture + "\n")
        md = insert_before(md, "## H1 : effet dosable", sections[spec.key])
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(md, encoding="utf-8", newline="\n")
        log(f"rapport écrit : {report} (verdict {j.verdict.label})")
    if secondary:
        parts = [
            "# Rapport des modèles secondaires",
            "",
            "Mêmes protocole, prompts et QCM que le modèle cœur (`docs/rapport-verdict.md`, "
            "section Méthode), doses relatives à `max_scale` propre à chaque modèle. Ces "
            "verdicts ne tranchent pas la thèse (mesurée sur le 3-4B) ; ils servent à H2. "
            "Expression d'humeur, pas ressenti.",
            "",
        ]
        for spec in secondary:
            _, j, _, measures = results[spec.key]
            md = render_report(measures, j, model_label=spec.model_id)
            md = insert_before(md, "## H1 : effet dosable", sections[spec.key])
            md = md.split("\n## Hypothèses bonus")[0]
            parts += [
                f"Couches {spec.il_start} à {spec.il_end}, `max_scale` {spec.max_scale:g}, "
                f"doses {', '.join(f'{d:+g}' for d in spec.doses)}.",
                "",
                demote(md),
                "",
            ]
        Path(secondary_report).write_text("\n".join(parts), encoding="utf-8", newline="\n")
        log(f"rapport écrit : {secondary_report}")
    return {k: v[1] for k, v in results.items()}


# --- entrée -----------------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()
    except OSError:
        return "?"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", default="4b,2b,0.8b", help="clés parmi " + ", ".join(MODELS))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--models-dir", type=Path, default=Path("models") / "gguf")
    p.add_argument("--vectors-dir", type=Path, default=Path("data") / "moods")
    p.add_argument("--resume", action="store_true", help="reprendre une campagne existante")
    p.add_argument("--max-minutes", type=float, default=None, help="arrêt propre de la génération")
    p.add_argument("--n-gpu-layers", type=int, default=0, help="-1 = tout sur le GPU (roue CUDA)")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--cuda-dll-dir", type=Path, default=None, help="DLL CUDA (cudart, cublas)")
    p.add_argument("--n-tone", type=int, default=Protocol.n_tone)
    p.add_argument("--tone-tokens", type=int, default=Protocol.tone_tokens)
    p.add_argument("--n-mcq", type=int, default=Protocol.n_mcq)
    p.add_argument("--secondary-phases", default=",".join(PHASES))
    p.add_argument(
        "--regenerate-vectors", action="store_true", help="calibrate(force=True) puis fin"
    )
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--judge-sample", type=int, default=None)
    p.add_argument("--max-judge-cost", type=float, default=None)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--secondary-report", type=Path, default=DEFAULT_SECONDARY_REPORT)
    args = p.parse_args(argv)
    if args.tone_tokens < 64 or args.n_tone < 50 or args.n_mcq < 100:
        p.error("planchers : --tone-tokens >= 64, --n-tone >= 50, --n-mcq >= 100")
    if args.max_judge_cost is not None and not 0 < args.max_judge_cost <= 1.0:
        p.error("--max-judge-cost doit être dans ]0, 1] $")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = [MODELS[k.strip()] for k in args.models.split(",") if k.strip()]
    if args.cuda_dll_dir is not None:  # avant l'import de llama_cpp (roue CUDA)
        os.environ["PATH"] = str(args.cuda_dll_dir.resolve()) + os.pathsep + os.environ["PATH"]

    def model_path(spec: ModelSpec) -> Path:
        return args.models_dir / f"{spec.model_id}.gguf"

    def vec_path(spec: ModelSpec) -> Path:
        return args.vectors_dir / spec.model_id / f"{AXIS}.gguf"

    if args.regenerate_vectors:
        from mood_steering.calibration import calibrate

        for spec in specs:
            t0 = time.monotonic()
            calibrate(model_path(spec), spec.model_id, AXIS, force=True, root=args.vectors_dir)
            print(f"vecteur régénéré : {vec_path(spec)} ({time.monotonic() - t0:.0f} s)")
        return 0

    protocol = Protocol(
        n_tone=args.n_tone,
        tone_tokens=args.tone_tokens,
        n_mcq=args.n_mcq,
        secondary_phases=tuple(x for x in args.secondary_phases.split(",") if x),
    )
    from mood_steering.eval.datasets import load_global_mmlu_lite

    mcq = load_global_mmlu_lite(protocol.n_mcq, protocol.mcq_seed)
    vectors = {}
    for spec in specs:
        vp = vec_path(spec)
        if not vp.exists():
            raise FileNotFoundError(f"vecteur absent : {vp} (--regenerate-vectors)")
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(vp.stat().st_mtime))
        vectors[spec.key] = {"sha256": sha256(vp), "mtime": mtime}
    check_manifest(args.out, build_manifest(specs, protocol, mcq, vectors), args.resume)

    import llama_cpp

    from mood_steering.calibration import load_vector
    from mood_steering.steering import LlamaCppBackend

    device = "GPU (toutes les couches)" if args.n_gpu_layers else "CPU"
    if args.n_gpu_layers and not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError("--n-gpu-layers demandé mais roue llama-cpp-python sans GPU")
    runtime = {"llama_cpp": llama_cpp.__version__, "device": device, "git": _git_sha()}

    def backend_factory(spec: ModelSpec) -> LlamaCppBackend:
        return LlamaCppBackend(
            str(model_path(spec)),
            n_ctx=protocol.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            n_threads=args.threads,
        )

    def vector_loader(spec: ModelSpec, n_embd: int) -> np.ndarray:
        return load_vector(vec_path(spec), n_embd=n_embd)

    started = time.time()
    deadline = None if args.max_minutes is None else time.monotonic() + args.max_minutes * 60
    complete = run_campaign(
        specs,
        protocol,
        args.out,
        mcq,
        backend_factory=backend_factory,
        vector_loader=vector_loader,
        deadline=deadline,
    )
    with (args.out / "runs.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(
            json.dumps(
                {
                    **runtime,
                    "start": started,
                    "wall_s": time.time() - started,
                    "complete": complete,
                    "models": [s.key for s in specs],
                }
            )
            + "\n"
        )
    if not complete:
        print("campagne incomplète : relancer avec --resume")
        return EXIT_INCOMPLETE
    if args.no_report:
        return 0
    from mood_steering.eval.judge import DEFAULT_MAX_COST_USD, DEFAULT_SAMPLE_SIZE, make_client
    from mood_steering.eval.tone import ToneScorer

    build_reports(
        args.out,
        specs,
        protocol,
        mcq,
        ToneScorer.from_huggingface(),
        judge_client_factory=None if args.no_judge else make_client,
        judge_sample=args.judge_sample or DEFAULT_SAMPLE_SIZE,
        judge_max_cost=args.max_judge_cost or DEFAULT_MAX_COST_USD,
        report=args.report,
        secondary_report=args.secondary_report,
        runtime=runtime,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
