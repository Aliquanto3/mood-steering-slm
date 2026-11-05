"""Effet de taille (story 5.2, H2) : Qwen3.5 4B, 2B et 0.8B sur les données de la campagne 3.5.

Lecture seule des sorties de la campagne (``data/runs/campagne-3-5``, jamais versionné) : aucune
génération. Mesure par modèle la plage de doses utile, la stabilité du ton et la pente dose → ton,
puis compare les modèles par bootstrap sur les items (items communs aux bras, tirés ensemble).

    uv run python scripts/report_size.py --out data/runs/campagne-3-5 \
        --report docs/rapport-taille.md

``--report`` remplace le bloc entre ``<!-- mesures -->`` et ``<!-- /mesures -->`` ;
sans lui, le script imprime les mesures.

Expression d'humeur dans le texte, jamais un ressenti.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_bench import (  # noqa: E402  (import en lecture seule)
    MODELS,
    PHASE_TONE,
    ModelSpec,
    output_flags,
    read_jsonl,
    unit_path,
)

from mood_steering.eval.arms import KIND_NEUTRAL, KIND_PROMPT, KIND_STEERING  # noqa: E402
from mood_steering.eval.stats import (  # noqa: E402
    Interval,
    bootstrap_ci,
    excludes_zero,
    interval_from_draws,
    paired_shift,
)

DEFAULT_OUT = Path("data") / "runs" / "campagne-3-5"
DEFAULT_REPORT = Path("docs") / "rapport-taille.md"
MAX_DEGENERATE = 0.10  # une dose n'est utile que si au plus 10 % des sorties sont dégénérées
THRESHOLDS = (0.05, 0.10, 0.20)  # sensibilité de la plage utile au seuil de dégénérées
N_BOOT = 2000
MARK_START = "<!-- mesures -->"
MARK_END = "<!-- /mesures -->"
SIZE_ORDER = ("4b", "2b", "0.8b")  # du plus grand au plus petit


# --- chargement -------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmData:
    name: str
    kind: str
    dose: float
    tones: tuple[float, ...]  # alignés sur ``ModelData.ids``
    refusal: tuple[bool, ...]
    degenerate: tuple[bool, ...]


@dataclass(frozen=True)
class ModelData:
    spec: ModelSpec
    ids: tuple[str, ...]
    arms: tuple[ArmData, ...]  # neutre, prompt, puis steering

    def arm(self, name: str) -> ArmData:
        return next(a for a in self.arms if a.name == name)

    @property
    def neutral(self) -> ArmData:
        return next(a for a in self.arms if a.kind == KIND_NEUTRAL)

    @property
    def prompt(self) -> ArmData:
        return next(a for a in self.arms if a.kind == KIND_PROMPT)

    def steering(self, sign: int) -> list[ArmData]:
        """Bras steering d'un pôle (+1 ou -1), du plus faible au plus fort en module."""
        arms = [a for a in self.arms if a.kind == KIND_STEERING and a.dose * sign > 0]
        return sorted(arms, key=lambda a: abs(a.dose))


def load_model(out: Path, spec: ModelSpec) -> ModelData:
    """Ton, refus et sorties dégénérées par bras, sur les items communs à tous les bras."""
    scores = json.loads((Path(out) / spec.model_id / "scores.json").read_text(encoding="utf-8"))
    tone = scores[PHASE_TONE]
    per_arm = {}
    for arm in spec.arms():
        rows = {
            r["item_id"]: r for r in read_jsonl(unit_path(out, spec.model_id, PHASE_TONE, arm.name))
        }
        per_arm[arm.name] = (arm, rows, tone[arm.name])
    common = set.intersection(*(set(s) & set(r) for _, r, s in per_arm.values()))
    if not common:
        raise ValueError(f"{spec.model_id} : aucun item commun aux bras")
    ids = tuple(sorted(common))
    arms = []
    for arm, rows, s in per_arm.values():
        flags = [output_flags(rows[i]["output"]) for i in ids]
        arms.append(
            ArmData(
                name=arm.name,
                kind=arm.kind,
                dose=arm.dose,
                tones=tuple(float(s[i][0]) for i in ids),
                refusal=tuple(f[0] for f in flags),
                degenerate=tuple(f[1] for f in flags),
            )
        )
    return ModelData(spec, ids, tuple(arms))


# --- mesures par modèle -----------------------------------------------------------------------


def sd(values: Sequence[float]) -> float:
    """Écart-type d'échantillon (n - 1) ; 0 pour moins de deux valeurs."""
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def rate(flags: Sequence[bool]) -> float:
    return sum(flags) / len(flags) if flags else math.nan


def slope(points: Sequence[tuple[float, float]]) -> float:
    """Pente des moindres carrés de y sur x ; nan si x est constant ou moins de deux points."""
    n = len(points)
    if n < 2:
        return math.nan
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    if sxx == 0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in points) / sxx


@dataclass(frozen=True)
class ArmRow:
    name: str
    dose: float
    scale: float  # dose x max_scale
    tone: Interval
    sd: float
    refusal_rate: float
    degenerate_rate: float
    shifted: bool | None  # None : bras sans steering
    useful: bool | None  # None : bras sans steering ou sans ton mesuré

    def useful_at(self, max_degenerate: float) -> bool | None:
        if self.shifted is None:
            return None
        return self.shifted and self.degenerate_rate <= max_degenerate


def shifted_arm(arm: ArmData, neutral: ArmData, *, n_boot: int, seed: int) -> bool:
    """Ton déplacé dans le sens de la dose : critère unique de ``eval.stats.paired_shift``.

    IC95 du bootstrap apparié de la différence de ton (bras - neutre, mêmes prompts) qui exclut
    0 du côté du signe de la dose.
    """
    return paired_shift(
        arm.tones, neutral.tones, direction=1 if arm.dose > 0 else -1, n_boot=n_boot, seed=seed
    )


def useful_arm(
    arm: ArmData,
    neutral: ArmData,
    *,
    n_boot: int,
    seed: int,
    max_degenerate: float = MAX_DEGENERATE,
) -> bool:
    """Utile : ton déplacé dans le sens de la dose (IC95 apparié excluant 0), peu de dégénérées."""
    shifted = shifted_arm(arm, neutral, n_boot=n_boot, seed=seed)
    return shifted and rate(arm.degenerate) <= max_degenerate


def arm_rows(data: ModelData, *, n_boot: int = N_BOOT, seed: int = 0) -> list[ArmRow]:
    rows = []
    for a in data.arms:
        steered = a.kind == KIND_STEERING
        shifted = shifted_arm(a, data.neutral, n_boot=n_boot, seed=seed) if steered else None
        deg = rate(a.degenerate)
        rows.append(
            ArmRow(
                name=a.name,
                dose=a.dose,
                scale=a.dose * data.spec.max_scale,
                tone=bootstrap_ci(list(a.tones), n_boot=n_boot, seed=seed),
                sd=sd(a.tones),
                refusal_rate=rate(a.refusal),
                degenerate_rate=deg,
                shifted=shifted,
                useful=None if shifted is None else (shifted and deg <= MAX_DEGENERATE),
            )
        )
    return rows


@dataclass(frozen=True)
class UsefulRange:
    pole: str
    doses: tuple[float, ...]  # doses normalisées utiles, en module
    scales: tuple[float, ...]  # échelles réelles correspondantes

    @property
    def ratio(self) -> float | None:
        return self.doses[-1] / self.doses[0] if self.doses else None


def useful_ranges(
    data: ModelData, rows: Sequence[ArmRow], max_degenerate: float = MAX_DEGENERATE
) -> list[UsefulRange]:
    by = {r.name: r for r in rows}
    out = []
    for sign, pole in ((1, "positif"), (-1, "négatif")):
        ok = [a for a in data.steering(sign) if by[a.name].useful_at(max_degenerate)]
        doses = tuple(abs(a.dose) for a in ok)
        out.append(UsefulRange(pole, doses, tuple(d * data.spec.max_scale for d in doses)))
    return out


def tone_points(data: ModelData, sign: int | None, idx: Sequence[int]) -> list[tuple[float, float]]:
    """(dose, ton moyen sur ``idx``) pour le neutre et les bras d'un pôle (tous si None)."""
    arms = [data.neutral] + [
        a for a in data.arms if a.kind == KIND_STEERING and (sign is None or a.dose * sign > 0)
    ]
    return [(a.dose, sum(a.tones[i] for i in idx) / len(idx)) for a in arms]


def prompt_gap(data: ModelData, *, n_boot: int = N_BOOT, seed: int = 0) -> str:
    """Ton du bras steering le plus fort face au prompt joyeux (recouvrement des IC95)."""
    top = data.steering(1)[-1]
    ci, cp = (bootstrap_ci(list(x.tones), n_boot=n_boot, seed=seed) for x in (top, data.prompt))
    overlap = ci.low <= cp.high and cp.low <= ci.high
    rel = "recouvre" if overlap else ("reste sous" if ci.high < cp.low else "dépasse")
    return (
        f"ton du prompt joyeux {_iv(cp)}, ton à la dose +1 {_iv(ci)} : l'IC95 du steering {rel} "
        "celui du prompt"
    )


# --- bootstrap sur les items -------------------------------------------------------------------

Stat = Callable[[ModelData, Sequence[int]], float]


def boot_stat(data: ModelData, stat: Stat, *, n_boot: int, seed: int) -> list[float]:
    """Distribution bootstrap d'une statistique (mêmes items tirés pour tous les bras)."""
    rng = random.Random(seed)
    n = len(data.ids)
    return [stat(data, rng.choices(range(n), k=n)) for _ in range(n_boot)]


def paired_diff_draws(
    a: ModelData, b: ModelData, stat: Stat, *, n_boot: int, seed: int
) -> list[float]:
    """Tirages de stat(a) - stat(b), appariés : les mêmes prompts rééchantillonnés pour a et b."""
    if a.ids != b.ids:
        raise ValueError(
            f"{a.spec.model_id} et {b.spec.model_id} : prompts différents, appariement impossible"
        )
    rng = random.Random(seed)
    n = len(a.ids)
    out = []
    for _ in range(n_boot):
        idx = rng.choices(range(n), k=n)
        out.append(stat(a, idx) - stat(b, idx))
    return out


def stat_slope(sign: int | None, real_scale: bool = False) -> Stat:
    """Pente du ton sur la dose normalisée, ou sur l'échelle réelle (dose x ``max_scale``)."""

    def f(d: ModelData, idx: Sequence[int]) -> float:
        k = d.spec.max_scale if real_scale else 1.0
        return slope([(x * k, y) for x, y in tone_points(d, sign, idx)])

    return f


def stat_tone(sign: int) -> Stat:
    """Ton moyen à la dose ±1 (le bras le plus fort du pôle)."""
    return lambda d, idx: _mean_at(d.steering(sign)[-1].tones, idx)


def stat_degenerate(sign: int) -> Stat:
    return lambda d, idx: _mean_at([float(f) for f in d.steering(sign)[-1].degenerate], idx)


def stat_sd(sign: int) -> Stat:
    return lambda d, idx: sd([d.steering(sign)[-1].tones[i] for i in idx])


def _mean_at(values: Sequence[float], idx: Sequence[int]) -> float:
    return sum(values[i] for i in idx) / len(idx)


# nom, statistique, unité de lecture
COMPARED: tuple[tuple[str, Stat], ...] = (
    ("pente dose → ton, pôle positif (par dose normalisée)", stat_slope(1)),
    ("pente dose → ton, pôle négatif (par dose normalisée)", stat_slope(-1)),
    ("pente dose → ton, tous bras (par dose normalisée)", stat_slope(None)),
    ("pente dose → ton, pôle positif (par unité d'échelle réelle)", stat_slope(1, True)),
    ("pente dose → ton, pôle négatif (par unité d'échelle réelle)", stat_slope(-1, True)),
    ("pente dose → ton, tous bras (par unité d'échelle réelle)", stat_slope(None, True)),
    ("ton à la dose +1", stat_tone(1)),
    ("ton à la dose −1", stat_tone(-1)),
    ("écart-type du ton à la dose +1", stat_sd(1)),
    ("écart-type du ton à la dose −1", stat_sd(-1)),
    ("part de sorties dégénérées à +1", stat_degenerate(1)),
    ("part de sorties dégénérées à −1", stat_degenerate(-1)),
)


@dataclass(frozen=True)
class ModelStats:
    data: ModelData
    rows: list[ArmRow]
    ranges: list[UsefulRange]
    slopes: dict[str, Interval]  # pôle -> pente par dose normalisée
    prompt: str
    sensitivity: dict[float, list[UsefulRange]]  # seuil de dégénérées -> plages utiles
    n_boot: int = N_BOOT
    seed: int = 0


def model_stats(data: ModelData, *, n_boot: int = N_BOOT, seed: int = 0) -> ModelStats:
    rows = arm_rows(data, n_boot=n_boot, seed=seed)
    n = range(len(data.ids))
    slopes = {}
    for name, stat in COMPARED[:3]:
        draws = boot_stat(data, stat, n_boot=n_boot, seed=seed)
        slopes[name] = interval_from_draws(draws, stat(data, list(n)))
    return ModelStats(
        data,
        rows,
        useful_ranges(data, rows),
        slopes,
        prompt_gap(data, n_boot=n_boot, seed=seed),
        {t: useful_ranges(data, rows, t) for t in THRESHOLDS},
        n_boot,
        seed,
    )


@dataclass(frozen=True)
class Comparison:
    stat: str
    bigger: str
    smaller: str
    diff: Interval  # grand - petit
    decided: bool  # IC95 de l'écart sans 0


def compare(a: ModelStats, b: ModelStats) -> list[Comparison]:
    """Écart (a - b) de chaque statistique ; bootstrap apparié sur les prompts communs."""
    out = []
    for name, stat in COMPARED:
        point = stat(a.data, list(range(len(a.data.ids)))) - stat(
            b.data, list(range(len(b.data.ids)))
        )
        diffs = paired_diff_draws(a.data, b.data, stat, n_boot=a.n_boot, seed=a.seed)
        iv = interval_from_draws(diffs, point)
        decided = excludes_zero(iv)
        out.append(Comparison(name, a.data.spec.model_id, b.data.spec.model_id, iv, decided))
    return out


# --- rapport ----------------------------------------------------------------------------------


def _iv(iv: Interval | None, digits: int = 3) -> str:
    if iv is None or math.isnan(iv.low):
        return "n.d."
    return f"{iv.mean:.{digits}f} [{iv.low:.{digits}f} ; {iv.high:.{digits}f}]"


def _p(x: float) -> str:
    return "n.d." if math.isnan(x) else f"{x * 100:.0f} %"


def _range_text(r: UsefulRange) -> str:
    if not r.doses:
        return "aucune"
    doses = ", ".join(f"{d:g}" for d in r.doses)
    scales = ", ".join(f"{s:.3g}" for s in r.scales)
    ratio = f" (rapport {r.ratio:.2f})" if len(r.doses) > 1 else ""
    return f"doses {doses}{ratio} ; échelles {scales}"


def model_section(ms: ModelStats) -> str:
    spec = ms.data.spec
    lines = [
        f"### {spec.model_id}",
        "",
        f"`max_scale` {spec.max_scale:g}, couches {spec.il_start} à {spec.il_end}, "
        f"{len(ms.data.ids)} prompts de ton communs aux bras.",
        "",
        "| Bras | Dose | Échelle réelle | Ton (IC95) | Écart-type | Refus | Dégénérées "
        "| Dose utile |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ms.rows:
        useful = "—" if r.useful is None else ("oui" if r.useful else "non")
        lines.append(
            f"| {r.name} | {r.dose:+g} | {r.scale:+.3g} | {_iv(r.tone)} | {r.sd:.3f} "
            f"| {_p(r.refusal_rate)} | {_p(r.degenerate_rate)} | {useful} |"
        )
    lines += ["", "Plage de doses utile :", ""]
    lines += [f"- pôle {r.pole} : {_range_text(r)}" for r in ms.ranges]
    lines += ["", "Sensibilité au seuil de sorties dégénérées (doses utiles par pôle) :", ""]
    for t, ranges in ms.sensitivity.items():
        per_pole = " ; ".join(
            f"{r.pole} {', '.join(f'{d:g}' for d in r.doses) or 'aucune'}" for r in ranges
        )
        lines.append(f"- seuil {t * 100:.0f} % : {per_pole}")
    lines += ["", "Pente dose → ton (tous les bras d'un pôle, plus le neutre à la dose 0) :", ""]
    for name, iv in ms.slopes.items():
        per_scale = Interval(
            iv.mean / spec.max_scale, iv.low / spec.max_scale, iv.high / spec.max_scale, iv.n
        )
        lines.append(
            f"- {name.split(',', 1)[1].split('(')[0].strip()} : {_iv(iv)} par dose normalisée, "
            f"{_iv(per_scale)} par unité d'échelle réelle"
        )
    lines += ["", f"Face au prompt : {ms.prompt}.", ""]
    return "\n".join(lines)


def comparison_table(comps: Sequence[Comparison]) -> str:
    lines = [
        f"### {comps[0].bigger} moins {comps[0].smaller}",
        "",
        "| Statistique | Écart (IC95 bootstrap) | Lecture |",
        "|---|---|---|",
    ]
    for c in comps:
        reading = "écart non nul" if c.decided else "indéterminé (l'IC contient 0)"
        lines.append(f"| {c.stat} | {_iv(c.diff)} | {reading} |")
    return "\n".join(lines) + "\n"


def build_report(stats: Mapping[str, ModelStats]) -> str:
    """Corps de la section H2 : un tableau par modèle puis les comparaisons deux à deux."""
    keys = [k for k in SIZE_ORDER if k in stats]
    parts = ["## Mesures par modèle", ""]
    parts += [model_section(stats[k]) for k in keys]
    parts += ["## Comparaisons entre modèles", ""]
    for i, ka in enumerate(keys):
        for kb in keys[i + 1 :]:
            parts.append(comparison_table(compare(stats[ka], stats[kb])))
    return "\n".join(parts)


def splice(doc: str, body: str) -> str:
    """Remplace le bloc entre les marqueurs ``mesures`` ; le texte écrit à la main est conservé."""
    a, b = doc.find(MARK_START), doc.find(MARK_END)
    if a < 0 or b < a:
        raise ValueError(f"marqueurs {MARK_START} / {MARK_END} absents du rapport")
    return doc[: a + len(MARK_START)] + "\n\n" + body.strip("\n") + "\n\n" + doc[b:]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="dossier de la campagne 3.5")
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="rapport existant dont le bloc de mesures est remplacé",
    )
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # cp1252 sous Windows quand la sortie est redirigée
        sys.stdout.reconfigure(encoding="utf-8")
    stats = {
        k: model_stats(load_model(args.out, MODELS[k]), n_boot=args.n_boot) for k in SIZE_ORDER
    }
    body = build_report(stats)
    if args.report is None:
        print(body)
    else:
        doc = args.report.read_text(encoding="utf-8")
        args.report.write_text(splice(doc, body), encoding="utf-8", newline="\n")
        print(f"mesures écrites dans : {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
