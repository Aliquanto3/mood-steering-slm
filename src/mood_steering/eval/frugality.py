"""Frugalité (H7) : latence, énergie et CO2 du steering face au prompt joyeux.

Le traqueur est injectable (protocole ``EnergyTracker``) ; l'implémentation réelle
``CodeCarbonTracker`` importe ``codecarbon`` à la demande (extra ``frugal``).
Les lectures par bras sont écrites dans ``energy.json`` du run, à côté des JSONL.

Branchement prévu dans ``scripts/run_bench.py`` (option ``--energy``) ::

    from mood_steering.eval.frugality import (
        CodeCarbonTracker, frugality_section, frugality_summary)

    tracker = CodeCarbonTracker() if args.energy else None
    paths = run_bench(arms, items, steerer, run_id=..., model_id=..., energy_tracker=tracker)
    ...
    if args.energy:
        extra_sections["H7"] = frugality_section(frugality_summary(run_dir))
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from mood_steering.eval.arms import KIND_PROMPT, KIND_STEERING
from mood_steering.eval.runner import load_run
from mood_steering.eval.stats import Interval, bootstrap_ci

ENERGY_FILE = "energy.json"
DEFAULT_COUNTRY_ISO = "FRA"
DEFAULT_TRACKER_DIR = Path("data") / "codecarbon"

LIMIT_NOTE = (
    "CodeCarbon estime la consommation de la machine entière (CPU, GPU, mémoire) à partir "
    "de modèles et de facteurs d'émission du pays : c'est une estimation, pas une mesure "
    "au wattmètre. Une autre charge simultanée sur la machine fausse la mesure ; les "
    "bras doivent être mesurés sur une machine par ailleurs au repos."
)


@dataclass(frozen=True)
class EnergyReading:
    energy_kwh: float
    emissions_kg: float
    duration_s: float


class EnergyTracker(Protocol):
    def start(self) -> None: ...

    def stop(self) -> EnergyReading: ...


class CodeCarbonTracker:
    """Traqueur CodeCarbon hors ligne (aucun appel réseau pour la géolocalisation).

    ``output_dir`` reste sous ``data/`` (ignoré par git). ``save_to_file=False`` par
    défaut : seules les lectures de ``energy.json`` sont conservées ; ``True`` fait
    écrire à CodeCarbon son ``emissions.csv`` dans ``output_dir``.
    """

    def __init__(
        self,
        country_iso_code: str = DEFAULT_COUNTRY_ISO,
        *,
        output_dir: Path = DEFAULT_TRACKER_DIR,
        save_to_file: bool = False,
    ) -> None:
        self.country_iso_code = country_iso_code
        self.output_dir = Path(output_dir)
        self.save_to_file = save_to_file
        self._tracker: Any = None

    def start(self) -> None:
        try:
            from codecarbon import OfflineEmissionsTracker
        except ImportError as exc:  # pragma: no cover - extra absent
            raise RuntimeError("codecarbon absent : uv sync --extra frugal") from exc
        # CodeCarbon exige que le dossier existe, même sans écriture de fichier.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._tracker = OfflineEmissionsTracker(
            country_iso_code=self.country_iso_code,
            output_dir=str(self.output_dir),
            save_to_file=self.save_to_file,
            log_level="error",
            allow_multiple_runs=True,  # un verrou périmé ne doit pas faire échouer le run
        )
        self._tracker.start()

    def stop(self) -> EnergyReading:
        if self._tracker is None:
            raise RuntimeError("stop() sans start()")
        tracker, self._tracker = self._tracker, None
        emissions = tracker.stop()
        data = getattr(tracker, "final_emissions_data", None)
        if data is None:
            raise RuntimeError(
                "CodeCarbon n'a fourni aucune donnée d'émissions (autre instance active ?)"
            )
        return EnergyReading(
            energy_kwh=float(data.energy_consumed),
            emissions_kg=float(emissions if emissions is not None else data.emissions),
            duration_s=float(data.duration),
        )


def write_energy(out_dir: Path, readings: Mapping[str, EnergyReading]) -> Path:
    path = Path(out_dir) / ENERGY_FILE
    payload = {name: asdict(r) for name, r in readings.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_energy(out_dir: Path) -> dict[str, EnergyReading]:
    path = Path(out_dir) / ENERGY_FILE
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: EnergyReading(**vals) for name, vals in raw.items()}


def estimate_tokens(row: Mapping[str, Any]) -> int:
    """``n_tokens`` de la ligne s'il existe, sinon nombre de mots (approximation)."""
    n = row.get("n_tokens")
    if isinstance(n, int) and n > 0:
        return n
    return len(str(row.get("output", "")).split())


@dataclass(frozen=True)
class ArmFrugality:
    arm: str
    kind: str
    n: int
    latency_s: Interval
    latency_per_token_s: Interval | None
    energy_kwh: float | None
    emissions_kg: float | None
    energy_kwh_per_generation: float | None
    emissions_kg_per_generation: float | None


@dataclass(frozen=True)
class Overhead:
    """Surcoût d'un bras steering face au prompt joyeux (None si non mesuré)."""

    arm: str
    latency_ratio: float | None
    latency_diff_s: float
    energy_ratio: float | None
    energy_diff_kwh: float | None
    emissions_ratio: float | None
    emissions_diff_kg: float | None


@dataclass(frozen=True)
class FrugalitySummary:
    arms: dict[str, ArmFrugality]
    overhead: dict[str, Overhead]
    reference: str | None


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def frugality_summary(run_dir: Path, *, n_boot: int = 2000, seed: int = 0) -> FrugalitySummary:
    rows_by_arm = load_run(run_dir)
    energy = load_energy(run_dir)
    arms: dict[str, ArmFrugality] = {}
    for name, all_rows in rows_by_arm.items():
        rows = [r for r in all_rows if isinstance(r.get("latency_s"), (int, float))]
        if not rows:
            continue
        lats = [float(r["latency_s"]) for r in rows]
        per_tok = [
            float(r["latency_s"]) / tokens for r in rows if (tokens := estimate_tokens(r)) > 0
        ]
        reading = energy.get(name)
        n = len(rows)
        arms[name] = ArmFrugality(
            arm=name,
            kind=str(rows[0].get("kind", "")),
            n=n,
            latency_s=bootstrap_ci(lats, n_boot=n_boot, seed=seed),
            latency_per_token_s=(
                bootstrap_ci(per_tok, n_boot=n_boot, seed=seed) if per_tok else None
            ),
            energy_kwh=reading.energy_kwh if reading else None,
            emissions_kg=reading.emissions_kg if reading else None,
            energy_kwh_per_generation=reading.energy_kwh / n if reading else None,
            emissions_kg_per_generation=reading.emissions_kg / n if reading else None,
        )
    ref = next((a for a in arms.values() if a.kind == KIND_PROMPT), None)
    overhead: dict[str, Overhead] = {}
    if ref is not None:
        for a in arms.values():
            if a.kind != KIND_STEERING:
                continue
            overhead[a.arm] = Overhead(
                arm=a.arm,
                latency_ratio=_ratio(a.latency_s.mean, ref.latency_s.mean),
                latency_diff_s=a.latency_s.mean - ref.latency_s.mean,
                energy_ratio=_ratio(a.energy_kwh_per_generation, ref.energy_kwh_per_generation),
                energy_diff_kwh=_diff(a.energy_kwh_per_generation, ref.energy_kwh_per_generation),
                emissions_ratio=_ratio(
                    a.emissions_kg_per_generation, ref.emissions_kg_per_generation
                ),
                emissions_diff_kg=_diff(
                    a.emissions_kg_per_generation, ref.emissions_kg_per_generation
                ),
            )
    return FrugalitySummary(arms=arms, overhead=overhead, reference=ref.arm if ref else None)


def _f(value: float | None, fmt: str = ".4g") -> str:
    return "n/d" if value is None else format(value, fmt)


def _ci(i: Interval | None) -> str:
    return "n/d" if i is None else f"{i.mean:.4g} [{i.low:.4g} ; {i.high:.4g}]"


def frugality_section(summary: FrugalitySummary) -> str:
    """Section Markdown (français) pour ``extra_sections["H7"]``."""
    lines = [
        "Latence par génération (moyenne, IC95 bootstrap), énergie et CO2 par bras.",
        "",
        "| Bras | n | Latence (s) | Latence par mot (s/mot) | Énergie (kWh) | CO2 (kg) "
        "| Énergie/gén. (kWh) | CO2/gén. (kg) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in summary.arms.values():
        lines.append(
            f"| {a.arm} | {a.n} | {_ci(a.latency_s)} | {_ci(a.latency_per_token_s)} "
            f"| {_f(a.energy_kwh)} | {_f(a.emissions_kg)} "
            f"| {_f(a.energy_kwh_per_generation)} | {_f(a.emissions_kg_per_generation)} |"
        )
    lines.append("")
    if summary.reference is None or not summary.overhead:
        lines.append("Surcoût non calculable : bras prompt joyeux ou bras steering absent.")
    else:
        lines += [
            f"Surcoût du steering face à `{summary.reference}` (ratio ; différence) :",
            "",
            "| Bras | Latence | Énergie/gén. | CO2/gén. |",
            "|---|---|---|---|",
        ]
        for o in summary.overhead.values():
            lines.append(
                f"| {o.arm} | x{_f(o.latency_ratio)} ; {o.latency_diff_s:+.4g} s "
                f"| x{_f(o.energy_ratio)} ; {_f(o.energy_diff_kwh, '+.4g')} kWh "
                f"| x{_f(o.emissions_ratio)} ; {_f(o.emissions_diff_kg, '+.4g')} kg |"
            )
    lines += ["", f"Limite de mesure : {LIMIT_NOTE}"]
    if any(a.latency_per_token_s is not None for a in summary.arms.values()):
        lines.append(
            "Les tokens sont comptés par mots de la sortie (approximation) en l'absence "
            "de `n_tokens` dans le JSONL."
        )
    return "\n".join(lines) + "\n"
