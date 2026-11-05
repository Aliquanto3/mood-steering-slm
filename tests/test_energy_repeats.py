"""A4 : mesure H7 répétée (``--energy-repeats``), ordre des bras alterné, traqueur simulé."""

import json
import re

import pytest
from test_run_bench import FakeBackend, FakeTracker, _tone_kw, mcq_items, rb

from mood_steering.eval.frugality import EnergyReading, system_load


class ThreeArms(rb.ModelSpec):
    """Modèle simulé à 3 bras : neutre, prompt joyeux, un bras steering."""

    def arms(self):
        return super().arms()[:3]


def three_arm_spec():
    base = rb.MODELS["0.8b"]
    return ThreeArms("t3", "Tiny-3", base.il_start, base.il_end, base.max_scale, (0.1,))


class ScriptedTracker(FakeTracker):
    """Lecture k (1, 2, …) : k x 0.001 kWh, k x 0.0001 kg, 2 s."""

    def stop(self):
        self.stops += 1
        k = self.stops
        return EnergyReading(energy_kwh=k * 0.001, emissions_kg=k * 0.0001, duration_s=2.0)


class CountingBackend(FakeBackend):
    """Échoue au ``fail_at``-ième appel (arrêt simulé au milieu d'un tour)."""

    def __init__(self, fail_at=None):
        super().__init__()
        self.fail_at = fail_at

    def generate(self, prompt, system, max_tokens, seed):
        if self.fail_at is not None and len(self.calls) + 1 == self.fail_at:
            raise RuntimeError("arrêt simulé")
        return super().generate(prompt, system, max_tokens, seed)


def proto():
    return rb.Protocol(n_tone=10, n_mcq=6, secondary_phases=("tone",))


def campaign(path, spec, backend=None, tracker=None, repeats=1, log=None):
    kw = _tone_kw()
    if backend is not None:
        kw["backend_factory"] = lambda s: backend
    if log is not None:
        kw["log"] = log
    return rb.run_campaign(
        [spec], proto(), path, mcq_items(), energy_tracker=tracker, energy_repeats=repeats, **kw
    )


def test_repeats_alternate_arm_order_and_keep_single_scored_pass(tmp_path):
    spec = three_arm_spec()
    names = [a.name for a in spec.arms()]
    tracker = ScriptedTracker()
    logs = []
    assert campaign(tmp_path, spec, tracker=tracker, repeats=3, log=logs.append)
    units = rb.load_energy_units(tmp_path)
    assert set(units) == {rb.round_key(spec.model_id, n, r) for n in names for r in range(3)}
    assert tracker.starts == tracker.stops == 9
    # ordre de mesure : tour 0 (chauffe) et tour 1 dans l'ordre des bras, tour 2 inversé
    seen = []
    for m in logs:
        hit = re.match(r"Tiny-3 tone (?:tour (\d+) )?(\S+) : ", m)
        assert hit, m
        seen.append((int(hit.group(1) or 0), hit.group(2)))
    assert seen == (
        [(0, n) for n in names] + [(1, n) for n in names] + [(2, n) for n in reversed(names)]
    )
    # la k-ième lecture vaut k x 0.001 : le bras steering est lu en 3e, 6e puis 7e
    steer = names[2]
    for rnd, kwh in ((0, 0.003), (1, 0.006), (2, 0.007)):
        got = units[rb.round_key(spec.model_id, steer, rnd)]["energy_kwh"]
        assert got == pytest.approx(kwh)
    # sorties notées : un seul passage (tour 0) dans tone/ ; les tours suivants à part
    tone_dir = tmp_path / spec.model_id / "tone"
    assert sorted(p.name for p in tone_dir.glob("*.jsonl")) == sorted(f"{n}.jsonl" for n in names)
    assert all(len(rb.read_jsonl(p)) == 10 for p in tone_dir.glob("*.jsonl"))
    r1 = rb.read_jsonl(rb.round_path(tmp_path, spec.model_id, steer, 1))
    assert len(r1) == 10 and {r["energy_round"] for r in r1} == {1}
    assert rb.round_path(tmp_path, spec.model_id, steer, 0) == tone_dir / f"{steer}.jsonl"
    assert not any(p.startswith("tone_r") for p in rb.PHASES)


def test_repeat_one_is_unchanged_and_ignored_without_tracker(tmp_path):
    spec = three_arm_spec()
    tracker = FakeTracker()
    campaign(tmp_path, spec, tracker=tracker)
    assert tracker.stops == 3 and len(rb.load_energy_units(tmp_path)) == 3
    assert not list((tmp_path / spec.model_id).glob("tone_r*"))
    other = tmp_path / "sans-traqueur"
    campaign(other, spec, repeats=3)
    assert not list((other / spec.model_id).glob("tone_r*"))
    assert rb.repeat_units(spec, proto(), other, 1) == []


def test_repeats_resume_in_the_middle_of_round_two(tmp_path):
    spec = three_arm_spec()
    names = [a.name for a in spec.arms()]
    tracker = ScriptedTracker()
    # tours 0 et 1 : 60 appels ; tour 2 (inversé) : bras 2 complet (70), échec au 5e item du
    # bras 1 (4 faits)
    with pytest.raises(RuntimeError, match="arrêt simulé"):
        campaign(tmp_path, spec, CountingBackend(fail_at=75), tracker, repeats=3)
    before = rb.load_energy_units(tmp_path)
    assert len(before) == 3 + 3 + 2  # tours 0 et 1 finis ; tour 2 : bras 2 fini, bras 1 partiel
    k1 = rb.round_key(spec.model_id, names[1], 2)
    assert rb.round_path(tmp_path, spec.model_id, names[2], 2).exists()
    assert not rb.round_path(tmp_path, spec.model_id, names[1], 2).exists()
    backend = CountingBackend()
    assert campaign(tmp_path, spec, backend, tracker, repeats=3)
    assert len(backend.calls) == 16  # 6 items restants du bras 1 (4 faits) + 10 du bras 0
    after = rb.load_energy_units(tmp_path)
    assert len(after) == 9
    for r in range(2):  # lectures des tours finis conservées à l'identique
        for n in names:
            key = rb.round_key(spec.model_id, n, r)
            assert after[key] == before[key]
    # le tour partiel cumule la lecture d'avant l'arrêt et celle de la reprise
    assert after[k1]["energy_kwh"] == pytest.approx(before[k1]["energy_kwh"] + 0.009)
    rows = rb.read_jsonl(rb.round_path(tmp_path, spec.model_id, names[1], 2))
    assert [r["item_id"] for r in rows] == [f"tone{i:03d}" for i in range(10)]


def test_repeats_resume_with_other_k_is_refused(tmp_path):
    spec = three_arm_spec()
    rt = {"llama_cpp": "0.3", "device": "CPU", "energy": True, "energy_country": "FRA"}
    k3 = {**rt, "energy_repeats": 3}
    (tmp_path / "runs.jsonl").write_text(json.dumps(k3) + "\n", "utf-8")
    rb.check_runtime(tmp_path, k3)
    for other in (rt, {**rt, "energy_repeats": 5}):
        with pytest.raises(ValueError, match="energy-repeats"):
            rb.check_runtime(tmp_path, other)
    # campagne K = 1 sans le champ : reprise à K = 1 acceptée, à K = 3 refusée
    (tmp_path / "runs.jsonl").write_text(json.dumps(rt) + "\n", "utf-8")
    rb.check_runtime(tmp_path, {**rt, "energy_repeats": 1})
    with pytest.raises(ValueError, match="energy-repeats"):
        rb.check_runtime(tmp_path, k3)
    # le manifeste fige K et la règle d'ordre, sans changer les manifestes à K = 1
    args = ([spec], proto(), mcq_items(), {spec.key: {}})
    plain = rb.build_manifest(*args)
    k3m = rb.build_manifest(*args, energy_repeats=3)
    assert "energy_repeats" not in plain and "energy_round_order" not in plain
    assert k3m["energy_repeats"] == 3 and k3m["energy_round_order"] == rb.ENERGY_ROUND_ORDER
    rb.check_manifest(tmp_path / "m", k3m, resume=False)
    with pytest.raises(ValueError, match="energy_repeats"):
        rb.check_manifest(tmp_path / "m", plain, resume=True)


def test_h7_repeated_gives_means_ci_and_paired_ratio_without_verdict(tmp_path):
    spec = three_arm_spec()
    names = [a.name for a in spec.arms()]
    campaign(tmp_path, spec, tracker=ScriptedTracker(), repeats=3)
    load = {"gpu": {"utilization_pct": 3, "memory_used_mib": 900}, "cpu_percent": 7.0}
    (tmp_path / "runs.jsonl").write_text(json.dumps({"load": load}) + "\n", "utf-8")
    text = rb.h7_section(tmp_path, [spec], proto())
    # tour 0 (chauffe) hors statistiques : tours 1 et 2 seulement
    assert "sur 2 tour(s) de mesure (tours 1 à 2" in text and "Student" in text
    # joyeux : lectures 5 et 8 (x 0.001 kWh pour 10 générations) -> 6.5e-4 ; IC de Student à
    # 1 ddl : 6.5e-4 -/+ 12.706 x sd / sqrt(2), sd = 2.1213e-4 -> [-1.256e-3 ; 2.556e-3]
    assert "| prompt_joyful | 2 | 0.00065 [-0.001256 ; 0.002556] |" in text
    # ratio apparié par tour steering / joyeux : 6/5 et 7/8 -> moyenne 1.0375
    assert f"| {names[2]} | 2 | x1.038 [" in text
    assert "Ratio apparié par tour face à `prompt_joyful`" in text
    assert "Pas de verdict H7" in text and "A6" in text
    assert "Relevé de charge au lancement : GPU 3 % (900 MiB), CPU 7 %." in text
    assert "ordre inverse" in text
    assert "une seule lecture par bras" not in text


def test_h7_single_reading_report_has_no_round_section(tmp_path):
    spec = three_arm_spec()
    campaign(tmp_path, spec, tracker=FakeTracker())
    text = rb.h7_section(tmp_path, [spec], proto())
    assert "tour(s) de mesure" not in text and "une seule lecture par bras" in text
    units = rb.load_energy_units(tmp_path)
    assert set(units) == {f"{spec.model_id}/tone/{a.name}" for a in spec.arms()}
    assert rb.repeated_energy(tmp_path, spec, units) == {}


def test_cli_energy_repeats_and_tone_only():
    assert rb.parse_args([]).energy_repeats == 1
    assert rb.parse_args(["--energy", "--energy-repeats", "5"]).energy_repeats == 5
    for bad in (["--energy-repeats", "3"], ["--energy", "--energy-repeats", "0"], ["--tone-only"]):
        with pytest.raises(SystemExit):
            rb.parse_args(bad)
    # phase ton seule : cœur autorisé, plancher des QCM levé, planchers du ton maintenus
    ok = ["--energy", "--tone-only", "--models", "4b"]
    assert rb.parse_args([*ok, "--n-mcq", "10"]).tone_only
    with pytest.raises(SystemExit):
        rb.parse_args([*ok, "--n-tone", "10"])


def test_tone_only_phases_and_manifest():
    spec = rb.MODELS["4b"]
    only = rb.Protocol(tone_only=True)
    assert rb.phases_for(spec, only) == ("tone",)
    assert rb.phases_for(spec, rb.Protocol()) == rb.PHASES
    plain = rb.build_manifest([spec], rb.Protocol(), mcq_items(), {spec.key: {}})
    tone = rb.build_manifest([spec], only, mcq_items(), {spec.key: {}})
    assert "tone_only" not in plain["protocol"] and tone["protocol"]["tone_only"] is True
    assert tone["models"]["4b"]["phases"] == ["tone"]


def test_tone_only_report_header(tmp_path):
    target = tmp_path / "f.md"
    rb.write_frugality_report(
        target,
        "corps H7",
        [rb.MODELS["4b"]],
        rb.Protocol(tone_only=True),
        {"device": "GPU (toutes les couches)", "energy_repeats": 5},
        log=lambda s: None,
    )
    text = target.read_text("utf-8")
    assert (
        "Mesure H7, Qwen3.5-4B-Q4_K_M, GPU (toutes les couches), phase ton seule, 5 tour(s)" in text
    )
    assert "Pas de verdict" in text and text.rstrip().endswith("corps H7")


class Done:
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.returncode = stdout, returncode


def test_system_load_reads_gpu_and_cpu():
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return Done("37, 1024\n")

    got = system_load(run=run, cpu_percent=lambda: 12.5)
    assert got == {"gpu": {"utilization_pct": 37, "memory_used_mib": 1024}, "cpu_percent": 12.5}
    assert calls[0][0] == "nvidia-smi"


def test_system_load_degrades_without_nvidia_smi_or_on_garbage():
    def missing(cmd, **kw):
        raise FileNotFoundError("nvidia-smi")

    assert system_load(run=missing, cpu_percent=lambda: 1.0) == {"gpu": None, "cpu_percent": 1.0}
    assert system_load(run=lambda c, **k: Done("", 9), cpu_percent=lambda: 2.0)["gpu"] is None
    assert (
        system_load(run=lambda c, **k: Done("N/A, N/A\n"), cpu_percent=lambda: 2.0)["gpu"] is None
    )
    assert system_load(run=lambda c, **k: Done(""), cpu_percent=lambda: 2.0)["gpu"] is None


def test_student_ci_for_few_rounds():
    from mood_steering.eval.stats import student_ci

    iv = student_ci([1.0, 2.0, 3.0])  # sd 1, t(2 ddl) = 4.303 -> demi-largeur 2.4843
    assert iv.mean == pytest.approx(2.0)
    assert (iv.low, iv.high) == pytest.approx((-0.4843, 4.4843), abs=1e-4)
    single = student_ci([0.5])
    assert single.mean == 0.5 and single.low != single.low  # nan : pas d'IC sur une valeur


def test_round_started_without_reading_is_remeasured_and_missing_reading_reported(tmp_path):
    spec = three_arm_spec()
    names = [a.name for a in spec.arms()]
    assert campaign(tmp_path, spec, tracker=ScriptedTracker(), repeats=3)
    # arrêt brutal simulé au tour 2 : JSONL partiel sans lecture
    path = rb.round_path(tmp_path, spec.model_id, names[0], 2)
    rows = path.read_text("utf-8").splitlines(keepends=True)
    path.unlink()
    path.with_name(path.name + ".part").write_text("".join(rows[:3]), "utf-8")
    units = rb.load_energy_units(tmp_path)
    key = rb.round_key(spec.model_id, names[0], 2)
    del units[key]
    rb.save_energy_units(tmp_path, units)
    assert not (tmp_path / (rb.ENERGY_UNITS_FILE + ".part")).exists()  # écriture atomique
    backend, logs = CountingBackend(), []
    assert campaign(tmp_path, spec, backend, ScriptedTracker(), repeats=3, log=logs.append)
    assert len(backend.calls) == 10  # tour remesuré en entier
    assert key in rb.load_energy_units(tmp_path)
    assert any("remesuré" in m for m in logs)
    # tour publié sans lecture (traqueur en échec) : signalé dans le rapport
    units = rb.load_energy_units(tmp_path)
    del units[rb.round_key(spec.model_id, names[1], 1)]
    rb.save_energy_units(tmp_path, units)
    text = rb.h7_section(tmp_path, [spec], proto())
    assert (
        "Tours sans lecture d'énergie (hors statistiques, à remesurer avec --resume) : "
        f"tour 1 {names[1]}." in text
    )


def test_compute_minutes_ignores_energy_rounds(tmp_path):
    spec = three_arm_spec()
    campaign(tmp_path, spec, tracker=FakeTracker())
    base = rb.compute_minutes(tmp_path, spec)
    extra = tmp_path / spec.model_id / "tone_r1" / "neutral.jsonl"
    extra.parent.mkdir()
    extra.write_text(json.dumps({"latency_s": 6000.0}) + "\n", "utf-8")
    assert rb.compute_minutes(tmp_path, spec) == base


def test_cli_even_repeats_refused_and_tone_only_report_kept_out_of_docs(tmp_path):
    with pytest.raises(SystemExit):
        rb.parse_args(["--energy", "--energy-repeats", "4"])
    ok = ["--energy", "--tone-only", "--models", "4b", "--out", str(tmp_path)]
    assert rb.parse_args(ok).report == tmp_path / "frugalite.md"
    with pytest.raises(SystemExit):
        rb.parse_args([*ok, "--report", str(rb.DEFAULT_REPORT)])


def test_system_load_never_raises_on_cpu_error():
    def denied():
        raise PermissionError("accès refusé")

    got = system_load(run=lambda c, **k: Done(""), cpu_percent=denied)
    assert got == {"gpu": None, "cpu_percent": None}
