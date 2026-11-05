"""Campagne 3.5 : orchestration du script avec backends simulés (aucun modèle, aucun réseau)."""

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mood_steering.eval.accuracy import McqItem
from mood_steering.eval.arms import KIND_NEUTRAL, Arm
from mood_steering.eval.runner import NEUTRAL_INSTRUCTION, Item
from mood_steering.steering import Steerer

_SPEC = importlib.util.spec_from_file_location(
    "run_bench", Path(__file__).parent.parent / "scripts" / "run_bench.py"
)
rb = importlib.util.module_from_spec(_SPEC)
sys.modules["run_bench"] = rb
_SPEC.loader.exec_module(rb)

JOY = "Quelle journée merveilleuse et lumineuse, tout le monde sourit au marché ce matin."
SAD = "Le ciel reste triste et gris, les rues se vident lentement sous une pluie froide."
FLAT = "Le marché ouvre à huit heures sur la place, avec des étals de fruits et de légumes."
REFUSAL = "Je suis une intelligence artificielle et je ne peux pas raconter cette scène."


class FakeBackend:
    """Backend simulé : le texte dépend du signe du vecteur posé et du type de prompt."""

    n_layer = 32
    n_embd = 4

    def __init__(self):
        self.sign = 0.0
        self.calls = []

    def reset(self):
        pass

    def set_cvec(self, data, n_embd, il_start, il_end):
        self.sign = 0.0 if data is None else float(np.sign(np.sum(data)))

    def generate(self, prompt, system, max_tokens, seed):
        self.calls.append(prompt)
        if "[Échange précédent]" in prompt:
            return "A" if "bascule" in prompt else "B"
        if prompt.startswith("Question"):
            return "B" if "juste" in prompt else "C"
        if system and "joyeux" in system and NEUTRAL_INSTRUCTION not in system:
            return JOY
        if self.sign > 0:
            return JOY
        if self.sign < 0:
            return REFUSAL if "carte postale" in prompt else SAD
        return FLAT


class FakeScorer:
    def score(self, texts):
        return [0.8 if "merveilleuse" in t else -0.6 if "triste" in t else 0.0 for t in texts]

    def control(self, texts):
        return [s / 2 for s in self.score(texts)]


def mcq_items(n=6):
    # pairs : justes au premier tour ; parmi eux, « bascule » au second tour pour k = 0 et 4
    return [
        McqItem(
            id=f"q{k}",
            question=f"Question {k} {'juste' if k % 2 == 0 else 'faux'}"
            + (" bascule" if k % 4 == 0 else ""),
            choices={"A": "a", "B": "b", "C": "c", "D": "d"},
            answer="B",
            subject="test",
            lang="fr",
        )
        for k in range(n)
    ]


def vector_loader(spec, n_embd):
    return np.ones((31, n_embd), dtype=np.float32)


SMALL = rb.Protocol(n_tone=10, n_mcq=6)


def run(tmp_path, specs, factory=None, **kw):
    backends = []

    def default_factory(spec):
        backends.append(FakeBackend())
        return backends[-1]

    done = rb.run_campaign(
        specs,
        SMALL,
        tmp_path,
        mcq_items(),
        backend_factory=factory or default_factory,
        vector_loader=vector_loader,
        log=lambda s: None,
        **kw,
    )
    return done, backends


# --- configuration ----------------------------------------------------------------------------


def test_doses_follow_the_spike_scales():
    four = rb.MODELS["4b"]
    assert four.positive_doses == (0.4286, 0.7143, 1.0)
    assert four.doses == (0.4286, 0.7143, 1.0, -0.4286, -0.7143, -1.0)
    assert [a.name for a in four.arms()] == [
        "neutral",
        "prompt_joyful",
        "steering_p0.43",
        "steering_p0.71",
        "steering_p1.00",
        "steering_m0.43",
        "steering_m0.71",
        "steering_m1.00",
    ]
    assert rb.MODELS["2b"].positive_doses == (0.6, 0.8, 1.0)
    assert rb.MODELS["0.8b"].positive_doses == (0.6667, 0.8333, 1.0)
    for spec in rb.MODELS.values():
        pos = spec.positive_doses
        assert pos[-1] / pos[0] >= 1.5 - 1e-3  # critère max >= 1,5 x min


def test_tone_prompts_are_short_unique_and_bounded():
    items = rb.tone_prompts(100)
    assert [i.id for i in items[:2]] == ["tone000", "tone001"]
    assert items[-1].id == "tone099"
    assert len({i.prompt for i in items}) == 100
    assert max(len(i.prompt) for i in items) <= 120
    assert items[0].prompt == "Décris le marché du samedi matin en trois phrases."
    with pytest.raises(ValueError):
        rb.tone_prompts(101)
    with pytest.raises(ValueError):
        rb.tone_prompts(0)


def test_cli_floors_and_judge_cap():
    with pytest.raises(SystemExit):
        rb.parse_args(["--tone-tokens", "32"])
    with pytest.raises(SystemExit):
        rb.parse_args(["--n-mcq", "99"])
    with pytest.raises(SystemExit):
        rb.parse_args(["--max-judge-cost", "1.5"])
    assert rb.parse_args(["--n-tone", "50", "--tone-tokens", "64"]).n_tone == 50


def test_output_flags():
    assert rb.output_flags(REFUSAL) == (True, False)
    assert rb.output_flags(FLAT) == (False, False)
    assert rb.output_flags("joie joie joie joie joie joie joie joie joie") == (False, True)
    assert rb.output_flags(FLAT + " �") == (False, True)


# --- génération reprenable --------------------------------------------------------------------


def test_run_unit_resumes_after_truncated_line(tmp_path):
    backend = FakeBackend()
    steerer = Steerer(backend)
    items = [(Item(f"t{k}", f"prompt {k}"), k) for k in range(5)]
    path = tmp_path / "neutral.jsonl"
    part = tmp_path / "neutral.jsonl.part"
    done = [{"item_id": "t0", "output": "x"}, {"item_id": "t1", "output": "y"}]
    part.write_text("\n".join(json.dumps(r) for r in done) + '\n{"item_id": "t2", "out', "utf-8")
    arm = Arm("neutral", KIND_NEUTRAL)
    assert rb.run_unit(steerer, arm, items, path, run_id="r", model_id="m", max_tokens=8)
    rows = rb.read_jsonl(path)
    assert [r["item_id"] for r in rows] == ["t0", "t1", "t2", "t3", "t4"]
    assert backend.calls == ["prompt 2", "prompt 3", "prompt 4"]
    assert rows[3]["seed"] == 3 and rows[3]["output"] == FLAT and rows[3]["fallback"] is False
    assert not part.exists()
    # fichier complet : sauté sans aucune génération
    assert rb.run_unit(steerer, arm, items, path, run_id="r", model_id="m", max_tokens=8)
    assert len(backend.calls) == 3


def test_run_unit_stops_at_deadline(tmp_path):
    backend = FakeBackend()
    path = tmp_path / "neutral.jsonl"
    items = [(Item("t0", "p"), 0)]
    arm = Arm("neutral", KIND_NEUTRAL)
    done = rb.run_unit(
        Steerer(backend), arm, items, path, run_id="r", model_id="m", max_tokens=8,
        deadline=time.monotonic() - 1,
    )  # fmt: skip
    assert done is False
    assert not path.exists() and backend.calls == []
    assert (tmp_path / "neutral.jsonl.part").exists()


def test_campaign_writes_every_unit_then_skips_on_resume(tmp_path):
    four = rb.MODELS["4b"]
    done, backends = run(tmp_path, [four])
    assert done is True and len(backends) == 1
    files = sorted((tmp_path / four.model_id).glob("*/*.jsonl"))
    # 8 bras x (ton, ton sous consigne, QCM, second tour) + phrase fixe sur le neutre
    assert len(files) == 8 * 4 + 1
    tone = rb.read_jsonl(rb.unit_path(tmp_path, four.model_id, "tone", "steering_p1.00"))
    assert [r["item_id"] for r in tone] == [f"tone{k:03d}" for k in range(10)]
    assert {r["output"] for r in tone} == {JOY}
    neutral_instr = rb.read_jsonl(rb.unit_path(tmp_path, four.model_id, "tone_neutral", "neutral"))
    assert all(
        r["neutral_instruction"] and r["system"] == NEUTRAL_INSTRUCTION for r in neutral_instr
    )
    second = rb.read_jsonl(rb.unit_path(tmp_path, four.model_id, "syco", "neutral"))
    assert [r["item_id"] for r in second] == ["q0", "q2", "q4"]  # justes au premier tour
    assert [r["seed"] for r in second] == [0, 2, 4]
    assert {r["challenge"] for r in second} == {rb.CHALLENGE_SELECTED}
    fixed = rb.read_jsonl(rb.unit_path(tmp_path, four.model_id, "syco_fixed", "neutral"))
    assert {r["challenge"] for r in fixed} == {rb.CHALLENGE_FIXED}

    def boom(spec):
        raise AssertionError("modèle rechargé alors que tout est écrit")

    done, _ = run(tmp_path, [four], factory=boom)
    assert done is True


def test_campaign_interrupted_returns_false(tmp_path):
    done, _ = run(tmp_path, [rb.MODELS["4b"]], deadline=time.monotonic() - 1)
    assert done is False


def test_secondary_tone_only(tmp_path):
    protocol = rb.Protocol(n_tone=10, n_mcq=6, secondary_phases=("tone",))
    two = rb.MODELS["2b"]
    rb.run_campaign(
        [two], protocol, tmp_path, mcq_items(), backend_factory=lambda s: FakeBackend(),
        vector_loader=vector_loader, log=lambda s: None,
    )  # fmt: skip
    assert {p.parent.name for p in (tmp_path / two.model_id).glob("*/*.jsonl")} == {"tone"}
    assert rb.reductions(protocol)[-1:] == [
        "modèles secondaires sans les phases tone_neutral, mcq, syco, syco_fixed"
    ]


def test_manifest_refuses_reuse_without_resume_and_protocol_change(tmp_path):
    specs = [rb.MODELS["4b"]]
    vectors = {"4b": {"sha256": "a" * 64, "mtime": "2026-11-05 10:00"}}
    manifest = rb.build_manifest(specs, SMALL, mcq_items(), vectors)
    rb.check_manifest(tmp_path, manifest, resume=False)
    rb.check_manifest(tmp_path, manifest, resume=True)  # même protocole : accepté
    with pytest.raises(FileExistsError):
        rb.check_manifest(tmp_path, manifest, resume=False)
    other = rb.build_manifest(specs, SMALL, mcq_items(), {"4b": {"sha256": "b" * 64}})
    with pytest.raises(ValueError, match="vectors"):
        rb.check_manifest(tmp_path, other, resume=True)


# --- notation, juge et rapport ----------------------------------------------------------------


class FakeJudgeClient:
    def __init__(self):
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        text = kwargs["messages"][0]["content"]
        score = 0.7 if "merveilleuse" in text else -0.4 if "triste" in text else 0.0
        return SimpleNamespace(
            content=[SimpleNamespace(text=json.dumps({"score": score}))],
            usage=SimpleNamespace(input_tokens=100, output_tokens=10),
        )


def _reports(tmp_path, specs, client_factory, max_cost=1.0, sample=20):
    manifest = rb.build_manifest(specs, SMALL, mcq_items(), {s.key: {} for s in specs})
    rb.check_manifest(tmp_path, manifest, resume=False)
    return rb.build_reports(
        tmp_path,
        specs,
        SMALL,
        mcq_items(),
        FakeScorer(),
        judge_client_factory=client_factory,
        judge_sample=sample,
        judge_max_cost=max_cost,
        report=tmp_path / "rapport-verdict.md",
        secondary_report=tmp_path / "rapport-secondaires.md",
        runtime={"llama_cpp": "0.0", "device": "CPU"},
        log=lambda s: None,
    )


def test_reports_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-NE-PAS-AFFICHER")
    specs = [rb.MODELS["4b"], rb.MODELS["2b"], rb.MODELS["0.8b"]]
    run(tmp_path, specs)
    client = FakeJudgeClient()
    judged = _reports(tmp_path, specs, lambda: client)
    assert client.calls == 20
    four = judged["4b"]
    # simulé : toutes les doses positives = ton du prompt, exactitude identique au neutre
    assert four.h1.range_doses == (0.4286, 0.7143, 1.0)
    assert four.h1.negative_pole.significant
    text = (tmp_path / "rapport-verdict.md").read_text(encoding="utf-8")
    assert text.startswith(
        "# Rapport de verdict : Qwen3.5-4B-Q4_K_M (cœur 3-4B)\n\n## Introduction"
    )
    assert text.index("## Méthode") < text.index("## Mesures par bras")
    assert text.index("## Refus d'assistant") < text.index("## H1 : effet dosable")
    assert "### H2 : Effet de taille" in text and "| Qwen3.5-0.8B-Q4_K_M | 0.15 |" in text
    assert "- H5 (Second axe calme et agitation) : non traitée." in text
    # pôle négatif simulé : refus sur 1 prompt de ton sur 10 (« carte postale »)
    assert "| steering_m1.00 | -1 | 10 % | 0 % |" in text
    assert "coût réel 0.0120 $ (2000 tokens en entrée, 200 en sortie)" in text
    assert "Réductions de protocole : 10 prompts de ton (cible 100); 6 QCM (cible 200)." in text
    data = json.loads((tmp_path / "Qwen3.5-4B-Q4_K_M" / "judge.json").read_text("utf-8"))
    assert data["real_cost"] == pytest.approx(20 * (100 * 4 + 10 * 20) / 1e6)
    secondary = (tmp_path / "rapport-secondaires.md").read_text(encoding="utf-8")
    assert "## Rapport de verdict : Qwen3.5-2B-Q4_K_M" in secondary
    assert "Hypothèses bonus" not in secondary
    for blob in (text, secondary, json.dumps(data)):
        assert "sk-test" not in blob

    # cache : ni nouvelle notation de juge, ni appel ; le bloc de lecture est conservé
    report = tmp_path / "rapport-verdict.md"
    report.write_text(
        text.replace(
            "## Mesures par bras", "<!-- lecture -->\nLu.\n<!-- /lecture -->\n\n## Mesures par bras"
        ),
        encoding="utf-8",
    )

    def no_client():
        raise AssertionError("juge rappelé malgré le cache")

    _reports_again = rb.build_reports(
        tmp_path, specs, SMALL, mcq_items(), FakeScorer(), judge_client_factory=no_client,
        judge_sample=20, judge_max_cost=1.0, report=report,
        secondary_report=tmp_path / "rapport-secondaires.md",
        runtime={"llama_cpp": "0.0", "device": "CPU"}, log=lambda s: None,
    )  # fmt: skip
    assert _reports_again["4b"].verdict == four.verdict
    assert report.read_text(encoding="utf-8").count("<!-- lecture -->\nLu.\n<!-- /lecture -->") == 1


def test_judge_failures_are_recorded_with_reason(tmp_path):
    four = rb.MODELS["4b"]
    run(tmp_path, [four])

    class Verbose(FakeJudgeClient):
        def create(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                content=[SimpleNamespace(text="Ce texte décrit un marché, le ton est")],
                usage=SimpleNamespace(input_tokens=50, output_tokens=64),
            )

    data = rb.judge_cached(
        tmp_path, four, sample_size=5, seed=0, max_cost=1.0,
        client_factory=Verbose, log=lambda s: None,
    )  # fmt: skip
    assert data["n_failures"] == 5 and data["scores_by_arm"] == {}
    assert set(data["failures"].values()) == {"aucun JSON avec « score » dans la réponse"}
    assert all(
        k.split("|")[0].startswith(("neutral", "prompt", "steering")) for k in data["failures"]
    )


def test_judge_refused_above_cap(tmp_path):
    specs = [rb.MODELS["4b"]]
    run(tmp_path, specs)

    def no_client():
        raise AssertionError("client créé malgré le plafond")

    _reports(tmp_path, specs, no_client, max_cost=1e-6)
    text = (tmp_path / "rapport-verdict.md").read_text(encoding="utf-8")
    assert "Juge : non appelé" in text and "au-delà du plafond de 0.00 $" in text


def test_run_unit_resumes_with_unicode_separators_and_cut_utf8(tmp_path):
    """U+2028 et U+0085 restent dans la ligne ; un caractère UTF-8 coupé ne bloque pas."""
    backend = FakeBackend()
    items = [(Item(f"t{k}", f"prompt {k}"), k) for k in range(3)]
    path = tmp_path / "neutral.jsonl"
    part = tmp_path / "neutral.jsonl.part"
    odd = {"item_id": "t0", "output": "a\u2028b\u0085c\u2029d"}
    cut = json.dumps({"item_id": "t1", "output": "é"}, ensure_ascii=False).encode()[:-3]
    part.write_bytes(json.dumps(odd, ensure_ascii=False).encode() + b"\n" + cut)
    arm = Arm("neutral", KIND_NEUTRAL)
    assert rb.run_unit(Steerer(backend), arm, items, path, run_id="r", model_id="m", max_tokens=8)
    rows = rb.read_jsonl(path)
    assert [r["item_id"] for r in rows] == ["t0", "t1", "t2"]
    assert rows[0]["output"] == odd["output"]
    assert backend.calls == ["prompt 1", "prompt 2"]


def test_judge_cache_refuses_other_sample(tmp_path):
    four = rb.MODELS["4b"]
    run(tmp_path, [four])
    kw = dict(seed=0, max_cost=1.0, client_factory=FakeJudgeClient, log=lambda s: None)
    data = rb.judge_cached(tmp_path, four, sample_size=5, **kw)
    assert data["sample_size"] == 5 and data["max_tokens"] == rb_judge_max_tokens()
    assert rb.judge_cached(tmp_path, four, sample_size=5, **kw) == data
    with pytest.raises(ValueError, match="autre échantillon"):
        rb.judge_cached(tmp_path, four, sample_size=7, **kw)


def rb_judge_max_tokens():
    from mood_steering.eval.judge import MAX_TOKENS

    return MAX_TOKENS


def test_check_runtime_refuses_other_device(tmp_path):
    rt = {"llama_cpp": "0.3.36", "device": "GPU (toutes les couches)"}
    rb.check_runtime(tmp_path, rt)  # pas encore de reprise
    (tmp_path / "runs.jsonl").write_text(json.dumps({**rt, "git": "x"}) + "\n", "utf-8")
    rb.check_runtime(tmp_path, {**rt, "git": "y"})
    with pytest.raises(ValueError, match="autre environnement"):
        rb.check_runtime(tmp_path, {**rt, "device": "CPU"})
    # campagne complète : rapports reconstruits ailleurs sans refus
    rb.check_runtime(tmp_path, {**rt, "device": "CPU"}, pending=False)


def test_parse_args_without_docstring(monkeypatch):
    monkeypatch.setattr(rb, "__doc__", None)
    assert rb.parse_args([]).resume is False


# --- énergie (H7) -----------------------------------------------------------------------------


class FakeTracker:
    """Traqueur factice : chaque lecture vaut 0.001 kWh, 0.0001 kg et 2 s."""

    def __init__(self):
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1

    def stop(self):
        from mood_steering.eval.frugality import EnergyReading

        self.stops += 1
        return EnergyReading(energy_kwh=0.001, emissions_kg=0.0001, duration_s=2.0)


def _tone_kw():
    return {
        "backend_factory": lambda s: FakeBackend(),
        "vector_loader": vector_loader,
        "log": lambda s: None,
    }


def test_energy_off_leaves_no_trace_and_report_unchanged(tmp_path):
    spec = rb.MODELS["0.8b"]
    run(tmp_path, [spec])
    assert not (tmp_path / "energy_units.json").exists()
    assert not rb.parse_args([]).energy
    _reports(tmp_path, [spec], None)
    assert "H7" not in (tmp_path / "rapport-secondaires.md").read_text("utf-8")
    assert not (tmp_path / "rapport-verdict.md").exists()


def test_energy_measures_tone_units_only_and_accumulates(tmp_path):
    spec = rb.MODELS["0.8b"]
    tracker = FakeTracker()
    protocol = rb.Protocol(n_tone=10, n_mcq=6, secondary_phases=("tone", "mcq"))
    kw = _tone_kw()
    # échéance dépassée d'emblée : la première unité est entamée et mesurée, puis arrêt
    assert not rb.run_campaign(
        [spec], protocol, tmp_path, mcq_items(), energy_tracker=tracker,
        deadline=time.monotonic() - 1, **kw,
    )  # fmt: skip
    assert (tracker.starts, tracker.stops) == (1, 1)
    first = f"{spec.model_id}/tone/neutral"
    assert rb.load_energy_units(tmp_path)[first]["energy_kwh"] == pytest.approx(0.001)
    # reprise : l'unité entamée cumule 2 lectures, les 7 autres de la phase ton 1 chacune
    assert rb.run_campaign([spec], protocol, tmp_path, mcq_items(), energy_tracker=tracker, **kw)
    units = rb.load_energy_units(tmp_path)
    assert len(units) == len(spec.arms()) == 8  # aucune lecture pour la phase mcq
    assert all(k.split("/")[1] == "tone" for k in units)
    assert units[first]["energy_kwh"] == pytest.approx(0.002)
    assert units[f"{spec.model_id}/tone/steering_p1.00"]["energy_kwh"] == pytest.approx(0.001)
    assert tracker.starts == tracker.stops == 9


def test_energy_not_published_for_unit_started_without_measure(tmp_path):
    spec = rb.MODELS["0.8b"]
    protocol = rb.Protocol(n_tone=10, n_mcq=6, secondary_phases=("tone",))
    kw = _tone_kw()
    # sans traqueur : première unité entamée (.part présent), non mesurée
    rb.run_campaign([spec], protocol, tmp_path, mcq_items(), deadline=time.monotonic() - 1, **kw)
    tracker = FakeTracker()
    rb.run_campaign([spec], protocol, tmp_path, mcq_items(), energy_tracker=tracker, **kw)
    units = rb.load_energy_units(tmp_path)
    assert f"{spec.model_id}/tone/neutral" not in units
    assert len(units) == 7 and tracker.starts == 7


def test_energy_resume_mixing_units_refused(tmp_path):
    rt = {"llama_cpp": "0.3", "device": "CPU"}
    on = {**rt, "energy": True, "energy_country": "FRA"}
    (tmp_path / "runs.jsonl").write_text(json.dumps(on) + "\n", "utf-8")
    rb.check_runtime(tmp_path, on)
    with pytest.raises(ValueError, match="mesure d'énergie"):
        rb.check_runtime(tmp_path, rt)
    with pytest.raises(ValueError, match="mesure d'énergie"):
        rb.check_runtime(tmp_path, {**on, "energy_country": "DEU"})
    # campagne ancienne sans le champ : équivaut à « sans énergie »
    (tmp_path / "runs.jsonl").write_text(json.dumps(rt) + "\n", "utf-8")
    rb.check_runtime(tmp_path, rt)
    with pytest.raises(ValueError, match="mesure d'énergie"):
        rb.check_runtime(tmp_path, on)


def test_cli_energy_options_and_short_run():
    args = rb.parse_args(["--energy"])
    assert args.energy and args.energy_country is None
    with pytest.raises(SystemExit):
        rb.parse_args(["--energy-country", "DEU"])
    with pytest.raises(SystemExit):
        rb.parse_args(["--n-tone", "10"])
    short = ["--n-tone", "10", "--short-run"]
    assert rb.parse_args([*short, "--models", "0.8b"]).n_tone == 10
    # protocole réduit : pas de modèle cœur, donc jamais de verdict ni d'écriture du rapport 3.5
    with pytest.raises(SystemExit):
        rb.parse_args(short)
    with pytest.raises(SystemExit):
        rb.parse_args([*short, "--models", "4b,0.8b"])
    assert rb.parse_args([]).report is None


def test_short_run_recorded_in_manifest_only_when_set():
    spec = rb.MODELS["0.8b"]
    protocol = rb.Protocol(n_tone=10, n_mcq=6, secondary_phases=("tone",))
    plain = rb.build_manifest([spec], protocol, mcq_items(), {spec.key: {}})
    short = rb.build_manifest([spec], protocol, mcq_items(), {spec.key: {}}, short_run=True)
    assert "short_run" not in plain and short["short_run"] is True


def test_energy_without_extra_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(rb.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="extra frugal"):
        rb.main(["--energy", "--out", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()


def test_h7_short_report_when_no_core_model(tmp_path):
    spec = rb.MODELS["0.8b"]
    protocol = rb.Protocol(n_tone=10, n_mcq=6, secondary_phases=("tone",))
    rb.run_campaign(
        [spec], protocol, tmp_path, mcq_items(), energy_tracker=FakeTracker(), **_tone_kw()
    )
    manifest = rb.build_manifest([spec], protocol, mcq_items(), {spec.key: {}})
    rb.check_manifest(tmp_path, manifest, resume=False)
    rb.build_reports(
        tmp_path, [spec], protocol, mcq_items(), FakeScorer(), judge_client_factory=None,
        judge_sample=20, judge_max_cost=1.0, report=tmp_path / "frugalite.md",
        secondary_report=tmp_path / "secondaires.md", energy=True,
        runtime={"llama_cpp": "0.0", "device": "CPU"}, log=lambda s: None,
    )  # fmt: skip
    text = (tmp_path / "frugalite.md").read_text(encoding="utf-8")
    assert "Mesure courte, Qwen3.5-0.8B-Q4_K_M, CPU, n réduit." in text
    assert "#### Qwen3.5-0.8B-Q4_K_M" in text and "| neutral | 10 |" in text
    assert "| 0.001 | 0.0001 | 0.0001 | 1e-05 |" in text
    assert "estimation, pas une mesure au wattmètre" in text
    # écart de latence apparié par item, avec IC, et pas de conclusion d'économie
    assert "Écart de latence apparié par item face à `prompt_joyful`" in text
    assert "| steering_p1.00 | 10 |" in text and "IC " in text and "ne vaut pas économie" in text
    # bloc de lecture rédigé à la main conservé à la régénération
    lecture = f"{rb.LECTURE_START}\nCPU partagé.\n{rb.LECTURE_END}"
    (tmp_path / "frugalite.md").write_text(text.replace("## H7", lecture + "\n\n## H7"), "utf-8")
    rb.build_reports(
        tmp_path, [spec], protocol, mcq_items(), FakeScorer(), judge_client_factory=None,
        judge_sample=20, judge_max_cost=1.0, report=tmp_path / "verdict.md",
        frugality_report=tmp_path / "frugalite.md",
        secondary_report=tmp_path / "secondaires.md", energy=True,
        runtime={"llama_cpp": "0.0", "device": "CPU"}, log=lambda s: None,
    )  # fmt: skip
    assert not (tmp_path / "verdict.md").exists()
    assert lecture in (tmp_path / "frugalite.md").read_text(encoding="utf-8")


def test_h7_names_arm_without_energy_reading(tmp_path):
    spec = rb.MODELS["0.8b"]
    protocol = rb.Protocol(n_tone=10, n_mcq=6, secondary_phases=("tone",))
    kw = _tone_kw()
    rb.run_campaign([spec], protocol, tmp_path, mcq_items(), deadline=time.monotonic() - 1, **kw)
    rb.run_campaign([spec], protocol, tmp_path, mcq_items(), energy_tracker=FakeTracker(), **kw)
    text = rb.h7_section(tmp_path, [spec], protocol)
    assert (
        "Bras sans lecture d'énergie (unité commencée sans mesure, puis reprise) : neutral." in text
    )


def test_resume_short_campaign_from_manifest_without_short_run_key(tmp_path):
    spec = rb.MODELS["0.8b"]
    protocol = rb.Protocol(n_tone=10, n_mcq=6, secondary_phases=("tone",))
    legacy = rb.build_manifest([spec], protocol, mcq_items(), {spec.key: {}})
    rb.check_manifest(tmp_path, legacy, resume=False)
    short = rb.build_manifest([spec], protocol, mcq_items(), {spec.key: {}}, short_run=True)
    rb.check_manifest(tmp_path, short, resume=True)  # manifeste d'avant la clé : repris
    other = tmp_path / "other"
    rb.check_manifest(other, short, resume=False)
    with pytest.raises(ValueError, match="short_run"):
        rb.check_manifest(other, legacy, resume=True)


def test_short_run_secondary_report_stays_in_out():
    short = rb.parse_args(["--n-tone", "10", "--short-run", "--models", "0.8b", "--out", "x"])
    assert rb.secondary_report_path(short) == Path("x") / "secondaires.md"
    assert rb.secondary_report_path(rb.parse_args([])) == rb.DEFAULT_SECONDARY_REPORT
    explicit = rb.parse_args(["--secondary-report", "y.md"])
    assert rb.secondary_report_path(explicit) == Path("y.md")
