"""Story 5.4 : script d'axe calme/agitation et fichier de paires, sans modèle ni réseau."""

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mood_steering.calibration import _read_pairs

_SPEC = importlib.util.spec_from_file_location(
    "run_axis", Path(__file__).parent.parent / "scripts" / "run_axis.py"
)
ra = importlib.util.module_from_spec(_SPEC)
sys.modules["run_axis"] = ra
_SPEC.loader.exec_module(ra)

PAIRS = Path(__file__).parent.parent / "src" / "mood_steering" / "pairs" / "calm_agitation.jsonl"
N_EMBD = 4
LAYERS = 31


# --- paires -----------------------------------------------------------------------------------


def test_pairs_format_and_count():
    pos, neg = _read_pairs(PAIRS)
    assert 100 <= len(pos) == len(neg) <= 110
    assert len(set(pos)) == len(pos)
    assert len(set(neg)) == len(neg)
    assert all(p != n for p, n in zip(pos, neg, strict=True))
    for line in PAIRS.read_text(encoding="utf-8").split("\n"):
        if line:
            assert set(json.loads(line)) == {"positive", "negative"}


def test_pairs_do_not_vary_joy_sadness_lexicon():
    # la paire ne doit varier que l'activation : pas de vocabulaire explicite de joie/tristesse
    banned = ("joie", "joyeu", "triste", "tristesse", "chagrin", "heureu", "bonheur")
    text = PAIRS.read_text(encoding="utf-8").lower()
    assert [w for w in banned if w in text] == []


# --- bras et vecteurs --------------------------------------------------------------------------


def test_build_arms_names_and_scales():
    arms = ra.build_arms((0.3, 0.5))
    assert [a.name for a in arms] == [
        "neutral",
        "calm_p0.30", "agit_m0.30", "joy_p0.30", "joycalm_p0.30",
        "calm_p0.50", "agit_m0.50", "joy_p0.50", "joycalm_p0.50",
    ]  # fmt: skip
    by = {a.name: a for a in arms}
    assert (by["agit_m0.30"].joy, by["agit_m0.30"].calm) == (0.0, -0.3)
    assert (by["joycalm_p0.50"].joy, by["joycalm_p0.50"].calm) == (0.5, 0.5)
    assert by["neutral"].is_neutral and not by["calm_p0.30"].is_neutral


@pytest.mark.parametrize("bad", [(), (0.0,), (-0.3,), (0.3, 0.3)])
def test_build_arms_rejects_bad_scales(bad):
    with pytest.raises(ValueError, match="échelles"):
        ra.build_arms(bad)


def test_arm_vector_sums_with_each_dose_and_divides_by_max_scale():
    joy = np.full((LAYERS, N_EMBD), 2.0, dtype=np.float32)
    calm = np.full((LAYERS, N_EMBD), 10.0, dtype=np.float32)
    assert ra.arm_vector(ra.AxisArm("neutral"), joy, calm, 0.7) is None
    v = ra.arm_vector(ra.AxisArm("c", joy=0.4, calm=0.5), joy, calm, 0.5)
    assert v.dtype == np.float32
    np.testing.assert_allclose(v, (0.4 * 2 + 0.5 * 10) / 0.5)
    neg = ra.arm_vector(ra.AxisArm("a", calm=-0.3), joy, calm, 0.5)
    np.testing.assert_allclose(neg, -0.3 * 10 / 0.5)
    with pytest.raises(ValueError, match="formes différentes"):
        ra.arm_vector(ra.AxisArm("c", joy=0.1, calm=0.1), joy, calm[:5], 0.7)


def test_cosine_by_layer_values_and_errors():
    a = np.zeros((4, 2), dtype=np.float32)
    b = np.zeros((4, 2), dtype=np.float32)
    a[0], b[0] = (1, 0), (1, 0)
    a[1], b[1] = (1, 0), (0, 1)
    a[2], b[2] = (1, 0), (-1, 0)
    a[3], b[3] = (1, 1), (0, 0)
    cs = ra.cosine_by_layer(a, b, 1, 4)
    assert cs[1] == pytest.approx(1.0)
    assert cs[2] == pytest.approx(0.0)
    assert cs[3] == pytest.approx(-1.0)
    assert math.isnan(cs[4])
    assert list(ra.cosine_by_layer(a, b, 2, 3)) == [2, 3]
    with pytest.raises(ValueError, match="plage"):
        ra.cosine_by_layer(a, b, 0, 3)
    with pytest.raises(ValueError, match="plage"):
        ra.cosine_by_layer(a, b, 2, 5)
    with pytest.raises(ValueError, match="formes"):
        ra.cosine_by_layer(a, b[:3], 1, 3)


def test_distinct_threshold():
    assert ra.distinct({8: 0.3, 9: -0.69})
    assert not ra.distinct({8: 0.3, 9: 0.7})  # strict
    assert not ra.distinct({8: 0.3, 9: math.nan})
    assert not ra.distinct({})


# --- proxies -----------------------------------------------------------------------------------


def test_text_proxies():
    assert ra.exclamation_rate("Vite ! Vite ! Courez.") == pytest.approx(2 / 3)
    assert ra.exclamation_rate("") == 0.0
    assert ra.mean_sentence_len("Un deux trois. Quatre cinq.") == pytest.approx(2.5)
    assert math.isnan(ra.mean_sentence_len("..."))
    # 2 mots d'agitation (vite, panique) sur 5 mots -> +40 ; 1 mot de calme sur 4 -> -25
    assert ra.lexicon_score("on court vite dans la panique") > 0
    assert ra.lexicon_score("vite vite panique doucement") == pytest.approx(50.0)
    assert ra.lexicon_score("un matin calme et lent") == pytest.approx(-40.0)
    assert ra.lexicon_score("") == 0.0


def test_activation_share_normalises_by_total():
    probs = {"anger": 0.2, "fear": 0.1, "surprise": 0.1, "joy": 0.3, "sadness": 0.3}
    assert ra.activation_share(probs) == pytest.approx(0.4)
    assert ra.activation_share({"joy": 0.0}) == 0.0


def test_score_text_fields():
    probs = {"anger": 0.4, "fear": 0.0, "surprise": 0.0, "joy": 0.5, "sadness": 0.1}
    s = ra.score_text("Une phrase assez longue pour être cohérente et lisible par tous !", probs)
    assert s["valence"] == pytest.approx(0.4)
    assert s["act_cls"] == pytest.approx(0.4)
    assert s["exclam"] == 1.0
    assert s["degenerate"] == 0.0
    assert ra.score_text("ok", probs)["degenerate"] == 1.0


# --- statistiques ------------------------------------------------------------------------------


def test_paired_boot_is_deterministic_and_brackets_mean():
    cols = {"x": np.arange(20, dtype=float)}
    stat = lambda c: float(c["x"].mean())  # noqa: E731
    r1 = ra.paired_boot(cols, stat, n_boot=300, seed=1)
    assert r1 == ra.paired_boot(cols, stat, n_boot=300, seed=1)
    assert r1[0] == 9.5 and r1[1] < 9.5 < r1[2]
    with pytest.raises(ValueError, match="longueurs"):
        ra.paired_boot({"a": np.ones(3), "b": np.ones(2)}, stat)


def _scores(effects, n=40):
    """Scores synthétiques ``{bras: {item: proxies}}`` : bruit fixe, effets additifs par bras."""
    rng = np.random.default_rng(0)
    base = {
        "act_cls": rng.normal(0.3, 0.05, n),
        "exclam": rng.normal(0.2, 0.05, n),
        "lex": rng.normal(0.0, 1.0, n),
        "sent_len": rng.normal(12, 2, n),
        "valence": rng.normal(0.0, 0.1, n),
    }
    out = {}
    for arm, (d_act, d_val) in effects.items():
        rows = {}
        for i in range(n):
            rows[f"t{i:03d}"] = {
                "act_cls": base["act_cls"][i] + d_act * 0.05,
                "exclam": base["exclam"][i] + d_act * 0.05,
                "lex": base["lex"][i] + d_act * 1.0,
                "sent_len": base["sent_len"][i],
                "valence": base["valence"][i] + d_val * 0.1,
                "degenerate": 0.0,
                "refusal": 0.0,
            }
        out[arm] = rows
    return out


def test_analyse_holds_when_axes_are_separate_and_combine():
    eff = {
        "neutral": (0, 0),
        "calm_p0.30": (-2, 0.1), "agit_m0.30": (2, -0.1),
        "joy_p0.30": (0.1, 2), "joycalm_p0.30": (-1.8, 1.9),
    }  # fmt: skip
    stats, verdicts, sd = ra.analyse(_scores(eff), (0.3,), n_boot=300)
    assert sd["valence"] == pytest.approx(0.1, rel=0.3)
    # écarts en écarts-types du neutre : l'échelle dépend du bruit tiré, le signe et l'ordre non
    assert stats["calm_p0.30"].act.mean == pytest.approx(-2, rel=0.35)
    assert stats["joy_p0.30"].val.mean == pytest.approx(2, rel=0.35)
    assert stats["calm_p0.30"].act.high < 0 < stats["joy_p0.30"].val.low
    v = verdicts[0.3]
    assert (v.valid, v.calm_ok, v.agit_ok, v.joy_ok, v.a_held) == (True,) * 5
    assert v.b_held and v.held
    assert v.retention_act == pytest.approx(0.9, abs=0.01)
    assert v.retention_val == pytest.approx(0.95, abs=0.01)


def test_analyse_fails_when_calm_moves_valence_as_much():
    eff = {
        "neutral": (0, 0),
        "calm_p0.30": (-1, 3), "agit_m0.30": (1, -3),
        "joy_p0.30": (0.1, 2), "joycalm_p0.30": (-1, 2),
    }  # fmt: skip
    _, verdicts, _ = ra.analyse(_scores(eff), (0.3,), n_boot=300)
    v = verdicts[0.3]
    assert not v.calm_ok and not v.agit_ok and not v.a_held and not v.held


def test_analyse_b_fails_when_combined_loses_activation():
    eff = {
        "neutral": (0, 0),
        "calm_p0.30": (-2, 0.1), "agit_m0.30": (2, -0.1),
        "joy_p0.30": (0.1, 2), "joycalm_p0.30": (-0.4, 2),
    }  # fmt: skip
    _, verdicts, _ = ra.analyse(_scores(eff), (0.3,), n_boot=300)
    v = verdicts[0.3]
    assert v.a_held and not v.b_held and not v.held
    assert v.retention_act == pytest.approx(0.2, abs=0.01)


def test_analyse_invalid_when_degenerate_rate_jumps():
    eff = {
        "neutral": (0, 0),
        "calm_p0.30": (-2, 0.1), "agit_m0.30": (2, -0.1),
        "joy_p0.30": (0.1, 2), "joycalm_p0.30": (-1.8, 1.9),
    }  # fmt: skip
    scores = _scores(eff)
    for row in scores["agit_m0.30"].values():
        row["degenerate"] = 0.5
    _, verdicts, _ = ra.analyse(scores, (0.3,), n_boot=200)
    assert not verdicts[0.3].valid and not verdicts[0.3].held


# --- génération reprenable ---------------------------------------------------------------------


class FakeBackend:
    n_layer = LAYERS + 1
    n_embd = N_EMBD

    def __init__(self):
        self.cvecs = []  # (tampon ou None, il_start, il_end) à chaque set_cvec
        self.generated = 0

    def reset(self):
        pass

    def set_cvec(self, data, n_embd, il_start, il_end):
        self.cvecs.append((None if data is None else np.array(data), il_start, il_end))

    def generate(self, prompt, system, max_tokens, seed):
        self.generated += 1
        return f"sortie {self.generated}"


def _loader(axis, n_embd):
    base = {"joy_sadness": 1.0, "calm_agitation": 100.0}[axis]
    return np.full((LAYERS, n_embd), base, dtype=np.float32)


SPEC = SimpleNamespace(model_id="m", il_start=2, il_end=5, max_scale=0.5)


def test_run_axis_poses_one_summed_vector_per_arm(tmp_path):
    backend = FakeBackend()
    logs = []
    ok = ra.run_axis(
        SPEC, (0.3,), tmp_path, backend_factory=lambda: backend, vector_loader=_loader,
        n_prompts=2, tokens=8, log=logs.append,
    )  # fmt: skip
    assert ok
    names = [a.name for a in ra.build_arms((0.3,))]
    for name in names:
        rows = ra_rows(tmp_path, name)
        assert [r["item_id"] for r in rows] == ["tone000", "tone001"]
        assert [r["seed"] for r in rows] == [0, 1]
        assert rows[0]["scale_joy"] is not None and rows[0]["arm"] == name
    comb = ra_rows(tmp_path, "joycalm_p0.30")[0]
    assert (comb["scale_joy"], comb["scale_calm"]) == (0.3, 0.3)
    # le dernier tampon posé avant la génération du bras combiné : somme des deux doses
    # (0.3 * 1 + 0.3 * 100), couches 2..5 seulement, couche 1 mise à zéro
    posed = [c for c in backend.cvecs if c[0] is not None]
    combined = posed[-1]
    assert (combined[1], combined[2]) == (2, 5)
    buf = combined[0].reshape(5, N_EMBD)
    np.testing.assert_allclose(buf[0], 0.0)
    np.testing.assert_allclose(buf[1:], 30.3, rtol=1e-5)
    # la sortie du bras agitation est posée avec le signe négatif
    assert any(np.allclose(c[0][N_EMBD:], -30.0, rtol=1e-5) for c in posed)


def ra_rows(out, arm):
    return [json.loads(x) for x in ra.unit_path(out, "m", arm).read_text("utf-8").split("\n") if x]


def test_run_axis_resumes_after_deadline(tmp_path):
    backend = FakeBackend()
    kw = dict(backend_factory=lambda: backend, vector_loader=_loader, n_prompts=3, tokens=8)
    # échéance dépassée d'emblée : rien n'est écrit, le premier bras reste en .part
    assert ra.run_axis(SPEC, (0.3,), tmp_path, deadline=-1.0, log=lambda m: None, **kw) is False
    assert backend.generated == 0
    assert ra.run_axis(SPEC, (0.3,), tmp_path, log=lambda m: None, **kw) is True
    total = 5 * 3
    assert backend.generated == total
    # relance : tout est écrit, aucun chargement de backend ni génération
    calls = []
    assert ra.run_axis(
        SPEC, (0.3,), tmp_path, backend_factory=lambda: calls.append(1), vector_loader=_loader,
        n_prompts=3, tokens=8, log=lambda m: None,
    )  # fmt: skip
    assert calls == [] and backend.generated == total


def test_run_axis_resumes_partial_arm_item_by_item(tmp_path):
    backend = FakeBackend()
    kw = dict(backend_factory=lambda: backend, vector_loader=_loader, tokens=8)
    arm0 = ra.unit_path(tmp_path, "m", "neutral")
    arm0.parent.mkdir(parents=True)
    row = {"run_id": "x", "item_id": "tone000", "output": "déjà fait"}
    arm0.with_name(arm0.name + ".part").write_text(json.dumps(row) + "\n", encoding="utf-8")
    ra.run_axis(SPEC, (0.3,), tmp_path, n_prompts=2, log=lambda m: None, **kw)
    rows = ra_rows(tmp_path, "neutral")
    assert [r["output"] for r in rows][0] == "déjà fait"
    assert [r["item_id"] for r in rows] == ["tone000", "tone001"]


def test_manifest_pins_protocol_and_vectors():
    m = ra.build_manifest(SPEC, (0.3, 0.5), 50, 128, {"joy_sadness": "aa", "calm_agitation": "bb"})
    assert m["n_prompts"] == 50 and m["tokens"] == 128 and m["seed"] == 0
    assert "scales" not in m and "arms" not in m  # bras ajoutables à une campagne reprise
    assert m["vectors_sha256"] == {"joy_sadness": "aa", "calm_agitation": "bb"}
    assert len(m["prompts_sha256"]) == 64


def test_score_arms_caches_and_scores(tmp_path):
    backend = FakeBackend()
    ra.run_axis(
        SPEC, (0.3,), tmp_path, backend_factory=lambda: backend, vector_loader=_loader,
        n_prompts=2, tokens=8, log=lambda m: None,
    )  # fmt: skip
    calls = []

    def classify(texts):
        calls.append(len(texts))
        return [{"anger": 0.1, "fear": 0.1, "surprise": 0.0, "joy": 0.6, "sadness": 0.2}] * len(
            texts
        )

    scores = ra.score_arms(tmp_path, SPEC, (0.3,), classify)
    assert set(scores) == {a.name for a in ra.build_arms((0.3,))}
    assert scores["neutral"]["tone000"]["valence"] == pytest.approx(0.4)
    assert scores["neutral"]["tone000"]["act_cls"] == pytest.approx(0.2 / 1.0)
    assert calls == [2] * 5
    ra.score_arms(tmp_path, SPEC, (0.3,), classify)  # relu du cache
    assert calls == [2] * 5


def test_render_report_mentions_verdict_and_cosine(tmp_path):
    eff = {
        "neutral": (0, 0),
        "calm_p0.30": (-2, 0.1), "agit_m0.30": (2, -0.1),
        "joy_p0.30": (0.1, 2), "joycalm_p0.30": (-1.8, 1.9),
    }  # fmt: skip
    scores = _scores(eff)
    stats, verdicts, sd = ra.analyse(scores, (0.3,), n_boot=200)
    cos = {8: 0.3, 9: 0.31}
    md = ra.render_report(
        spec=SPEC, scales=(0.3,), cosines=cos, scores=scores, stats=stats, verdicts=verdicts,
        sd=sd, runtime={"llama_cpp": "0.3.36", "device": "CPU"}, n_prompts=40, tokens=128,
        lecture=None, extracts={"neutral": "x"},
    )  # fmt: skip
    assert "Verdict H5 : tenue à l'échelle 0.3" in md
    assert "se distingue" in md and "|cos| maximal 0.31" in md
    assert "<!-- lecture -->" in md
    bad = ra.render_report(
        spec=SPEC, scales=(0.3,), cosines={8: 0.9}, scores=scores, stats=stats,
        verdicts=verdicts, sd=sd, runtime={}, n_prompts=40, tokens=128, lecture="LECTURE",
        extracts={},
    )  # fmt: skip
    assert "Verdict H5 : non tenue" in bad and "ne se distingue pas" in bad
    assert "LECTURE" in bad


def test_lexicon_handles_elision_and_calm_phrases():
    # « s'agite » compte comme « agite » ; « sans se presser » compte calme, pas « presser »
    assert ra.lexicon_score("elle s'agite") == pytest.approx(50.0)
    assert ra.lexicon_score("il marche sans se presser") == pytest.approx(-20.0)
    assert ra.lexicon_score("il avance sans hâte") == pytest.approx(-25.0)


def test_raw_proxy_ignores_items_without_sentence():
    eff = {"neutral": (0, 0), "calm_p0.30": (0, 0), "agit_m0.30": (0, 0),
           "joy_p0.30": (0, 0), "joycalm_p0.30": (0, 0)}  # fmt: skip
    scores = _scores(eff)
    for i, row in enumerate(scores["calm_p0.30"].values()):
        row["sent_len"] = math.nan if i % 2 else scores["neutral"][f"t{i:03d}"]["sent_len"] + 5
    stats, _, _ = ra.analyse(scores, (0.3,), n_boot=100)
    assert stats["calm_p0.30"].raw["sent_len"].mean == pytest.approx(5.0)  # pas tiré vers 0


def test_score_arms_refreshes_text_proxies_without_reclassifying(tmp_path):
    backend = FakeBackend()
    ra.run_axis(
        SPEC, (0.3,), tmp_path, backend_factory=lambda: backend, vector_loader=_loader,
        n_prompts=2, tokens=8, log=lambda m: None,
    )  # fmt: skip
    probs = {"anger": 0.1, "fear": 0.1, "surprise": 0.0, "joy": 0.6, "sadness": 0.2}
    ra.score_arms(tmp_path, SPEC, (0.3,), lambda t: [probs] * len(t))
    cache = tmp_path / SPEC.model_id / "axis_scores.json"
    data = json.loads(cache.read_text(encoding="utf-8"))
    data["neutral"]["tone000"]["lex"] = 999.0  # ancien lexique
    data["neutral"]["tone000"]["valence"] = -0.5  # score du classifieur, à garder
    cache.write_text(json.dumps(data), encoding="utf-8")

    def no_classify(texts):
        raise AssertionError("classifieur rappelé")

    scores = ra.score_arms(tmp_path, SPEC, (0.3,), no_classify)
    assert scores["neutral"]["tone000"]["lex"] != 999.0
    assert scores["neutral"]["tone000"]["valence"] == -0.5


def test_render_report_inconclusive_when_distinct_but_criteria_fail():
    eff = {
        "neutral": (0, 0),
        "calm_p0.30": (-1, 3), "agit_m0.30": (1, -3),
        "joy_p0.30": (0.1, 2), "joycalm_p0.30": (0, 0),
    }  # fmt: skip
    scores = _scores(eff)
    stats, verdicts, sd = ra.analyse(scores, (0.3,), n_boot=200)
    md = ra.render_report(
        spec=SPEC, scales=(0.3,), cosines={8: 0.3}, scores=scores, stats=stats,
        verdicts=verdicts, sd=sd, runtime={}, n_prompts=40, tokens=128, lecture=None,
        extracts={},
    )  # fmt: skip
    assert "Verdict H5 : non concluante" in md


def test_paired_boot_and_ci_use_shared_stats():
    from mood_steering.eval.stats import Interval, excludes_zero, paired_bootstrap_stat

    x = np.arange(20, dtype=float)
    cols = {"x": x}
    stat = lambda c: float(c["x"].mean())  # noqa: E731
    iv = paired_bootstrap_stat(cols, lambda c: float(np.mean(c["x"])), n_boot=300, seed=1)
    assert ra.paired_boot(cols, stat, n_boot=300, seed=1) == (iv.mean, iv.low, iv.high)
    ci = ra.Ci(*ra.paired_boot(cols, stat, n_boot=300, seed=1))
    assert ci.excludes_zero() == excludes_zero(Interval(ci.mean, ci.low, ci.high, 20)) is True
    assert not ra.Ci(0.0, -0.1, 0.2).excludes_zero()
    assert ra.Ci(-1.0, -2.0, -0.5).excludes_zero()
