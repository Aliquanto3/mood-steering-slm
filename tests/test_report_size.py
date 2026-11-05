"""Effet de taille (5.2) : mesures et comparaisons sur des données synthétiques."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "report_size", Path(__file__).parent.parent / "scripts" / "report_size.py"
)
rs = importlib.util.module_from_spec(_SPEC)
sys.modules["report_size"] = rs
_SPEC.loader.exec_module(rs)

GOOD = "Le marché ouvre à huit heures sur la place, avec des étals de fruits et de légumes."
REFUSAL = "Je suis une intelligence artificielle et je ne peux pas raconter cette scène."
BAD = "oui"


def _arm(name, kind, dose, tones, degenerate=None, refusal=None):
    n = len(tones)
    return rs.ArmData(
        name,
        kind,
        dose,
        tuple(tones),
        tuple(refusal or [False] * n),
        tuple(degenerate or [False] * n),
    )


def _wave(center, n=40, spread=0.1):
    """Valeurs déterministes autour d'un centre, moyenne exactement ``center``."""
    return [center + spread * ((i % 4) - 1.5) / 1.5 for i in range(n)]


def _model(key="4b", gain=0.6, bad_last=0):
    """Ton = gain x dose, plus ou moins de sorties dégénérées à la dose la plus forte."""
    spec = rs.MODELS[key]
    arms = [_arm("neutral", "neutral", 0.0, _wave(0.0)), _arm("prompt", "prompt", 0.0, _wave(0.7))]
    for sign in (1, -1):
        for d in (0.5, 1.0):
            n = 40
            deg = [i < bad_last for i in range(n)] if d == 1.0 else None
            arms.append(
                _arm(f"s{sign}{d}", "steering", sign * d, _wave(sign * gain * d), degenerate=deg)
            )
    ids = tuple(f"i{i:02d}" for i in range(40))
    return rs.ModelData(spec, ids, tuple(arms))


def test_sd_and_rate_and_slope_exact_values():
    assert rs.sd([1.0, 3.0]) == pytest.approx(math.sqrt(2))
    assert rs.sd([5.0]) == 0.0
    assert rs.rate([True, False, False, False]) == 0.25
    assert math.isnan(rs.rate([]))
    assert rs.slope([(0.0, 0.0), (0.5, 0.25), (1.0, 0.5)]) == pytest.approx(0.5)
    assert rs.slope([(0.0, 1.0), (1.0, 0.0)]) == pytest.approx(-1.0)
    assert math.isnan(rs.slope([(1.0, 0.0), (1.0, 2.0)]))
    assert math.isnan(rs.slope([(1.0, 0.0)]))


def test_useful_arm_needs_shift_in_dose_direction_and_few_degenerate():
    data = _model()
    n_boot = 200
    pos, neg = data.steering(1)[-1], data.steering(-1)[-1]
    assert rs.useful_arm(pos, data.neutral, n_boot=n_boot, seed=0)
    assert rs.useful_arm(neg, data.neutral, n_boot=n_boot, seed=0)
    # un déplacement de signe opposé à la dose n'est pas utile
    flipped = rs.ArmData("x", "steering", -1.0, pos.tones, pos.refusal, pos.degenerate)
    assert not rs.useful_arm(flipped, data.neutral, n_boot=n_boot, seed=0)
    # 11 % de sorties dégénérées : au-dessus du seuil de 10 %
    noisy = _model(bad_last=5).steering(1)[-1]
    assert rs.rate(noisy.degenerate) == 0.125
    assert not rs.useful_arm(noisy, data.neutral, n_boot=n_boot, seed=0)
    ok = _model(bad_last=4).steering(1)[-1]
    assert rs.rate(ok.degenerate) == 0.10
    assert rs.useful_arm(ok, data.neutral, n_boot=n_boot, seed=0)


def test_useful_ranges_real_scale_and_ratio():
    data = _model("2b", bad_last=8)  # 2B : max_scale 0.5 ; la dose 1 devient dégénérée
    rows = rs.arm_rows(data, n_boot=200)
    pos, neg = rs.useful_ranges(data, rows)
    assert (pos.pole, neg.pole) == ("positif", "négatif")
    assert pos.doses == (0.5,) and pos.scales == (0.25,) and pos.ratio == 1.0
    assert neg.doses == (0.5,)
    full = rs.useful_ranges(_model("0.8b"), rs.arm_rows(_model("0.8b"), n_boot=200))[0]
    assert full.doses == (0.5, 1.0)
    assert full.scales == (0.075, 0.15)
    assert full.ratio == 2.0
    empty = rs.UsefulRange("positif", (), ())
    assert empty.ratio is None and rs._range_text(empty) == "aucune"
    assert "rapport 2.00" in rs._range_text(full)


def test_tone_points_per_pole_and_slope_recovers_gain():
    data = _model(gain=0.6)
    idx = list(range(40))
    pts = rs.tone_points(data, 1, idx)
    assert [d for d, _ in pts] == [0.0, 0.5, 1.0]
    assert rs.slope(pts) == pytest.approx(0.6)
    assert rs.slope(rs.tone_points(data, -1, idx)) == pytest.approx(0.6)
    assert rs.slope(rs.tone_points(data, None, idx)) == pytest.approx(0.6)
    # un sous-ensemble d'items change la moyenne : un seul item
    assert rs.tone_points(data, 1, [0])[0][1] == data.neutral.tones[0]


def test_boot_stat_is_deterministic_and_interval_brackets_point():
    data = _model()
    a = rs.boot_stat(data, rs.stat_tone(1), n_boot=50, seed=3)
    b = rs.boot_stat(data, rs.stat_tone(1), n_boot=50, seed=3)
    assert a == b and len(a) == 50
    point = rs.stat_tone(1)(data, list(range(40)))
    assert point == pytest.approx(0.6)
    iv = rs.interval_from_draws(a, point)
    assert iv.low <= point <= iv.high
    assert iv.n == 50
    nan_iv = rs.interval_from_draws([math.nan], 1.0)
    assert math.isnan(nan_iv.low) and nan_iv.n == 0


def test_compare_detects_gap_and_keeps_equal_models_undecided():
    big = rs.model_stats(_model("4b", gain=0.7), n_boot=200)
    small = rs.model_stats(_model("2b", gain=0.3, bad_last=20), n_boot=200)
    by = {c.stat: c for c in rs.compare(big, small)}
    tone = by["ton à la dose +1"]
    assert tone.diff.mean == pytest.approx(0.4)
    assert tone.decided and tone.diff.low > 0
    deg = by["part de sorties dégénérées à +1"]
    assert deg.diff.mean == pytest.approx(-0.5)
    assert deg.decided and deg.diff.high < 0
    same = {c.stat: c for c in rs.compare(big, rs.model_stats(_model("2b", gain=0.7), n_boot=200))}
    assert not same["ton à la dose +1"].decided
    assert same["ton à la dose +1"].diff.mean == pytest.approx(0.0)
    assert same["pente dose → ton, tous bras (par dose normalisée)"].diff.mean == pytest.approx(0)


def test_model_stats_slopes_and_prompt_gap():
    ms = rs.model_stats(_model(gain=0.6), n_boot=200)
    for iv in ms.slopes.values():
        assert iv.mean == pytest.approx(0.6)
        assert iv.low - 1e-9 <= 0.6 <= iv.high + 1e-9
    assert "recouvre" in ms.prompt or "reste sous" in ms.prompt or "dépasse" in ms.prompt
    assert "reste sous" in rs.prompt_gap(_model(gain=0.3), n_boot=200)
    assert "dépasse" in rs.prompt_gap(_model(gain=1.6), n_boot=200)
    assert "recouvre" in rs.prompt_gap(_model(gain=0.7), n_boot=200)


def _write_campaign(out: Path, spec):
    base = out / spec.model_id
    scores = {"tone": {}}
    for arm in spec.arms():
        d = base / "tone"
        d.mkdir(parents=True, exist_ok=True)
        rows, sc = [], {}
        for i in range(5):
            text = {0: REFUSAL, 1: BAD}.get(i, GOOD) if arm.dose < 0 else GOOD
            rows.append({"item_id": f"t{i}", "output": text, "latency_s": 1.0})
            sc[f"t{i}"] = [arm.dose * 0.5 + i * 0.01, 0.0]
        sc.pop("t4")  # un item sans score : exclu des items communs
        (d / f"{arm.name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        scores["tone"][arm.name] = sc
    (base / "scores.json").write_text(json.dumps(scores), encoding="utf-8")


def test_load_model_reads_flags_and_common_items(tmp_path):
    spec = rs.MODELS["0.8b"]
    _write_campaign(tmp_path, spec)
    data = rs.load_model(tmp_path, spec)
    assert data.ids == ("t0", "t1", "t2", "t3")
    assert data.neutral.tones == (0.0, 0.01, 0.02, 0.03)
    assert data.arm("steering_p1.00").tones == (0.5, 0.51, 0.52, 0.53)
    m = data.arm("steering_m1.00")
    assert m.refusal == (True, False, False, False)
    assert m.degenerate == (False, True, False, False)  # le refus est cohérent, « oui » non
    assert data.arm("steering_p1.00").degenerate == (False,) * 4
    assert [a.dose for a in data.steering(-1)] == [-0.6667, -0.8333, -1.0]
    assert data.prompt.kind == "prompt"


def test_load_model_without_common_items_raises(tmp_path):
    spec = rs.MODELS["2b"]
    _write_campaign(tmp_path, spec)
    path = tmp_path / spec.model_id / "scores.json"
    scores = json.loads(path.read_text(encoding="utf-8"))
    scores["tone"]["neutral"] = {"autre": [0.0, 0.0]}
    path.write_text(json.dumps(scores), encoding="utf-8")
    with pytest.raises(ValueError, match="aucun item commun"):
        rs.load_model(tmp_path, spec)


def test_report_text_and_splice(tmp_path):
    stats = {
        k: rs.model_stats(_model(k, gain=g), n_boot=100) for k, g in (("4b", 0.7), ("2b", 0.4))
    }
    body = rs.build_report(stats)
    assert "### Qwen3.5-4B-Q4_K_M" in body and "### Qwen3.5-2B-Q4_K_M" in body
    assert "0.8B" not in body
    assert body.count("| Statistique |") == 1
    assert "### Qwen3.5-4B-Q4_K_M moins Qwen3.5-2B-Q4_K_M" in body
    assert "écart non nul" in body
    doc = f"avant\n{rs.MARK_START}\nancien\n{rs.MARK_END}\naprès\n"
    new = rs.splice(doc, body)
    assert new.startswith("avant\n" + rs.MARK_START + "\n\n## Mesures par modèle")
    assert new.endswith(rs.MARK_END + "\naprès\n") and "ancien" not in new
    with pytest.raises(ValueError, match="marqueurs"):
        rs.splice("rien", body)


def test_main_prints_without_report(tmp_path, capsys):
    for key in rs.SIZE_ORDER:
        _write_campaign(tmp_path, rs.MODELS[key])
    assert rs.main(["--out", str(tmp_path), "--n-boot", "50"]) == 0
    out = capsys.readouterr().out
    assert "## Comparaisons entre modèles" in out
    report = tmp_path / "r.md"
    report.write_text(
        f"# Titre écrit à la main\n{rs.MARK_START}\n{rs.MARK_END}\n", encoding="utf-8"
    )
    assert rs.main(["--out", str(tmp_path), "--report", str(report), "--n-boot", "50"]) == 0
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Titre écrit à la main\n" + rs.MARK_START)
    assert "## Mesures par modèle" in text and text.endswith(rs.MARK_END + "\n")


def _shifted(data, delta, spec=None):
    """Même modèle, tons décalés de ``delta`` prompt par prompt."""
    arms = tuple(
        rs.ArmData(
            a.name, a.kind, a.dose, tuple(t + delta for t in a.tones), a.refusal, a.degenerate
        )
        for a in data.arms
    )
    return rs.ModelData(spec or data.spec, data.ids, arms)


def test_compare_is_paired_on_the_same_resampled_prompts():
    base = _model("4b", gain=0.6)
    noisy = tuple(
        rs.ArmData(
            a.name,
            a.kind,
            a.dose,
            tuple(t + (i % 7) * 0.3 for i, t in enumerate(a.tones)),
            a.refusal,
            a.degenerate,
        )
        for a in base.arms
    )
    a = rs.ModelData(base.spec, base.ids, noisy)
    b = _shifted(a, -0.1, rs.MODELS["2b"])
    by = {
        c.stat: c for c in rs.compare(rs.model_stats(a, n_boot=200), rs.model_stats(b, n_boot=200))
    }
    tone = by["ton à la dose +1"].diff
    # apparié : l'écart vaut 0,1 sur chaque tirage ; indépendant, l'IC serait large
    assert tone.low == pytest.approx(0.1) and tone.high == pytest.approx(0.1)


def test_paired_draws_refuse_different_prompts():
    a = _model("4b")
    b = rs.ModelData(rs.MODELS["2b"], tuple(f"x{i:02d}" for i in range(40)), a.arms)
    with pytest.raises(ValueError, match="appariement"):
        rs.paired_diff_draws(a, b, rs.stat_tone(1), n_boot=10, seed=0)


def test_real_scale_slope_divides_by_max_scale_and_sensitivity_to_threshold():
    data = _model("0.8b", gain=0.6, bad_last=6)  # 15 % de dégénérées à ±1
    idx = list(range(len(data.ids)))
    assert rs.stat_slope(1, True)(data, idx) == pytest.approx(0.6 / data.spec.max_scale)
    ms = rs.model_stats(data, n_boot=200)
    doses = {t: [r.doses for r in ranges] for t, ranges in ms.sensitivity.items()}
    assert doses[0.05] == doses[0.10] == [(0.5,), (0.5,)]
    assert doses[0.20] == [(0.5, 1.0), (0.5, 1.0)]
    assert "seuil 20 % : positif 0.5, 1 ; négatif 0.5, 1" in rs.model_section(ms)


def test_shifted_arm_uses_paired_criterion_of_stats(monkeypatch):
    from mood_steering.eval import stats

    # petit écart constant sur des prompts très dispersés : IC séparés qui se recouvrent,
    # mais IC95 de la différence appariée loin de 0
    base = _wave(0.0, spread=2.0)
    neutral = _arm("neutral", "neutral", 0.0, base)
    up = _arm("up", "steering", 1.0, [x + 0.3 for x in base])
    down = _arm("down", "steering", -1.0, [x - 0.3 for x in base])
    kw = {"n_boot": 200, "seed": 0}
    ci, ci0 = (stats.bootstrap_ci(list(a.tones), **kw) for a in (up, neutral))
    assert ci.low <= ci0.high  # l'ancien critère (IC disjoints) ne voyait rien
    assert rs.shifted_arm(up, neutral, **kw)
    assert rs.shifted_arm(down, neutral, **kw)
    assert rs.shifted_arm(up, neutral, **kw) == stats.paired_shift(
        up.tones, neutral.tones, direction=1, **kw
    )
    # mauvais sens : pas utile ; l'appel passe bien par la fonction partagée
    flipped = _arm("f", "steering", -1.0, up.tones)
    assert not rs.shifted_arm(flipped, neutral, **kw)
    seen = []
    real = stats.paired_shift
    monkeypatch.setattr(
        rs, "paired_shift", lambda *a, **k: seen.append(k["direction"]) or real(*a, **k)
    )
    rs.shifted_arm(down, neutral, **kw)
    assert seen == [-1]
