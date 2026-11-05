import json

import pytest

from mood_steering.eval.arms import KIND_NEUTRAL, KIND_PROMPT, KIND_STEERING, Arm
from mood_steering.eval.frugality import (
    ENERGY_FILE,
    CodeCarbonTracker,
    EnergyReading,
    estimate_tokens,
    frugality_section,
    frugality_summary,
    load_energy,
    write_energy,
)
from mood_steering.eval.runner import Item, load_run, run_bench
from mood_steering.steering import Steerer


class FakeBackend:
    n_layer = 8
    n_embd = 4

    def reset(self):
        pass

    def set_cvec(self, data, n_embd, il_start, il_end):
        pass

    def generate(self, prompt, system, max_tokens, seed):
        return f"out:{prompt}:{seed}"


ARMS = [
    Arm("neutral", KIND_NEUTRAL),
    Arm("prompt_joyful", KIND_PROMPT),
    Arm("steering_p0.50", KIND_STEERING, 0.5),
]
ITEMS = [Item("a", "pa"), Item("b", "pb")]


class FakeTracker:
    def __init__(self):
        self.events = []
        self.n = 0

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")
        self.n += 1
        return EnergyReading(energy_kwh=self.n * 1e-3, emissions_kg=self.n * 1e-4, duration_s=1.5)


def _steerer():
    return Steerer(FakeBackend())


def test_tracker_called_around_each_arm_and_energy_json_written(tmp_path):
    tracker = FakeTracker()
    run_bench(
        ARMS, ITEMS, _steerer(), run_id="r", model_id="m", out_dir=tmp_path, energy_tracker=tracker
    )
    assert tracker.events == ["start", "stop"] * 3
    data = json.loads((tmp_path / ENERGY_FILE).read_text(encoding="utf-8"))
    assert data["neutral"] == {"energy_kwh": 1e-3, "emissions_kg": 1e-4, "duration_s": 1.5}
    assert set(data) == {a.name for a in ARMS}
    assert load_energy(tmp_path)["steering_p0.50"].energy_kwh == pytest.approx(3e-3)


def test_jsonl_unchanged_with_or_without_tracker(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run_bench(ARMS, ITEMS, _steerer(), run_id="r", model_id="m", out_dir=a)
    run_bench(
        ARMS, ITEMS, _steerer(), run_id="r", model_id="m", out_dir=b, energy_tracker=FakeTracker()
    )
    assert not (a / ENERGY_FILE).exists()
    for name in ("neutral", "prompt_joyful", "steering_p0.50"):
        ra, rb = load_run(a)[name], load_run(b)[name]
        for x, y in zip(ra, rb, strict=True):
            x.pop("latency_s"), y.pop("latency_s")
            assert x == y
    assert sorted(p.name for p in b.glob("*.jsonl")) == sorted(p.name for p in a.glob("*.jsonl"))


def test_tracker_stopped_when_arm_fails(tmp_path):
    class Boom(FakeBackend):
        def generate(self, *args, **kwargs):
            raise RuntimeError("boom")

    tracker = FakeTracker()
    with pytest.raises(RuntimeError, match="boom"):
        run_bench(
            ARMS,
            ITEMS,
            Steerer(Boom()),
            run_id="r",
            model_id="m",
            out_dir=tmp_path,
            energy_tracker=tracker,
        )
    assert tracker.events == ["start", "stop"]


def test_overwrite_without_tracker_removes_stale_energy(tmp_path):
    run_bench(
        ARMS,
        ITEMS,
        _steerer(),
        run_id="r",
        model_id="m",
        out_dir=tmp_path,
        energy_tracker=FakeTracker(),
    )
    run_bench(ARMS, ITEMS, _steerer(), run_id="r", model_id="m", out_dir=tmp_path, overwrite=True)
    assert not (tmp_path / ENERGY_FILE).exists()


def _write_run(tmp_path):
    rows = {
        "neutral": ("neutral", [1.0, 1.0]),
        "prompt_joyful": ("prompt", [2.0, 4.0]),
        "steering_p0.50": ("steering", [3.0, 6.0]),
    }
    for arm, (kind, lats) in rows.items():
        lines = [
            {
                "arm": arm,
                "kind": kind,
                "item_id": str(i),
                "output": "un deux trois quatre",
                "latency_s": lat,
            }
            for i, lat in enumerate(lats)
        ]
        (tmp_path / f"{arm}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in lines) + "\n", encoding="utf-8"
        )
    write_energy(
        tmp_path,
        {
            "neutral": EnergyReading(0.002, 0.0002, 2.0),
            "prompt_joyful": EnergyReading(0.004, 0.0004, 6.0),
            "steering_p0.50": EnergyReading(0.006, 0.0006, 9.0),
        },
    )


def test_summary_per_arm_and_overhead(tmp_path):
    _write_run(tmp_path)
    s = frugality_summary(tmp_path, n_boot=50)
    assert s.reference == "prompt_joyful"
    p = s.arms["prompt_joyful"]
    assert p.n == 2
    assert p.latency_s.mean == 3.0
    assert p.latency_per_token_s.mean == pytest.approx((0.5 + 1.0) / 2)
    assert p.energy_kwh == 0.004
    assert p.energy_kwh_per_generation == pytest.approx(0.002)
    assert p.emissions_kg_per_generation == pytest.approx(0.0002)
    assert "neutral" not in s.overhead
    o = s.overhead["steering_p0.50"]
    assert o.latency_ratio == pytest.approx(1.5)
    assert o.latency_diff_s == pytest.approx(1.5)
    assert o.energy_ratio == pytest.approx(1.5)
    assert o.energy_diff_kwh == pytest.approx(0.001)
    assert o.emissions_ratio == pytest.approx(1.5)
    assert o.emissions_diff_kg == pytest.approx(0.0001)


def test_summary_without_energy_file_gives_none(tmp_path):
    _write_run(tmp_path)
    (tmp_path / ENERGY_FILE).unlink()
    s = frugality_summary(tmp_path, n_boot=10)
    o = s.overhead["steering_p0.50"]
    assert s.arms["neutral"].energy_kwh is None
    assert o.energy_ratio is None and o.energy_diff_kwh is None
    assert o.latency_ratio == pytest.approx(1.5)
    assert "n/d" in frugality_section(s)


def test_summary_without_prompt_arm_has_no_overhead(tmp_path):
    _write_run(tmp_path)
    (tmp_path / "prompt_joyful.jsonl").unlink()
    s = frugality_summary(tmp_path, n_boot=10)
    assert s.reference is None and s.overhead == {}
    assert "Surcoût non calculable" in frugality_section(s)


def test_estimate_tokens_prefers_n_tokens():
    assert estimate_tokens({"output": "a b c", "n_tokens": 10}) == 10
    assert estimate_tokens({"output": "a b c"}) == 3
    assert estimate_tokens({"output": ""}) == 0


def test_section_markdown(tmp_path):
    _write_run(tmp_path)
    text = frugality_section(frugality_summary(tmp_path, n_boot=50))
    assert "| steering_p0.50 | 2 |" in text
    assert "Surcoût du steering face à `prompt_joyful`" in text
    assert "x1.5 ; +1.5 s" in text
    assert "machine entière" in text
    assert "Une autre charge simultanée" in text
    assert text.endswith("\n")


def test_codecarbon_tracker_defaults_and_stop_without_start(tmp_path):
    t = CodeCarbonTracker()
    assert t.country_iso_code == "FRA"
    assert t.save_to_file is False
    assert t.output_dir.parts[0] == "data"
    with pytest.raises(RuntimeError, match="sans start"):
        t.stop()
