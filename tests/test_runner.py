import numpy as np
import pytest

from mood_steering.eval.arms import KIND_STEERING, Arm, default_arms
from mood_steering.eval.runner import (
    NEUTRAL_INSTRUCTION,
    Item,
    load_run,
    run_bench,
)
from mood_steering.steering import JOYFUL_SYSTEM_PROMPT, Steerer


class FakeBackend:
    n_layer = 8
    n_embd = 4

    def __init__(self):
        self.calls = []
        self.cvec = None

    def reset(self):
        pass

    def set_cvec(self, data, n_embd, il_start, il_end):
        self.cvec = data

    def generate(self, prompt, system, max_tokens, seed):
        cvec = None if self.cvec is None else self.cvec.copy()
        self.calls.append((prompt, system, seed, cvec))
        return f"out:{prompt}:{seed}"


ITEMS = [Item("a", "pa", {"k": 1}), Item("b", "pb", {}), Item("c", "pc", {})]
ARMS = default_arms((0.5, -0.5))


def _steerer(backend, with_vector=True):
    vec = np.ones((2, 4), dtype=np.float32) if with_vector else None
    return Steerer(backend, vec)


def test_default_arms():
    assert [a.name for a in ARMS] == [
        "neutral",
        "prompt_joyful",
        "steering_p0.50",
        "steering_m0.50",
    ]
    assert [a.kind for a in ARMS] == ["neutral", "prompt", "steering", "steering"]
    assert [a.dose for a in ARMS] == [0.0, 0.0, 0.5, -0.5]
    with pytest.raises(ValueError):
        default_arms((0.5, 1.0))
    with pytest.raises(ValueError):
        default_arms((0.5, 0.5, -0.5))


def test_arm_validation():
    with pytest.raises(ValueError):
        Arm("a+b", "neutral")
    with pytest.raises(ValueError):
        Arm("x", "prompt", 0.3)
    with pytest.raises(ValueError):
        Arm("x", "bizarre")
    with pytest.raises(ValueError):
        Arm("x", KIND_STEERING, 1.5)


def test_same_items_order_and_seed_for_all_arms(tmp_path):
    paths = run_bench(
        ARMS, ITEMS, _steerer(FakeBackend()), run_id="r", model_id="m", out_dir=tmp_path, seed=10
    )
    assert {n: p.name for n, p in paths.items()} == {a.name: f"{a.name}.jsonl" for a in ARMS}
    run = load_run(tmp_path)
    assert set(run) == {a.name for a in ARMS}
    for rows in run.values():
        assert [r["item_id"] for r in rows] == ["a", "b", "c"]
        assert [r["seed"] for r in rows] == [10, 11, 12]
        assert [r["output"] for r in rows] == ["out:pa:10", "out:pb:11", "out:pc:12"]


def test_row_format_and_dose_applied(tmp_path):
    be = FakeBackend()
    run_bench(ARMS, ITEMS[:1], _steerer(be), run_id="r1", model_id="mod", out_dir=tmp_path)
    run = load_run(tmp_path)
    row = run["steering_m0.50"][0]
    assert set(row) == {
        "run_id",
        "model_id",
        "arm",
        "kind",
        "dose",
        "item_id",
        "seed",
        "system",
        "output",
        "fallback",
        "neutral_instruction",
        "latency_s",
    }
    assert row["run_id"] == "r1" and row["model_id"] == "mod"
    assert row["arm"] == "steering_m0.50" and row["kind"] == "steering"
    assert row["dose"] == -0.5 and row["fallback"] is False
    assert row["neutral_instruction"] is False and row["system"] == ""
    assert isinstance(row["latency_s"], float) and row["latency_s"] >= 0
    cv = [c[3] for c in be.calls]
    assert cv[0] is None and cv[1] is None
    assert np.all(cv[2] == 0.5)
    assert np.all(cv[3] == -0.5)
    assert run["prompt_joyful"][0]["system"] == JOYFUL_SYSTEM_PROMPT
    assert be.calls[1][1] == JOYFUL_SYSTEM_PROMPT
    assert be.calls[0][1] is None


def test_neutral_instruction_added_to_every_arm(tmp_path):
    be = FakeBackend()
    run_bench(
        ARMS,
        ITEMS[:1],
        _steerer(be),
        run_id="r",
        model_id="m",
        out_dir=tmp_path,
        neutral_instruction=True,
    )
    run = load_run(tmp_path)
    assert [c[1] for c in be.calls] == [
        NEUTRAL_INSTRUCTION,
        f"{JOYFUL_SYSTEM_PROMPT}\n\n{NEUTRAL_INSTRUCTION}",
        NEUTRAL_INSTRUCTION,
        NEUTRAL_INSTRUCTION,
    ]
    assert all(rows[0]["neutral_instruction"] is True for rows in run.values())
    assert run["prompt_joyful"][0]["system"].endswith(NEUTRAL_INSTRUCTION)


def test_fallback_without_vector_is_recorded(tmp_path):
    be = FakeBackend()
    run_bench(
        ARMS,
        ITEMS[:1],
        _steerer(be, with_vector=False),
        run_id="r",
        model_id="m",
        out_dir=tmp_path,
        neutral_instruction=True,
    )
    run = load_run(tmp_path)
    assert all(rows[0]["fallback"] is True for rows in run.values())
    joyful_neutral = f"{JOYFUL_SYSTEM_PROMPT}\n\n{NEUTRAL_INSTRUCTION}"
    # le prompt joyeux n'est jamais doublé
    assert [c[1] for c in be.calls] == [joyful_neutral] * 4
    assert all(rows[0]["system"] == joyful_neutral for rows in run.values())


def test_default_out_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_bench(ARMS, ITEMS[:1], _steerer(FakeBackend()), run_id="abc", model_id="m")
    assert (tmp_path / "data" / "runs" / "abc" / "neutral.jsonl").is_file()


def test_duplicate_inputs_rejected(tmp_path):
    steerer = _steerer(FakeBackend())
    dup_arms = [Arm("x", "neutral"), Arm("x", "prompt")]
    with pytest.raises(ValueError, match="bras"):
        run_bench(dup_arms, ITEMS, steerer, run_id="r", model_id="m", out_dir=tmp_path)
    with pytest.raises(ValueError, match="items"):
        run_bench(
            [Arm("x", "neutral")],
            [ITEMS[0], ITEMS[0]],
            steerer,
            run_id="r",
            model_id="m",
            out_dir=tmp_path,
        )
