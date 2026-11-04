import numpy as np
import pytest

from mood_steering.eval.arms import KIND_STEERING, Arm, default_arms
from mood_steering.eval.runner import (
    NEUTRAL_INSTRUCTION,
    Item,
    load_run,
    run_bench,
)
from mood_steering.steering import (
    JOYFUL_SYSTEM_PROMPT,
    SAD_SYSTEM_PROMPT,
    Steerer,
    compose_system,
)


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
    for bad in ("CON", "nul", "com1", "Lpt9.txt", "aux.jsonl", ".cache", "x."):
        with pytest.raises(ValueError):
            Arm(bad, "neutral")
    Arm("console", "neutral")  # préfixe réservé seul : accepté


@pytest.mark.parametrize("run_id", ["../x", "a/b", "NUL", "", ".."])
def test_unsafe_run_id_rejected(tmp_path, monkeypatch, run_id):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        run_bench(ARMS, ITEMS[:1], _steerer(FakeBackend()), run_id=run_id, model_id="m")
    assert not (tmp_path / "data").exists()


def test_case_collision_rejected(tmp_path):
    arms = [Arm("Neutral", "neutral"), Arm("neutral", "neutral")]
    with pytest.raises(ValueError, match="bras"):
        run_bench(arms, ITEMS, _steerer(FakeBackend()), run_id="r", model_id="m", out_dir=tmp_path)


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


@pytest.mark.parametrize("neutral", [False, True])
def test_fallback_without_vector_only_for_steering_arms(tmp_path, neutral):
    be = FakeBackend()
    run_bench(
        ARMS,
        ITEMS[:1],
        _steerer(be, with_vector=False),
        run_id="r",
        model_id="m",
        out_dir=tmp_path,
        neutral_instruction=neutral,
    )
    run = load_run(tmp_path)
    extra = NEUTRAL_INSTRUCTION if neutral else None
    expected = [
        extra,  # neutre : ni vecteur ni prompt d'humeur
        compose_system(JOYFUL_SYSTEM_PROMPT, extra),
        compose_system(JOYFUL_SYSTEM_PROMPT, extra),  # dose > 0 : repli joyeux
        compose_system(SAD_SYSTEM_PROMPT, extra),  # dose < 0 : repli triste
    ]
    assert [c[1] for c in be.calls] == expected
    assert all(c[3] is None for c in be.calls)
    assert {n: rows[0]["fallback"] for n, rows in run.items()} == {
        "neutral": False,
        "prompt_joyful": False,
        "steering_p0.50": True,
        "steering_m0.50": True,
    }
    assert [run[a.name][0]["system"] for a in ARMS] == [e or "" for e in expected]


def test_existing_run_not_overwritten(tmp_path):
    steerer = _steerer(FakeBackend())
    run_bench(ARMS, ITEMS, steerer, run_id="r", model_id="m", out_dir=tmp_path)
    (tmp_path / "old_arm.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_bench(ARMS[:1], ITEMS[:1], steerer, run_id="r", model_id="m", out_dir=tmp_path)
    run_bench(
        ARMS[:1], ITEMS[:1], steerer, run_id="r", model_id="m", out_dir=tmp_path, overwrite=True
    )
    run = load_run(tmp_path)
    assert set(run) == {"neutral"} and len(run["neutral"]) == 1


def test_interrupted_arm_leaves_no_jsonl(tmp_path):
    class Boom(FakeBackend):
        def generate(self, prompt, system, max_tokens, seed):
            if prompt == "pb":
                raise RuntimeError("panne")
            return super().generate(prompt, system, max_tokens, seed)

    with pytest.raises(RuntimeError):
        run_bench(ARMS[:1], ITEMS, _steerer(Boom()), run_id="r", model_id="m", out_dir=tmp_path)
    assert not (tmp_path / "neutral.jsonl").exists()
    assert (tmp_path / "neutral.jsonl.part").exists()
    assert load_run(tmp_path) == {}


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
