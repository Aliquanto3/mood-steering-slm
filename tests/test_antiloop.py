import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mood_steering.eval.accuracy import McqItem

_SPEC = importlib.util.spec_from_file_location(
    "run_antiloop", Path(__file__).resolve().parents[1] / "scripts" / "run_antiloop.py"
)
antiloop = importlib.util.module_from_spec(_SPEC)
sys.modules["run_antiloop"] = antiloop
_SPEC.loader.exec_module(antiloop)


def _items(n=4):
    return [
        McqItem(f"q{i}", f"Q{i} ?", dict(zip("ABCD", "wxyz", strict=True)), "B", "s", "fr")
        for i in range(n)
    ]


def test_is_truncated():
    assert antiloop.is_truncated([1, 2, 3], 3, [99])
    assert not antiloop.is_truncated([1, 2, 99], 3, [99])  # fin au plafond exact
    assert not antiloop.is_truncated([1, 2], 3, [99])
    assert not antiloop.is_truncated([1, 99, 3], 3, [98, 99])


def test_build_arms_order_and_filter():
    arms = antiloop.build_arms([0.5, 1.0])
    assert [a.name for a in arms] == ["origin", "ablation@0.5", "ablation@1", "surgery"]
    assert [a.alpha for a in arms] == [None, 0.5, 1.0, None]
    assert [a.name for a in antiloop.build_arms([1.0], ["surgery"])] == ["surgery"]
    with pytest.raises(ValueError, match="bras inconnus"):
        antiloop.build_arms([1.0], ["nope"])


def test_summarize_arm_aggregation():
    items = _items(4)
    outs = ["B", "A", "<think>sans fin", "Réponse : B"]
    recs = [
        {"item_id": i.id, "output": o, "n_tokens": t, "truncated": tr}
        for i, o, t, tr in zip(
            items, outs, [10, 20, 30, 40], [False, False, True, True], strict=True
        )
    ]
    s = antiloop.summarize_arm(recs, items)
    assert s["n"] == 4
    assert s["truncation_rate"]["mean"] == 0.5
    assert s["accuracy"]["mean"] == 0.5  # q0 et q3 corrects
    assert s["n_not_extracted"] == 1
    assert s["mean_tokens"] == 25
    assert s["accuracy"]["low"] <= 0.5 <= s["accuracy"]["high"]
    with pytest.raises(ValueError, match="manquants"):
        antiloop.summarize_arm(recs[:3], items)


def _fake(calls):
    def generate(prompt):
        calls.append(prompt)
        return "B", 5, True

    return generate


def test_run_arm_resume_and_same_items(tmp_path):
    items = _items(4)
    arm = antiloop.Arm("origin", "origin")
    calls: list[str] = []
    antiloop.run_arm(arm, items[:2], _fake(calls), tmp_path, 32)
    assert len(calls) == 2
    # Reprise : seuls les items restants sont générés, les premiers sont conservés.
    recs = antiloop.run_arm(arm, items, _fake(calls), tmp_path, 32)
    assert len(calls) == 4
    assert [r["item_id"] for r in recs] == ["q0", "q1", "q2", "q3"]
    lines = (tmp_path / "origin.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert json.loads(lines[0])["truncated"] is False
    # Autre bras : mêmes prompts, dans le même ordre.
    calls2: list[str] = []
    antiloop.run_arm(
        antiloop.Arm("ablation@1", "ablation", 1.0), items, _fake(calls2), tmp_path, 32
    )
    assert calls2 == calls[:2] + calls[2:4]
    assert (tmp_path / "ablation_a1.jsonl").is_file()


def test_run_arm_truncated_flag_and_partial_line(tmp_path):
    items = _items(2)
    arm = antiloop.Arm("origin", "origin")
    path = tmp_path / "origin.jsonl"
    good = {"item_id": "q0", "output": "x", "n_tokens": 1, "truncated": False}
    path.write_text(json.dumps(good) + '\n{"item_id": "q1", "ou', encoding="utf-8")
    recs = antiloop.run_arm(arm, items, lambda _p: ("boucle", 32, False), tmp_path, 32)
    assert recs[0] == good
    assert recs[1]["truncated"] is True


def test_run_arm_inconsistent_resume(tmp_path):
    arm = antiloop.Arm("origin", "origin")
    antiloop.run_arm(arm, _items(2), _fake([]), tmp_path, 32)
    with pytest.raises(ValueError, match="incohérente"):
        antiloop.run_arm(arm, list(reversed(_items(2))), _fake([]), tmp_path, 32)
