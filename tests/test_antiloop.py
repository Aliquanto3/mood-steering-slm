import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mood_steering.eval.accuracy import McqItem
from mood_steering.eval.resumable import read_rows

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
    assert s["accuracy"]["mean"] == 0.25  # q0 seul : q3 donne B mais est tronquée
    assert s["n_not_extracted"] == 1
    assert s["mean_tokens"] == 25
    assert s["accuracy"]["low"] <= 0.25 <= s["accuracy"]["high"]
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


class _Cfg:
    def __init__(self, eos):
        self.eos_token_id = eos


class _Tok:
    unk_token_id = 3

    def __init__(self, eos, vocab):
        self.eos_token_id = eos
        self._vocab = vocab

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)


class _Model:
    def __init__(self, eos):
        self.generation_config = _Cfg(eos)


def test_collect_eos_ids_adds_im_end():
    # Cas Qwen3.5 : config.json donne <|endoftext|>, le tour se termine par <|im_end|>.
    tok = _Tok(None, {"<|im_end|>": 248046})
    assert antiloop.collect_eos_ids(tok, _Model(248044)) == [248044, 248046]
    tok = _Tok(7, {})  # <|im_end|> inconnu : l'id unk n'est pas pris pour une fin
    assert antiloop.collect_eos_ids(tok, _Model([1, 2])) == [1, 2, 7]
    with pytest.raises(ValueError, match="aucun token de fin"):
        antiloop.collect_eos_ids(_Tok(None, {}), _Model(None))
    # Réponse finie par <|im_end|> au plafond exact : non tronquée.
    eos = antiloop.collect_eos_ids(_Tok(None, {"<|im_end|>": 9}), _Model(8))
    assert not antiloop.is_truncated([1, 2, 9], 3, eos)


def test_check_run_config_refuses_other_config(tmp_path):
    cfg = {"max_new_tokens": 32, "thinking": True, "item_sources": ("gmmlu-fr",)}
    antiloop.check_run_config(tmp_path, cfg)
    antiloop.check_run_config(tmp_path, dict(cfg))  # même config : reprise acceptée
    with pytest.raises(ValueError, match="max_new_tokens"):
        antiloop.check_run_config(tmp_path, {**cfg, "max_new_tokens": 64})
    with pytest.raises(ValueError, match="torch"):
        antiloop.check_run_config(tmp_path, {**cfg, "torch": "2.15.0"})


def test_item_sources():
    items = [
        McqItem(f"{p}:{i}", "Q", dict.fromkeys("ABCD", "x"), "A", "s", "fr")
        for p, i in [("gmmlu-fr", 1), ("gmmlu-fr", 2), ("mmmlu-fr", 3)]
    ]
    assert antiloop.item_sources(items) == ["gmmlu-fr", "mmmlu-fr"]


def test_run_arms_ablator_wiring_and_summary_merge(tmp_path):
    torch = pytest.importorskip("torch")

    class Layer(torch.nn.Module):
        def forward(self, h):
            return h

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(Layer() for _ in range(20))

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    models = {"origin": Fake(), "surgery": Fake()}
    loads: list[str] = []

    def load(key):
        loads.append(key)
        return None, models[key]

    seen: list[tuple[str, int, float]] = []

    def make_gen(_tok, model):
        def gen(_prompt):
            hooked = sum(len(layer._forward_hooks) for layer in model.model.layers)
            h = torch.ones(1, 4)
            out = model.model.layers[15](h)
            name = "surgery" if model is models["surgery"] else "origin"
            seen.append((name, hooked, float(out[0, 0])))
            return "B", 5, True

        return gen

    dirs = {15: torch.tensor([1.0, 0, 0, 0]).numpy(), 19: torch.tensor([0, 1.0, 0, 0]).numpy()}
    items = _items(1)
    arms = antiloop.build_arms([1.0])
    summary = antiloop.run_arms(arms, items, load, make_gen, dirs, tmp_path, 32, {"arms": {}})
    # Hooks posés (couches 15 et 19) sur le seul bras ablation, retirés avant le suivant.
    assert seen == [("origin", 0, 1.0), ("origin", 2, 0.0), ("surgery", 0, 1.0)]
    assert loads == ["origin", "surgery"]
    assert all(len(layer._forward_hooks) == 0 for layer in models["origin"].model.layers)
    assert set(summary["arms"]) == {"origin", "ablation@1", "surgery"}
    # Relance d'un seul bras : summary.json garde les bras déjà mesurés.
    antiloop.run_arms(
        antiloop.build_arms([1.0], ["surgery"]),
        items,
        load,
        make_gen,
        dirs,
        tmp_path,
        32,
        {"arms": {}},
    )
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(saved["arms"]) == {"origin", "ablation@1", "surgery"}


def _rec(item_id, output):
    return {
        "item_id": item_id,
        "arm": "origin",
        "output": output,
        "n_tokens": 3,
        "truncated": False,
        "max_new_tokens": 32,
    }


def test_run_arm_reprise_avec_separateurs_unicode(tmp_path):
    items = _items(3)
    arm = antiloop.Arm("origin", "origin")
    path = tmp_path / "origin.jsonl"
    first = [_rec("q0", "a b c"), _rec("q1", "x\u0085y")]
    raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in first)
    path.write_bytes(raw.encode("utf-8"))
    assert read_rows(path) == first
    calls: list[str] = []
    recs = antiloop.run_arm(arm, items, _fake(calls), tmp_path, 32)
    assert len(calls) == 1  # seul q2 est généré
    assert recs[:2] == first
    assert path.read_bytes().startswith(raw.encode("utf-8"))
    assert read_rows(path) == recs


def test_run_arm_reecriture_atomique_via_part(tmp_path, monkeypatch):
    items = _items(2)
    arm = antiloop.Arm("origin", "origin")
    path = tmp_path / "origin.jsonl"
    good = json.dumps(_rec("q0", "ok"), ensure_ascii=False) + "\n"
    path.write_bytes(good.encode() + b'{"item_id": "q1", "ou')
    before = path.read_bytes()

    def boom(src, dst):
        raise OSError("arrêt simulé")

    monkeypatch.setattr(antiloop.os, "replace", boom)
    with pytest.raises(OSError, match="arrêt simulé"):
        antiloop.run_arm(arm, items, _fake([]), tmp_path, 32)
    assert path.read_bytes() == before  # fichier d'origine intact
    assert (tmp_path / "origin.jsonl.part").read_bytes() == good.encode()
    monkeypatch.undo()
    # Le .part résiduel est écrasé ; la ligne tronquée est écartée.
    calls: list[str] = []
    recs = antiloop.run_arm(arm, items, _fake(calls), tmp_path, 32)
    assert len(calls) == 1 and [r["item_id"] for r in recs] == ["q0", "q1"]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert not (tmp_path / "origin.jsonl.part").exists()
