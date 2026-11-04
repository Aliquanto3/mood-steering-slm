"""Logique pure du pic 1.2 : aucun modèle, llama_cpp simulé par un faux module."""

import importlib.util
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "cvec_between_tokens", Path(__file__).parent.parent / "spikes" / "cvec_between_tokens.py"
)
spike = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(spike)


def test_vector_shape_and_norm_per_layer():
    vec = spike.make_synthetic_vector(n_embd=16, n_layers=3, seed=1, scale=8.0)
    assert len(vec) == 48
    for layer in range(3):
        chunk = vec[layer * 16 : (layer + 1) * 16]
        assert math.sqrt(sum(x * x for x in chunk)) == pytest.approx(8.0)


def test_vector_is_deterministic_per_seed():
    a = spike.make_synthetic_vector(8, 2, seed=3, scale=1.0)
    assert a == spike.make_synthetic_vector(8, 2, seed=3, scale=1.0)
    assert a != spike.make_synthetic_vector(8, 2, seed=4, scale=1.0)


@pytest.mark.parametrize("n_embd,n_layers", [(0, 1), (4, 0), (-1, 2)])
def test_vector_rejects_bad_dimensions(n_embd, n_layers):
    with pytest.raises(ValueError, match="positifs"):
        spike.make_synthetic_vector(n_embd, n_layers, seed=0, scale=1.0)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ([1, 2, 3], [1, 2, 3], None),
        ([1, 2, 3], [1, 9, 3], 1),
        ([5], [6], 0),
        ([1, 2], [1, 2, 3], 2),
        ([1, 2, 3], [1, 2], 2),
        ([], [], None),
    ],
)
def test_first_divergence(a, b, expected):
    assert spike.first_divergence(a, b) == expected


def test_summarize_reports_each_pair():
    runs = {
        "base": [1, 2, 3, 4],
        "full": [1, 9, 9, 9],
        "late": [1, 2, 3, 7],
        "off": [1, 9, 3, 4],
        "replay": [1, 9, 3, 5],
    }
    assert spike.summarize(runs, 3) == {
        "late_vs_base": 3,
        "full_vs_base": 1,
        "off_vs_full": 2,
        "off_vs_base": 1,
        "late_vs_full": 1,
        "off_vs_replay": 3,
        "n_switch": 3,
    }


class _FakeLib:
    """Faux binding : enregistre les appels de llama_set_adapter_cvec."""

    def __init__(self, rc=0):
        self.calls = []
        self.rc = rc

    def llama_set_adapter_cvec(self, ctx, data, length, n_embd, il_start, il_end):
        self.calls.append((ctx, None if data is None else list(data), length, il_start, il_end))
        return self.rc


def test_set_cvec_passes_buffer_and_range():
    lib = _FakeLib()
    spike._set_cvec(lib, "ctx", [1.0, 2.0, 3.0, 4.0], 2, 1, 2)
    assert lib.calls == [("ctx", [1.0, 2.0, 3.0, 4.0], 4, 1, 2)]


def test_set_cvec_none_clears_with_null():
    lib = _FakeLib()
    spike._set_cvec(lib, "ctx", None, 2, 1, 2)
    assert lib.calls == [("ctx", None, 0, 0, 0)]


def test_set_cvec_raises_on_error_code():
    with pytest.raises(RuntimeError, match="-1"):
        spike._set_cvec(_FakeLib(rc=-1), "ctx", None, 2, 1, 2)


class _FakeLlm:
    """Faux Llama : journalise reset, eval et sample dans le même journal que le binding."""

    def __init__(self, log):
        self.log = log
        self._ctx = type("C", (), {"ctx": "ctx"})()
        self.next_tok = 100

    def reset(self):
        self.log.append("reset")

    def tokenize(self, data, add_bos):
        return [1, 2]

    def eval(self, toks):
        self.log.append(("eval", list(toks)))

    def sample(self, top_k, temp):
        self.next_tok += 1
        self.log.append(("sample", self.next_tok))
        return self.next_tok


class _LoggingLib(_FakeLib):
    def __init__(self, log):
        super().__init__()
        self.log = log

    def llama_set_adapter_cvec(self, ctx, data, length, n_embd, il_start, il_end):
        self.log.append(("cvec", None if data is None else list(data)))
        return 0


def test_generate_sets_vector_before_sampling_and_clears_at_end():
    log = []
    vec = [0.5, 0.5]
    out = spike.generate(_FakeLlm(log), _LoggingLib(log), "p", 3, {1: vec, 2: None}, 2, 1, 1)
    assert out == [101, 102, 103]
    assert log == [
        "reset",
        ("cvec", None),  # prompt évalué sans vecteur
        ("eval", [1, 2]),
        ("sample", 101),
        ("eval", [101]),
        ("cvec", vec),  # posé avant le token 1 : agit sur l'évaluation de 101, donc sur 102
        ("sample", 102),
        ("eval", [102]),
        ("cvec", None),
        ("sample", 103),
        ("eval", [103]),
        ("cvec", None),  # retrait final : pas de fuite vers l'exécution suivante
    ]


def test_generate_forced_prefix_skips_sampling():
    log = []
    out = spike.generate(_FakeLlm(log), _LoggingLib(log), "p", 3, {}, 2, 1, 1, forced=[7, 8])
    assert out == [7, 8, 101]
    assert [e for e in log if e[0] == "sample"] == [("sample", 101)]
    assert ("eval", [7]) in log and ("eval", [8]) in log


def test_summarize_reports_replay_control():
    runs = {"base": [1], "full": [2], "late": [1], "off": [2, 3], "replay": [2, 4]}
    assert spike.summarize(runs, 1)["off_vs_replay"] == 1
