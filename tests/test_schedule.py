"""Tests du module schedule et de son branchement dans le Steerer (backend simulé)."""

from types import SimpleNamespace

import numpy as np
import pytest

from mood_steering.schedule import constant, pulse, sample
from mood_steering.steering import (
    ARM_PROMPT_JOYFUL,
    ARM_PROMPT_SAD,
    ARM_STEERING,
    JOYFUL_SYSTEM_PROMPT,
    SAD_SYSTEM_PROMPT,
    Steerer,
)

N_EMBD = 4


class FakeBackend:
    n_embd = N_EMBD
    n_layer = 6

    def __init__(self):
        self.events = []

    def reset(self):
        self.events.append("reset")

    def set_cvec(self, data, n_embd, il_start, il_end):
        self.events.append(("set", None if data is None else float(data.sum())))

    def generate(self, prompt, system, max_tokens, seed):
        return "x"

    def stream(self, prompt, system, max_tokens, seed):
        self.system = system
        yield from ["a", "b"]

    def stream_scheduled(self, prompt, system, max_tokens, seed, on_token_start):
        self.system = system
        for t in range(max_tokens):
            on_token_start(t)
            self.events.append(("token", t))
            yield f"t{t}"


def vec():
    return np.ones((3, N_EMBD), dtype=np.float32)  # somme par dose d : 12 * d * max_scale


def sets(backend):
    return [e for e in backend.events if isinstance(e, tuple) and e[0] == "set"]


def test_constant():
    s = constant(0.4)
    assert all(s(t) == 0.4 for t in (0, 1, 50, 10_000))
    assert sample(s, 3) == [0.4, 0.4, 0.4]


@pytest.mark.parametrize("bad", [1.01, -1.01, float("nan")])
def test_constant_hors_bornes(bad):
    with pytest.raises(ValueError, match="dose"):
        constant(bad)


def test_pulse_monte_culmine_decroit():
    s = pulse(4, 0.8, 2)
    assert sample(s, 5) == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])
    assert s(4) == 0.8  # sommet exact
    assert s(6) == pytest.approx(0.4)  # une demi-vie après le sommet
    assert s(8) == pytest.approx(0.2)
    vals = sample(s, 40)
    assert vals.index(max(vals)) == 4
    assert vals == sorted(vals[:5]) + sorted(vals[5:], reverse=True)
    assert vals[-1] < 1e-5
    assert all(-1 <= v <= 1 for v in vals)


def test_pulse_start_et_pic_negatif():
    assert sample(pulse(2, 0.5, 1, start=0.1), 3) == pytest.approx([0.1, 0.3, 0.5])
    neg = sample(pulse(2, -1.0, 3), 40)
    assert neg[2] == -1.0
    assert all(-1 <= v <= 0 for v in neg)


def test_pulse_rise_zero():
    s = pulse(0, 0.6, 1)
    assert sample(s, 3) == pytest.approx([0.6, 0.3, 0.15])


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        ({"rise": -1, "peak": 0.5, "half_life": 1}, "rise"),
        ({"rise": float("nan"), "peak": 0.5, "half_life": 1}, "rise"),
        ({"rise": 1, "peak": 0.5, "half_life": 0}, "half_life"),
        ({"rise": 1, "peak": 0.5, "half_life": -2}, "half_life"),
        ({"rise": 1, "peak": 1.5, "half_life": 1}, "peak"),
        ({"rise": 1, "peak": -1.5, "half_life": 1}, "peak"),
        ({"rise": 1, "peak": 0.5, "half_life": 1, "start": 2.0}, "start"),
    ],
)
def test_pulse_validation(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        pulse(**kwargs)


def test_sample_n_negatif():
    with pytest.raises(ValueError, match="n"):
        sample(constant(0), -1)
    assert sample(constant(0), 0) == []


def test_binding_rappele_seulement_aux_changements():
    b = FakeBackend()
    st = Steerer(b, vec(), max_scale=0.5)
    # doses par token : 0.5, 0.5, 0.5, 0.2, 0.2, 0.0 (le dernier retire le vecteur)
    table = [0.5, 0.5, 0.5, 0.2, 0.2, 0.0]
    ts = st.open_stream("q", schedule=table.__getitem__, max_tokens=6)
    assert list(ts) == [f"t{i}" for i in range(6)]
    # initial None (reset), puis set aux tokens 0 (0.5), 3 (0.2), 5 (retrait)
    assert sets(b) == [("set", None), ("set", 3.0), ("set", 1.2000000476837158), ("set", None)]
    assert ts.doses == table


def test_ordre_set_avant_token():
    b = FakeBackend()
    st = Steerer(b, vec(), max_scale=1.0)
    ts = st.open_stream("q", schedule=lambda t: 0.1 if t < 2 else 0.2, max_tokens=3)
    list(ts)
    assert b.events == [
        "reset",
        ("set", None),
        ("set", pytest.approx(1.2)),
        ("token", 0),
        ("token", 1),
        ("set", pytest.approx(2.4)),
        ("token", 2),
    ]


def test_arrondi_evite_les_rappels():
    b = FakeBackend()
    st = Steerer(b, vec())
    ts = st.open_stream("q", schedule=lambda t: 0.3 + t * 0.0001, max_tokens=5)
    list(ts)
    assert len(sets(b)) == 2  # None initial + un seul set (0.3, 0.3001... arrondis à 0.3)
    assert ts.doses == [0.3] * 5


def test_stream_schedule_remplace_dose():
    b = FakeBackend()
    st = Steerer(b, vec())
    assert list(st.stream("q", 1.0, schedule=constant(0.5), max_tokens=2)) == ["t0", "t1"]
    assert sets(b)[-1] == ("set", pytest.approx(6.0))


def test_dose_hors_bornes_par_token():
    b = FakeBackend()
    st = Steerer(b, vec())
    ts = st.open_stream("q", schedule=lambda t: 0.1 if t == 0 else 1.5, max_tokens=3)
    it = iter(ts)
    assert next(it) == "t0"
    with pytest.raises(ValueError, match="dose"):
        next(it)


def test_schedule_initial_hors_bornes_leve_a_l_ouverture():
    st = Steerer(FakeBackend(), vec())
    with pytest.raises(ValueError, match="dose"):
        st.open_stream("q", schedule=lambda t: 2.0)


def test_repli_signe_du_premier_token():
    b = FakeBackend()
    st = Steerer(b)  # sans vecteur
    ts = st.open_stream("q", schedule=pulse(2, 0.8, 3), system="Sois bref.", max_tokens=4)
    # dose 0 au token 0 : le repli suit la première dose non nulle (token 1)
    assert (ts.arm, ts.fallback, ts.dose) == (ARM_PROMPT_JOYFUL, True, 0.0)
    list(ts)
    assert b.system == f"{JOYFUL_SYSTEM_PROMPT}\n\nSois bref."

    ts = st.open_stream("q", schedule=pulse(2, -0.8, 3), max_tokens=4)
    assert (ts.arm, ts.fallback) == (ARM_PROMPT_SAD, True)
    list(ts)

    ts = st.open_stream("q", schedule=constant(0.0), system="Sois bref.", max_tokens=4)
    assert (ts.arm, ts.fallback) == (ARM_STEERING, False)  # profil nul : pas de repli
    list(ts)
    assert b.system == "Sois bref."

    # première dose non nulle au-delà de max_tokens : pas de repli
    ts = st.open_stream("q", schedule=lambda t: 0.0 if t < 5 else 0.5, max_tokens=5)
    assert (ts.arm, ts.fallback) == (ARM_STEERING, False)
    list(ts)

    ts = st.open_stream("q", schedule=pulse(0, 0.8, 3), system="Sois bref.", max_tokens=4)
    assert (ts.arm, ts.fallback) == (ARM_PROMPT_JOYFUL, True)
    list(ts)
    assert b.system == f"{JOYFUL_SYSTEM_PROMPT}\n\nSois bref."
    assert ts.doses == pytest.approx([0.8, 0.635, 0.504, 0.4], abs=1e-3)
    assert all(e[1] is None for e in sets(b))  # aucun vecteur posé

    ts = st.open_stream("q", schedule=constant(-0.5), max_tokens=2)
    assert (ts.arm, ts.fallback) == (ARM_PROMPT_SAD, True)
    list(ts)
    assert b.system == SAD_SYSTEM_PROMPT


def test_appels_existants_sans_schedule():
    b = FakeBackend()
    st = Steerer(b, vec(), max_scale=0.5)
    ts = st.open_stream("q", 1.0)
    assert (ts.arm, ts.dose, ts.fallback, ts.doses) == (ARM_STEERING, 1.0, False, [])
    assert list(ts) == ["a", "b"]
    assert sets(b) == [("set", 6.0)]
    assert list(st.stream("q", 0.0)) == ["a", "b"]


class _FakeLlama:
    """Faux Llama (frontière externe) : 7 = 0xC3 seul, 8 = suite de « é », 6 = 0xFF
    invalide, autres = « ok » ; fins de génération : 99 (EOS) et 98 (EOT)."""

    def __init__(self, tokens, n_ctx=2048, n_prompt=2):
        self.tokens = tokens
        self.evals = []
        self.resets = 0
        self.metadata = {}
        self._n_ctx = n_ctx
        self._n_prompt = n_prompt
        self._model = SimpleNamespace(vocab="vocab")

    def n_ctx(self):
        return self._n_ctx

    def reset(self):
        self.resets += 1
        self.evals = []

    def tokenize(self, data, add_bos=False, special=False):
        return list(range(1, self._n_prompt + 1))

    def token_eos(self):
        return 99

    def eval(self, toks):
        self.evals.append(list(toks))

    def sample(self, top_k, temp):
        return self.tokens[len(self.evals) - 1]

    def detokenize(self, toks):
        return {7: b"\xc3", 8: "é".encode()[1:], 6: b"\xff"}.get(toks[0], b"ok")


def _backend(tokens, **kw):
    from mood_steering.steering import LlamaCppBackend

    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._llm = _FakeLlama(tokens, **kw)
    b._cpp = SimpleNamespace(llama_vocab_is_eog=lambda vocab, tok: vocab == "vocab" and tok > 97)

    class _Fmt:
        prompt = "P"

    b._chat_formatter = lambda: lambda messages: _Fmt
    return b


def test_llama_boucle_s_arrete_a_max_tokens_sans_utf8_complet():
    b = _backend([7] * 50)
    calls = []
    out = list(b.stream_scheduled("q", None, 5, 0, calls.append))
    assert out == ["\ufffd"] * 4  # 0xC3 suivi de 0xC3 : invalide ; le dernier, coupé, abandonné
    assert calls == [0, 1, 2, 3, 4]
    assert len(b._llm.evals) == 5  # prompt + 4 tokens : pas un de plus
    assert b._llm.resets == 1


def test_llama_boucle_recolle_les_octets_et_s_arrete_a_eos():
    b = _backend([7, 8, 5, 99, 5])
    assert list(b.stream("q", None, 10, 0)) == ["é", "ok"]
    assert b.generate("q", None, 10, 0) == "éok"
    assert b._llm.evals[-1] == [5]


def test_llama_octet_invalide_ne_bloque_pas_la_suite():
    b = _backend([6, 5, 7, 8, 5, 99])
    assert b.generate("q", None, 10, 0) == "\ufffdokéok"


def test_llama_arret_sur_tout_token_de_fin():
    b = _backend([5, 98, 5])  # 98 : fin de tour (EOT), pas EOS
    assert list(b.stream("q", None, 10, 0)) == ["ok"]


def test_llama_max_tokens_borne_par_le_contexte():
    b = _backend([5] * 50, n_ctx=10, n_prompt=7)
    calls = []
    assert list(b.stream_scheduled("q", None, 100, 0, calls.append)) == ["ok"] * 3
    assert calls == [0, 1, 2]


def test_llama_prompt_trop_long_erreur_claire_a_l_ouverture():
    b = _backend([5], n_ctx=8, n_prompt=8)
    with pytest.raises(ValueError, match="contexte de 8 tokens"):
        b.stream_scheduled("q", None, 4, 0, lambda t: None)  # avant toute itération


def test_llama_formateur_sans_bos():
    from mood_steering.steering import LlamaCppBackend

    b = LlamaCppBackend.__new__(LlamaCppBackend)
    seen = []
    b._llm = SimpleNamespace(_model=SimpleNamespace(token_get_text=lambda t: seen.append(t) or "x"))
    assert b._token_text(-1) == "" and seen == []
    assert b._token_text(3) == "x" and seen == [3]


def test_dose_validee_sans_tolerance_a_l_ouverture_et_en_flux():
    st = Steerer(FakeBackend(), vec())
    with pytest.raises(ValueError, match="dose"):
        st.open_stream("q", schedule=lambda t: 1.0004)
    ts = st.open_stream("q", schedule=lambda t: 1.0 if t == 0 else 1.0004, max_tokens=3)
    it = iter(ts)
    next(it)
    with pytest.raises(ValueError, match="dose"):
        next(it)


def test_repli_sans_vecteur_valide_le_profil_a_l_ouverture():
    st = Steerer(FakeBackend())
    with pytest.raises(ValueError, match="dose"):
        st.open_stream("q", schedule=lambda t: 0.0 if t < 2 else 1.5, max_tokens=4)
