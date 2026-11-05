"""Tests du Steerer avec un backend simulé (aucun modèle réel)."""

import numpy as np
import pytest

from mood_steering.steering import (
    JOYFUL_SYSTEM_PROMPT,
    SAD_SYSTEM_PROMPT,
    Steerer,
    apply_dose,
)

N_EMBD = 4
N_LAYER = 6


class FakeBackend:
    n_embd = N_EMBD
    n_layer = N_LAYER

    loads = 0  # chargements du modèle : un par instance

    def __init__(self):
        type(self).loads += 1
        self.cvec_calls = []
        self.gen_calls = []
        self.events = []
        self.current = None
        self.cached = None  # dose sous laquelle les K/V du contexte ont été calculés

    def reset(self):
        self.events.append("reset")
        self.cached = None

    def set_cvec(self, data, n_embd, il_start, il_end):
        self.events.append("set_cvec")
        self.cvec_calls.append((None if data is None else data.copy(), n_embd, il_start, il_end))
        self.current = None if data is None else data.copy()

    def generate(self, prompt, system, max_tokens, seed):
        self.events.append("generate")
        self.gen_calls.append((prompt, system, max_tokens, seed))
        # Modélise la réutilisation de préfixe : sans reset, la sortie dépend des K/V
        # calculés au premier appel.
        if self.cached is None:
            self.cached = self.current
        tag = "neutre" if self.cached is None else f"v{float(self.cached.sum()):g}"
        return f"out:{prompt}:{tag}"


def make_vector(layers=3):
    return np.arange(layers * N_EMBD, dtype=np.float32).reshape(layers, N_EMBD) + 1


def zeros(layers):
    return np.zeros((layers, N_EMBD), dtype=np.float32)


def test_apply_dose_exact():
    v = make_vector()
    out = apply_dose(v, 0.5, 4.0)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, v * 2.0)


@pytest.mark.parametrize("dose", [-1.0, 1.0, 0.0])
def test_apply_dose_bornes(dose):
    np.testing.assert_array_equal(apply_dose(make_vector(), dose, 3.0), make_vector() * dose * 3.0)


@pytest.mark.parametrize("dose", [1.01, -1.5, float("nan")])
def test_apply_dose_hors_bornes(dose):
    with pytest.raises(ValueError, match="dose hors de"):
        apply_dose(make_vector(), dose, 1.0)


def test_steerer_envoie_tampon_plat_dose_fois_echelle():
    b = FakeBackend()
    v = make_vector(3)
    res = Steerer(b, v, max_scale=2.0).generate("salut", 0.5, seed=7, max_tokens=10)
    data, n_embd, start, end = b.cvec_calls[-1]
    np.testing.assert_array_equal(data, v.reshape(-1))
    assert data.shape == (3 * N_EMBD,)
    assert (n_embd, start, end) == (N_EMBD, 1, 3)
    assert b.gen_calls[-1] == ("salut", None, 10, 7)
    assert (res.arm, res.dose, res.fallback) == ("steering", 0.5, False)
    assert res.text.startswith("out:salut")


def test_dose_negative():
    b = FakeBackend()
    v = make_vector(2)
    Steerer(b, v, max_scale=1.0).generate("x", -1.0)
    np.testing.assert_array_equal(b.cvec_calls[-1][0], -v.reshape(-1))


def test_dose_zero_retire_le_vecteur():
    b = FakeBackend()
    res = Steerer(b, make_vector()).generate("x", 0.0)
    assert b.cvec_calls == [(None, N_EMBD, 0, 0)]
    assert res.arm == "steering" and not res.fallback


def test_dose_hors_bornes_steerer():
    b = FakeBackend()
    with pytest.raises(ValueError, match="dose hors de"):
        Steerer(b, make_vector()).generate("x", 2.0)
    assert b.cvec_calls == [] and b.gen_calls == []


def test_repli_sans_vecteur():
    b = FakeBackend()
    res = Steerer(b, None).generate("x", 0.8)
    assert res.fallback is True and res.arm == "prompt_joyful"
    assert b.gen_calls[-1][1] == JOYFUL_SYSTEM_PROMPT
    assert b.cvec_calls == [(None, N_EMBD, 0, 0)]


def test_repli_negatif_sans_vecteur():
    b = FakeBackend()
    res = Steerer(b, None).generate("x", -0.5, system="Consigne.")
    assert res.fallback is True and res.arm == "prompt_sad"
    assert b.gen_calls[-1][1] == SAD_SYSTEM_PROMPT + "\n\nConsigne."
    assert b.cvec_calls == [(None, N_EMBD, 0, 0)]


@pytest.mark.parametrize("system", [None, "Consigne."])
def test_dose_zero_sans_vecteur_sans_repli(system):
    b = FakeBackend()
    res = Steerer(b, None).generate("x", 0.0, system=system)
    assert res.fallback is False and res.arm == "steering"
    assert b.gen_calls[-1][1] == system
    assert b.cvec_calls == [(None, N_EMBD, 0, 0)]


def test_un_seul_chargement_pour_deux_doses():
    FakeBackend.loads = 0
    b = FakeBackend()
    s = Steerer(b, make_vector())
    s.generate("a", 0.2)
    s.generate("b", 0.9)
    assert FakeBackend.loads == 1
    assert len(b.cvec_calls) == 2 and len(b.gen_calls) == 2
    assert not np.array_equal(b.cvec_calls[0][0], b.cvec_calls[1][0])


@pytest.mark.parametrize(
    ("vector", "kwargs", "msg"),
    [
        (np.zeros(N_EMBD, dtype=np.float32), {}, "forme"),
        (np.zeros((2, N_EMBD + 1), dtype=np.float32), {}, "n_embd"),
        (zeros(2), {"il_start": 0}, "plage"),
        (zeros(2), {"il_start": N_LAYER - 1}, "plage"),
        (zeros(2), {"il_end": N_LAYER}, "plage"),
        (zeros(2), {"il_start": 3, "il_end": 2}, "plage"),
        (zeros(2), {"il_end": 4}, "couches fournies"),
    ],
)
def test_validation(vector, kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        Steerer(FakeBackend(), vector, **kwargs)


def test_plage_maximale_acceptee():
    assert Steerer(FakeBackend(), zeros(N_LAYER - 1)).has_vector


def test_reset_avant_chaque_generation():
    b = FakeBackend()
    s = Steerer(b, make_vector())
    s.generate("x", 0.5)
    s.generate("x", 0.0)
    assert b.events == ["reset", "set_cvec", "generate"] * 2


def test_dose_zero_apres_dose_a_egale_neutre_neuf():
    neutre = Steerer(FakeBackend(), make_vector()).generate("x", 0.0).text
    s = Steerer(FakeBackend(), make_vector())
    assert s.generate("x", 1.0).text != neutre
    assert s.generate("x", 0.0).text == neutre


def test_repli_compose_la_consigne_appelant():
    b = FakeBackend()
    Steerer(b, None).generate("x", 0.5, system="Réponds de façon neutre.")
    assert b.gen_calls[-1][1] == JOYFUL_SYSTEM_PROMPT + "\n\nRéponds de façon neutre."


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_vecteur_non_fini_rejete(bad):
    v = zeros(2)
    v[1, 2] = bad
    with pytest.raises(ValueError, match="non finies"):
        Steerer(FakeBackend(), v)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_max_scale_invalide(scale):
    with pytest.raises(ValueError, match="max_scale"):
        Steerer(FakeBackend(), make_vector(), max_scale=scale)
