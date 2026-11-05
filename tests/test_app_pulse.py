"""Profil de dose « pulse » de la démo (story 5.1) : moteur simulé, aucun modèle chargé."""

import pytest
from fastapi.testclient import TestClient
from test_app import StreamBackend, events, make, vec

from mood_steering.steering import SAD_SYSTEM_PROMPT
from mood_steering_demo.app import MAX_TOKENS_LIMIT, create_app


class PulseBackend(StreamBackend):
    """Moteur simulé qui rappelle ``on_token_start(t)`` avant chaque token, comme llama.cpp."""

    def __init__(self):
        super().__init__()
        self.cvecs = []

    def set_cvec(self, data, n_embd, il_start, il_end):
        super().set_cvec(data, n_embd, il_start, il_end)
        self.cvecs.append(None if data is None else float(data[0]))

    def stream_scheduled(self, prompt, system, max_tokens, seed, on_token_start):
        self.systems.append(system)
        for t in range(max_tokens):
            on_token_start(t)
            yield f"t{t}"


def make_pulse(vector=None, **kw):
    backends = []

    def factory():
        b = PulseBackend()
        backends.append(b)
        return b

    return TestClient(create_app(factory, vector=vector, **kw)), backends


def steering_tokens(evs):
    return [e for e in evs if e.get("arm") == "steering" and e["type"] == "token"]


def test_pulse_envoie_la_dose_appliquee_a_chaque_token():
    client, backends = make_pulse(vec(), max_scale=1.0)
    evs = events(client, dose=1.0, profile="pulse", rise=4, half_life=2, max_tokens=8)
    start = next(e for e in evs if e["type"] == "start" and e["arm"] == "steering")
    assert start["profile"] == "pulse" and start["dose"] == 0.0 and start["fallback"] is False
    doses = [e["dose"] for e in steering_tokens(evs)]
    # montée linéaire 0 -> 1 en 4 tokens, pic au token 4, puis demi-vie de 2 tokens
    assert doses == [0.0, 0.25, 0.5, 0.75, 1.0, 0.707, 0.5, 0.354]
    # le vecteur est posé avec la dose arrondie (vecteur de 1 : première composante = dose)
    # (None : état initial ; la dose 0 du token 0 ne pose rien)
    assert backends[2].cvecs[0] is None
    assert backends[2].cvecs[1:] == pytest.approx([0.25, 0.5, 0.75, 1.0, 0.707, 0.5, 0.354])


def test_pulse_ne_touche_pas_les_autres_bras():
    client, _ = make_pulse(vec())
    evs = events(client, dose=1.0, profile="pulse", rise=2, half_life=2, max_tokens=3)
    for arm in ("neutral", "prompt_joyful"):
        mine = [e for e in evs if e.get("arm") == arm]
        assert all("dose" not in e for e in mine if e["type"] == "token")
        assert "profile" not in mine[0] and mine[0]["dose"] == 0.0
        assert [e["text"] for e in mine if e["type"] == "token"] == ["n0", "n1", "n2"]


def test_constant_garde_le_comportement_historique():
    client, _ = make(vec())
    ref = events(client)
    explicite = events(client, profile="constant", rise=3, half_life=5)
    assert ref == explicite
    assert all("dose" not in e for e in ref if e["type"] == "token")
    assert all("profile" not in e for e in ref)


def test_pulse_negatif_et_repli_sans_vecteur():
    client, backends = make_pulse(None)
    evs = events(client, dose=-0.5, profile="pulse", rise=2, half_life=2, max_tokens=4)
    start = next(e for e in evs if e["type"] == "start" and e["arm"] == "steering")
    assert start["fallback"] is True
    assert backends[2].systems == [SAD_SYSTEM_PROMPT]
    assert [e["dose"] for e in steering_tokens(evs)] == [0.0, -0.25, -0.5, -0.354]


@pytest.mark.parametrize(
    "params",
    [
        {"profile": "sinus"},
        {"profile": "pulse", "rise": -1},
        {"profile": "pulse", "rise": "nan"},
        {"profile": "pulse", "half_life": 0},
        {"profile": "pulse", "half_life": -3},
        {"profile": "pulse", "rise": MAX_TOKENS_LIMIT + 1},
        {"profile": "pulse", "dose": 1.5},
    ],
)
def test_pulse_parametres_invalides_422(params):
    client, _ = make_pulse(vec())
    r = client.get("/stream", params={"prompt": "Bonjour", "dose": 0.5, **params})
    assert r.status_code == 422


def test_pulse_rise_zero_accepte_et_config_annonce_les_profils():
    client, _ = make_pulse(vec())
    evs = events(client, dose=0.5, profile="pulse", rise=0, half_life=1, max_tokens=2)
    assert [e["dose"] for e in steering_tokens(evs)] == [0.5, 0.25]
    cfg = client.get("/config").json()
    assert cfg["profiles"] == ["constant", "pulse"]
    assert cfg["default_rise"] == 24 and cfg["default_half_life"] == 32.0
