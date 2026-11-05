"""Tests du serveur de démo : backends simulés, aucun modèle chargé."""

import json
import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mood_steering.steering import JOYFUL_SYSTEM_PROMPT, SAD_SYSTEM_PROMPT, Steerer
from mood_steering_demo.app import ARM_LABELS, ARMS, create_app

N_EMBD, N_LAYER = 4, 6


class StreamBackend:
    n_embd = N_EMBD
    n_layer = N_LAYER

    def __init__(self):
        self.cvec = None
        self.systems = []
        self.resets = 0
        self.fail = False

    def reset(self):
        self.resets += 1

    def set_cvec(self, data, n_embd, il_start, il_end):
        self.cvec = None if data is None else data.copy()

    def generate(self, prompt, system, max_tokens, seed):
        return "".join(self.stream(prompt, system, max_tokens, seed))

    def stream(self, prompt, system, max_tokens, seed):
        self.systems.append(system)
        tag = "v" if self.cvec is not None else "n"
        for i in range(min(max_tokens, 3)):
            if self.fail and i == 1:
                raise RuntimeError("boum")
            yield f"{tag}{i}"


def make(vector=None, **kw):
    backends = []

    def factory():
        b = StreamBackend()
        backends.append(b)
        return b

    return TestClient(create_app(factory, vector=vector, **kw)), backends


def events(client, **params):
    params = {"prompt": "Bonjour", "dose": 0.5, **params}
    with client.stream("GET", "/stream", params=params) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in r.iter_lines() if ln]
    assert all(ln.startswith("data: ") for ln in lines)
    return [json.loads(ln[6:]) for ln in lines]


def vec():
    return np.ones((N_LAYER - 1, N_EMBD), dtype=np.float32)


def test_trois_bras_etiquetes_et_ordre():
    client, backends = make(vec())
    assert len(backends) == 3
    evs = events(client)
    assert evs[-1] == {"type": "done"}
    assert [e["type"] for e in evs].count("done") == 1
    for arm in ARMS:
        mine = [e for e in evs if e.get("arm") == arm]
        assert [e["type"] for e in mine] == ["start", "token", "token", "token", "end"]
        assert mine[0]["label"] == ARM_LABELS[arm]
        assert mine[0]["fallback"] is False
    starts = {e["arm"]: e["dose"] for e in evs if e["type"] == "start"}
    assert starts == {"neutral": 0.0, "prompt_joyful": 0.0, "steering": 0.5}
    texts = {
        a: "".join(e["text"] for e in evs if e.get("arm") == a and e["type"] == "token")
        for a in ARMS
    }
    assert texts == {"neutral": "n0n1n2", "prompt_joyful": "n0n1n2", "steering": "v0v1v2"}
    assert [b.systems for b in backends] == [[None], [JOYFUL_SYSTEM_PROMPT], [None]]
    assert all(b.resets == 1 for b in backends)


def test_repli_signale_sans_vecteur():
    client, backends = make(None)
    evs = events(client, dose=-0.5)
    start = next(e for e in evs if e["type"] == "start" and e["arm"] == "steering")
    assert start["fallback"] is True
    assert backends[2].systems == [SAD_SYSTEM_PROMPT]
    others = [e for e in evs if e["type"] == "start" and e["arm"] != "steering"]
    assert all(e["fallback"] is False for e in others)


def test_dose_zero_sans_vecteur_pas_de_repli():
    client, _ = make(None)
    evs = events(client, dose=0)
    start = next(e for e in evs if e["type"] == "start" and e["arm"] == "steering")
    assert start["fallback"] is False


def test_erreur_isolee_a_un_bras():
    client, backends = make(vec())
    backends[2].fail = True
    evs = events(client)
    err = [e for e in evs if e["type"] == "error"]
    assert err == [{"type": "error", "arm": "steering", "message": "boum"}]
    assert not [e for e in evs if e["type"] == "end" and e["arm"] == "steering"]
    assert [e for e in evs if e["type"] == "end"] and evs[-1] == {"type": "done"}


@pytest.mark.parametrize(
    "params",
    [{"dose": 1.5}, {"dose": -1.01}, {"prompt": "  "}, {"max_tokens": 0}, {"dose": "x"}],
)
def test_422(params):
    client, _ = make(vec())
    r = client.get("/stream", params={"prompt": "a", "dose": 0, **params})
    assert r.status_code == 422


def test_prompt_absent_422():
    client, _ = make(vec())
    assert client.get("/stream").status_code == 422


def test_bornes_acceptees():
    client, _ = make(vec())
    assert events(client, dose=1)[-1] == {"type": "done"}
    assert events(client, dose=-1)[-1] == {"type": "done"}


def test_health_et_config():
    client, _ = make(vec(), max_scale=2.5, model="m.gguf")
    assert client.get("/health").json() == {"status": "ok"}
    cfg = client.get("/config").json()
    assert cfg["model"] == "m.gguf"
    assert (cfg["dose_min"], cfg["dose_max"], cfg["max_scale"]) == (-1.0, 1.0, 2.5)
    assert cfg["has_vector"] is True
    assert cfg["arms"] == ["neutral", "prompt_joyful", "steering"]
    assert cfg["labels"] == ARM_LABELS
    assert cfg["register"] == "expression d'humeur"
    assert make(None)[0].get("/config").json()["has_vector"] is False


def test_steerer_stream_et_open_stream():
    b = StreamBackend()
    s = Steerer(b)
    st = s.open_stream("q", 0.7, system="x")
    assert (st.arm, st.fallback, st.dose) == ("prompt_joyful", True, 0.7)
    assert b.resets == 1 and b.systems == []  # préparé, aucun token encore produit
    assert list(st) == ["n0", "n1", "n2"]
    assert b.systems == [f"{JOYFUL_SYSTEM_PROMPT}\n\nx"]
    assert list(s.stream("q", 0)) == ["n0", "n1", "n2"]
    assert b.resets == 2
    with pytest.raises(ValueError):
        s.stream("q", 2.0)


def test_aucun_modele_charge():
    code = (
        "import sys; import mood_steering_demo.app; "
        "sys.exit(1 if 'llama_cpp' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
