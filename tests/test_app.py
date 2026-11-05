"""Tests du serveur de démo : backends simulés, aucun modèle chargé."""

import json
import subprocess
import sys
import threading
import time

import anyio
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mood_steering.steering import JOYFUL_SYSTEM_PROMPT, SAD_SYSTEM_PROMPT, Steerer
from mood_steering_demo.app import (
    ARM_LABELS,
    ARMS,
    BUSY_MESSAGE,
    MAX_PROMPT_CHARS,
    MAX_TOKENS_LIMIT,
    create_app,
)

N_EMBD, N_LAYER = 4, 6


class StreamBackend:
    n_embd = N_EMBD
    n_layer = N_LAYER

    def __init__(self):
        self.cvec = None
        self.systems = []
        self.resets = 0
        self.fail = False
        self.texts = None

    def reset(self):
        self.resets += 1

    def set_cvec(self, data, n_embd, il_start, il_end):
        self.cvec = None if data is None else data.copy()
        self.cvec_range = (il_start, il_end)

    def generate(self, prompt, system, max_tokens, seed):
        return "".join(self.stream(prompt, system, max_tokens, seed))

    def stream(self, prompt, system, max_tokens, seed):
        self.systems.append(system)
        if self.texts is not None:
            yield from self.texts
            return
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


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_base_exception_signalee_au_bras():
    client, backends = make(vec())

    def interrupted(*a):
        raise SystemExit(3)
        yield

    backends[2].stream = interrupted
    evs = events(client)
    err = [e for e in evs if e["type"] == "error"]
    assert err == [{"type": "error", "arm": "steering", "message": "interrompu (SystemExit)"}]
    assert evs[-1] == {"type": "done"}
    assert events(client)[-1] == {"type": "done"}  # verrou rendu


@pytest.mark.parametrize(
    "params",
    [{"max_tokens": MAX_TOKENS_LIMIT + 1}, {"prompt": "a" * (MAX_PROMPT_CHARS + 1)}],
)
def test_bornes_max_422(params):
    client, _ = make(vec())
    r = client.get("/stream", params={"prompt": "a", **params})
    assert r.status_code == 422


def test_bornes_max_acceptees():
    client, _ = make(vec())
    evs = events(client, prompt="a" * MAX_PROMPT_CHARS, max_tokens=MAX_TOKENS_LIMIT)
    assert evs[-1] == {"type": "done"}
    cfg = client.get("/config").json()
    assert (cfg["max_tokens_limit"], cfg["max_prompt_chars"]) == (512, 2000)
    assert "mmap" in cfg["memory_note"]


def test_separateurs_unicode_echappes():
    client, backends = make(vec())
    for b in backends:
        b.texts = ["a b", "c d", "e\u0085f"]
    with client.stream("GET", "/stream", params={"prompt": "x"}) as r:
        raw = b"".join(r.iter_bytes()).decode("utf-8")
    assert not any(c in raw for c in "  \u0085")
    evs = [json.loads(ln[6:]) for ln in raw.split("\n") if ln]
    toks = [e["text"] for e in evs if e["type"] == "token" and e["arm"] == "neutral"]
    assert toks == ["a b", "c d", "e\u0085f"]


class SlowBackend(StreamBackend):
    """Produit des tokens sans fin (jusqu'à max_tokens), chacun après une courte pause."""

    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.produced = 0
        self.closed = False
        self.gate = None  # Event : bloque les tokens tant qu'il n'est pas levé

    def stream(self, prompt, system, max_tokens, seed):
        self.started.set()
        try:
            for i in range(max_tokens):
                if self.gate is not None:
                    self.gate.wait(10)
                time.sleep(0.01)
                self.produced += 1
                yield f"t{i}"
        finally:
            self.closed = True


def make_slow():
    backends = []

    def factory():
        b = SlowBackend()
        backends.append(b)
        return b

    return create_app(factory, vector=vec()), backends


def test_seconde_generation_409():
    app, backends = make_slow()
    gate = threading.Event()  # la 1re génération reste en cours jusqu'au contrôle du 409
    for b in backends:
        b.gate = gate
    first = {}

    def run_first():
        first["r"] = TestClient(app).get("/stream", params={"prompt": "a", "max_tokens": 60})

    t = threading.Thread(target=run_first)
    t.start()
    assert all(b.started.wait(5) for b in backends)
    r = TestClient(app).get("/stream", params={"prompt": "b"})
    assert r.status_code == 409
    assert r.json() == {"detail": BUSY_MESSAGE}
    gate.set()
    t.join(10)
    assert first["r"].status_code == 200
    assert first["r"].text.rstrip().endswith('data: {"type": "done"}')
    # la génération finie, le serveur accepte de nouveau
    assert (
        TestClient(app).get("/stream", params={"prompt": "c", "max_tokens": 2}).status_code == 200
    )


async def _asgi_disconnect(app, query: str, spec_version: str, after_chunks: int):
    """Client ASGI brut : se déconnecte après ``after_chunks`` morceaux du corps."""
    got = []
    disconnected = anyio.Event()
    sent_request = False

    async def receive():
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            if disconnected.is_set():
                raise OSError("client parti")
            got.append(message["body"])
            if len(got) >= after_chunks:
                disconnected.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/stream",
        "raw_path": b"/stream",
        "query_string": query.encode(),
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 80),
        "root_path": "",
    }
    try:
        await app(scope, receive, send)
    except Exception:
        pass  # ClientDisconnect en ASGI 2.4
    return got


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_deconnexion_arrete_les_bras_et_rend_le_verrou(spec_version):
    app, backends = make_slow()
    got = anyio.run(_asgi_disconnect, app, "prompt=a&max_tokens=500", spec_version, 4)
    assert got  # le flux a bien commencé
    deadline = time.monotonic() + 5
    client = TestClient(app)
    while time.monotonic() < deadline:
        r = client.get("/stream", params={"prompt": "b", "max_tokens": 1})
        if r.status_code == 200:
            break
        time.sleep(0.05)
    assert r.status_code == 200, "verrou non rendu après la déconnexion"
    # arrêt coopératif : loin des 500 tokens demandés, générateurs du moteur fermés
    assert all(b.produced < 200 for b in backends), [b.produced for b in backends]
    assert all(b.closed for b in backends)


def test_aucun_modele_charge():
    code = (
        "import sys; import mood_steering_demo.app; "
        "sys.exit(1 if 'llama_cpp' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_plage_de_couches_avec_vecteur_complet():
    full = np.arange((N_LAYER - 1) * N_EMBD, dtype=np.float32).reshape(N_LAYER - 1, N_EMBD) + 1
    client, backends = make(full, max_scale=1.0, il_start=2, il_end=3)
    events(client, dose=1.0)
    steering = next(b for b in backends if b.cvec is not None)
    assert steering.cvec_range == (2, 3)
    np.testing.assert_array_equal(steering.cvec[:N_EMBD], np.zeros(N_EMBD))
    np.testing.assert_array_equal(steering.cvec[N_EMBD:], full[1:3].reshape(-1))
