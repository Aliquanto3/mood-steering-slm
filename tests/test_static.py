"""Tests de l'interface statique : servie par l'app, sans navigateur."""

import json
import re

import numpy as np
from fastapi.testclient import TestClient
from test_app import StreamBackend

from mood_steering_demo.app import STATIC_DIR, create_app

FORBIDDEN = [
    "ressent",
    "émotion ressentie",
    "emotion ressentie",
    "sentiment",
    "éprouv",
    "eprouv",
    "conscience",
    "souffr",
    "se sent",
    "je me sens",
]


def client():
    return TestClient(create_app(StreamBackend, vector=np.ones((5, 4), dtype=np.float32)))


def test_index_et_app_js_servis():
    c = client()
    r = c.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert 'lang="fr"' in r.text and 'src="app.js"' in r.text
    assert "expression d'humeur" in r.text.lower()
    js = c.get("/app.js")
    assert js.status_code == 200
    # Flux lu par fetch (code HTTP connu, pas de reconnexion automatique d'EventSource).
    assert "getReader" in js.text and "AbortController" in js.text
    assert "new EventSource" not in js.text and "drawDoseCurve" in js.text
    assert c.get("/style.css").status_code == 200


def test_curseur_et_accessibilite():
    html = client().get("/").text
    assert re.search(r'type="range" min="-1" max="1" step="0.05"', html)
    # Une seule zone annoncée (statut) : pas d'annonce à chaque token dans les colonnes.
    assert html.count("aria-live=") == 1 and 'id="status" role="status"' in html
    for arm in ("neutral", "prompt_joyful", "steering"):
        assert f'role="region" aria-labelledby="title-{arm}"' in html
    assert 'for="prompt"' in html and 'for="dose"' in html


def test_aucun_terme_de_ressenti():
    fichiers = list(STATIC_DIR.iterdir())
    assert {p.name for p in fichiers} >= {"index.html", "app.js", "style.css"}
    for path in fichiers:
        text = path.read_text(encoding="utf-8").lower()
        for word in FORBIDDEN:
            assert word not in text, f"{word!r} dans {path.name}"


def test_trames_sse_au_format_du_parseur_js():
    """Le parseur d'app.js découpe sur une ligne vide et lit les lignes « data: {json} »."""
    r = client().get("/stream", params={"prompt": "bonjour", "dose": 0.5})
    assert r.status_code == 200
    body = r.text
    assert body.endswith("\n\n") and "\r" not in body
    frames = body[:-2].split("\n\n")
    events = []
    for frame in frames:
        lines = frame.split("\n")
        assert len(lines) == 1 and lines[0].startswith("data: "), frame
        events.append(json.loads(lines[0][len("data: ") :]))
    assert events[-1] == {"type": "done"}
    assert {e["type"] for e in events} <= {"start", "token", "end", "error", "done"}
    steering = [e for e in events if e.get("arm") == "steering"]
    assert steering[0]["type"] == "start" and steering[0]["dose"] == 0.5
    assert any(e["type"] == "token" for e in steering)
