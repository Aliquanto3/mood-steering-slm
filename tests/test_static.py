"""Tests de l'interface statique : servie par l'app, sans navigateur."""

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
    assert "EventSource" in js.text and "drawDoseCurve" in js.text
    assert c.get("/style.css").status_code == 200


def test_curseur_et_accessibilite():
    html = client().get("/").text
    assert re.search(r'type="range" min="-1" max="1" step="0.05"', html)
    assert html.count('aria-live="polite"') >= 4
    assert 'for="prompt"' in html and 'for="dose"' in html


def test_aucun_terme_de_ressenti():
    fichiers = list(STATIC_DIR.iterdir())
    assert {p.name for p in fichiers} >= {"index.html", "app.js", "style.css"}
    for path in fichiers:
        text = path.read_text(encoding="utf-8").lower()
        for word in FORBIDDEN:
            assert word not in text, f"{word!r} dans {path.name}"
