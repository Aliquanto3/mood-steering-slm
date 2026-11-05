"""Boucle bas niveau sur un vrai GGUF (extra ``llama`` et modèle local, sinon ignoré).

Modèle : ``MOOD_STEERING_TEST_GGUF`` ou ``models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf`` ; vecteur
facultatif : ``MOOD_STEERING_TEST_CVEC`` ou ``data/moods/Qwen3.5-0.8B-Q4_K_M/joy_sadness.gguf``.
Rien n'est téléchargé.
"""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(
    os.environ.get("MOOD_STEERING_TEST_GGUF", ROOT / "models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
)
CVEC = Path(
    os.environ.get(
        "MOOD_STEERING_TEST_CVEC", ROOT / "data/moods/Qwen3.5-0.8B-Q4_K_M/joy_sadness.gguf"
    )
)


@pytest.fixture(scope="module")
def backend():
    pytest.importorskip("llama_cpp")
    if not MODEL.is_file():
        pytest.skip(f"modèle absent : {MODEL}")
    from mood_steering.steering import LlamaCppBackend

    return LlamaCppBackend(str(MODEL), n_ctx=512)


def test_vraie_boucle_quelques_tokens(backend):
    from mood_steering.schedule import constant
    from mood_steering.steering import Steerer

    st = Steerer(backend)
    ts = st.open_stream("Dis bonjour.", schedule=constant(0), max_tokens=6)
    text = "".join(ts)
    assert text and ts.doses == [0.0] * len(ts.doses)
    assert 1 <= len(ts.doses) <= 6
    assert st.generate("Dis bonjour.", 0.0, max_tokens=6).text == text  # glouton, reset


def test_vrai_vecteur_complet_plage_partielle(backend):
    if not CVEC.is_file():
        pytest.skip(f"vecteur absent : {CVEC}")
    from mood_steering.calibration import load_vector
    from mood_steering.schedule import pulse
    from mood_steering.steering import Steerer

    vec = load_vector(CVEC, n_embd=backend.n_embd)
    st = Steerer(backend, vec, max_scale=0.15, il_start=6, il_end=min(18, backend.n_layer - 1))
    ts = st.open_stream("Dis bonjour.", schedule=pulse(2, 1.0, 4), max_tokens=6)
    assert "".join(ts)
    assert ts.doses[:3] == [0.0, 0.5, 1.0]
