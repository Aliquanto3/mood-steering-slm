"""Lecture d'un GGUF de vecteur de contrôle écrit avec le nommage de llama-cvector-generator."""

import importlib.util
import sys
from pathlib import Path

import gguf
import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "read_cvector_gguf", Path(__file__).parent.parent / "spikes" / "read_cvector_gguf.py"
)
rcg = importlib.util.module_from_spec(_SPEC)
sys.modules["read_cvector_gguf"] = rcg
_SPEC.loader.exec_module(rcg)

N_EMBD = 8
N_LAYER = 5


def _write(path, tensors: dict[str, np.ndarray], layer_count=N_LAYER):
    w = gguf.GGUFWriter(str(path), "controlvector")
    w.add_string("controlvector.model_hint", "qwen35")
    w.add_int32("controlvector.layer_count", layer_count)
    for name, data in tensors.items():
        w.add_tensor(name, data)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _rows():
    rng = np.random.default_rng(0)
    return {il: rng.standard_normal(N_EMBD).astype(np.float32) for il in (1, 2, 4)}


def test_read_floats_and_metadata(tmp_path):
    rows = _rows()
    _write(tmp_path / "v.gguf", {f"direction.{il}": v for il, v in rows.items()})
    cv = rcg.read_cvector(tmp_path / "v.gguf")
    assert cv.n_embd == N_EMBD
    assert cv.layer_count == N_LAYER
    assert cv.model_hint == "qwen35"
    assert sorted(cv.directions) == [1, 2, 4]
    for il, v in rows.items():
        assert cv.directions[il].dtype == np.float32
        np.testing.assert_array_equal(cv.directions[il], v)
    assert cv.missing_layers == [3, 5]


def test_flat_buffer_layout_layer1_first_missing_zero(tmp_path):
    rows = _rows()
    _write(tmp_path / "v.gguf", {f"direction.{il}": v for il, v in rows.items()})
    flat = rcg.read_cvector(tmp_path / "v.gguf").flat(N_LAYER)
    assert flat.shape == (N_LAYER * N_EMBD,)
    np.testing.assert_array_equal(flat[:N_EMBD], rows[1])
    np.testing.assert_array_equal(flat[N_EMBD : 2 * N_EMBD], rows[2])
    assert not flat[2 * N_EMBD : 3 * N_EMBD].any()
    np.testing.assert_array_equal(flat[3 * N_EMBD : 4 * N_EMBD], rows[4])
    assert not flat[4 * N_EMBD :].any()


def test_real_layout_last_layer_absent(tmp_path):
    # Cas réel du générateur : direction.1..n_layer-1, layer_count = n_layer - 1.
    rng = np.random.default_rng(1)
    rows = {il: rng.standard_normal(N_EMBD).astype(np.float32) for il in range(1, N_LAYER)}
    _write(tmp_path / "v.gguf", {f"direction.{il}": v for il, v in rows.items()}, N_LAYER - 1)
    cv = rcg.read_cvector(tmp_path / "v.gguf")
    assert cv.layer_count == N_LAYER - 1
    assert cv.missing_layers == []
    flat = cv.flat(N_LAYER)
    np.testing.assert_array_equal(flat[(N_LAYER - 2) * N_EMBD : (N_LAYER - 1) * N_EMBD], rows[4])
    assert not flat[(N_LAYER - 1) * N_EMBD :].any()


def test_rejects_direction_beyond_layer_count(tmp_path):
    _write(tmp_path / "v.gguf", {"direction.3": _rows()[1]}, layer_count=2)
    with pytest.raises(ValueError, match="au-delà de layer_count"):
        rcg.read_cvector(tmp_path / "v.gguf")


def test_flat_rejects_too_few_layers(tmp_path):
    _write(tmp_path / "v.gguf", {"direction.4": _rows()[4]})
    with pytest.raises(ValueError, match="au-delà de 3 couches"):
        rcg.read_cvector(tmp_path / "v.gguf").flat(3)


@pytest.mark.parametrize(
    ("name", "data", "msg"),
    [
        ("direction.0", np.zeros(N_EMBD, np.float32), "nom de tenseur invalide"),
        ("blk.1.weight", np.zeros(N_EMBD, np.float32), "nom de tenseur invalide"),
        ("direction.1", np.zeros(N_EMBD, np.float16), "type non F32"),
        ("direction.1", np.zeros((2, N_EMBD), np.float32), "forme non 1D"),
    ],
)
def test_rejects_what_cli_rejects(tmp_path, name, data, msg):
    _write(tmp_path / "v.gguf", {name: data})
    with pytest.raises(ValueError, match=msg):
        rcg.read_cvector(tmp_path / "v.gguf")


def test_rejects_mixed_dimensions(tmp_path):
    _write(
        tmp_path / "v.gguf",
        {"direction.1": np.zeros(4, np.float32), "direction.2": np.zeros(6, np.float32)},
    )
    with pytest.raises(ValueError, match="dimensions"):
        rcg.read_cvector(tmp_path / "v.gguf")
