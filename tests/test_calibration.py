"""Calibration : chemin, cache, lecture synthétique, erreurs (aucun binaire, aucun modèle)."""

from pathlib import Path

import gguf
import numpy as np
import pytest

from mood_steering import calibration as cal

N_EMBD = 8
N_LAYER = 5


def _write(path, tensors, layer_count=N_LAYER):
    path.parent.mkdir(parents=True, exist_ok=True)
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


@pytest.fixture
def pairs(tmp_path):
    p = tmp_path / "pairs.jsonl"
    p.write_text(
        '{"positive": "Super journée", "negative": "Journée morne"}\n\n'
        '{"positive": "Quel bonheur", "negative": "Quel chagrin"}\n',
        encoding="utf-8",
    )
    return p


def _fake_generator(calls):
    def gen(cmd):
        calls.append(list(cmd))
        out = cmd[cmd.index("-o") + 1]
        _write(Path(out), {"direction.1": np.ones(N_EMBD, np.float32)})

    return gen


def test_vector_path(tmp_path):
    expected = tmp_path / "qwen" / "joy_sadness.gguf"
    assert cal.vector_path("qwen", "joy_sadness", tmp_path) == expected
    assert cal.vector_path("m", "a").as_posix() == "data/moods/m/a.gguf"


@pytest.mark.parametrize("bad", ["../x", "a/b", "a\\b", "..", "", "a..b", "/abs", "C:x"])
def test_vector_path_rejects_unsafe_names(bad):
    with pytest.raises(ValueError, match="model_id invalide"):
        cal.vector_path(bad, "joy_sadness")
    with pytest.raises(ValueError, match="axis invalide"):
        cal.vector_path("m", bad)


def test_load_vector_shape_and_zero_fill(tmp_path):
    rows = _rows()
    p = tmp_path / "v.gguf"
    _write(p, {f"direction.{il}": v for il, v in rows.items()})
    out = cal.load_vector(p, n_embd=N_EMBD)
    assert out.dtype == np.float32
    assert out.shape == (N_LAYER, N_EMBD)
    np.testing.assert_array_equal(out[0], rows[1])
    np.testing.assert_array_equal(out[3], rows[4])
    assert not out[2].any()  # couche 3 absente
    assert not out[4].any()  # couche 5 absente


def test_load_vector_without_layer_count_uses_top_layer(tmp_path):
    p = tmp_path / "v.gguf"
    w = gguf.GGUFWriter(str(p), "controlvector")
    w.add_tensor("direction.3", np.ones(N_EMBD, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    assert cal.load_vector(p).shape == (3, N_EMBD)


def test_load_vector_inconsistent_dims(tmp_path):
    p = tmp_path / "v.gguf"
    _write(
        p,
        {
            "direction.1": np.zeros(N_EMBD, dtype=np.float32),
            "direction.2": np.zeros(N_EMBD + 1, dtype=np.float32),
        },
    )
    with pytest.raises(ValueError, match="dimensions incohérentes"):
        cal.load_vector(p)


def test_load_vector_wrong_n_embd(tmp_path):
    p = tmp_path / "v.gguf"
    _write(p, {"direction.1": np.zeros(N_EMBD, dtype=np.float32)})
    with pytest.raises(ValueError, match="n_embd=16"):
        cal.load_vector(p, n_embd=16)


def test_load_vector_unexpected_tensor_name(tmp_path):
    p = tmp_path / "v.gguf"
    _write(p, {"blk.1": np.zeros(N_EMBD, dtype=np.float32)})
    with pytest.raises(ValueError, match="nom de tenseur inattendu"):
        cal.load_vector(p)


def test_load_vector_layer_beyond_count(tmp_path):
    p = tmp_path / "v.gguf"
    _write(p, {"direction.6": np.zeros(N_EMBD, dtype=np.float32)})
    with pytest.raises(ValueError, match="au-delà de layer_count=5"):
        cal.load_vector(p)


def test_calibrate_generates_then_uses_cache(tmp_path, pairs):
    calls = []
    gen = _fake_generator(calls)
    path = cal.calibrate(
        "m.gguf", "qwen", pairs_path=pairs, generator=gen, generator_exe="gen.exe", root=tmp_path
    )
    assert path == tmp_path / "qwen" / "joy_sadness.gguf"
    assert cal.load_vector(path).shape == (N_LAYER, N_EMBD)
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "gen.exe"
    assert cmd[cmd.index("-m") + 1] == "m.gguf"
    assert cmd[cmd.index("--method") + 1] == "mean"
    # Second appel : cache relu, générateur non relancé.
    again = cal.calibrate("m.gguf", "qwen", pairs_path=pairs, generator=gen, root=tmp_path)
    assert again == path
    assert len(calls) == 1


def test_calibrate_writes_pair_files(tmp_path, pairs):
    seen = {}

    def gen(cmd):
        seen["pos"] = Path(cmd[cmd.index("--positive-file") + 1]).read_text(encoding="utf-8")
        seen["neg"] = Path(cmd[cmd.index("--negative-file") + 1]).read_text(encoding="utf-8")
        _fake_generator([])(cmd)

    cal.calibrate("m.gguf", "q", pairs_path=pairs, generator=gen, root=tmp_path)
    assert seen["pos"] == "Super journée\nQuel bonheur\n"
    assert seen["neg"] == "Journée morne\nQuel chagrin\n"


def test_calibrate_force_reruns_and_overwrites(tmp_path, pairs):
    calls = []
    gen = _fake_generator(calls)
    path = cal.calibrate("m", "q", pairs_path=pairs, generator=gen, root=tmp_path)
    cal.calibrate(
        "m", "q", pairs_path=pairs, generator=gen, force=True, method="pca", root=tmp_path
    )
    assert len(calls) == 2
    assert calls[1][calls[1].index("--method") + 1] == "pca"
    assert path.exists()


def test_calibrate_validates_ids_before_running(pairs):
    calls = []
    with pytest.raises(ValueError, match="axis invalide"):
        cal.calibrate("m", "ok", "../evil", pairs_path=pairs, generator=_fake_generator(calls))
    assert calls == []


def test_calibrate_without_binary(tmp_path, pairs, monkeypatch):
    monkeypatch.delenv(cal.GENERATOR_ENV, raising=False)
    with pytest.raises(RuntimeError, match="MOOD_CVECTOR_GENERATOR"):
        cal.calibrate("m", "q", pairs_path=pairs, root=tmp_path)


def test_calibrate_generator_without_output(tmp_path, pairs):
    with pytest.raises(RuntimeError, match="n'a pas produit"):
        cal.calibrate("m", "q", pairs_path=pairs, generator=lambda cmd: None, root=tmp_path)
    assert not (tmp_path / "q" / "joy_sadness.gguf").exists()


def test_calibrate_invalid_pairs(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"positive": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        cal.calibrate("m", "q", pairs_path=bad, generator=lambda c: None, root=tmp_path)
