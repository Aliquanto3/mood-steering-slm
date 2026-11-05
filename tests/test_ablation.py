"""Tests de l'ablation de direction (story 6.1) : tableaux synthétiques, aucun modèle réel."""

import subprocess
import sys

import numpy as np
import pytest

from mood_steering.ablation import (
    ORTHOGONAL_ATOL,
    QWEN35_0_8B_HIDDEN_SIZE,
    ablate,
    check_alpha,
    load_direction,
)

DIM = 16


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def setup(rng):
    u = rng.normal(size=DIM).astype(np.float32)
    u /= np.linalg.norm(u)
    h = rng.normal(size=(5, 3, DIM)).astype(np.float32)
    return h, u


def _split(h, u):
    c = h @ u
    return c, h - c[..., None] * u


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_ablate_scales_component_and_keeps_rest(setup, alpha):
    h, u = setup
    out = ablate(h, u, alpha)
    c0, r0 = _split(h, u)
    c1, r1 = _split(out, u)
    np.testing.assert_allclose(c1, c0 * (1 - alpha), atol=ORTHOGONAL_ATOL)
    np.testing.assert_allclose(r1, r0, atol=ORTHOGONAL_ATOL)


def test_alpha_zero_is_identity(setup):
    h, u = setup
    np.testing.assert_array_equal(ablate(h, u, 0.0), h)


def test_unnormalized_direction_same_result(setup):
    h, u = setup
    np.testing.assert_allclose(ablate(h, 7.3 * u, 0.5), ablate(h, u, 0.5), atol=1e-6)


@pytest.mark.parametrize("bad", [-0.1, 1.01, 2.0, float("nan"), float("inf")])
def test_alpha_validation(setup, bad):
    h, u = setup
    with pytest.raises(ValueError, match=r"alpha doit être dans \[0, 1\]"):
        ablate(h, u, bad)
    with pytest.raises(ValueError):
        check_alpha(bad)


def test_alpha_bounds_accepted():
    assert check_alpha(0) == 0.0
    assert check_alpha(1) == 1.0


def test_zero_and_bad_shape_direction(setup):
    h, u = setup
    with pytest.raises(ValueError, match="nulle"):
        ablate(h, np.zeros(DIM), 0.5)
    with pytest.raises(ValueError, match="1D"):
        ablate(h, u.reshape(2, -1), 0.5)
    with pytest.raises(ValueError, match="dimension des activations"):
        ablate(h, u[:-1], 0.5)


def test_hidden_size_constant():
    assert QWEN35_0_8B_HIDDEN_SIZE == 1024


def test_load_direction_npy_ok(tmp_path, rng):
    v = rng.normal(size=QWEN35_0_8B_HIDDEN_SIZE).astype(np.float64)
    np.save(tmp_path / "d.npy", v)
    out = load_direction(tmp_path / "d.npy", expected_dim=QWEN35_0_8B_HIDDEN_SIZE)
    assert out.dtype == np.float32 and out.shape == (1024,)
    np.testing.assert_allclose(out, v, rtol=1e-6)


def test_load_direction_default_dim_is_qwen(tmp_path):
    np.save(tmp_path / "d.npy", np.ones(512))
    with pytest.raises(ValueError, match="512 != dimension cachée attendue 1024"):
        load_direction(tmp_path / "d.npy")


def test_load_direction_wrong_dim_and_ndim(tmp_path):
    np.save(tmp_path / "a.npy", np.ones(10))
    with pytest.raises(ValueError, match="10 != dimension cachée attendue 12"):
        load_direction(tmp_path / "a.npy", expected_dim=12)
    np.save(tmp_path / "b.npy", np.ones((2, 12)))
    with pytest.raises(ValueError, match="1D"):
        load_direction(tmp_path / "b.npy", expected_dim=12)


def test_load_direction_bad_format_and_missing(tmp_path):
    (tmp_path / "d.txt").write_text("x")
    with pytest.raises(ValueError, match="format non pris en charge"):
        load_direction(tmp_path / "d.txt", expected_dim=3)
    with pytest.raises(FileNotFoundError):
        load_direction(tmp_path / "absent.npy", expected_dim=3)


def test_load_direction_safetensors_and_pt(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.numpy")
    v = np.arange(8, dtype=np.float32)
    st.save_file({"direction": v}, str(tmp_path / "d.safetensors"))
    np.testing.assert_array_equal(load_direction(tmp_path / "d.safetensors", expected_dim=8), v)
    st.save_file({"a": v, "b": v}, str(tmp_path / "two.safetensors"))
    with pytest.raises(ValueError, match="préciser key"):
        load_direction(tmp_path / "two.safetensors", expected_dim=8)
    np.testing.assert_array_equal(
        load_direction(tmp_path / "two.safetensors", expected_dim=8, key="b"), v
    )
    torch.save(torch.from_numpy(v), tmp_path / "d.pt")
    np.testing.assert_array_equal(load_direction(tmp_path / "d.pt", expected_dim=8), v)
    with pytest.raises(ValueError, match="8 != dimension cachée attendue 9"):
        load_direction(tmp_path / "d.pt", expected_dim=9)


def test_base_import_does_not_load_torch():
    code = (
        "import sys, mood_steering, mood_steering.ablation\n"
        "bad = [m for m in ('torch', 'transformers') if m in sys.modules]\n"
        "sys.exit(1 if bad else 0)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# --- version torch, module factice ---


def _toy_model(torch, tuple_output):
    class Layer(torch.nn.Module):
        def forward(self, x):
            y = x * 2.0
            return (y, "cache") if tuple_output else y

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(), Layer()])

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

        def forward(self, x):
            for layer in self.model.layers:
                out = layer(x)
                x = out[0] if isinstance(out, tuple) else out
            return x

    return Toy()


@pytest.mark.parametrize("tuple_output", [False, True])
def test_direction_ablator_hooks(rng, tuple_output):
    torch = pytest.importorskip("torch")
    from mood_steering.ablation import DirectionAblator

    u = rng.normal(size=DIM).astype(np.float32)
    h = rng.normal(size=(2, 4, DIM)).astype(np.float32)
    model = _toy_model(torch, tuple_output)
    x = torch.from_numpy(h)
    base = model(x).detach().numpy()
    np.testing.assert_allclose(base, h * 4.0, rtol=1e-5)

    # Ablation en sortie de la couche 1 (la dernière) seulement.
    ab = DirectionAblator(model, 3.0 * u, alpha=1.0, layers=[1])
    np.testing.assert_allclose(model(x).detach().numpy(), ablate(h * 4.0, u, 1.0), atol=1e-4)

    ab.alpha = 0.5
    np.testing.assert_allclose(model(x).detach().numpy(), ablate(h * 4.0, u, 0.5), atol=1e-4)
    with pytest.raises(ValueError):
        ab.alpha = 1.5
    assert ab.alpha == 0.5

    ab.remove()
    np.testing.assert_allclose(model(x).detach().numpy(), base, rtol=1e-5)


def test_direction_ablator_bad_layer():
    torch = pytest.importorskip("torch")
    from mood_steering.ablation import DirectionAblator

    with pytest.raises(IndexError):
        DirectionAblator(_toy_model(torch, False), np.ones(DIM), 0.5, layers=[2])


def test_direction_ablator_per_layer_dict(rng):
    torch = pytest.importorskip("torch")
    from mood_steering.ablation import DirectionAblator

    u0 = rng.normal(size=DIM).astype(np.float32)
    u1 = rng.normal(size=DIM).astype(np.float32)
    h = rng.normal(size=(2, 4, DIM)).astype(np.float32)
    model = _toy_model(torch, True)
    ab = DirectionAblator(model, {0: u0, 1: u1}, alpha=1.0)
    assert ab.layers == [0, 1]
    # Couche 0 : ablation de u0 sur 2h ; couche 1 : x2 puis ablation de u1.
    expected = ablate(2.0 * ablate(2.0 * h, u0, 1.0), u1, 1.0)
    got = model(torch.from_numpy(h)).detach().numpy()
    np.testing.assert_allclose(got, expected, atol=1e-4)
    with pytest.raises(ValueError, match="layers doit être None"):
        DirectionAblator(model, {0: u0}, alpha=1.0, layers=[0])
    with pytest.raises(IndexError):
        DirectionAblator(model, {5: u0}, alpha=1.0)


def test_direction_ablator_bf16_projection_in_float32(rng):
    torch = pytest.importorskip("torch")
    from mood_steering.ablation import DirectionAblator

    u = rng.normal(size=DIM).astype(np.float32)
    h = rng.normal(size=(2, 4, DIM)).astype(np.float32)
    model = _toy_model(torch, False)
    DirectionAblator(model, u, alpha=1.0, layers=[1])
    x = torch.from_numpy(h).to(torch.bfloat16)
    out = model(x)
    assert out.dtype == torch.bfloat16
    # Tolérance bf16 (8 bits de mantisse, ~4e-3 relatif) sur des valeurs d'ordre 4 : 5e-2.
    ref = ablate(4.0 * x.float().numpy(), u, 1.0)
    np.testing.assert_allclose(out.float().numpy(), ref, atol=5e-2)
    un = u / np.linalg.norm(u)
    assert np.abs(out.float().numpy() @ un).max() < 5e-2


def test_direction_ablator_checks_hidden_size_from_config():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from mood_steering.ablation import DirectionAblator

    model = _toy_model(torch, False)
    model.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=DIM + 1))
    with pytest.raises(ValueError, match="hidden_size"):
        DirectionAblator(model, np.ones(DIM), 0.5)
    model.config = SimpleNamespace(hidden_size=DIM)
    DirectionAblator(model, np.ones(DIM), 0.5).remove()


def test_load_direction_safetensors_bf16(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    v = torch.linspace(-1, 1, 8).to(torch.bfloat16)
    st.save_file({"direction": v}, str(tmp_path / "bf16.safetensors"))
    got = load_direction(tmp_path / "bf16.safetensors", expected_dim=8)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, v.float().numpy())


def test_load_direction_safetensors_bf16_without_torch(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    st.save_file({"d": torch.ones(8, dtype=torch.bfloat16)}, str(tmp_path / "bf16.safetensors"))
    monkeypatch.setitem(sys.modules, "safetensors.torch", None)  # simule l'absence de torch
    with pytest.raises(ValueError, match="BF16"):
        load_direction(tmp_path / "bf16.safetensors", expected_dim=8)
