"""Ablation de direction à l'inférence (pile C, lot anti-boucle, SPEC CAP-9).

Retire à la dose ``alpha`` la composante des activations sur une direction :
``h - alpha * (h . u) u`` avec ``u`` la direction normalisée. À ``alpha = 1`` la composante est
supprimée, à ``0`` rien ne change. La direction est reprise d'ailleurs (``slm-weight-editing``,
lot 1, ou diff du MIT) : elle n'est pas recherchée ici.

`import mood_steering.ablation` n'importe ni torch ni transformers : ils ne sont chargés que par
``DirectionAblator`` et par la lecture des formats ``.pt`` / ``.safetensors`` (extra ``ablation``).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

# Dimension cachée de Qwen3.5-0.8B : ``text_config.hidden_size`` du ``config.json`` du modèle
# (RikiyaT/Qwen3.5-0.8B-weight-space-programming, copie locale de slm-weight-editing
# ``models/hf/qwen3.5-base/config.json``) : 1024.
QWEN35_0_8B_HIDDEN_SIZE = 1024

# Tolérance numérique documentée : la composante orthogonale est inchangée à cette précision
# absolue près (float32, activations d'ordre 1) ; les tests s'y réfèrent.
ORTHOGONAL_ATOL = 1e-5

SUPPORTED_SUFFIXES = (".npy", ".pt", ".safetensors")


def _unit(direction: np.ndarray) -> np.ndarray:
    d = np.asarray(direction)
    if d.ndim != 1:
        raise ValueError(f"la direction doit être un vecteur 1D, forme reçue {d.shape}")
    norm = float(np.linalg.norm(d.astype(np.float64)))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("la direction est nulle ou non finie : normalisation impossible")
    return d / norm


def check_alpha(alpha: float) -> float:
    """Valide la dose : ``0 <= alpha <= 1``. Une dose > 1 (sur-correction, inversion de la
    composante) est refusée, tout comme une valeur non finie."""
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 <= a <= 1.0:
        raise ValueError(f"alpha doit être dans [0, 1], reçu {alpha!r}")
    return a


def ablate(activations: np.ndarray, direction: np.ndarray, alpha: float) -> np.ndarray:
    """Retourne ``h - alpha * (h . u) u`` ; ``activations`` de forme ``(..., dim)``.

    La composante sur ``u`` est multipliée par ``(1 - alpha)`` ; la composante orthogonale est
    inchangée à ``ORTHOGONAL_ATOL`` près. ``direction`` n'a pas besoin d'être normalisée.
    """
    alpha = check_alpha(alpha)
    h = np.asarray(activations)
    u = _unit(direction)
    if h.shape[-1] != u.shape[0]:
        raise ValueError(f"dimension des activations {h.shape[-1]} != direction {u.shape[0]}")
    coeff = h @ u
    return h - alpha * coeff[..., None] * u


def load_direction(
    path: str | Path, *, expected_dim: int = QWEN35_0_8B_HIDDEN_SIZE, key: str | None = None
) -> np.ndarray:
    """Charge une direction 1D (float32) et vérifie sa dimension contre ``expected_dim``.

    Formats : ``.npy`` (numpy), ``.pt`` (torch, ``weights_only``) et ``.safetensors`` ; les deux
    derniers exigent l'extra ``ablation`` et un seul tenseur, sinon ``key`` nomme le tenseur.

    Direction anti-boucle (état au 2026-11-05) : ``slm-weight-editing`` ne stocke AUCUN fichier de
    direction. Elle se déduit du diff du MIT (``models/hf/qwen3.5-wsp`` contre
    ``models/hf/qwen3.5-base``, lot 0) : seuls
    ``model.language_model.layers.15.mlp.down_proj.weight`` et ``...layers.19...`` diffèrent
    (forme 1024 x 3584, 64 colonnes touchées) ; chaque delta est de rang 1 (1re valeur singulière
    1,75 et 1,40, contre ~1e-3 pour les suivantes). La direction est son 1er vecteur singulier
    gauche, de dimension 1024 = hidden_size de Qwen3.5-0.8B (vérifié). À exporter en ``.npy``
    hors du dépôt (jamais versionnée ici : AGENTS.md).
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"format non pris en charge {suffix!r} ; attendus : {SUPPORTED_SUFFIXES}")
    if not p.is_file():
        raise FileNotFoundError(p)
    if suffix == ".npy":
        arr = np.load(p, allow_pickle=False)
    elif suffix == ".safetensors":
        from safetensors.numpy import load_file

        arr = _pick(load_file(str(p)), key, p)
    else:
        import torch

        obj = torch.load(p, map_location="cpu", weights_only=True)
        arr = _pick(obj, key, p) if isinstance(obj, dict) else obj
        arr = arr.detach().float().numpy() if hasattr(arr, "numpy") else np.asarray(arr)
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"direction attendue 1D, forme reçue {arr.shape} dans {p.name}")
    if arr.shape[0] != expected_dim:
        raise ValueError(
            f"dimension de la direction {arr.shape[0]} != dimension cachée attendue "
            f"{expected_dim} ({p.name})"
        )
    return arr.astype(np.float32)


def _pick(tensors: dict[str, Any], key: str | None, path: Path) -> Any:
    if key is not None:
        if key not in tensors:
            raise KeyError(f"tenseur {key!r} absent de {path.name} : {sorted(tensors)}")
        return tensors[key]
    if len(tensors) != 1:
        raise ValueError(f"{path.name} contient {len(tensors)} tenseurs : préciser key=")
    return next(iter(tensors.values()))


class DirectionAblator:
    """Pose des hooks ``forward`` qui ablatent ``direction`` en sortie des couches de décodeur.

    ``model`` : modèle `transformers` (ou module factice) ; les couches sont lues dans
    ``model.model.layers`` (repli ``model.model.language_model.layers``). ``layers`` : indices à
    traiter (``None`` = toutes). ``alpha`` est modifiable sans recharger (``ablator.alpha = 0.5``) ;
    ``remove()`` retire les hooks. Les sorties en tuple des couches HF sont gérées : seul le
    premier élément (états cachés) est modifié.
    """

    def __init__(
        self, model: Any, direction: np.ndarray, alpha: float, layers: Iterable[int] | None = None
    ) -> None:
        import torch

        self._alpha = check_alpha(alpha)
        self._unit = torch.as_tensor(_unit(np.asarray(direction, dtype=np.float32)))
        stack = _find_layers(model)
        idx = list(range(len(stack))) if layers is None else list(layers)
        for i in idx:
            if not 0 <= i < len(stack):
                raise IndexError(f"couche {i} hors de [0, {len(stack)})")
        self.layers = idx
        self._handles = [stack[i].register_forward_hook(self._hook) for i in idx]

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._alpha = check_alpha(value)

    def _ablate(self, h: Any) -> Any:
        u = self._unit.to(device=h.device, dtype=h.dtype)
        return h - self._alpha * (h @ u).unsqueeze(-1) * u

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        if self._alpha == 0.0:
            return output
        if isinstance(output, tuple):
            return (self._ablate(output[0]), *output[1:])
        return self._ablate(output)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _find_layers(model: Any) -> Any:
    inner = getattr(model, "model", model)
    for owner in (inner, getattr(inner, "language_model", None)):
        layers = getattr(owner, "layers", None)
        if layers is not None:
            return layers
    raise AttributeError("couches de décodeur introuvables (model.model.layers)")
