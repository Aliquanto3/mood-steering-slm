"""Story 6.1 : exporter les directions anti-boucle du diff du MIT, une par couche.

Compare les ``mlp.down_proj.weight`` de deux copies HF de Qwen3.5-0.8B (modèle édité contre
modèle de base, par ex. ``slm-weight-editing/models/hf/qwen3.5-wsp`` et ``.../qwen3.5-base``) ;
pour chaque couche dont le poids diffère, la direction est le premier vecteur singulier gauche
du delta (dimension ``hidden_size``, écriture dans le flux résiduel). Écrit ``layer{L}.npy`` dans
``--out`` (défaut ``data/directions/``, ignoré par git : aucune direction versionnée).

    uv run --extra ablation python spikes/export_mit_direction.py EDITE BASE [--out DOSSIER]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

DOWN_PROJ = re.compile(r"layers\.(\d+)\.mlp\.down_proj\.weight$")


def _shards(model_dir: Path) -> dict[str, Path]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return {k: model_dir / v for k, v in index["weight_map"].items() if DOWN_PROJ.search(k)}


def export(edited: Path, base: Path, out: Path, rtol: float = 0.0) -> dict[int, Path]:
    import torch
    from safetensors import safe_open

    a, b = _shards(edited), _shards(base)
    if set(a) != set(b):
        raise ValueError("les deux modèles n'ont pas les mêmes tenseurs down_proj")
    out.mkdir(parents=True, exist_ok=True)
    written: dict[int, Path] = {}
    for key in sorted(a, key=lambda k: int(DOWN_PROJ.search(k).group(1))):
        with safe_open(str(a[key]), "pt") as fa, safe_open(str(b[key]), "pt") as fb:
            delta = fa.get_tensor(key).float() - fb.get_tensor(key).float()
        if not torch.any(delta != 0):
            continue
        layer = int(DOWN_PROJ.search(key).group(1))
        u, s, _ = torch.linalg.svd(delta, full_matrices=False)
        direction = u[:, 0].numpy().astype(np.float32)
        path = out / f"layer{layer}.npy"
        np.save(path, direction)
        written[layer] = path
        print(
            f"couche {layer} : delta {tuple(delta.shape)}, s1={s[0]:.3f}, s2={s[1]:.2e}, "
            f"direction {direction.shape[0]} -> {path}"
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("edited", type=Path, help="dossier HF du modèle édité (MIT)")
    parser.add_argument("base", type=Path, help="dossier HF du modèle de base")
    parser.add_argument("--out", type=Path, default=Path("data/directions"))
    args = parser.parse_args()
    written = export(args.edited, args.base, args.out)
    if not written:
        raise SystemExit("aucune couche down_proj ne diffère")
    if len(written) > 1:
        dirs = {k: np.load(v) for k, v in written.items()}
        layers = sorted(dirs)
        for i, li in enumerate(layers):
            for lj in layers[i + 1 :]:
                print(f"|cos(couche {li}, couche {lj})| = {abs(float(dirs[li] @ dirs[lj])):.3f}")


if __name__ == "__main__":
    main()
