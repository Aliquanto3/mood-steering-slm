"""Pic 1.3 : relire en flottants le GGUF de `llama-cvector-generator`.

Sous-commandes :
  show PATH                      tenseurs par couche (nom, forme, type) et métadonnées
  generate MODELE SORTIE         génère un vecteur joie/tristesse avec llama-cvector-generator
  equiv MODELE VECTEUR           `llama_set_adapter_cvec` (llama-cpp-python) contre la CLI

`show` et `read_cvector` n'ont besoin que de `gguf` et numpy ; `equiv` demande l'extra `llama`.
Les binaires viennent de l'environnement (LLAMA_CVECTOR_GENERATOR, LLAMA_COMPLETION).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gguf import GGUFReader

_DIRECTION = re.compile(r"^direction\.(\d+)$")

# Paires contrastées : expression de joie contre expression de tristesse.
PAIRS = [
    (
        "Je suis ravi, tout se passe merveilleusement bien aujourd'hui !",
        "Je suis abattu, plus rien ne va depuis ce matin.",
    ),
    (
        "Quelle belle journée, le soleil brille et je souris.",
        "Quelle journée grise, la pluie tombe et je soupire.",
    ),
    (
        "J'ai reçu une excellente nouvelle, c'est formidable !",
        "J'ai reçu une mauvaise nouvelle, c'est désolant.",
    ),
    (
        "Nous avons gagné, tout le monde rit et applaudit.",
        "Nous avons perdu, tout le monde se tait et baisse les yeux.",
    ),
    (
        "Avec enthousiasme, je commence ce nouveau projet.",
        "Avec lassitude, je regarde ce projet inachevé.",
    ),
    (
        "Un grand sourire illumine son visage.",
        "Un voile de chagrin couvre son visage.",
    ),
]
PROMPT = "Décris ta journée en deux phrases."


@dataclass(frozen=True)
class CVector:
    """Vecteur de contrôle relu : une direction float32 par couche présente."""

    n_embd: int
    layer_count: int | None
    model_hint: str | None
    directions: dict[int, np.ndarray]  # indice de couche (1..n_layer) -> (n_embd,)

    @property
    def missing_layers(self) -> list[int]:
        """Couches 1..layer_count sans direction (la couche 0 n'existe jamais)."""
        n = self.layer_count if self.layer_count is not None else max(self.directions, default=0)
        return [il for il in range(1, n + 1) if il not in self.directions]

    def flat(self, n_layer: int) -> np.ndarray:
        """Tampon (n_layer * n_embd) attendu par `llama_set_adapter_cvec` : couche 1 en tête."""
        if self.directions and max(self.directions) > n_layer:
            raise ValueError(f"direction {max(self.directions)} au-delà de {n_layer} couches")
        buf = np.zeros(n_layer * self.n_embd, dtype=np.float32)
        for il, vec in self.directions.items():
            buf[(il - 1) * self.n_embd : il * self.n_embd] = vec
        return buf


def _field_value(reader: GGUFReader, key: str):
    field = reader.fields.get(key)
    if field is None:
        return None
    return field.contents()


def read_cvector(path: str | Path) -> CVector:
    """Lit un GGUF `controlvector` ; refuse ce que `--control-vector` refuserait."""
    reader = GGUFReader(str(path))
    directions: dict[int, np.ndarray] = {}
    n_embd = -1
    for t in reader.tensors:
        m = _DIRECTION.match(t.name)
        if m is None or int(m.group(1)) == 0:
            raise ValueError(f"nom de tenseur invalide : {t.name!r}")
        if t.tensor_type.name != "F32":
            raise ValueError(f"type non F32 : {t.name} ({t.tensor_type.name})")
        if len(t.shape) != 1:
            raise ValueError(f"forme non 1D : {t.name} {list(t.shape)}")
        if n_embd == -1:
            n_embd = int(t.shape[0])
        elif int(t.shape[0]) != n_embd:
            raise ValueError("dimensions de directions différentes")
        directions[int(m.group(1))] = np.array(t.data, dtype=np.float32).reshape(-1)
    layer_count = _field_value(reader, "controlvector.layer_count")
    hint = _field_value(reader, "controlvector.model_hint")
    return CVector(
        n_embd=n_embd,
        layer_count=int(layer_count) if layer_count is not None else None,
        model_hint=str(hint) if hint is not None else None,
        directions=directions,
    )


def show(path: str | Path) -> CVector:
    reader = GGUFReader(str(path))
    for t in reader.tensors:
        print(f"{t.name}\tforme={[int(s) for s in t.shape]}\ttype={t.tensor_type.name}")
    cv = read_cvector(path)
    print(f"n_embd={cv.n_embd} layer_count={cv.layer_count} model_hint={cv.model_hint}")
    print(f"couches présentes={sorted(cv.directions)} absentes={cv.missing_layers}")
    return cv


def generate(model: str, out: str) -> None:
    exe = os.environ.get("LLAMA_CVECTOR_GENERATOR")
    if not exe:
        sys.exit("définir LLAMA_CVECTOR_GENERATOR")
    with tempfile.TemporaryDirectory() as tmp:
        pos, neg = Path(tmp, "pos.txt"), Path(tmp, "neg.txt")
        pos.write_text("\n".join(p for p, _ in PAIRS) + "\n", encoding="utf-8")
        neg.write_text("\n".join(n for _, n in PAIRS) + "\n", encoding="utf-8")
        cmd = [
            exe, "-m", model,
            "--positive-file", str(pos), "--negative-file", str(neg),
            "-o", out, "-ngl", "0", "--method", "mean",
        ]  # fmt: skip
        subprocess.run(cmd, check=True)


def _cli_text(model: str, vec: str | None, n_predict: int, seed: int, scale: float) -> str:
    exe = os.environ.get("LLAMA_COMPLETION")
    if not exe:
        sys.exit("définir LLAMA_COMPLETION")
    cmd = [
        exe, "-m", model, "-p", PROMPT, "-n", str(n_predict), "--temp", "0", "-s", str(seed),
        "-ngl", "0", "-no-cnv", "--no-display-prompt", "--simple-io", "-t", "4",
    ]  # fmt: skip
    if vec:
        # Le séparateur `:` de --control-vector-scaled casse sur « C:/ » : nom relatif + cwd.
        cmd += ["--control-vector-scaled", f"{Path(vec).name}:{scale}"]
    cwd = Path(vec).parent if vec else None
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", cwd=cwd)
    return out.stdout


def _py_text(model: str, cv: CVector | None, n_predict: int, seed: int, scale: float) -> str:
    import ctypes

    import llama_cpp

    llm = llama_cpp.Llama(model, n_ctx=512, seed=seed, n_threads=4, verbose=False)
    if cv is not None:
        n_layer = llama_cpp.llama_model_n_layer(llm.model)
        buf = cv.flat(n_layer) * np.float32(scale)  # doit rester vivant pendant l'appel
        ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = llama_cpp.llama_set_adapter_cvec(llm.ctx, ptr, buf.size, cv.n_embd, 1, n_layer)
        if rc != 0:
            raise RuntimeError(f"llama_set_adapter_cvec a échoué ({rc})")
    out = llm.create_completion(PROMPT, max_tokens=n_predict, temperature=0.0, seed=seed)
    return out["choices"][0]["text"]


def equiv(model: str, vec: str, n_predict: int = 40, seed: int = 42, scale: float = 1.0) -> bool:
    cv = read_cvector(vec)
    cli_with = _cli_text(model, vec, n_predict, seed, scale)
    cli_without = _cli_text(model, None, n_predict, seed, scale)
    py_with = _py_text(model, cv, n_predict, seed, scale)
    py_without = _py_text(model, None, n_predict, seed, scale)
    for label, text in [
        ("cli +vec", cli_with),
        ("py  +vec", py_with),
        ("cli base", cli_without),
        ("py  base", py_without),
    ]:
        print(f"--- {label}\n{text.strip()}")
    same = cli_with.strip() == py_with.strip()
    print(f"\néquivalence (CLI --control-vector = set_adapter_cvec x1) : {same}")
    print(f"le vecteur modifie la sortie : {cli_with.strip() != cli_without.strip()}")
    return same


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show").add_argument("path")
    g = sub.add_parser("generate")
    g.add_argument("model")
    g.add_argument("out")
    e = sub.add_parser("equiv")
    e.add_argument("model")
    e.add_argument("vector")
    e.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    if a.cmd == "show":
        show(a.path)
    elif a.cmd == "generate":
        generate(a.model, a.out)
    else:
        sys.exit(0 if equiv(a.model, a.vector, scale=a.scale) else 1)


if __name__ == "__main__":
    main()
