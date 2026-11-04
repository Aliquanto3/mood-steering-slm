"""Steering d'humeur par vecteur de contrôle, avec repli sur le prompt joyeux.

Le module expose l'expression d'une humeur (ton du texte produit), jamais un ressenti.
Toute validation (forme, plage de couches, dose) se fait ici, en Python : le binding
``llama_set_adapter_cvec`` ne valide presque rien (voir ``docs/pics/cvec-entre-tokens.md``).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Protocol

import numpy as np

JOYFUL_SYSTEM_PROMPT = (
    "Tu t'exprimes sur un ton joyeux et enthousiaste, avec des phrases positives "
    "et chaleureuses, sans rien changer au fond de ta réponse."
)

# Valeur prudente par défaut, à remplacer par celle du pic de calibration 1.4.
DEFAULT_MAX_SCALE = 1.0
ARM_STEERING = "steering"
ARM_PROMPT_JOYFUL = "prompt_joyful"


class SteeringBackend(Protocol):
    """Frontière externe : le moteur d'inférence."""

    n_layer: int
    n_embd: int

    def reset(self) -> None:
        """Vide le contexte (cache KV, état récurrent) : aucun préfixe réutilisé."""

    def set_cvec(self, data: np.ndarray | None, n_embd: int, il_start: int, il_end: int) -> None:
        """Pose (tampon plat float32) ou retire (``None``) le vecteur de contrôle."""

    def generate(self, prompt: str, system: str | None, max_tokens: int, seed: int) -> str:
        """Génère le texte."""


@dataclass(frozen=True)
class SteeringResult:
    """Résultat d'une génération du bras steering."""

    arm: str
    dose: float
    text: str
    fallback: bool


def _check_dose(dose: float) -> None:
    if not -1.0 <= dose <= 1.0:  # rejette aussi NaN
        raise ValueError(f"dose hors de [-1, 1] : {dose!r}")


def apply_dose(vector: np.ndarray, dose: float, max_scale: float) -> np.ndarray:
    """Rend ``dose × max_scale × vector`` (float32), dose dans [-1, 1]."""
    _check_dose(dose)
    return np.float32(dose * max_scale) * np.asarray(vector, dtype=np.float32)


def compose_system(base: str, extra: str | None) -> str:
    """Prompt système du repli : le prompt joyeux, puis la consigne de l'appelant."""
    return f"{base}\n\n{extra}" if extra else base


class Steerer:
    """Applique une dose de vecteur à un backend, ou se replie sur le prompt joyeux.

    ``max_scale`` : échelle de la dose 1 (défaut prudent 1.0, à fixer par le pic 1.4).
    ``il_start`` / ``il_end`` : plage de couches inclusive ; défaut ``il_start=1`` et
    ``il_end`` déduit du nombre de couches du vecteur. Le backend est chargé une seule
    fois, par l'appelant, puis réutilisé pour toutes les doses.
    """

    def __init__(
        self,
        backend: SteeringBackend,
        vector: np.ndarray | None = None,
        *,
        max_scale: float = DEFAULT_MAX_SCALE,
        il_start: int = 1,
        il_end: int | None = None,
    ) -> None:
        if not (np.isfinite(max_scale) and max_scale > 0):
            raise ValueError(f"max_scale doit être fini et > 0 : {max_scale!r}")
        self._backend = backend
        self._max_scale = max_scale
        self._il_start = il_start
        self._il_end = il_end
        self._vector: np.ndarray | None = None
        if vector is not None:
            self._vector, self._il_end = self._validate(vector, il_start, il_end)

    def _validate(
        self, vector: np.ndarray, il_start: int, il_end: int | None
    ) -> tuple[np.ndarray, int]:
        vec = np.asarray(vector)
        if vec.ndim != 2:
            raise ValueError(f"vecteur de forme (couches, n_embd) attendu, reçu {vec.shape}")
        n_embd = self._backend.n_embd
        if vec.shape[1] != n_embd:
            raise ValueError(f"dimension {vec.shape[1]} différente de n_embd={n_embd}")
        end = il_start + vec.shape[0] - 1 if il_end is None else il_end
        if not 1 <= il_start <= end <= self._backend.n_layer - 1:
            raise ValueError(
                f"plage de couches invalide : {il_start}..{end} "
                f"(attendu 1 <= début <= fin <= {self._backend.n_layer - 1})"
            )
        if vec.shape[0] != end - il_start + 1:
            raise ValueError(f"{vec.shape[0]} couches fournies pour la plage {il_start}..{end}")
        if not np.all(np.isfinite(vec)):
            raise ValueError("vecteur avec des valeurs non finies (NaN ou inf)")
        return vec.astype(np.float32, copy=False), end

    @property
    def has_vector(self) -> bool:
        return self._vector is not None

    def generate(
        self,
        prompt: str,
        dose: float,
        *,
        system: str | None = None,
        max_tokens: int = 128,
        seed: int = 0,
    ) -> SteeringResult:
        """Bras steering ; sans vecteur, repli sur le prompt joyeux (``fallback=True``).

        Chaque génération part d'un contexte vide (``reset``) : sinon le moteur
        réutiliserait les K/V du prompt calculés à la dose précédente. Au repli, la
        consigne ``system`` de l'appelant suit le prompt joyeux.
        """
        _check_dose(dose)
        backend = self._backend
        backend.reset()
        if self._vector is None or self._il_end is None:
            backend.set_cvec(None, backend.n_embd, 0, 0)
            joyful = compose_system(JOYFUL_SYSTEM_PROMPT, system)
            text = backend.generate(prompt, joyful, max_tokens, seed)
            return SteeringResult(ARM_PROMPT_JOYFUL, dose, text, True)
        if dose == 0:
            backend.set_cvec(None, backend.n_embd, 0, 0)
        else:
            scaled = apply_dose(self._vector, dose, self._max_scale)
            backend.set_cvec(
                np.ascontiguousarray(scaled.reshape(-1)),
                backend.n_embd,
                self._il_start,
                self._il_end,
            )
        text = backend.generate(prompt, system, max_tokens, seed)
        return SteeringResult(ARM_STEERING, dose, text, False)


class LlamaCppBackend:
    """Backend réel (extra ``llama``) ; seul endroit qui touche ``Llama._ctx.ctx``."""

    def __init__(self, model_path: str, *, n_ctx: int = 2048) -> None:
        import llama_cpp  # import paresseux : extra optionnel

        self._cpp = llama_cpp
        self._llm = llama_cpp.Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)
        self.n_embd = int(self._llm.n_embd())
        self.n_layer = int(llama_cpp.llama_model_n_layer(self._llm._model.model))

    def reset(self) -> None:
        # Llama.reset remet n_tokens à 0 : generate n'essaie plus de réutiliser le
        # préfixe KV et eval purge le cache depuis la position 0.
        self._llm.reset()

    def set_cvec(self, data: np.ndarray | None, n_embd: int, il_start: int, il_end: int) -> None:
        ctx = self._llm._ctx.ctx
        if data is None:
            rc = self._cpp.llama_set_adapter_cvec(ctx, None, 0, n_embd, 0, 0)
        else:
            flat = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
            buf = (ctypes.c_float * flat.size)(*flat.tolist())
            rc = self._cpp.llama_set_adapter_cvec(ctx, buf, flat.size, n_embd, il_start, il_end)
        if rc != 0:
            raise RuntimeError(f"llama_set_adapter_cvec a échoué (code {rc})")

    def generate(self, prompt: str, system: str | None, max_tokens: int, seed: int) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        out = self._llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=0.0, seed=seed
        )
        return out["choices"][0]["message"]["content"]
