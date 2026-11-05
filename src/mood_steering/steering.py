"""Steering d'humeur par vecteur de contrôle, avec repli sur un prompt d'humeur.

Le module expose l'expression d'une humeur (ton du texte produit), jamais un ressenti.
Toute validation (forme, plage de couches, dose) se fait ici, en Python : le binding
``llama_set_adapter_cvec`` ne valide presque rien (voir ``docs/pics/cvec-entre-tokens.md``).
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from mood_steering.schedule import DoseSchedule

JOYFUL_SYSTEM_PROMPT = (
    "Tu t'exprimes sur un ton joyeux et enthousiaste, avec des phrases positives "
    "et chaleureuses, sans rien changer au fond de ta réponse."
)
SAD_SYSTEM_PROMPT = (
    "Tu t'exprimes sur un ton triste et abattu, avec des phrases mélancoliques "
    "et retenues, sans rien changer au fond de ta réponse."
)

# Précision de la dose appliquée par un profil : une variation plus fine que 10^-3
# ne rappelle pas le binding (la dose appliquée est la dose arrondie).
DOSE_DECIMALS = 3

# Valeur prudente par défaut, à remplacer par celle du pic de calibration 1.4.
DEFAULT_MAX_SCALE = 1.0
ARM_STEERING = "steering"
ARM_PROMPT_JOYFUL = "prompt_joyful"
ARM_PROMPT_SAD = "prompt_sad"


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

    def stream(self, prompt: str, system: str | None, max_tokens: int, seed: int) -> Iterator[str]:
        """Génère le texte token par token (fragments de texte, dans l'ordre)."""

    def stream_scheduled(
        self,
        prompt: str,
        system: str | None,
        max_tokens: int,
        seed: int,
        on_token_start: Callable[[int], None],
    ) -> Iterator[str]:
        """Comme ``stream``, en appelant ``on_token_start(t)`` avant le token ``t``.

        Le rappel est appelé pour chaque token, dans l'ordre (t = 0, 1, ...), avant que le
        moteur calcule les logits dont ce token est tiré ; il peut donc appeler ``set_cvec``.
        """


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
    """Prompt système du repli : le prompt d'humeur, puis la consigne de l'appelant."""
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
        """Bras steering ; sans vecteur, repli sur un prompt d'humeur (``fallback=True``).

        Le repli ne vaut que pour une demande de steering (dose non nulle) : prompt
        joyeux si dose > 0, prompt triste si dose < 0, suivi de la consigne ``system``
        de l'appelant. À dose 0, aucun vecteur n'est posé et la consigne est passée
        telle quelle, avec ou sans vecteur (``fallback=False``).

        Chaque génération part d'un contexte vide (``reset``) : sinon le moteur
        réutiliserait les K/V du prompt calculés à la dose précédente.
        """
        arm, fallback, eff_system = self._prepare(dose, system)
        text = self._backend.generate(prompt, eff_system, max_tokens, seed)
        return SteeringResult(arm, dose, text, fallback)

    def _prepare(self, dose: float, system: str | None) -> tuple[str, bool, str | None]:
        """Valide la dose, vide le contexte et pose (ou retire) le vecteur.

        Rend ``(bras, repli, consigne système effective)``.
        """
        _check_dose(dose)
        backend = self._backend
        backend.reset()
        if dose != 0 and (self._vector is None or self._il_end is None):
            backend.set_cvec(None, backend.n_embd, 0, 0)
            if dose > 0:
                arm, base = ARM_PROMPT_JOYFUL, JOYFUL_SYSTEM_PROMPT
            else:
                arm, base = ARM_PROMPT_SAD, SAD_SYSTEM_PROMPT
            return arm, True, compose_system(base, system)
        if dose == 0 or self._vector is None or self._il_end is None:
            backend.set_cvec(None, backend.n_embd, 0, 0)
        else:
            scaled = apply_dose(self._vector, dose, self._max_scale)
            backend.set_cvec(
                np.ascontiguousarray(scaled.reshape(-1)),
                backend.n_embd,
                self._il_start,
                self._il_end,
            )
        return ARM_STEERING, False, system

    def open_stream(
        self,
        prompt: str,
        dose: float = 0.0,
        *,
        system: str | None = None,
        max_tokens: int = 128,
        seed: int = 0,
        schedule: DoseSchedule | None = None,
    ) -> TokenStream:
        """Prépare une génération en flux : même contrat que ``generate``.

        La préparation (validation de la dose, ``reset``, vecteur) est immédiate ; le
        résultat de repli (``arm``, ``fallback``) est donc connu avant le premier token.
        Les tokens ne sont produits que lorsqu'on itère le ``TokenStream`` rendu.

        ``schedule`` (profil ``dose_at(t)``) remplace ``dose``. Le vecteur est (re)posé
        avant chaque token dont la dose arrondie à ``DOSE_DECIMALS`` décimales change
        (jamais rappelé sinon) ; la dose du token ``t`` agit dès les logits de ce token.
        ``TokenStream.doses`` se remplit au fil des tokens (dose appliquée, arrondie).
        Sans vecteur : le signe de ``schedule(0)`` choisit le prompt joyeux (> 0) ou
        triste (< 0), ``fallback=True``, et le profil est sans effet sur le texte
        (``doses`` garde la dose demandée, arrondie, pour le tracé).
        """
        if schedule is not None:
            return self._open_scheduled(prompt, schedule, system, max_tokens, seed)
        arm, fallback, eff_system = self._prepare(dose, system)
        tokens = self._backend.stream(prompt, eff_system, max_tokens, seed)
        return TokenStream(arm, dose, fallback, tokens)

    def _open_scheduled(
        self,
        prompt: str,
        schedule: DoseSchedule,
        system: str | None,
        max_tokens: int,
        seed: int,
    ) -> TokenStream:
        backend = self._backend
        first = round(float(schedule(0)), DOSE_DECIMALS)
        _check_dose(first)
        backend.reset()
        backend.set_cvec(None, backend.n_embd, 0, 0)  # état initial : aucune dose posée
        vector, il_end = self._vector, self._il_end
        fallback = first != 0 and (vector is None or il_end is None)
        eff_system = system
        arm = ARM_STEERING
        if fallback:
            arm, base = ARM_PROMPT_JOYFUL, JOYFUL_SYSTEM_PROMPT
            if first < 0:
                arm, base = ARM_PROMPT_SAD, SAD_SYSTEM_PROMPT
            eff_system = compose_system(base, system)
        doses: list[float] = []
        applied = 0.0  # dose actuellement posée dans le moteur (0 = aucun vecteur)

        def on_token_start(t: int) -> None:
            nonlocal applied
            raw = float(schedule(t))
            _check_dose(raw)
            dose = round(raw, DOSE_DECIMALS)
            doses.append(dose)
            if fallback or vector is None or il_end is None or dose == applied:
                return
            if dose == 0:
                backend.set_cvec(None, backend.n_embd, 0, 0)
            else:
                scaled = apply_dose(vector, dose, self._max_scale)
                backend.set_cvec(
                    np.ascontiguousarray(scaled.reshape(-1)),
                    backend.n_embd,
                    self._il_start,
                    il_end,
                )
            applied = dose

        tokens = backend.stream_scheduled(prompt, eff_system, max_tokens, seed, on_token_start)
        return TokenStream(arm, first, fallback, tokens, doses)

    def stream(
        self,
        prompt: str,
        dose: float = 0.0,
        *,
        system: str | None = None,
        max_tokens: int = 128,
        seed: int = 0,
        schedule: DoseSchedule | None = None,
    ) -> Iterator[str]:
        """Comme ``generate``, mais rend les tokens un par un (``open_stream`` pour le repli)."""
        return iter(
            self.open_stream(
                prompt, dose, system=system, max_tokens=max_tokens, seed=seed, schedule=schedule
            )
        )


@dataclass
class TokenStream:
    """Flux de tokens d'une génération ; ``arm``, ``dose`` et ``fallback`` sont connus d'avance.

    Avec un profil, ``dose`` est la dose du token 0 et ``doses`` la dose de chaque token
    déjà démarré (une entrée par token, dans l'ordre) ; vide sans profil.
    """

    arm: str
    dose: float
    fallback: bool
    tokens: Iterator[str]
    doses: list[float] = field(default_factory=list)

    def __iter__(self) -> Iterator[str]:
        return iter(self.tokens)


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
        # Même boucle bas niveau que le flux : create_chat_completion (llama-cpp-python
        # 0.3.36) saute le test de max_tokens sur les octets UTF-8 incomplets et peut
        # remplir le contexte (« failed to find a memory slot »).
        return "".join(self.stream(prompt, system, max_tokens, seed))

    def stream(self, prompt: str, system: str | None, max_tokens: int, seed: int) -> Iterator[str]:
        return self.stream_scheduled(prompt, system, max_tokens, seed, lambda t: None)

    def stream_scheduled(
        self,
        prompt: str,
        system: str | None,
        max_tokens: int,
        seed: int,
        on_token_start: Callable[[int], None],
    ) -> Iterator[str]:
        """Boucle bas niveau ``eval`` / ``sample`` gloutonne, rappel avant chaque token.

        ``create_chat_completion(stream=True)`` ne rend pas la main avant l'échantillonnage.
        Ici ``on_token_start(t)`` est appelé juste avant l'``eval`` dont les logits donnent
        le token ``t`` (avant l'évaluation du prompt pour ``t = 0``) : la dose du token
        ``t`` agit donc sur ses propres logits. Toujours après ``reset()``. Échantillonnage
        glouton (température 0) : ``seed`` est accepté pour l'interface, sans effet.
        """
        llm = self._llm
        llm.reset()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        formatted = self._chat_formatter()(messages=messages).prompt
        pending = llm.tokenize(formatted.encode("utf-8"), add_bos=False, special=True)
        eos = llm.token_eos()
        out = bytearray()
        for t in range(max_tokens):
            on_token_start(t)
            llm.eval(pending)
            tok = llm.sample(top_k=1, temp=0.0)
            if tok == eos:
                break
            pending = [tok]
            out += llm.detokenize([tok])
            try:
                text = bytes(out).decode("utf-8")
            except UnicodeDecodeError:
                continue  # fragment UTF-8 incomplet : attend le token suivant
            out.clear()
            if text:
                yield text

    def _chat_formatter(self):
        from llama_cpp import llama_chat_format

        llm = self._llm
        return llama_chat_format.Jinja2ChatFormatter(
            template=llm.metadata["tokenizer.chat_template"],
            eos_token=llm._model.token_get_text(llm.token_eos()),
            bos_token=llm._model.token_get_text(llm.token_bos()),
            add_generation_prompt=True,
        )
