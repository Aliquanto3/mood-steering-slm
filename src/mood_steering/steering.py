"""Steering d'humeur par vecteur de contrôle, avec repli sur un prompt d'humeur.

Le module expose l'expression d'une humeur (ton du texte produit), jamais un ressenti.
Toute validation (forme, plage de couches, dose) se fait ici, en Python : le binding
``llama_set_adapter_cvec`` ne valide presque rien (voir ``docs/pics/cvec-entre-tokens.md``).
"""

from __future__ import annotations

import codecs
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
    ``vector`` : le vecteur COMPLET, tel que rendu par ``calibration.load_vector``, de forme
    ``(lignes, n_embd)`` avec la ligne 0 = couche 1 (llama.cpp lit toujours son tampon à
    l'offset ``n_embd × (il − 1)``, quelle que soit ``il_start``).
    ``il_start`` / ``il_end`` : plage de couches inclusive ; défaut ``il_start=1`` et
    ``il_end = min(lignes, n_layer − 1)``. Exige
    ``1 <= il_start <= il_end <= min(lignes, n_layer − 1)``. Le tampon envoyé au binding a
    ``il_end`` lignes (couches 1..il_end) ; les lignes des couches < ``il_start`` sont à zéro,
    les couches > ``il_end`` absentes. Le backend est chargé une seule fois, par
    l'appelant, puis réutilisé pour toutes les doses.
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
        max_end = min(vec.shape[0], self._backend.n_layer - 1)
        end = max_end if il_end is None else il_end
        if not 1 <= il_start <= end <= max_end:
            raise ValueError(
                f"plage de couches invalide : {il_start}..{end} (attendu 1 <= début <= fin <= "
                f"{max_end} = min({vec.shape[0]} lignes du vecteur, n_layer - 1))"
            )
        if not np.all(np.isfinite(vec)):
            raise ValueError("vecteur avec des valeurs non finies (NaN ou inf)")
        # Tampon de il_end lignes (ligne 0 = couche 1) : llama.cpp lit la couche il à
        # l'offset n_embd × (il − 1) ; les couches < il_start sont mises à zéro.
        buf = np.array(vec[:end], dtype=np.float32)
        buf[: il_start - 1] = 0.0
        return buf, end

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
        réutiliserait les K/V du prompt calculés à la dose précédente. Génération
        gloutonne avec ``LlamaCppBackend`` : ``seed`` y est sans effet.
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
        (jamais rappelé sinon). ``TokenStream.doses`` se remplit au fil des tokens (dose
        appliquée, arrondie).

        Calendrier : la dose du token ``t`` est posée juste avant l'évaluation
        (``llama_decode``) dont les logits donnent le token ``t`` ; elle agit donc sur ses
        propres logits. Pour ``t = 0``, cette évaluation est celle du prompt : la dose du
        token 0 s'applique aussi au prompt (un profil constant équivaut à
        ``--control-vector`` en CLI). Écart avec le protocole du pic 1.2
        (``docs/pics/cvec-entre-tokens.md``), où le prompt est évalué sans vecteur et où une
        dose posée « avant le token N » n'agit qu'à partir du token N+1 : ici l'indice du
        profil est celui du token dont la dose change les logits (décalage d'un cran).

        Validation : chaque ``schedule(t)`` doit être dans [-1, 1] avant arrondi, sans
        tolérance, à l'ouverture comme en flux (une dose hors bornes en cours de flux lève
        ``ValueError`` à l'itération). Sans vecteur : le signe de la première dose arrondie
        non nulle pour ``t < max_tokens`` choisit le prompt joyeux (> 0) ou triste (< 0),
        ``fallback=True`` ; profil nul partout : pas de repli. Le profil est alors sans
        effet sur le texte (``doses`` garde la dose demandée, arrondie, pour le tracé).
        Échantillonnage glouton : ``seed`` est sans effet sur le texte.
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

        def dose_at(t: int) -> float:
            raw = float(schedule(t))
            _check_dose(raw)
            return round(raw, DOSE_DECIMALS)

        first = dose_at(0)
        vector, il_end = self._vector, self._il_end
        sign = 0.0
        if vector is None or il_end is None:
            # Repli décidé par la première dose non nulle (un pulse peut partir de 0).
            sign = next((d for d in map(dose_at, range(max_tokens)) if d != 0), 0.0)
        fallback = sign != 0
        backend.reset()
        backend.set_cvec(None, backend.n_embd, 0, 0)  # état initial : aucune dose posée
        eff_system = system
        arm = ARM_STEERING
        if fallback:
            arm, base = ARM_PROMPT_JOYFUL, JOYFUL_SYSTEM_PROMPT
            if sign < 0:
                arm, base = ARM_PROMPT_SAD, SAD_SYSTEM_PROMPT
            eff_system = compose_system(base, system)
        doses: list[float] = []
        applied = 0.0  # dose actuellement posée dans le moteur (0 = aucun vecteur)

        def on_token_start(t: int) -> None:
            nonlocal applied
            dose = dose_at(t)
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

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
    ) -> None:
        """``n_gpu_layers`` : couches déchargées sur le GPU (0 = CPU seul, -1 = toutes ;
        sans effet avec la roue CPU). ``n_threads`` : défaut de llama-cpp-python si ``None``."""
        import llama_cpp  # import paresseux : extra optionnel

        self._cpp = llama_cpp
        self._llm = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False,
        )
        self.n_embd = int(self._llm.n_embd())
        self.n_layer = int(llama_cpp.llama_model_n_layer(self._llm._model.model))
        # cause de fin de la dernière boucle : "eog" (token de fin), "budget" (max_tokens ou
        # contexte plein, le dernier token démarré est un vrai token), None avant toute boucle
        self.last_stop: str | None = None

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
            # Pointeur sur le tableau (vivant pendant l'appel, llama.cpp copie le tampon) :
            # pas de liste Python reconstruite à chaque changement de dose par token.
            buf = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
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
        ``t`` agit donc sur ses propres logits, et celle du token 0 aussi sur le prompt.
        Toujours après ``reset()``. Échantillonnage glouton (température 0) : ``seed`` est
        accepté pour l'interface, sans effet.

        Préparation immédiate (à l'appel, pas à la première itération) : ``max_tokens`` est
        borné à ``n_ctx − len(prompt)`` (comme ``create_completion``) ; un prompt qui
        remplit seul le contexte lève ``ValueError``. Arrêt sur tout token de fin de
        génération (``llama_vocab_is_eog``). Décodage UTF-8 incrémental : un octet
        invalide devient U+FFFD sans bloquer la suite ; un caractère coupé par
        ``max_tokens`` est abandonné.
        """
        llm = self._llm
        llm.reset()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        formatted = self._chat_formatter()(messages=messages).prompt
        pending = llm.tokenize(formatted.encode("utf-8"), add_bos=False, special=True)
        n_ctx = int(llm.n_ctx())
        if len(pending) >= n_ctx:
            raise ValueError(
                f"prompt de {len(pending)} tokens : dépasse le contexte de {n_ctx} tokens"
            )
        budget = min(max_tokens, n_ctx - len(pending))
        return self._loop(pending, budget, on_token_start)

    def _loop(
        self, pending: list[int], max_tokens: int, on_token_start: Callable[[int], None]
    ) -> Iterator[str]:
        llm = self._llm
        vocab = llm._model.vocab
        is_eog = self._cpp.llama_vocab_is_eog
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.last_stop = None
        for t in range(max_tokens):
            on_token_start(t)
            llm.eval(pending)
            tok = llm.sample(top_k=1, temp=0.0)
            if is_eog(vocab, tok):
                self.last_stop = "eog"
                break
            pending = [tok]
            text = decoder.decode(llm.detokenize([tok]))
            if text:
                yield text
        else:
            self.last_stop = "budget"

    def _chat_formatter(self):
        from llama_cpp import llama_chat_format

        llm = self._llm
        return llama_chat_format.Jinja2ChatFormatter(
            template=llm.metadata["tokenizer.chat_template"],
            eos_token=self._token_text(llm.token_eos()),
            bos_token=self._token_text(llm.token_bos()),
            add_generation_prompt=True,
        )

    def _token_text(self, token: int) -> str:
        # Un vocabulaire sans BOS (ou sans EOS) rend -1 : pas de texte à lire.
        return "" if token < 0 else self._llm._model.token_get_text(token)
