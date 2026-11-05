"""Serveur FastAPI de la démo : trois flux de tokens multiplexés dans un seul flux SSE.

Le démonstrateur montre l'expression d'une humeur (ton du texte), jamais un ressenti.

Protocole ``GET /stream?prompt=...&dose=...&max_tokens=...&seed=...`` (``text/event-stream``).
Chaque événement est une ligne ``data: <JSON>`` suivie d'une ligne vide (JSON en ASCII :
U+2028, U+2029 et U+0085 sortent échappés) :

- ``{"type":"start","arm":A,"dose":D,"fallback":B,"label":L}`` : une fois par bras, avant ses
  tokens (``fallback`` vrai : pas de vecteur, la colonne steering suit un prompt d'humeur) ;
- ``{"type":"token","arm":A,"text":T}`` : un fragment de texte ;
- ``{"type":"end","arm":A}`` : fin du bras ;
- ``{"type":"error","arm":A,"message":M}`` : échec du bras (pas de ``end`` ensuite) ;
- ``{"type":"done"}`` : dernier événement, une fois les trois bras terminés.

Bras ``A`` : ``neutral`` (dose 0, sans consigne), ``prompt_joyful`` (dose 0, prompt joyeux),
``steering`` (dose demandée, dans [-1, 1]). Dans un bras, l'ordre est start, tokens, end ;
entre bras, les événements s'entrelacent. Dose hors bornes, prompt vide ou de plus de
``MAX_PROMPT_CHARS`` caractères, ``max_tokens`` hors de [1, ``MAX_TOKENS_LIMIT``] : réponse 422.

Une seule génération à la fois (démo à un présentateur, backends partagés) : une requête
``/stream`` reçue pendant une génération en cours reçoit 409 (``{"detail": ...}``), sans
attente. Si le client se déconnecte, les trois bras s'arrêtent au token suivant et le
serveur redevient libre dès qu'ils ont rendu la main. ``GET /health`` et ``GET /config``
complètent l'API.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import anyio
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send

from mood_steering.steering import (
    DEFAULT_MAX_SCALE,
    JOYFUL_SYSTEM_PROMPT,
    Steerer,
    SteeringBackend,
)

ARM_NEUTRAL = "neutral"
ARM_PROMPT_JOYFUL = "prompt_joyful"
ARM_STEERING = "steering"
ARMS = (ARM_NEUTRAL, ARM_PROMPT_JOYFUL, ARM_STEERING)
ARM_LABELS = {
    ARM_NEUTRAL: "Neutre",
    ARM_PROMPT_JOYFUL: "Prompt joyeux",
    ARM_STEERING: "Steering d'activations",
}
REGISTER = "expression d'humeur"
STATIC_DIR = Path(__file__).parent / "static"
MAX_TOKENS_LIMIT = 512
MAX_PROMPT_CHARS = 2000
BUSY_MESSAGE = "une génération est déjà en cours ; réessayez quand elle est terminée"
MEMORY_NOTE = (
    "trois instances du modèle (une par bras) ; les poids sont projetés en mémoire (mmap) "
    "et partagés par le système, seuls le cache KV et les contextes sont triplés"
)
_POLL_SECONDS = 0.25


def _sse(event: dict) -> str:
    # ensure_ascii : U+2028, U+2029 et U+0085 ne peuvent pas passer pour une fin de ligne.
    return f"data: {json.dumps(event, ensure_ascii=True)}\n\n"


class _Generation:
    """Cycle de vie d'une génération : threads des bras, drapeau d'arrêt, verrou de l'app."""

    def __init__(self, lock: threading.Lock) -> None:
        self.stop = threading.Event()
        self.threads: list[threading.Thread] = []
        self._lock = lock
        self._mu = threading.Lock()
        self._closed = False

    def close(self) -> None:
        """Lève l'arrêt et rend le verrou dès que les bras ont fini (idempotent)."""
        with self._mu:
            if self._closed:
                return
            self._closed = True
        self.stop.set()
        threads = list(self.threads)

        def release() -> None:
            for t in threads:
                t.join()
            self._lock.release()

        if any(t.is_alive() for t in threads):
            threading.Thread(target=release, daemon=True).start()
        else:
            release()


class _GenerationResponse(StreamingResponse):
    """Rend la génération même si le flux est annulé avant ou pendant son itération."""

    def __init__(self, content, generation: _Generation, **kw) -> None:
        super().__init__(content, **kw)
        self._generation = generation

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._generation.close()


def create_app(
    backend_factory: Callable[[], SteeringBackend],
    *,
    vector: np.ndarray | None = None,
    max_scale: float = DEFAULT_MAX_SCALE,
    il_start: int = 1,
    il_end: int | None = None,
    model: str = "inconnu",
) -> FastAPI:
    """Crée l'app ; ``backend_factory`` est appelée trois fois (un backend par bras).

    Mémoire : ``LlamaCppBackend`` garde ``use_mmap=True`` (défaut de llama-cpp-python), les
    pages des poids du GGUF sont donc partagées par l'OS entre les trois instances ; seuls le
    cache KV et les contextes sont triplés (``/config`` le rappelle dans ``memory_note``).
    Les backends sont partagés par toutes les requêtes : une génération à la fois (409 sinon).
    """
    steerers = {
        ARM_NEUTRAL: Steerer(backend_factory()),
        ARM_PROMPT_JOYFUL: Steerer(backend_factory()),
        ARM_STEERING: Steerer(
            backend_factory(), vector, max_scale=max_scale, il_start=il_start, il_end=il_end
        ),
    }
    # (dose imposée ou None = dose demandée, consigne système)
    arm_specs = {
        ARM_NEUTRAL: (0.0, None),
        ARM_PROMPT_JOYFUL: (0.0, JOYFUL_SYSTEM_PROMPT),
        ARM_STEERING: (None, None),
    }
    busy = threading.Lock()
    app = FastAPI(title="mood-steering-slm")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/config")
    def config() -> dict:
        return {
            "model": model,
            "dose_min": -1.0,
            "dose_max": 1.0,
            "max_scale": max_scale,
            "has_vector": steerers[ARM_STEERING].has_vector,
            "arms": list(ARMS),
            "labels": ARM_LABELS,
            "register": REGISTER,
            "max_tokens_limit": MAX_TOKENS_LIMIT,
            "max_prompt_chars": MAX_PROMPT_CHARS,
            "memory_note": MEMORY_NOTE,
        }

    def run_arm(arm, prompt, dose, max_tokens, seed, out: queue.Queue, stop) -> None:
        tokens = None
        try:
            fixed, system = arm_specs[arm]
            real_dose = dose if fixed is None else fixed
            stream = steerers[arm].open_stream(
                prompt, real_dose, system=system, max_tokens=max_tokens, seed=seed
            )
            tokens = iter(stream)
            out.put(
                {
                    "type": "start",
                    "arm": arm,
                    "dose": real_dose,
                    "fallback": stream.fallback,
                    "label": ARM_LABELS[arm],
                }
            )
            for text in tokens:
                if stop.is_set():  # client parti : arrêt coopératif entre deux tokens
                    break
                out.put({"type": "token", "arm": arm, "text": text})
            out.put({"type": "end", "arm": arm})
        except Exception as exc:  # un bras en échec ne doit pas bloquer les autres
            out.put({"type": "error", "arm": arm, "message": str(exc)})
        except BaseException as exc:  # SystemExit, KeyboardInterrupt : signalé puis relancé
            out.put({"type": "error", "arm": arm, "message": f"interrompu ({type(exc).__name__})"})
            raise
        finally:
            close = getattr(tokens, "close", None)
            if close is not None:
                try:
                    close()  # libère le générateur du moteur (llama-cpp arrête la génération)
                except Exception:
                    pass
            out.put(None)

    @app.get("/stream")
    async def stream(
        request: Request,
        prompt: str = Query(..., max_length=MAX_PROMPT_CHARS),
        dose: float = Query(0.0),
        max_tokens: int = Query(128, ge=1, le=MAX_TOKENS_LIMIT),
        seed: int = Query(0),
    ) -> StreamingResponse:
        if not prompt.strip():
            raise HTTPException(422, "prompt vide")
        if not -1.0 <= dose <= 1.0:
            raise HTTPException(422, f"dose hors de [-1, 1] : {dose!r}")
        if not busy.acquire(blocking=False):
            raise HTTPException(409, BUSY_MESSAGE)
        generation = _Generation(busy)

        async def events() -> AsyncIterator[str]:
            out: queue.Queue = queue.Queue()
            try:
                if generation.stop.is_set():
                    return
                generation.threads = [
                    threading.Thread(
                        target=run_arm,
                        args=(arm, prompt, dose, max_tokens, seed, out, generation.stop),
                        daemon=True,
                    )
                    for arm in ARMS
                ]
                for t in generation.threads:
                    t.start()
                pending = len(generation.threads)
                while pending:
                    try:
                        item = await anyio.to_thread.run_sync(out.get, True, _POLL_SECONDS)
                    except queue.Empty:
                        if await request.is_disconnected():
                            return
                        continue
                    if item is None:
                        pending -= 1
                    else:
                        yield _sse(item)
                for t in generation.threads:
                    t.join()  # immédiat : chaque bras sort juste après son None
                generation.close()  # verrou rendu avant « done » : la requête suivante passe
                yield _sse({"type": "done"})
            finally:
                generation.close()

        return _GenerationResponse(
            events(),
            generation,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mood_steering_demo", description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="chemin du GGUF")
    parser.add_argument("--vector", help="GGUF controlvector (sinon repli sur le prompt)")
    parser.add_argument("--max-scale", type=float, default=DEFAULT_MAX_SCALE)
    parser.add_argument("--il-start", type=int, default=1, help="première couche (incluse)")
    parser.add_argument(
        "--il-end", type=int, help="dernière couche (incluse ; défaut : min(lignes, n_layer - 1))"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    import uvicorn

    from mood_steering.calibration import load_vector
    from mood_steering.steering import LlamaCppBackend

    vector = load_vector(args.vector) if args.vector else None
    app = create_app(
        lambda: LlamaCppBackend(args.model),
        vector=vector,
        max_scale=args.max_scale,
        il_start=args.il_start,
        il_end=args.il_end,
        model=Path(args.model).name,
    )
    uvicorn.run(app, host=args.host, port=args.port)
