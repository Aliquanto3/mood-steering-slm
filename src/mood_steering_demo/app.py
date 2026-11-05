"""Serveur FastAPI de la démo : trois flux de tokens multiplexés dans un seul flux SSE.

Le démonstrateur montre l'expression d'une humeur (ton du texte), jamais un ressenti.

Protocole ``GET /stream?prompt=...&dose=...&max_tokens=...&seed=...`` (``text/event-stream``).
Chaque événement est une ligne ``data: <JSON>`` suivie d'une ligne vide :

- ``{"type":"start","arm":A,"dose":D,"fallback":B,"label":L}`` : une fois par bras, avant ses
  tokens (``fallback`` vrai : pas de vecteur, la colonne steering suit un prompt d'humeur) ;
- ``{"type":"token","arm":A,"text":T}`` : un fragment de texte ;
- ``{"type":"end","arm":A}`` : fin du bras ;
- ``{"type":"error","arm":A,"message":M}`` : échec du bras (pas de ``end`` ensuite) ;
- ``{"type":"done"}`` : dernier événement, une fois les trois bras terminés.

Bras ``A`` : ``neutral`` (dose 0, sans consigne), ``prompt_joyful`` (dose 0, prompt joyeux),
``steering`` (dose demandée, dans [-1, 1]). Dans un bras, l'ordre est start, tokens, end ;
entre bras, les événements s'entrelacent. Dose hors bornes, prompt vide, ``max_tokens`` < 1 :
réponse 422. ``GET /health`` et ``GET /config`` complètent l'API.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

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


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app(
    backend_factory: Callable[[], SteeringBackend],
    *,
    vector: np.ndarray | None = None,
    max_scale: float = DEFAULT_MAX_SCALE,
    il_start: int = 1,
    il_end: int | None = None,
    model: str = "inconnu",
) -> FastAPI:
    """Crée l'app ; ``backend_factory`` est appelée trois fois (un backend par bras)."""
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
        }

    def run_arm(arm, prompt, dose, max_tokens, seed, out: queue.Queue) -> None:
        try:
            fixed, system = arm_specs[arm]
            real_dose = dose if fixed is None else fixed
            stream = steerers[arm].open_stream(
                prompt, real_dose, system=system, max_tokens=max_tokens, seed=seed
            )
            out.put(
                {
                    "type": "start",
                    "arm": arm,
                    "dose": real_dose,
                    "fallback": stream.fallback,
                    "label": ARM_LABELS[arm],
                }
            )
            for text in stream:
                out.put({"type": "token", "arm": arm, "text": text})
            out.put({"type": "end", "arm": arm})
        except Exception as exc:  # un bras en échec ne doit pas bloquer les autres
            out.put({"type": "error", "arm": arm, "message": str(exc)})
        finally:
            out.put(None)

    @app.get("/stream")
    def stream(
        prompt: str = Query(...),
        dose: float = Query(0.0),
        max_tokens: int = Query(128),
        seed: int = Query(0),
    ) -> StreamingResponse:
        if not prompt.strip():
            raise HTTPException(422, "prompt vide")
        if not -1.0 <= dose <= 1.0:
            raise HTTPException(422, f"dose hors de [-1, 1] : {dose!r}")
        if max_tokens < 1:
            raise HTTPException(422, "max_tokens doit être >= 1")

        def events() -> Iterator[str]:
            out: queue.Queue = queue.Queue()
            threads = [
                threading.Thread(
                    target=run_arm, args=(arm, prompt, dose, max_tokens, seed, out), daemon=True
                )
                for arm in ARMS
            ]
            for t in threads:
                t.start()
            pending = len(threads)
            while pending:
                item = out.get()
                if item is None:
                    pending -= 1
                else:
                    yield _sse(item)
            for t in threads:
                t.join()
            yield _sse({"type": "done"})

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mood_steering_demo", description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="chemin du GGUF")
    parser.add_argument("--vector", help="GGUF controlvector (sinon repli sur le prompt)")
    parser.add_argument("--max-scale", type=float, default=DEFAULT_MAX_SCALE)
    parser.add_argument("--il-start", type=int, default=1)
    parser.add_argument("--il-end", type=int)
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
