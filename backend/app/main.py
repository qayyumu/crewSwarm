"""FastAPI server: chat endpoint that streams swarm events as SSE.

A conversation is a pheromone board. Every question in the same session
keeps building on the same evaporating trail, so follow-ups cost one short
round instead of a full fresh search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .engine import run_swarm
from .pheromones import store

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

app = FastAPI(title="crewSwarm — predict anything", version="0.1.0")

# last question per session, so follow-ups can forage with context
_last_question: dict[str, str] = {}


class PredictRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    mode: str = "auto"  # auto | heuristic | openai | gemini
    simulate: bool = True


def _resolve_mode(requested: str) -> tuple[Optional[bool], str, Optional[str]]:
    """Map the UI's mode choice onto engine settings.

    Returns (use_llm, provider, notice): use_llm None = auto-detect; notice
    is set when a requested mode had to fall back.
    """
    from .llm_adapter import crewai_status, default_provider

    if requested == "heuristic":
        return False, "auto", None
    status = crewai_status()
    if requested in ("openai", "gemini"):
        if status["available"] and status["providers"].get(requested):
            return True, requested, None
        reason = "crewai not installed" if not status["available"] else f"no {requested.upper()}_API_KEY set"
        return False, "auto", f"{requested} mode unavailable ({reason}) — fell back to the heuristic engine."
    # auto
    if status["available"]:
        return None, default_provider() or "auto", None
    return None, "auto", None


@app.get("/")
def index():
    return FileResponse(f"{FRONTEND_DIR}/index.html")


@app.get("/api/health")
def health():
    from .llm_adapter import crewai_status

    return {"ok": True, "crewai": crewai_status()}


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Non-streaming fallback: collects the full swarm run into one JSON."""
    import asyncio

    sid, board = store.get(req.session_id)
    prior = _last_question.get(sid, "")
    _last_question[sid] = req.question.strip()
    use_llm, provider, notice = _resolve_mode(req.mode)
    events = []
    if notice:
        events.append({"type": "notice", "message": notice})
    async def _collect():
        async for ev in run_swarm(req.question.strip(), board, use_llm=use_llm,
                                  provider=provider, prior_question=prior,
                                  simulate=req.simulate):
            events.append(ev)

    asyncio.run(_collect())
    return {"session_id": sid, "events": events}


@app.get("/api/predict/stream")
def predict_stream(q: str = Query(..., min_length=3), session_id: Optional[str] = None,
                   mode: str = Query("auto", pattern="^(auto|heuristic|openai|gemini)$"),
                   simulate: bool = Query(True)):
    sid, board = store.get(session_id)
    use_llm, provider, notice = _resolve_mode(mode)

    async def gen():
        prior = _last_question.get(sid, "")
        _last_question[sid] = q.strip()
        yield f"event: session\ndata: {json.dumps({'session_id': sid})}\n\n"
        if notice:
            yield f"data: {json.dumps({'type': 'notice', 'message': notice})}\n\n"
        async for ev in run_swarm(q.strip(), board, use_llm=use_llm,
                                  provider=provider, prior_question=prior,
                                  simulate=simulate):
            yield f"data: {json.dumps(ev)}\n\n"
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/board/{session_id}")
def board_state(session_id: str):
    _, board = store.get(session_id)
    return {
        "session_id": session_id,
        "stats": board.stats(),
        "bits": [b.to_dict() for b in board.bits],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
