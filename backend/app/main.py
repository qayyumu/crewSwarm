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


@app.get("/")
def index():
    return FileResponse(f"{FRONTEND_DIR}/index.html")


@app.get("/api/health")
def health():
    from .llm_adapter import crewai_available

    return {"ok": True, "crewai_mode": crewai_available()}


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Non-streaming fallback: collects the full swarm run into one JSON."""
    import asyncio

    sid, board = store.get(req.session_id)
    prior = _last_question.get(sid, "")
    _last_question[sid] = req.question.strip()
    events = []
    async def _collect():
        async for ev in run_swarm(req.question.strip(), board, prior_question=prior):
            events.append(ev)

    asyncio.run(_collect())
    return {"session_id": sid, "events": events}


@app.get("/api/predict/stream")
def predict_stream(q: str = Query(..., min_length=3), session_id: Optional[str] = None):
    sid, board = store.get(session_id)

    async def gen():
        prior = _last_question.get(sid, "")
        _last_question[sid] = q.strip()
        yield f"event: session\ndata: {json.dumps({'session_id': sid})}\n\n"
        async for ev in run_swarm(q.strip(), board, prior_question=prior):
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
