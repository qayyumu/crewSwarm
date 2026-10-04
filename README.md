# crewSwarm 🐜 — predict anything

A multi-agent research swarm in the spirit of [MiroFish](https://mirofish.homes),
with an ant-colony twist: agents forage the web and communicate **only**
through short, evaporating pheromone trails — no agent ever sees another
agent's full transcript.

Chat with it like ChatGPT: ask it to predict anything, watch the colony
forage live, and keep asking follow-ups on the same trail.

![Main UI — chat with the swarm while agents forage](main_UI.png)

![Research results — oracle prediction with confidence and key signals](research_results.png)

## The swarm

| Agent    | Role                                                                 |
|----------|----------------------------------------------------------------------|
| scout    | searches the web, lays link pheromones                               |
| scraper  | follows the strongest link trails, drags page content back           |
| analyst  | distills content into signal pheromones                              |
| skeptic  | marks risky / contradicted trails                                    |
| oracle   | reads the whole board once, commits to a prediction with confidence  |

After the prediction, an optional **persona colony** (`backend/app/simulation.py`)
reacts to the forecast: five personas (cast per-question by the LLM, or templated
archetypes in heuristic mode — investor, regulator, analyst, competitor,
advocate) post reactions on a fresh board seeded with the prediction itself.
Same rules: each persona only smells a small window of the trail, aligned
reactions reinforce into coalitions, everything evaporates between rounds.
The second oracle pass forecasts the narrative: who supports, who resists,
who reframes, the likely trajectory, and the amplification risk of a backlash
cascade. Toggle it with the "simulate audience reactions" checkbox.

**Sparse information sharing.** Each forager can only *sniff* the top few
pheromones its role is tuned for (`backend/app/pheromones.py` → `ROLE_SCENTS`),
ranked by strength. **Evaporation.** Every round all pheromones decay and
fully-faded bits are pruned, so only reinforced information survives —
exactly how ant trails work.

## Run it

```bash
./run.sh          # → http://127.0.0.1:8000
```

No API key needed — the default **heuristic engine** does real web search and
scraping with rule-based analysis.

### Optional: LLM-driven agents via CrewAI

```bash
.venv/bin/pip install "crewai[google-genai]"   # the google-genai extra enables Gemini
# then put one or both in .env:
#   OPENAI_API_KEY=sk-...
#   GEMINI_API_KEY=...        (or GOOGLE_API_KEY)
./run.sh
```

With crewai installed and a provider key set, the same pheromone loop is driven by
CrewAI agents (classic Agent/Task/Crew API, `backend/app/llm_adapter.py`) —
each agent still only perceives the board through the `sniff_trail` tool.
Models default to `gpt-4o-mini` and `gemini/gemini-2.5-flash`; override with
`OPENAI_MODEL` / `GEMINI_MODEL`. If the LLM fails at any step, the colony
falls back gracefully.

The UI engine selector (top right) lists each provider separately: **auto**
picks OpenAI → Gemini → heuristic in that order, or force any of them.
Forcing a provider without its key streams a notice and falls back to the
heuristic engine for that run only.

## Layout

```
backend/app/
  pheromones.py   the shared trail: deposit / sniff / evaporate
  tools.py        keyless search (ddgs + DDG HTML fallback) and scraping
  engine.py       the swarm loop + heuristic oracle + simulation merge
  simulation.py   persona reaction colony (heuristic + LLM casting)
  llm_adapter.py  CrewAI mode (optional)
  main.py         FastAPI, SSE streaming
frontend/
  index.html      ChatGPT-style UI, streams the colony live
```

## API

- `GET  /api/predict/stream?q=...&session_id=...&mode=auto|heuristic|openai|gemini&simulate=1` — SSE event stream
- `POST /api/predict` `{question, session_id?, mode?}` — one-shot JSON
- `GET  /api/board/{session_id}` — inspect the pheromone board
- `GET  /api/health` — crewai availability + reason

A `session_id` is a pheromone board: follow-up questions keep foraging on
the same evaporating trail.
