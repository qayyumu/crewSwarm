"""CrewAI adapter: drive the same pheromone loop with LLM agents.

Used only when crewai is importable AND OPENAI_API_KEY is set. Each forager
is a CrewAI agent whose only window into the rest of the swarm is the
PheromoneBoard (via the sniff/deposit tools) — same sparse communication
as the heuristic engine. The oracle agent reads the whole board and writes
the prediction report.

Written against the classic Agent/Task/Crew API, which is stable from
crewai 0.11 through current releases. Any failure here falls back to the
heuristic engine, so the app never hard-depends on an LLM.
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator, Optional

from .pheromones import PheromoneBoard
from . import tools


def crewai_available() -> bool:
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        import crewai  # noqa: F401

        return True
    except Exception:
        return False


def _make_tools(board: PheromoneBoard, role: str):
    def sniff_trail(k: int = 5) -> str:
        """Smell the pheromone trail: returns the few info bits your role can perceive."""
        bits = board.sniff(role, k=int(k))
        if not bits:
            return "(the trail is empty)"
        return "\n---\n".join(
            f"[{b.id}] round {b.round} by {b.agent}: {b.content}"
            + (f" (url: {b.url})" if b.url else "")
            for b in bits
        )

    def deposit_bit(content: str, url: str = "", tags: str = "", strength: float = 1.0) -> str:
        """Lay down a pheromone bit for other agents to find. Keep it under 120 words."""
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        bit = board.deposit(role, role, content[:800], url=url or None,
                            tags=tag_list, strength=float(strength))
        return f"deposited bit {bit.id}"

    def search(query: str) -> str:
        """Search the web. Returns titles, urls and snippets."""
        rs = tools.search_web(query, max_results=5)
        if not rs:
            return "(no results)"
        return "\n".join(f"- {r.title} | {r.url} | {r.snippet[:200]}" for r in rs)

    def fetch(url: str) -> str:
        """Fetch and extract readable text from a page."""
        text = tools.scrape_url(url)
        return text[:4000] or "(could not extract text)"

    return [
        sniff_trail, deposit_bit, search, fetch,
    ]


AGENT_DEFS = {
    "scout": {
        "goal": "Find the most relevant and recent web sources about the user's question.",
        "backstory": "You are a scout ant. You leave the nest, search widely, and lay "
                     "link pheromones. You never analyze; you only find and mark trails.",
    },
    "scraper": {
        "goal": "Fetch the strongest link trails and extract the hard facts and quotes inside.",
        "backstory": "You are a worker ant following the scout's pheromone trails. You drag "
                     "raw content back to the colony. You never speculate; extract only what "
                     "the page actually says.",
    },
    "analyst": {
        "goal": "Distill scraped content into the key signals that bear on the prediction.",
        "backstory": "You are an ant that smells content trails and refines them into signal "
                     "pheromones. Drop anything that does not change the answer.",
    },
    "skeptic": {
        "goal": "Find counter-evidence, uncertainty and risks in the signal trails.",
        "backstory": "You are a guard ant. Your job is to mark trails that look weak, biased "
                     "or contradicted, so the colony does not follow a false scent.",
    },
}


async def run_llm_swarm(
    question: str,
    board: PheromoneBoard,
    rounds: int = 2,
) -> AsyncIterator[dict]:
    from crewai import Agent, Crew, Process, Task

    yield {"type": "start", "question": question, "rounds": rounds,
           "mode": "crewai", "terms": tools.key_terms(question)}

    def run_agent(role: str, task_desc: str) -> str:
        agent = Agent(
            role=role,
            goal=AGENT_DEFS[role]["goal"],
            backstory=AGENT_DEFS[role]["backstory"],
            tools=_make_tools(board, role),
            verbose=False,
            allow_delegation=False,
            max_iter=6,
        )
        task = Task(
            description=task_desc,
            expected_output="A concise result; anything important must be deposited "
                            "as pheromone bits via the deposit tool.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        return crew.kickoff()

    for r in range(rounds):
        yield {"type": "round", "round": r + 1}
        for role in ("scout", "scraper", "analyst", "skeptic"):
            yield {"type": "agent_start", "role": role}
            task_desc = {
                "scout": f"User's question: {question}\nSearch for strong sources, "
                         "then deposit 3-6 link pheromones with url tags.",
                "scraper": "Sniff the trail for link pheromones, fetch the most promising "
                           "ones, and deposit content pheromones with the key facts/quotes.",
                "analyst": "Sniff content pheromones and deposit signal pheromones: the "
                           "facts that most affect the answer to the user's question.",
                "skeptic": "Sniff signal pheromones and deposit risk pheromones wherever "
                           "you find counter-evidence, uncertainty or weak sourcing.",
            }[role]
            try:
                result = await __import__("asyncio").to_thread(run_agent, role, task_desc)
                yield {"type": "agent_done", "role": role, "result": str(result)[:400]}
            except Exception as e:  # LLM hiccup -> keep the colony moving
                yield {"type": "agent_error", "role": role, "error": str(e)[:200]}
            for bit in board.sniff("oracle", k=8):
                yield {"type": "pheromone", "role": role, "bit": bit.to_dict()}
            board.evaporate()
            yield {"type": "board", "stats": board.stats()}

    # oracle reads everything and predicts
    yield {"type": "agent_start", "role": "oracle"}
    trail = "\n---\n".join(
        f"[{b.id}] round {b.round} by {b.agent}: {b.content}"
        + (f" (url: {b.url})" if b.url else "")
        for b in board.sniff("oracle", k=60)
    )
    oracle = Agent(
        role="oracle",
        goal="Read the entire pheromone board and write the final prediction report.",
        backstory="You are the colony's brain. You see every trail the foragers laid. "
                  "You synthesize, weigh the skeptic's marks, and commit to a prediction "
                  "with an honest confidence level.",
        verbose=False,
        allow_delegation=False,
        max_iter=4,
    )
    task = Task(
        description=(
            f"User's question: {question}\n\nPheromone board:\n{trail}\n\n"
            "Write a JSON report with keys: summary (string), confidence "
            "(integer 0-100), signals (array of {text, url}), risks (array of "
            "{text, url}), followups (array of strings, exactly 2). "
            "Output ONLY JSON."
        ),
        expected_output="A JSON object only.",
        agent=oracle,
    )
    try:
        out = await __import__("asyncio").to_thread(
            lambda: Crew(agents=[oracle], tasks=[task], process=Process.sequential,
                         verbose=False).kickoff()
        )
        report = json.loads(str(out).strip().removeprefix("```json").removesuffix("```").strip())
    except Exception as e:
        yield {"type": "agent_error", "role": "oracle", "error": str(e)[:200]}
        report = {
            "summary": "The oracle could not reach the LLM. The board below is the raw "
                       "trail the swarm collected.",
            "confidence": 0,
            "signals": [], "risks": [],
            "followups": ["Retry with the LLM configured."],
        }
    report["question"] = question
    report["sources"] = [
        {"url": b.url, "title": b.content[:80]} for b in board.sniff("oracle", k=40) if b.url
    ][:10]
    report["board_stats"] = board.stats()
    yield {"type": "report", "report": report}
    yield {"type": "done", "board_stats": board.stats()}
