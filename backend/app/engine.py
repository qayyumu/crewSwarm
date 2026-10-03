"""The ant-pheromone swarm engine.

Five roles, mirroring a foraging ant colony:

  scout    — leaves the nest, searches the web, lays link pheromones
  scraper  — follows the strongest link trails, drags content back
  analyst  — smells the content trails, distills signal pheromones
  skeptic  — patrols the signal trails, marks risky/deceptive ones
  oracle   — the colony brain; reads the whole board once and predicts

Information sharing is deliberately sparse: each agent only *sniffs* the
top-k pheromones its role is tuned for, and every round the board
evaporates so only reinforced information survives.

The engine has two modes:
  - heuristic (default, no API key needed): real searching/scraping, with
    rule-based extraction and aggregation
  - crewai LLM mode (OPENAI_API_KEY set + crewai installed): the same loop
    and the same board, but each forager is a CrewAI agent that decides
    what to read and what to lay down.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

from .pheromones import PheromoneBoard
from . import tools
from .llm_adapter import crewai_available, run_llm_swarm

Event = dict


def _sentences(text: str) -> list[str]:
    out, buf = [], ""
    for ch in text:
        buf += ch
        if ch in ".!?\n":
            s = buf.strip()
            if len(s) > 40:
                out.append(s)
            buf = ""
    if len(buf.strip()) > 40:
        out.append(buf.strip())
    return out


def _best_sentences(text: str, terms: list[str], k: int = 3) -> list[str]:
    scored = []
    for s in _sentences(text):
        low = s.lower()
        score = sum(low.count(t) for t in terms) + min(len(s) / 500, 1)
        scored.append((score, s))
    scored.sort(key=lambda x: -x[0])
    return [s for sc, s in scored[:k] if sc > 0]


class Swarm:
    def __init__(self, board: PheromoneBoard, question: str,
                 prior_question: str = ""):
        self.board = board
        self.question = question
        prior_terms = tools.key_terms(prior_question) if prior_question else []
        self.terms = tools.key_terms(question) + [t for t in prior_terms
                                                  if t not in tools.key_terms(question)][:4]
        self.sources: dict[str, str] = {}  # url -> title

    # ------------------------------------------------------------------
    # foragers
    # ------------------------------------------------------------------
    def scout(self) -> list[Event]:
        events: list[Event] = []
        focus = " ".join(self.terms[:4])
        queries = [
            f"{self.question} {focus}".strip(),
            f"{focus} latest news",
            f"{focus} analysis forecast",
        ]
        seen = set()
        for q in dict.fromkeys(queries):
            for r in tools.search_web(q, max_results=4):
                if r.url in seen:
                    continue
                seen.add(r.url)
                self.sources[r.url] = r.title
                bit = self.board.deposit(
                    "scout", "scout",
                    f"{r.title} — {r.snippet[:240]}",
                    url=r.url, tags=["link", "source", "lead"],
                )
                events.append({"type": "pheromone", "role": "scout", "bit": bit.to_dict()})
            tools.polite_sleep(0.3)
        return events

    def scraper(self) -> list[Event]:
        events: list[Event] = []
        trails = [b for b in self.board.sniff("scraper", k=4) if b.url]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=4) as pool:
            texts = list(pool.map(tools.scrape_url, [b.url for b in trails]))
        for bit, text in zip(trails, texts):
            if not text:
                continue
            keep = _best_sentences(text, self.terms, k=3)
            if not keep:
                keep = _sentences(text)[:2]
            content = " ".join(keep)[:900] or text[:400]
            new = self.board.deposit(
                "scraper", "scraper", content,
                url=bit.url, tags=["content", "data"], strength=1.2,
            )
            events.append({"type": "pheromone", "role": "scraper", "bit": new.to_dict()})
        return events

    def analyst(self) -> list[Event]:
        events: list[Event] = []
        for bit in self.board.sniff("analyst", k=4):
            keep = _best_sentences(bit.content, self.terms, k=2)
            for s in keep:
                sig = self.board.deposit(
                    "analyst", "analyst", s[:400],
                    url=bit.url, tags=["signal", "fact"], strength=1.1,
                )
                events.append({"type": "pheromone", "role": "analyst", "bit": sig.to_dict()})
        return events

    def skeptic(self) -> list[Event]:
        events: list[Event] = []
        markers = ("however", "but ", "risk", "uncertain", "unknown", "decline",
                   "fall", "drop", "concern", "warn", "despite", "although",
                   "lawsuit", "ban", "probe", "short", "fraud")
        for bit in self.board.sniff("skeptic", k=5):
            low = bit.content.lower()
            hits = [m for m in markers if m in low]
            if hits:
                risk = self.board.deposit(
                    "skeptic", "skeptic", bit.content[:400],
                    url=bit.url, tags=["risk", "claim"], strength=1.3,
                )
                events.append({"type": "pheromone", "role": "skeptic", "bit": risk.to_dict()})
        return events

    # ------------------------------------------------------------------
    # oracle
    # ------------------------------------------------------------------
    def oracle(self) -> dict:
        all_bits = self.board.sniff("oracle", k=50)
        signals = [b for b in all_bits if "signal" in b.tags or b.role == "analyst"]
        risks = [b for b in all_bits if "risk" in b.tags]
        content = [b for b in all_bits if b.role == "scraper"]

        # heuristic confidence: source coverage × signal strength − risk load.
        # Deliberately capped low — a keyword-weighted gut feel, not a probability.
        coverage = len({b.url for b in all_bits if b.url})
        signal_mass = sum(b.strength for b in signals)
        risk_mass = sum(b.strength for b in risks)
        raw = (0.20 + 0.07 * min(coverage, 5) + 0.25 * min(signal_mass / 8, 1)
               - 0.20 * min(risk_mass / 4, 1))
        confidence = int(max(15, min(78, round(raw * 100))))

        cited = []
        for b in sorted(signals, key=lambda b: -b.strength)[:6]:
            cited.append({"text": b.content[:280], "url": b.url,
                          "source": self.sources.get(b.url or "", "")})

        uncovered = [t for t in self.terms
                     if not any(t in b.content.lower() for b in all_bits)][:4]
        followups = [
            f"What happens to {t} if the main assumption here is wrong?"
            for t in uncovered[:2]
        ] or [
            "Which single source would most change this prediction if it flipped?",
            "What is the earliest signal that would tell us this is going wrong?",
        ]

        summary_bits = sorted(signals, key=lambda b: -b.strength)[:3]
        summary = " ".join(b.content[:220] for b in summary_bits)[:500] or (
            "The swarm found limited trail on this question. Treat the "
            "prediction as low-confidence and try rephrasing with more "
            "specific actors, events, or a time horizon."
        )

        return {
            "question": self.question,
            "summary": summary,
            "confidence": confidence,
            "signals": cited,
            "risks": [{"text": b.content[:240], "url": b.url} for b in risks[:5]],
            "sources": [{"url": u, "title": t} for u, t in list(self.sources.items())[:10]],
            "board_stats": self.board.stats(),
            "followups": followups,
        }


async def run_swarm(
    question: str,
    board: PheromoneBoard,
    rounds: int = 2,
    use_llm: Optional[bool] = None,
    prior_question: str = "",
) -> AsyncIterator[Event]:
    """Drive the colony round by round, yielding UI events."""
    if use_llm is None:
        use_llm = crewai_available()

    board.evaporate()  # start a fresh round on this board

    if use_llm:
        async for ev in run_llm_swarm(question, board, rounds=rounds):
            yield ev
        return

    swarm = Swarm(board, question, prior_question=prior_question)
    yield {"type": "start", "question": question, "rounds": rounds,
           "mode": "heuristic", "terms": swarm.terms}

    for r in range(rounds):
        yield {"type": "round", "round": r + 1}
        for role, fn in (("scout", swarm.scout), ("scraper", swarm.scraper),
                         ("analyst", swarm.analyst), ("skeptic", swarm.skeptic)):
            yield {"type": "agent_start", "role": role}
            try:
                for ev in fn():
                    yield ev
            except Exception as e:  # one forager failing shouldn't stop the colony
                yield {"type": "agent_error", "role": role, "error": str(e)[:200]}
            await asyncio.sleep(0)
        board.evaporate()
        yield {"type": "board", "stats": board.stats()}

    yield {"type": "agent_start", "role": "oracle"}
    await asyncio.sleep(0.05)
    yield {"type": "report", "report": swarm.oracle()}
    yield {"type": "done", "board_stats": board.stats()}
