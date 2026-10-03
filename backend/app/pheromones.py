"""Pheromone board: the swarm's only shared memory.

Agents never see each other's full transcripts. They deposit short
"pheromone" bits onto a shared trail, and other agents only get to
*sniff* a small, relevance-ranked subset of it. Bits evaporate (decay)
each round, so stale information fades unless it keeps getting
reinforced — the same way ant pheromone trails work.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Pheromone:
    id: str
    round: int
    agent: str
    role: str
    content: str
    url: Optional[str] = None
    strength: float = 1.0
    tags: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "round": self.round,
            "agent": self.agent,
            "role": self.role,
            "content": self.content,
            "url": self.url,
            "strength": round(self.strength, 3),
            "tags": self.tags,
        }


# Which pheromone "scents" each role is tuned to sniff. This is what makes
# information sharing sparse: a role only picks up bits whose tags or text
# overlap with its interests, ranked by strength.
ROLE_SCENTS = {
    "scout": {"link", "source", "lead"},
    "scraper": {"content", "quote", "data", "source"},
    "analyst": {"signal", "trend", "fact", "content"},
    "skeptic": {"signal", "claim", "risk"},
    "oracle": set(),  # oracle reads everything at the end
}


class PheromoneBoard:
    def __init__(self, evaporation: float = 0.55):
        self.bits: list[Pheromone] = []
        self.evaporation = evaporation
        self.round = 0
        self._lock = threading.Lock()

    def deposit(
        self,
        agent: str,
        role: str,
        content: str,
        url: Optional[str] = None,
        tags: Optional[list[str]] = None,
        strength: float = 1.0,
    ) -> Pheromone:
        bit = Pheromone(
            id=uuid.uuid4().hex[:8],
            round=self.round,
            agent=agent,
            role=role,
            content=content.strip(),
            url=url,
            strength=strength,
            tags=[t.lower() for t in (tags or [])],
        )
        with self._lock:
            self.bits.append(bit)
        return bit

    def reinforce(self, bit_id: str, amount: float = 0.5) -> None:
        with self._lock:
            for bit in self.bits:
                if bit.id == bit_id:
                    bit.strength = min(bit.strength + amount, 3.0)
                    return

    def evaporate(self) -> None:
        with self._lock:
            self.round += 1
            for bit in self.bits:
                bit.strength *= self.evaporation
            # prune fully evaporated bits
            self.bits = [b for b in self.bits if b.strength >= 0.05]

    def sniff(self, role: str, k: int = 5, exclude_role: bool = True) -> list[Pheromone]:
        """Return the top-k bits this role can perceive.

        A role perceives a bit only if it tags it, shares a scent keyword,
        or the bit is strong enough to be "in the air". The oracle bypasses
        this filter entirely.
        """
        with self._lock:
            candidates = [b for b in self.bits if b.strength > 0.05]
            if exclude_role:
                candidates = [b for b in candidates if b.role != role]

            if role == "oracle" or ROLE_SCENTS.get(role) == set():
                scored = sorted(candidates, key=lambda b: b.strength, reverse=True)
                return scored[: max(k * 4, 20)]

            scents = ROLE_SCENTS.get(role, set())
            scored = []
            for b in candidates:
                score = b.strength
                if any(t in scents for t in b.tags):
                    score += 1.0
                text = (b.content + " " + " ".join(b.tags)).lower()
                if any(s in text for s in scents):
                    score += 0.5
                scored.append((score, b))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [b for _, b in scored[:k]]

    def stats(self) -> dict:
        with self._lock:
            by_role: dict[str, int] = {}
            for b in self.bits:
                by_role[b.role] = by_role.get(b.role, 0) + 1
            return {"bits": len(self.bits), "round": self.round, "by_role": by_role}


class BoardStore:
    """Keeps one board per conversation so follow-up questions can keep
    building on the same evaporating trail."""

    def __init__(self):
        self._boards: dict[str, PheromoneBoard] = {}
        self._lock = threading.Lock()

    def get(self, session_id: Optional[str]) -> tuple[str, PheromoneBoard]:
        with self._lock:
            sid = session_id or uuid.uuid4().hex[:12]
            if sid not in self._boards:
                self._boards[sid] = PheromoneBoard()
            return sid, self._boards[sid]


store = BoardStore()
