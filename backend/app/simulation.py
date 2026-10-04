"""Persona simulation swarm — the second colony.

After the oracle commits to a prediction, a cast of personas (retail
investor, regulator, journalist...) reacts to the forecast on a fresh
pheromone board seeded with the prediction itself. Same rules as the
foragers: each persona only sniffs a small window of the trail, reacts in
character, and everything evaporates between rounds. Aligned reactions
reinforce each other, so coalitions emerge the same way ant trails do.

The result is a `reactions` section for the report: who supports, who
resists, who reframes, where the narrative is likely to turn, and how
likely a backlash cascade is.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from .pheromones import PheromoneBoard
from . import tools

SIM_ROUNDS = 2


@dataclass
class Persona:
    name: str
    stance: str  # support | resist | reframe
    angle: str   # what this persona cares about


ARCHETYPES = [
    ("retail investor", "support",
     "If {t1} momentum holds and the {t2} catalysts land, this forecast plays "
     "out. I'm watching valuation, not vibes."),
    ("regulator", "resist",
     "This prediction underestimates oversight risk around {t1}. Expect "
     "scrutiny before any {t2} milestone is reached."),
    ("industry analyst", "reframe",
     "The data supports {t1}, but the real story is {t2} — the timeline "
     "assumes nothing breaks there."),
    ("competitor strategist", "resist",
     "We would actively exploit the gap in {t1}. That competitive pressure "
     "is exactly why this slips."),
    ("customer advocate", "support",
     "From the user side the {t1} momentum is real — communities already "
     "behave as if the forecast is right."),
]

FALLBACK_TERMS = ["the forecast", "the timeline"]


def _terms(question: str) -> list[str]:
    t = tools.key_terms(question)
    return (t + FALLBACK_TERMS)[:2]


def heuristic_personas(question: str) -> list[Persona]:
    t1, t2 = _terms(question)
    return [Persona(name=n, stance=s, angle=a.format(t1=t1, t2=t2))
            for n, s, a in ARCHETYPES]


def _seed_board(sim: PheromoneBoard, report: dict) -> None:
    """Plant the forecast itself on the trail so personas react to it."""
    sim.deposit("oracle", "oracle",
                f"Prediction: {report.get('summary', '')[:400]}",
                tags=["claim"], strength=2.5)
    for s in report.get("signals", [])[:4]:
        sim.deposit("analyst", "analyst", s.get("text", "")[:300],
                    url=s.get("url"), tags=["signal"], strength=2.0)
    for r in report.get("risks", [])[:3]:
        sim.deposit("skeptic", "skeptic", r.get("text", "")[:300],
                    url=r.get("url"), tags=["risk"], strength=2.0)


def _heuristic_reaction(persona: Persona, sim: PheromoneBoard,
                        round_no: int, question: str) -> list[dict]:
    """Template a reaction: round 1 responds to the forecast, round 2 to
    whichever persona reactions are strong enough to be smelled."""
    events: list[dict] = []
    if round_no == 1:
        text = persona.angle
        if persona.stance == "resist":
            risks = sorted((b for b in sim.bits if "risk" in b.tags),
                           key=lambda b: -b.strength)[:1]
            if risks:
                text += f" Exhibit A: {risks[0].content[:110]}"
        bit = sim.deposit(f"persona:{persona.name}", "persona", text[:400],
                          tags=["reaction", persona.stance], strength=1.4)
        events.append({"type": "reaction", "persona": persona.name,
                       "stance": persona.stance, "bit": bit.to_dict()})
        return events

    visible = sorted(
        (b for b in sim.bits
         if "reaction" in b.tags and b.agent != f"persona:{persona.name}"),
        key=lambda b: -b.strength)[:3]
    if visible:
        other = visible[0]
        agree = other.tags and other.tags[-1] == persona.stance
        verb = "Echoing" if agree else "Pushing back on"
        gist = other.content[:90]
        for prefix in ("Pushing back on the trail:", "Echoing the trail:"):
            if gist.startswith(prefix):
                gist = gist.split(":", 1)[1].strip()[:90]
        text = f"{verb} the trail: {gist} — but from where I sit, {persona.angle[:120]}"
        bit = sim.deposit(f"persona:{persona.name}", "persona", text[:400],
                          tags=["reaction", persona.stance], strength=1.2)
        events.append({"type": "reaction", "persona": persona.name,
                       "stance": persona.stance, "bit": bit.to_dict()})
    return events


def _reinforce_coalitions(sim: PheromoneBoard) -> None:
    """Aligned reactions reinforce each other — this is where clusters form."""
    bits = [b for b in sim.bits if "reaction" in b.tags]
    for i, a in enumerate(bits):
        for b in bits[i + 1:]:
            if a.agent != b.agent and a.tags and b.tags and a.tags[-1] == b.tags[-1]:
                sim.reinforce(a.id, 0.3)
                sim.reinforce(b.id, 0.3)


def aggregate(sim: PheromoneBoard, personas: list[Persona],
              resist_by_round: list[float], terms: list[str]) -> dict:
    bits = [b for b in sim.sniff("oracle", k=100) if "reaction" in b.tags]

    def top(stance: str) -> list[dict]:
        sel = sorted((b for b in bits if stance in b.tags),
                     key=lambda b: -b.strength)[:3]
        return [{"text": b.content[:220], "by": b.agent.replace("persona:", "")}
                for b in sel]

    support_n = sum(1 for b in bits if "support" in b.tags)
    resist_n = sum(1 for b in bits if "resist" in b.tags)
    t1 = terms[0] if terms else "the key claim"
    if resist_n > support_n:
        trajectory = (f"Narrative likely polarizes: resistance ({resist_n}) outnumbers "
                      f"support ({support_n}). The forecast survives only if {t1} "
                      "evidence lands early and visibly.")
    elif support_n > resist_n:
        trajectory = (f"Narrative likely consolidates around the forecast "
                      f"({support_n} support vs {resist_n} resistance), unless a "
                      f"single {t1} shock reframes the story.")
    else:
        trajectory = (f"Narrative stays contested ({support_n} vs {resist_n}). "
                      f"Watch which side first recruits a credible {t1} voice.")

    growing = len(resist_by_round) > 1 and resist_by_round[-1] > resist_by_round[0] * 1.2
    risk = "high" if growing and resist_n >= support_n else (
        "moderate" if resist_n else "low")

    return {
        "personas": [{"name": p.name, "stance": p.stance, "angle": p.angle}
                     for p in personas],
        "counts": {"support": support_n, "resist": resist_n,
                   "reframe": sum(1 for b in bits if "reframe" in b.tags)},
        "support": top("support"),
        "resist": top("resist"),
        "reframe": top("reframe"),
        "trajectory": trajectory,
        "amplification_risk": risk,
        "board_stats": sim.stats(),
    }


# ----------------------------------------------------------------------
# LLM-driven simulation
# ----------------------------------------------------------------------

def _make_llm(provider: str):
    from .llm_adapter import _make_llm as make

    return make(provider)


def _llm_persona_tools(sim: PheromoneBoard, persona_name: str):
    from crewai.tools import tool

    @tool("sniff_trail")
    def sniff_trail(k: int = 5) -> str:
        """Smell the trail: prediction claims, evidence and other personas' reactions."""
        bits = sorted(
            (b for b in sim.bits if b.strength > 0.05),
            key=lambda b: (0 if "reaction" in b.tags else 1, -b.strength),
        )[: int(k)]
        if not bits:
            return "(the trail is empty)"
        return "\n---\n".join(
            f"[{b.id}] by {b.agent}: {b.content}" + (f" (url: {b.url})" if b.url else "")
            for b in bits
        )

    @tool("deposit_reaction")
    def deposit_reaction(content: str, sentiment: str = "", strength: float = 1.2) -> str:
        """Post your reaction so other personas can smell it. Under 60 words, in character."""
        tags = ["reaction"] + ([sentiment] if sentiment else [])
        bit = sim.deposit(f"persona:{persona_name}", "persona", content[:500],
                          tags=tags, strength=float(strength))
        return f"posted reaction {bit.id}"

    return [sniff_trail, deposit_reaction]


def _cast_personas(question: str, report: dict, llm) -> list[Persona]:
    """One quick LLM call to cast 5 personas matched to the question."""
    from crewai import Agent, Crew, Process, Task

    agent = Agent(
        role="casting director",
        goal="Design a diverse cast of 5 personas who would react publicly to a prediction.",
        backstory="You cast simulated audiences. Each persona must have a distinct "
                  "stake, a clear stance, and one-sentence angle.",
        llm=llm, verbose=False, allow_delegation=False, max_iter=3,
    )
    task = Task(
        description=(
            f"Question: {question}\nPrediction summary: {report.get('summary', '')[:300]}\n\n"
            "Output ONLY a JSON list of 5 objects: "
            '{"name": str, "stance": one of support|resist|reframe, "angle": str}. '
            "Pick varied real-world roles (e.g. investor, regulator, journalist, "
            "competitor, affected community member)."
        ),
        expected_output="A JSON list only.", agent=agent,
    )
    try:
        out = Crew(agents=[agent], tasks=[task], process=Process.sequential,
                   verbose=False).kickoff()
        raw = str(out).strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw[raw.index("["):raw.rindex("]") + 1])
        personas = [Persona(name=d["name"][:40], stance=d.get("stance", "reframe"),
                            angle=d.get("angle", "")[:300]) for d in data[:6]]
        if personas:
            return personas
    except Exception:
        pass
    return heuristic_personas(question)


def _persona_kickoff(persona: Persona, sim: PheromoneBoard, llm,
                     question: str, summary: str, round_no: int) -> None:
    from crewai import Agent, Crew, Process, Task

    agent = Agent(
        role=persona.name,
        goal=f"React publicly to a prediction about: {question}. "
             f"Your stake: {persona.angle}",
        backstory=f"You are {persona.name}. You tend to {persona.stance} predictions "
                  "like this one, but you react to what you actually see on the trail, "
                  "not a script.",
        tools=_llm_persona_tools(sim, persona.name),
        llm=llm, verbose=False, allow_delegation=False, max_iter=4,
    )
    context = ("Round 1: react to the prediction itself." if round_no == 1 else
               "Round 2: sniff the trail first — react to other personas' reactions "
               "you disagree or agree with, and reinforce your camp.")
    task = Task(
        description=(
            f"Prediction: {summary[:400]}\n{context}\n"
            f"Deposit 1-2 reactions (sentiment='{persona.stance}'). Stay in character, "
            "under 60 words each, no hedging."
        ),
        expected_output="Reactions posted via the deposit tool.",
        agent=agent,
    )
    Crew(agents=[agent], tasks=[task], process=Process.sequential,
         verbose=False).kickoff()


def _llm_aggregate(sim: PheromoneBoard, personas: list[Persona], llm,
                   question: str, fallback: dict) -> dict:
    from crewai import Agent, Crew, Process, Task

    trail = "\n---\n".join(
        f"by {b.agent} [{' '.join(b.tags)}]: {b.content[:200]}"
        for b in sim.sniff("oracle", k=80) if "reaction" in b.tags
    )
    agent = Agent(
        role="narrative forecaster",
        goal="Read the persona reaction trail and forecast how the narrative develops.",
        backstory="You read crowds the way meteorologists read pressure systems.",
        llm=llm, verbose=False, allow_delegation=False, max_iter=3,
    )
    task = Task(
        description=(
            f"Question: {question}\nReactions:\n{trail}\n\n"
            "Output ONLY JSON: counts ({support:int, resist:int, reframe:int}), "
            "support/resist/reframe (arrays of {text, by}, top 3 each), trajectory "
            "(string), amplification_risk (high|moderate|low)."
        ),
        expected_output="A JSON object only.", agent=agent,
    )
    try:
        out = Crew(agents=[agent], tasks=[task], process=Process.sequential,
                   verbose=False).kickoff()
        raw = str(out).strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        return {
            "personas": [{"name": p.name, "stance": p.stance, "angle": p.angle}
                         for p in personas],
            "counts": data.get("counts", fallback["counts"]),
            "support": data.get("support", fallback["support"]),
            "resist": data.get("resist", fallback["resist"]),
            "reframe": data.get("reframe", fallback["reframe"]),
            "trajectory": data.get("trajectory", fallback["trajectory"]),
            "amplification_risk": data.get("amplification_risk",
                                           fallback["amplification_risk"]),
            "board_stats": sim.stats(),
        }
    except Exception:
        return fallback


# ----------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------

async def run_simulation(
    question: str,
    report: dict,
    use_llm: bool = False,
    provider: str = "openai",
) -> AsyncIterator[dict]:
    """Run the persona colony; final event is sim_done with the reactions."""
    terms = _terms(question)
    sim = PheromoneBoard(evaporation=0.6)
    _seed_board(sim, report)

    if use_llm:
        try:
            llm = _make_llm(provider)
            personas = await asyncio.to_thread(_cast_personas, question, report, llm)
        except Exception:
            personas = heuristic_personas(question)
    else:
        personas = heuristic_personas(question)

    yield {"type": "sim_start", "rounds": SIM_ROUNDS,
           "personas": [{"name": p.name, "stance": p.stance} for p in personas]}

    resist_by_round: list[float] = []
    summary = report.get("summary", "")

    for r in range(SIM_ROUNDS):
        yield {"type": "sim_round", "round": r + 1}
        for persona in personas:
            yield {"type": "persona_start", "name": persona.name, "stance": persona.stance}
            if use_llm:
                try:
                    before = {b.id for b in sim.bits}
                    await asyncio.to_thread(_persona_kickoff, persona, sim, llm,
                                            question, summary, r + 1)
                    for b in sim.bits:
                        if b.id not in before and b.agent == f"persona:{persona.name}":
                            yield {"type": "reaction", "persona": persona.name,
                                   "stance": persona.stance, "bit": b.to_dict()}
                except Exception as e:
                    yield {"type": "agent_error", "role": f"persona:{persona.name}",
                           "error": str(e)[:200]}
                    for ev in _heuristic_reaction(persona, sim, r + 1, question):
                        yield ev
            else:
                for ev in _heuristic_reaction(persona, sim, r + 1, question):
                    yield ev
            await asyncio.sleep(0)
        _reinforce_coalitions(sim)
        resist_by_round.append(sum(b.strength for b in sim.bits if "resist" in b.tags))
        sim.evaporate()

    fallback = aggregate(sim, personas, resist_by_round, terms)
    if use_llm:
        try:
            reactions = await asyncio.to_thread(_llm_aggregate, sim, personas, llm,
                                                question, fallback)
        except Exception:
            reactions = fallback
    else:
        reactions = fallback

    yield {"type": "sim_done", "reactions": reactions}
