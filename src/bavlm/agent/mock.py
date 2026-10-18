"""Deterministic mock for the three agent roles, so the loop demo and its
tests run on a laptop with no GPU (CLAUDE.md: mock mode first).

Why a separate mock instead of `models.backend.MockBackend`: the MCQ mock
speaks option letters; the planner, observer and reflector must speak JSON.
Keeping this in the agent package leaves `models/backend.py` untouched and
makes the boundary explicit -- this mock stands in for *judgement*, while the
plumbing under test (prompt build, JSON parse, budget clamp, span selection,
eid assignment, citation filtering, the round cap) is the real code.

Every reply is conditioned on the question, so the trace looks like what a
real loop produces: round 1 insufficient -> round 2 sufficient, with citations.
"""
from __future__ import annotations

import json
import pathlib
import random
import re

_REGIONS = ["mid front court", "mid court", "left front court", "right front court",
            "left mid court", "right mid court", "left back court",
            "right back court", "mid back court"]
_HITS = ["clear", "drop", "smash", "net shot", "lift", "drive"]


def _frame_time(path: str) -> float | None:
    parts = pathlib.Path(path).stem.split("_")
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1]) / 1000.0
    return None


def _span(start: float, end: float) -> list[float]:
    return [round(start, 2), round(max(end, start + 0.1), 2)]


class MockAgentBackend:
    """Answers the planner/observer/reflector prompts deterministically.

    answer_key maps the question TEXT to its option letter (the reflector is
    the role that produces the answer), so the demo can check extraction and
    citation plumbing while remaining honest: answers come from the key, not
    from any model. No accuracy may be claimed from a run of this mock.
    """
    name = "mock:agent"

    def __init__(self, answer_key: dict[str, str] | None = None, seed: int = 0) -> None:
        self.key = answer_key or {}
        self.rng = random.Random(seed)

    def generate(self, frames, prompt: str, **kw) -> str:
        if "You are the PLANNER" in prompt:
            return json.dumps(self._plan(prompt))
        if "You are the OBSERVER" in prompt:
            return json.dumps(self._observe(prompt, list(frames)))
        if "You are the REFLECTOR" in prompt:
            return json.dumps(self._reflect(prompt))
        raise ValueError("mock agent backend only answers the three agent roles")

    # ----------------------------------------------------------------- planner
    def _plan(self, prompt: str) -> dict:
        m = re.search(r"Rally length: ([\d.]+) seconds", prompt)
        dur = max(float(m.group(1)) if m else 10.0, 1.0)
        m = re.search(r"Frame budget for this round: (\d+)", prompt)
        budget = int(m.group(1)) if m else 25
        m = re.search(r"Question: (.*)", prompt)
        q = (m.group(1) if m else "").lower()
        second = "Previous rounds:" in prompt

        def plan(what: str, spans: list[list[float]], res: str = "low",
                 crop: str = "full") -> dict:
            total = sum(e - s for s, e in spans)
            fps = round(min(25.0, budget / total), 2) if total > 0 else 1.0
            return {"what": what, "spans": spans, "fps": fps,
                    "resolution": res, "crop": crop, "use_pose": False}

        if any(w in q for w in ("how many", "number of")):
            if not second:
                # counting wants broad coverage, low fps (planner guidance)
                return plan("every racket contact in the rally, for counting",
                            [_span(0.0, dur)])
            return plan("a recount of the contacts after the midpoint",
                        [_span(dur * 0.5, dur)])
        if "why" in q or "reason" in q:
            if not second:
                return plan("the rally outcome, coarse, across the whole rally",
                            [_span(0.0, dur)])
            return plan("the decisive shot at full detail", [_span(dur * 0.6, dur)],
                        res="high")
        if any(w in q for w in ("where", "position", "side", "court")):
            if not second:
                return plan("where the shuttle and players are", [_span(0.0, dur)],
                            res="high")
            return plan("the players' positions around contact",
                        [_span(dur * 0.3, dur * 0.8)], res="high", crop="player")
        # action / stroke identity: coarse first, then the contact window
        if not second:
            return plan("the stroke played, coarse, across the rally",
                        [_span(0.0, dur)])
        return plan("the racket-shuttle contact moment", [_span(dur * 0.4, dur * 0.7)],
                    res="high")

    # ----------------------------------------------------------------- observer
    def _observe(self, prompt: str, frames: list[str]) -> list[dict]:
        m = re.search(r"Seek exactly this: (.*)", prompt)
        what = m.group(1) if m else ""
        times = [t for t in (_frame_time(p) for p in frames) if t is not None]
        if len(times) >= 2:
            lo, hi = min(times), max(times) + 0.5
        else:
            lo, hi = 0.0, 3.0
        span = _span(lo, hi)

        def item(start: float, end: float, desc: str, **kw) -> dict:
            return {"start": round(start, 2), "end": round(end, 2),
                    "description": desc, "confidence": round(self.rng.uniform(0.6, 0.95), 2),
                    "hit_type": None, "court_region": None, **kw}

        if "count" in what or "recount" in what or "contact" in what:
            n = 3
            step = (hi - lo) / (n + 1)
            return [item(lo + step * (i + 1) - 0.2, lo + step * (i + 1) + 0.2,
                         f"racket meets the shuttle, contact {i + 1} visible",
                         hit_type=self.rng.choice(_HITS),
                         court_region=self.rng.choice(_REGIONS))
                    for i in range(n)]
        if "position" in what or "positions" in what or "where" in what:
            return [item(lo, hi,
                         "both players visible; shuttle over the mid court",
                         court_region=self.rng.choice(_REGIONS))]
        if "decisive" in what or "contact" in what:
            return [item(max(lo, hi - 0.6), hi,
                         "striker plays an attacking shot into the open court",
                         hit_type=self.rng.choice(_HITS),
                         court_region=self.rng.choice(_REGIONS))]
        return [item(lo, hi, "rally ends with the shuttle landing in court",
                     court_region=self.rng.choice(_REGIONS))]

    # ---------------------------------------------------------------- reflector
    def _reflect(self, prompt: str) -> dict:
        m = re.search(r"Round (\d+) of (\d+)", prompt)
        rnd, mx = (int(m.group(1)), int(m.group(2))) if m else (1, 3)
        m = re.search(r"Question: (.*)", prompt)
        q = m.group(1) if m else ""
        eids = re.findall(r"^(e\d+):", prompt, flags=re.M)

        # deterministic demo shape: round 1 always re-plans, round 2 answers --
        # the trace on the slide needs to SHOW the loop doing a second round.
        sufficient = bool(eids) and rnd >= 2
        if not sufficient:
            return {"sufficient": False, "confidence": 0.55,
                    "missing": "evidence covers only the opening; seek the later "
                               "part of the rally at higher density",
                    "answer": "", "cites": [], "explanation": ""}
        cites = eids[-2:] if eids else []
        answer = self.key.get(q, "")
        return {
            "sufficient": True, "confidence": 0.82, "missing": "",
            "answer": answer,
            "cites": cites,
            "explanation": f"The answer follows from {', '.join(cites)}: the "
                           "round-2 observations cover the window the first "
                           "round missed.",
        }
