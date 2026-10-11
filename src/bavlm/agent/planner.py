"""Planner -- turns a query into an observation plan.

Roadmap phase 3. The contribution lives here: `spans`, `fps` and `resolution`
are chosen from the question, within a fixed frame budget.
"""
from __future__ import annotations

import json
import pathlib

from bavlm.data.schema import Plan, Span

_PROMPT = (pathlib.Path(__file__).parent / "prompts" / "planner.txt").read_text()


def build_prompt(question: str, duration: float, budget: int, history: str = "") -> str:
    return _PROMPT.format(question=question, duration=duration, budget=budget,
                          history=f"\nPrevious rounds:\n{history}" if history else "")


def parse(raw: str, budget: int) -> Plan:
    """Parse the planner's JSON reply into a Plan, clamped to the budget."""
    obj = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    spans = [Span(float(a), float(b)) for a, b in obj["spans"]]
    plan = Plan(
        what=obj["what"],
        spans=spans,
        fps=min(float(obj.get("fps", 2.0)), 25.0),
        resolution=obj.get("resolution", "low"),
        crop=obj.get("crop", "full"),
        use_pose=bool(obj.get("use_pose", False)),
        budget=budget,
    )
    # the budget is a hard cap -- thin fps rather than trusting the model
    est = plan.estimated_frames()
    if est > budget and est > 0:
        plan.fps = plan.fps * budget / est
    return plan
