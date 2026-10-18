"""Reflector -- sufficiency scoring, citation selection, re-plan signal.

The reflector is asked whether the evidence is actually there, not whether an
answer can be argued generously: a "sufficient" verdict with no evidence behind
it would end the loop early and quietly turn the agent back into a single-shot
baseline.

Citations are tested by removal in the faithfulness study -- listing every
evidence id is not rewarded, so the prompt (prompts/reflector.txt) says so and
the loop records dropped/unknown ids rather than trusting them.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

_PROMPT = (pathlib.Path(__file__).parent / "prompts" / "reflector.txt").read_text()


@dataclass
class Judgement:
    """One reflector reply. `cites` are eids the answer depends on; `missing`
    drives the next planner round when `sufficient` is False."""
    sufficient: bool
    confidence: float
    missing: str = ""
    answer: str = ""
    cites: list[str] = field(default_factory=list)
    explanation: str = ""


def build_prompt(question: str, evidence_lines: list[str], round_num: int, max_rounds: int) -> str:
    """evidence_lines must carry eids (`e1: [0.00-1.00s] ...`) -- without them
    the reflector cannot cite, and citations are the faithfulness hook."""
    evidence = "\n".join(evidence_lines) or "(none yet)"
    return _PROMPT.format(question=question, evidence=evidence,
                          round_num=round_num, max_rounds=max_rounds)


def parse(raw: str) -> Judgement:
    """Parse the reflector's JSON reply.

    A parse failure is conservative: insufficient with no answer. The loop then
    re-plans or, at the cap, ends with an empty answer that scores as a refusal
    (never silently defaults to "A").
    """
    try:
        obj = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return Judgement(sufficient=False, confidence=0.0,
                         missing="reflector reply was not valid JSON")
    try:
        conf = min(1.0, max(0.0, float(obj.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    cites = [str(c) for c in obj.get("cites") or [] if str(c)]
    return Judgement(
        sufficient=bool(obj.get("sufficient", False)),
        confidence=conf,
        missing=str(obj.get("missing", "")),
        answer=str(obj.get("answer", "")),
        cites=cites,
        explanation=str(obj.get("explanation", "")),
    )
