"""Core dataclasses shared across the pipeline.

The Observer is required to return *structured* evidence rather than free prose.
That constraint is what makes each evidence item individually ablatable in
`bavlm.eval.faithfulness` -- it is a design decision, not a convenience.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Sequence
import json

Domain = Literal["count", "action", "position", "cognition"]
Granularity = Literal["foundational", "tactical", "decision"]


@dataclass(frozen=True)
class Span:
    """A half-open time interval within a rally clip, in seconds."""
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"empty span: {self.start}..{self.end}")

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class Plan:
    """What the Planner emits. `budget` is frames, not seconds.

    The whole contribution lives here: `spans` and `fps` are chosen from the
    query, whereas a query-agnostic selector would fix them in advance.
    """
    what: str                       # the sub-question this round is seeking
    spans: list[Span]               # where to look
    fps: float                      # temporal density within those spans
    resolution: Literal["low", "high"] = "low"
    crop: Literal["full", "player", "net", "rear"] = "full"
    use_pose: bool = False          # may the Observer call the pose tool?
    budget: int = 25                # hard cap on frames for this round

    def estimated_frames(self) -> int:
        return int(sum(s.end - s.start for s in self.spans) * self.fps)


@dataclass
class Evidence:
    """One atomic, ablatable observation."""
    span: Span
    description: str
    court_region: str | None = None     # one of the 9 FineBadminton regions
    hit_type: str | None = None
    confidence: float | None = None
    source: Literal["vlm", "pose", "tracker"] = "vlm"
    eid: str = ""                       # stable id, assigned by the loop

    def to_prompt_line(self) -> str:
        bits = [f"[{self.span.start:.2f}-{self.span.end:.2f}s]"]
        if self.court_region:
            bits.append(f"({self.court_region})")
        if self.hit_type:
            bits.append(f"<{self.hit_type}>")
        bits.append(self.description)
        return " ".join(bits)


@dataclass
class Reflection:
    sufficient: bool
    confidence: float
    missing: str = ""               # what to seek next, if insufficient


@dataclass
class Trace:
    """Full record of one query. Written to disk for every run -- the
    faithfulness evaluation replays these rather than re-running the loop."""
    question_id: str
    clip_id: str
    question: str
    domain: Domain | None = None
    rounds: list[dict] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)   # eids the answer cites
    answer: str = ""
    explanation: str = ""
    frames_used: int = 0
    input_tokens: int = 0
    wall_seconds: float = 0.0

    def cited_evidence(self) -> list[Evidence]:
        keep = set(self.cited)
        return [e for e in self.evidence if e.eid in keep]

    def uncited_evidence(self) -> list[Evidence]:
        keep = set(self.cited)
        return [e for e in self.evidence if e.eid not in keep]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def write_traces(traces: Sequence[Trace], path: str) -> None:
    with open(path, "w") as fh:
        for t in traces:
            fh.write(json.dumps(asdict(t), default=str) + "\n")
