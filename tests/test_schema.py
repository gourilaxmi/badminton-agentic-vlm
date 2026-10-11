import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from bavlm.data.schema import Evidence, Plan, Span, Trace


def test_span_rejects_empty():
    with pytest.raises(ValueError):
        Span(1.0, 1.0)


def test_plan_budget_estimate():
    p = Plan(what="contact", spans=[Span(0, 2), Span(5, 6)], fps=10)
    assert p.estimated_frames() == 30


def test_trace_partitions_evidence():
    ev = [Evidence(Span(0, 1), "a", eid="e1"), Evidence(Span(1, 2), "b", eid="e2")]
    t = Trace("q1", "c1", "why?", evidence=ev, cited=["e1"], answer="x")
    assert [e.eid for e in t.cited_evidence()] == ["e1"]
    assert [e.eid for e in t.uncited_evidence()] == ["e2"]


def test_evidence_prompt_line_is_checkable():
    e = Evidence(Span(1.25, 1.8), "racket meets shuttle above the head",
                 court_region="mid front court", hit_type="smash")
    line = e.to_prompt_line()
    assert "1.25-1.80s" in line and "mid front court" in line and "smash" in line
