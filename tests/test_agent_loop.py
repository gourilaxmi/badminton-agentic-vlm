"""The loop demo's contract: budget, spans, eids, citations, round cap.

These test the REAL loop plumbing against the mock agent backend -- parse,
clamp, frame selection, citation filtering, round cap, trace contents. They
make no claim about accuracy: the mock's answers come from the key by
construction (like oracle mode in the MCQ mock).

Run directly for readable output:  python tests/test_agent_loop.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bavlm.agent import observer, planner, reflector            # noqa: E402
from bavlm.agent.loop import AgentLoop, MAX_ROUNDS              # noqa: E402
from bavlm.agent.mock import MockAgentBackend                   # noqa: E402
from bavlm.data.schema import Plan, Span                        # noqa: E402
from bavlm.eval.fbbench import extract_choice                   # noqa: E402

# frames named like the real cache: {index:03d}_{t_ms:07d}.jpg, 2 fps
FRAMES = [f"clip/000_{i * 500:07d}.jpg" for i in range(20)]     # 0.0 .. 9.5 s


def test_planner_parse_clamps_fps_to_budget():
    raw = json.dumps({"what": "x", "spans": [[0.0, 10.0]], "fps": 25,
                      "resolution": "low", "crop": "full", "use_pose": False})
    p = planner.parse(raw, budget=25)
    assert p.estimated_frames() <= 25


def test_select_frames_respects_spans_and_budget():
    plan = Plan(what="x", spans=[Span(2.0, 4.0)], fps=25, budget=3)
    sel = observer.select_frames(FRAMES, plan)
    # frames at 2.0, 2.5, 3.0, 3.5 are inside -> budget caps at 3, uniform
    assert 0 < len(sel) <= 3
    for p in sel:
        t = int(p.split("_")[1].split(".")[0]) / 1000.0
        assert 2.0 <= t < 4.0


def test_select_frames_never_denser_than_the_cache():
    plan = Plan(what="x", spans=[Span(0.0, 5.0)], fps=25, budget=25)
    sel = observer.select_frames(FRAMES, plan)
    # 0..5 s at 2 fps = 10 frames, even though fps=25 would ask for 125
    assert len(sel) == 10


def test_select_frames_span_between_frames_still_looks_at_one():
    plan = Plan(what="x", spans=[Span(2.1, 2.4)], fps=25, budget=25)
    sel = observer.select_frames(FRAMES, plan)
    assert len(sel) == 1  # nearest frame to the midpoint, not nothing


def test_select_frames_with_no_frames_is_empty_not_an_error():
    plan = Plan(what="x", spans=[Span(0.0, 5.0)], fps=25)
    assert observer.select_frames([], plan) == []


def test_reflector_parse_failure_is_conservative():
    j = reflector.parse("I think it is probably B, though it is hard to say.")
    assert j.sufficient is False
    assert j.answer == ""


def test_reflector_prompt_carries_eids_for_citing():
    prompt = reflector.build_prompt("q?", ["e1: [0.00-1.00s] saw it",
                                           "e2: [1.00-2.00s] also"], 1, 3)
    assert "e1:" in prompt and "e2:" in prompt and "Round 1 of 3" in prompt


def test_loop_two_round_trace_with_intact_eids_and_citations():
    q = "How many times did the player smash?"
    backend = MockAgentBackend(answer_key={q: "B"})
    loop = AgentLoop(backend, budget=8, max_rounds=3)
    t = loop.run(qid="q1", clip_id="c1", question=q, frames=FRAMES, domain="count")

    assert len(t.rounds) == 2                     # mock: r1 re-plans, r2 answers
    assert t.rounds[0]["sufficient"] is False
    assert t.rounds[1]["sufficient"] is True
    assert t.rounds[1]["plan"] != t.rounds[0]["plan"]  # it actually re-planned

    eids = [e.eid for e in t.evidence]
    assert eids and len(eids) == len(set(eids))   # unique, stable
    assert set(t.cited) <= set(eids)              # only real evidence cited
    assert t.answer and extract_choice(t.answer, 4) == "B"

    total = sum(r["frames_this_round"] for r in t.rounds)
    assert t.frames_used == total
    assert t.frames_used <= 8 * len(t.rounds)     # budget per round, never over


def test_loop_respects_max_rounds_and_caps():
    class NeverSufficient(MockAgentBackend):
        def _reflect(self, prompt):
            j = super()._reflect(prompt)
            j["sufficient"] = False
            j["answer"] = ""
            return j

    loop = AgentLoop(NeverSufficient(), budget=8, max_rounds=3)
    t = loop.run(qid="q2", clip_id="c2", question="Where was the shuttle?",
                 frames=FRAMES, domain="position")
    assert len(t.rounds) == 3                     # hard cap
    assert t.answer == ""                         # no verdict -> refusal, not "A"
    assert extract_choice(t.answer, 4) is None


def test_loop_runs_all_domains_mock_answers_all_ten():
    questions = [
        ("How many rounds were won by the player in red?", "count", "C"),
        ("How many clear shots were played?", "count", "A"),
        ("What shot did the player play at the net?", "action", "B"),
        ("When did the striker play a drop shot?", "action", "D"),
        ("What is the tactical intent of this rally?", "action", "C"),
        ("Where did the shuttle land?", "position", "A"),
        ("Which side of the court was the player on?", "position", "D"),
        ("Where was the receiver standing?", "position", "B"),
        ("Why was the point lost?", "cognition", "C"),
        ("What comment best describes the round?", "cognition", "A"),
    ]
    backend = MockAgentBackend(answer_key={q: a for q, _, a in questions})
    loop = AgentLoop(backend, budget=10)
    for i, (q, domain, gold) in enumerate(questions):
        t = loop.run(qid=f"q{i}", clip_id=f"c{i}", question=q,
                     frames=FRAMES, domain=domain)
        assert extract_choice(t.answer, 4) == gold, q
        assert 1 <= len(t.rounds) <= MAX_ROUNDS
        assert t.frames_used <= 10 * len(t.rounds)
        assert set(t.cited) <= {e.eid for e in t.evidence}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok   {fn.__name__}")
    print(f"{len(fns)} passed")
