#!/usr/bin/env python3
"""Review-2 loop demonstration: plan -> evidence -> reflection -> round 2.

QUALITATIVE ONLY -- the deliverable is the trace, not an accuracy number
(ROADMAP Day 6; HANDOVER step 5). Planner, observer and reflector run against
the deterministic mock (src/bavlm/agent/mock.py), so this runs on a laptop with
no GPU and no frame cache; the SAME loop runs against real backends in roadmap
phase 3 (scripts/03_run_agent.py, a stub by design until then).

Question selection is the notebook's round-robin over the 12 sub-tasks,
deterministic, first n -- every teammate's run shows the same clips.

Usage:
    python scripts/05_loop_demo.py                 # questions only (no cache)
    python scripts/05_loop_demo.py --frames <dir>  # with the frame cache

Writes results/loop_demo_traces.jsonl (all traces) and
results/loop_demo_slide.md (one trace formatted for the slide).
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bavlm.agent.loop import AgentLoop                   # noqa: E402
from bavlm.agent.mock import MockAgentBackend            # noqa: E402
from bavlm.eval.fbbench import extract_choice, load_questions  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def round_robin(qs, n: int):
    """Spread the demo over every sub-task; deterministic, so all teammates
    show the same 10 clips (same convention as the e01-e06 runs)."""
    by_sub: dict = {}
    for q in qs:
        by_sub.setdefault(q.subtask, []).append(q)
    order = [q for grp in itertools.zip_longest(*by_sub.values()) for q in grp if q]
    return order[:n] if n else order


def _plan_line(p: dict) -> str:
    # asdict() turns Spans into dicts, so unpack by key, not position
    spans = ", ".join(
        f"{sp['start']}-{sp['end']}" if isinstance(sp, dict) else f"{sp[0]}-{sp[1]}"
        for sp in p["spans"])
    return (f"**Plan** -- seek `{p['what']}`  \n"
            f"spans [{spans}] @ {p['fps']} fps · resolution {p['resolution']} · "
            f"crop {p['crop']} · budget {p['budget']} frames")


def _slide(trace, assume_note: str) -> str:
    """One trace formatted for a slide: plan -> evidence -> reflection -> round 2."""
    lines = [
        "# Loop demo trace -- qualitative, mock agent, no accuracy claimed",
        "",
        f"**Question ({trace.domain}):** {trace.question}  ",
        f"**qid** `{trace.question_id}` · **clip** `{trace.clip_id}` · "
        f"**frames used** {trace.frames_used} · **rounds** {len(trace.rounds)}",
        "",
        f"*{assume_note}*",
        "",
    ]
    ev = {e.eid: e for e in trace.evidence}
    for r in trace.rounds:
        lines.append(f"### Round {r['round']}")
        lines.append(_plan_line(r["plan"]))
        lines.append("")
        lines.append(f"**Observer** -- {r['frames_this_round']} frames shown, evidence:")
        for eid in r["evidence_ids"]:
            e = ev[eid]
            lines.append(f"- `{eid}` {e.to_prompt_line()}")
        lines.append("")
        verdict = "sufficient" if r["sufficient"] else "**insufficient**"
        lines.append(f"**Reflection** -- {verdict} (confidence {r['confidence']})"
                     + (f": {r['missing']}" if r["missing"] else ""))
        lines.append("")
    lines.append(f"### Answer -- {trace.answer}")
    lines.append(f"- cites: {', '.join(f'`{c}`' for c in trace.cited) or '(none)'}")
    lines.append(f"- explanation: {trace.explanation}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=10, help="number of questions (default 10)")
    ap.add_argument("--frames", type=str, default=None,
                    help="frame cache root (frames/<video_uid>/*.jpg); omitted -> mock has no frames")
    ap.add_argument("--slide", type=int, default=None,
                    help="index of the trace to write as the slide (default: first cognition)")
    args = ap.parse_args()

    questions = load_questions(ROOT / "data" / "fbbench")
    demo = round_robin(questions, args.n)
    frames_root = pathlib.Path(args.frames) if args.frames else None
    assume_note = ("no frame cache given: frames=[] and duration assumed "
                   "10 s; with the cache the loop gets real frames and durations"
                   if frames_root is None else
                   f"frame cache: {frames_root}")

    backend = MockAgentBackend(answer_key={q.question: q.answer for q in demo}, seed=7)
    loop = AgentLoop(backend, budget=25, max_rounds=3)

    traces = []
    print(f"{len(demo)} questions, mock agent backend -- plumbing demo, "
          "answers come from the key, no accuracy claimed\n")
    for i, q in enumerate(demo):
        frames = q.load_frames(frames_root) if frames_root else []
        tr = loop.run(qid=q.qid, clip_id=q.clip_id, question=q.question,
                      frames=frames, domain=q.domain)
        traces.append(tr)
        verdicts = " -> ".join("ok" if r["sufficient"] else "re-plan" for r in tr.rounds)
        print(f"{i + 1:2} {q.subtask:<32} rounds {verdicts:<18} "
              f"frames {tr.frames_used:2}  evidence {len(tr.evidence)}  "
              f"answer {extract_choice(tr.answer, q.n_options)} gold {q.answer}")

    OUT.mkdir(exist_ok=True)
    (OUT / "loop_demo_traces.jsonl").write_text("".join(t.to_json() + "\n" for t in traces))

    if args.slide is None:
        pick = next((i for i, t in enumerate(traces) if t.domain == "cognition"), 0)
    else:
        pick = args.slide
    (OUT / "loop_demo_slide.md").write_text(_slide(traces[pick], assume_note))

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\ntraces  -> results/loop_demo_traces.jsonl ({len(traces)})")
    print(f"slide   -> results/loop_demo_slide.md (trace #{pick}: {traces[pick].question_id})")
    print(f"manifest row:\n| {stamp} | e07_loop_demo (qualitative, no accuracy claimed) "
          f"| mock:agent | agentic loop, budget 25/round, max 3 rounds | "
          f"{sum(t.frames_used for t in traces)} total | FBBench n={len(traces)} "
          f"| see git log | loop_demo_traces.jsonl |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
