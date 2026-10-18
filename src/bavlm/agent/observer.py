"""Observer -- executes a Plan and returns structured Evidence.

Wraps the VLM backend and the frame cache (roadmap phase 4 adds the pose tool).

Frame selection lives here rather than inside each backend so every round's
frame count is decided in exactly one place and recorded on the Trace: the
project's central claim is "same frames, chosen better" (rule 4), and a count
scattered across backends cannot be audited.

Evidence is structured, never free prose -- `schema.Evidence` is what makes
each observation individually ablatable in the faithfulness study. That
constraint is a design decision (see schema.py), not a convenience.
"""
from __future__ import annotations

import json
import pathlib

from bavlm.data.schema import Evidence, Plan, Span

_PROMPT = (pathlib.Path(__file__).parent / "prompts" / "observer.txt").read_text()


def _frame_time(path: str, pos: int, cache_fps: float) -> float:
    """Seconds into the clip. The cache writes `{index:03d}_{t_ms:07d}.jpg`
    (scripts/extract_frames.py), so the timestamp is exact when present; the
    fallbacks assume the cache contract of frames at 1/cache_fps apart."""
    stem = pathlib.Path(path).stem
    parts = stem.split("_")
    try:
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1]) / 1000.0
        if stem.isdigit():
            return int(stem) / cache_fps
    except ValueError:
        pass
    return pos / cache_fps


def select_frames(frames: list[str], plan: Plan, cache_fps: float = 2.0) -> list[str]:
    """Clip frames inside plan.spans, thinned to plan.fps, capped at plan.budget.

    Two rules the selection must not break:
    - never denser than the cache (a 2 fps cache cannot honour 10 fps;
      interpolating would invent frames the uniform baseline never had)
    - never over budget; an over-long selection is subsampled uniformly, the
      same treatment the baseline gives long clips (rule 4)
    """
    if not frames:
        return []
    times = [_frame_time(p, i, cache_fps) for i, p in enumerate(frames)]
    picked: list[str] = []
    seen: set[str] = set()
    for s in plan.spans:
        in_span = [p for t, p in zip(times, frames) if s.start <= t < s.end]
        if not in_span:
            # a span finer than the frame spacing must still look at something:
            # fall back to the single frame nearest the span's midpoint
            mid = (s.start + s.end) / 2.0
            in_span = [min(zip(times, frames), key=lambda tp: abs(tp[0] - mid))[1]]
        for p in in_span:
            if p not in seen:
                seen.add(p)
                picked.append(p)
    if not picked:
        return []
    if plan.fps > 0:
        picked = picked[::max(1, round(cache_fps / plan.fps))]
    if len(picked) > plan.budget:
        stride = -(-len(picked) // plan.budget)  # ceil: uniform, never over
        picked = picked[::stride][: plan.budget]
    return picked


def _parse_observations(raw: str) -> list[dict]:
    """The observer's JSON list. Unparseable output yields no evidence -- an
    observation that cannot be structured cannot be cited or ablated, and the
    reflector will see the round as empty and ask again."""
    i, j = raw.find("["), raw.rfind("]")
    if i == -1 or j <= i:
        return []
    try:
        obj = json.loads(raw[i: j + 1])
    except json.JSONDecodeError:
        return []
    return obj if isinstance(obj, list) else []


def execute(plan: Plan, frames: list[str], backend, cache_fps: float = 2.0) -> tuple[list[str], list[Evidence]]:
    """Run one observation round: select frames, ask the VLM, parse.

    Returns (frames actually shown, evidence found). Evidence comes back
    without eids -- the loop assigns them so ids stay stable across rounds and
    citations can point at any round's observations.
    """
    selected = select_frames(frames, plan, cache_fps=cache_fps)
    raw = backend.generate(selected, _PROMPT.format(what=plan.what))
    out: list[Evidence] = []
    for it in _parse_observations(raw):
        try:
            start, end = float(it["start"]), float(it["end"])
        except (KeyError, TypeError, ValueError):
            continue  # one malformed row must not kill the round
        if end <= start:
            end = start + 1e-3
        desc = str(it.get("description", "")).strip()
        if not desc:
            continue
        conf = it.get("confidence")
        try:
            conf = None if conf is None else min(1.0, max(0.0, float(conf)))
        except (TypeError, ValueError):
            conf = None
        out.append(Evidence(
            span=Span(start, end),
            description=desc,
            court_region=it.get("court_region") or None,
            hit_type=it.get("hit_type") or None,
            confidence=conf,
            source="vlm",
        ))
    return selected, out
