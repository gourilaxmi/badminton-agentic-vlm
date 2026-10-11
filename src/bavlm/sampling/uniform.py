"""Uniform sampling -- the baseline FineBadminton compares against.

2 FPS over a 12.4 s rally is about 25 frames. That number is the frame budget
every other strategy in this repo must respect, so that the claim stays
"given the same frames, we choose better ones".
"""
from __future__ import annotations


def uniform_indices(n_frames: int, src_fps: float, target_fps: float = 2.0,
                    cap: int | None = None) -> list[int]:
    if n_frames <= 0:
        return []
    step = max(1, round(src_fps / target_fps))
    idx = list(range(0, n_frames, step))
    if cap is not None and len(idx) > cap:
        # thin evenly rather than truncating the tail
        keep = [round(i * (len(idx) - 1) / (cap - 1)) for i in range(cap)]
        idx = [idx[k] for k in keep]
    return idx


def budget_for(duration_s: float, target_fps: float = 2.0) -> int:
    """The frame budget a rally of this length gets under the baseline."""
    return max(1, round(duration_s * target_fps))
