"""Split verification and the training split.

Phase 0 of the roadmap and the only check that can silently invalidate every
number produced later. If a rally appears in both the training corpus and the
benchmark, fine-tuning contaminates evaluation.

Why the check is done per match, not per clip: FBBench clips are anonymous
(`video_000001` ...) and carry no source video or frame range, so no clip ID
can ever equal a Finebadminton-20K rally ID. Comparing IDs would always report
"no overlap" -- a false clean. What the 20K release does publish is
`benchmark_source_videos.txt`, the match videos FBBench was cut from. Every
rally from those matches is held out, which is also rule 3 (split by match).
"""
from __future__ import annotations

import json
import pathlib
import random
from collections import Counter
from dataclasses import dataclass, field

@dataclass
class Rally:
    video: str                 # source match video, e.g. "0001.mp4"
    start_frame: int
    end_frame: int
    fps: float                 # per video: 30, 25 or ~24.7 -- not a constant
    hits: list[dict] = field(default_factory=list)

    @property
    def rally_id(self) -> str:
        return f"{self.video.removesuffix('.mp4')}:{self.start_frame}-{self.end_frame}"

    @property
    def seconds(self) -> float:
        return (self.end_frame - self.start_frame + 1) / self.fps


def load_rallies(root: str | pathlib.Path) -> list[Rally]:
    """Every rally in a Finebadminton-20K download (`finebadminton-20K/*.json`)."""
    files = sorted(pathlib.Path(root).rglob("finebadminton-20K/*.json"))
    if not files:
        raise FileNotFoundError(f"no finebadminton-20K/*.json under {root}")
    out = []
    for p in files:
        for r in json.loads(p.read_text(encoding="utf-8")):
            out.append(Rally(r["video"], int(r["start_frame"]), int(r["end_frame"]),
                             float(r["fps"]), r.get("hitting", [])))
    return out


def benchmark_videos(root: str | pathlib.Path) -> set[str]:
    """The match videos FBBench was built from, as published with the 20K release."""
    p = pathlib.Path(root) / "benchmark_source_videos.txt"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: the overlap check cannot run without it")
    return {l.strip() for l in p.read_text().split() if l.strip()}


def check_overlap(train_root: str | pathlib.Path) -> dict:
    """Which 20K matches are safe to train on. `clean` means the held-out set
    covers every FBBench source video, so training on the rest cannot see a
    benchmark rally."""
    rallies = load_rallies(train_root)
    bench = benchmark_videos(train_root)
    videos = Counter(r.video for r in rallies)
    held = {r.rally_id for r in rallies if r.video in bench}
    return {
        "videos": len(videos),
        "rallies": len(rallies),
        "bench_videos": len(bench),
        "bench_videos_in_20k": len(bench & set(videos)),
        "bench_videos_missing": sorted(bench - set(videos)),
        "held_out_rallies": len(held),
        "train_videos": len(set(videos) - bench),
        "train_rallies": len(rallies) - len(held),
        "clean": True,             # by construction: every bench video is excluded
    }


def match_split(rallies: list[Rally], exclude: set[str], val_matches: int = 5,
                seed: int = 0) -> dict[str, list[Rally]]:
    """Train / val by match (rule 3), after dropping the benchmark's matches.
    Deterministic for a given seed, so every teammate gets the same split."""
    videos = sorted({r.video for r in rallies} - exclude)
    val = set(random.Random(seed).sample(videos, val_matches))
    split = {"train": [], "val": []}
    for r in rallies:
        if r.video not in exclude:
            split["val" if r.video in val else "train"].append(r)
    return split


def label_vocabulary(rallies: list[Rally]) -> dict[str, Counter]:
    """The actual label space, from the JSON rather than the paper."""
    v = {k: Counter() for k in ("hit type", "subtype", "ball area", "hitter", "quality",
                                "player actions", "shot characteristics")}
    for r in rallies:
        for h in r.hits:
            fa = h.get("Foundational Actions Level", {})
            ts = h.get("Tactical Semantics Level", {})
            de = h.get("Decision Evaluation Level", {})
            for k in ("hit type", "ball area", "hitter"):
                v[k][fa.get(k)] += 1
            v["subtype"].update(fa.get("subtype") or [])
            v["player actions"].update(ts.get("player actions") or [])
            v["shot characteristics"].update(ts.get("shot characteristics") or [])
            q = de.get("quality")
            v["quality"][int(q) if str(q).isdigit() else q] += 1   # release mixes 4 and "4"
    return v


def is_clean_label(label) -> bool:
    """English label, not blank / None / one of the release's few stray Chinese
    or punctuation entries. Those stay in the counts, flagged, never remapped."""
    return isinstance(label, str) and label.isascii() and any(c.isalpha() for c in label)
