#!/usr/bin/env python3
"""Extract frames once, on CPU, so GPU quota is never spent decoding video.

Only clips that FBBench actually references are extracted -- not all 3,215.
Output is a flat per-clip folder of JPEGs plus a manifest, sized to upload as a
Kaggle Dataset (~1.2 GB at 224 px).

    python scripts/extract_frames.py \
        --fbbench data/fbbench \
        --clips   data/finebadminton-20k \
        --out     ./frame_cache \
        --fps 2 --size 224

Resumable: clips already extracted are skipped, so a killed run costs nothing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


# --------------------------------------------------------------------- discovery

def referenced_clip_ids(fbbench_root: pathlib.Path) -> set[str]:
    """Every clip id mentioned anywhere in the benchmark files.

    Field names are guessed because the exact FBBench schema is not known until
    you dump it (roadmap phase 0). Widen CANDIDATES after you have looked at a
    real record -- see scripts/00_verify_dataset.py output.
    """
    CANDIDATES = ("clip_id", "video_id", "video", "clip", "vid", "id", "video_name")
    ids: set[str] = set()

    for p in list(fbbench_root.rglob("*.json")) + list(fbbench_root.rglob("*.jsonl")):
        try:
            text = p.read_text()
        except OSError:
            continue
        records = ([json.loads(l) for l in text.splitlines() if l.strip()]
                   if p.suffix == ".jsonl" else json.loads(text))
        if isinstance(records, dict):
            records = records.get("data", records.get("questions", [records]))
        for r in records:
            if not isinstance(r, dict):
                continue
            for k in CANDIDATES:
                v = r.get(k)
                if isinstance(v, (str, int)):
                    ids.add(pathlib.Path(str(v)).stem)
                    break
    return ids


def index_videos(clips_root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Map clip id -> video path, keyed on filename stem."""
    return {p.stem: p for p in clips_root.rglob("*") if p.suffix.lower() in VIDEO_EXT}


# --------------------------------------------------------------------- extraction

def extract_one(video: pathlib.Path, out_dir: pathlib.Path,
                fps: float, size: int, quality: int = 88) -> tuple[str, int, str]:
    """Return (clip_id, n_frames, status)."""
    clip_id = video.stem
    dest = out_dir / clip_id

    if dest.exists() and any(dest.glob("*.jpg")):
        return clip_id, len(list(dest.glob("*.jpg"))), "skipped"

    try:
        import decord
        from PIL import Image
        import numpy as np

        vr = decord.VideoReader(str(video), num_threads=1)
        src_fps = float(vr.get_avg_fps()) or 25.0
        step = max(1, round(src_fps / fps))
        idx = list(range(0, len(vr), step))
        if not idx:
            return clip_id, 0, "empty"

        batch = vr.get_batch(idx).asnumpy()
        dest.mkdir(parents=True, exist_ok=True)

        for n, (frame, src_i) in enumerate(zip(batch, idx)):
            im = Image.fromarray(frame)
            w, h = im.size
            scale = size / min(w, h)                      # short side -> `size`
            im = im.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
            # timestamp in the filename: the sampler needs it, and it makes the
            # cache self-describing if the manifest is ever lost
            t_ms = round(src_i / src_fps * 1000)
            im.save(dest / f"{n:03d}_{t_ms:07d}.jpg", quality=quality)

        return clip_id, len(idx), "ok"

    except Exception as e:                                 # noqa: BLE001
        return clip_id, 0, f"error: {type(e).__name__}: {e}"


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbbench", required=True, type=pathlib.Path)
    ap.add_argument("--clips", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--size", type=int, default=224, help="short-side pixels")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="test on N clips first")
    a = ap.parse_args()

    for d in (a.fbbench, a.clips):
        if not d.exists():
            print(f"missing: {d}", file=sys.stderr)
            return 1

    wanted = referenced_clip_ids(a.fbbench)
    have = index_videos(a.clips)
    print(f"FBBench references {len(wanted)} clips; {len(have)} videos on disk")

    todo = [have[c] for c in sorted(wanted) if c in have]
    missing = sorted(wanted - set(have))
    if missing:
        print(f"WARNING {len(missing)} referenced clips not found, e.g. {missing[:5]}")
    if a.limit:
        todo = todo[: a.limit]
    print(f"extracting {len(todo)} clips at {a.fps} fps, short side {a.size}px")

    a.out.mkdir(parents=True, exist_ok=True)
    results, errors = [], []

    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(extract_one, v, a.out, a.fps, a.size): v for v in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            cid, n, status = fut.result()
            results.append({"clip_id": cid, "frames": n, "status": status})
            if status.startswith("error"):
                errors.append((cid, status))
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")

    (a.out / "manifest.json").write_text(json.dumps({
        "fps": a.fps, "size": a.size,
        "clips": len(results),
        "frames": sum(r["frames"] for r in results),
        "missing_videos": missing,
        "results": results,
    }, indent=1))

    total = sum(r["frames"] for r in results)
    mb = sum(f.stat().st_size for f in a.out.rglob("*.jpg")) / 1e6
    print(f"\ndone: {len(results)} clips, {total} frames, {mb:.0f} MB")
    if errors:
        print(f"{len(errors)} failed, first few: {errors[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
