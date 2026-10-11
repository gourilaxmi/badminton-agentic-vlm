#!/usr/bin/env python3
"""Screen several backbones on a small stratified subset, then pick one.

Objective 1 of the project: establish a zero-shot baseline across candidate
VLMs. The output table is itself a Review 2 deliverable.

Why screen rather than run 500 questions per model: five models at 500
questions is ~12 h of GPU. Five models at 150 is ~4 h, and 150 is enough to
eliminate anything clearly unsuitable. The full run then happens once, on the
winner.

    python scripts/01b_backbone_sweep.py --config configs/backbones.yaml
    python scripts/01b_backbone_sweep.py --config ... --mock oracle   # laptop

Resumable per model: a model whose JSONL is already complete is skipped, so a
killed Kaggle session costs only the model it died on.

IMPORTANT — every model sees the identical questions, the identical prompt,
the identical frames and the same seed. Vary any of those and you are comparing
prompts, not models.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bavlm.eval.fbbench import load_questions, extract_choice, score  # noqa: E402
from bavlm.models.backend import build_backend                        # noqa: E402

DOMAINS = ("count", "action", "position", "cognition")


# ----------------------------------------------------------------- sampling

def stratified(questions, n: int, seed: int = 0):
    """Equal share per domain, so a model strong on one domain cannot carry
    the overall number. Falls back gracefully if a domain is short."""
    import random
    rng = random.Random(seed)

    buckets = collections.defaultdict(list)
    for q in questions:
        buckets[q.domain or "unknown"].append(q)

    per = max(1, n // max(len(buckets), 1))
    picked = []
    for d in sorted(buckets):
        pool = buckets[d]
        rng.shuffle(pool)
        picked += pool[:per]

    # top up from the remainder if rounding left us short
    if len(picked) < n:
        rest = [q for q in questions if q not in picked]
        rng.shuffle(rest)
        picked += rest[: n - len(picked)]

    picked.sort(key=lambda q: q.clip_id)      # clip grouping -> prefix cache hits
    return picked[:n]


# ------------------------------------------------------------------ running

def run_model(spec: dict, questions, cfg: dict, out_dir: pathlib.Path,
              frame_root, mock: str | None) -> pathlib.Path:
    out = out_dir / f"screen_{spec['name']}.jsonl"

    done = set()
    if out.exists():
        done = {json.loads(l)["qid"] for l in open(out) if l.strip()}
        if len(done) >= len(questions):
            print(f"  {spec['name']}: already complete ({len(done)}), skipping")
            return out
        print(f"  {spec['name']}: resuming, {len(done)} done")

    key = {q.qid: q.answer for q in questions}
    backend = (build_backend({"mock": mock, "seed": cfg["seed"]}, answer_key=key)
               if mock else
               build_backend({"hf_id": spec["hf_id"],
                              "dtype": "float16",
                              "shard": spec.get("shard", True),
                              "loader": spec.get("loader"),
                              "max_pixels": cfg["max_pixels"]}))

    todo = [q for q in questions if q.qid not in done]
    t0 = time.time()

    with open(out, "a") as fh:
        for i, q in enumerate(todo, 1):
            t = time.time()
            frames = [] if mock else q.load_frames(frame_root)
            raw = backend.generate(frames, q.prompt, qid=q.qid)
            n_tok = getattr(backend, "last_input_tokens", None)
            fh.write(json.dumps({
                "model": spec["name"], "qid": q.qid, "clip_id": q.clip_id,
                "domain": q.domain, "gold": q.answer, "raw": raw,
                "pred": extract_choice(raw, q.n_options),
                "seconds": round(time.time() - t, 3), "input_tokens": n_tok,
            }) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"    {i}/{len(todo)}  {(time.time()-t0)/60:.1f} min")

    # free VRAM before the next model, or model 2 OOMs
    del backend
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass

    print(f"  {spec['name']}: done in {(time.time()-t0)/60:.1f} min")
    return out


# ----------------------------------------------------------------- reporting

def report(paths: list[pathlib.Path], n: int) -> None:
    import math

    rows = []
    for p in paths:
        if not p.exists():
            continue
        recs = [json.loads(l) for l in open(p) if l.strip()]
        if not recs:
            continue
        s = score(recs)
        secs = [r["seconds"] for r in recs if r.get("seconds")]
        rows.append({
            "model": recs[0]["model"],
            "n": s["n"],
            "overall": s["overall"],
            "by_domain": s["by_domain"],
            "sec_per_q": sum(secs) / len(secs) if secs else None,
            "unparsed": s["unparsed"],
        })

    rows.sort(key=lambda r: -r["overall"])

    hdr = f"{'model':26} {'n':>4} {'overall':>8} " + " ".join(f"{d[:4]:>6}" for d in DOMAINS) + f" {'s/q':>6} {'unp':>4}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = " ".join(f"{r['by_domain'].get(d, {}).get('accuracy', float('nan')):6.3f}"
                         for d in DOMAINS)
        sq = f"{r['sec_per_q']:6.1f}" if r["sec_per_q"] else "     -"
        print(f"{r['model']:26} {r['n']:4} {r['overall']:8.3f} {cells} {sq} {r['unparsed']:4}")

    # can we actually tell the top two apart?
    if len(rows) >= 2:
        a, b = rows[0], rows[1]
        gap = a["overall"] - b["overall"]
        p = (a["overall"] + b["overall"]) / 2
        se = math.sqrt(2 * p * (1 - p) / min(a["n"], b["n"]))
        print(f"\ntop-two gap {gap*100:.1f} pts;  ~95% threshold is {1.96*se*100:.1f} pts")
        if gap < 1.96 * se:
            print("  -> NOT separable at this sample size. Either run more questions\n"
                  "     on these two only, or choose on practical grounds: speed,\n"
                  "     memory, ecosystem maturity. Do not report a ranking you\n"
                  "     cannot defend.")
        else:
            print(f"  -> {a['model']} is ahead by more than noise. Run the full set on it.")

    print("\nReference points, FineBadminton (ACM MM 2025):")
    print("  7B open-source zero-shot     < 25%")
    print("  Gemini 2.5 Pro + S             38.6%")
    print("  Qwen2.5-VL-7B + S + C, ft      42.06%")


# ---------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/backbones.yaml")
    ap.add_argument("--fbbench", default="data/fbbench")
    ap.add_argument("--frames", default=None, help="frame cache root")
    ap.add_argument("--out", default="results/screening")
    ap.add_argument("--mock", default=None,
                    help="oracle | random | noisy -- laptop dry run, no GPU")
    ap.add_argument("--only", nargs="*", help="run just these model names")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    import yaml
    cfg_all = yaml.safe_load(open(a.config))
    cfg = cfg_all["screening"]
    specs = cfg_all["models"]
    if a.only:
        specs = [s for s in specs if s["name"] in a.only]

    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    allq = load_questions(a.fbbench)
    if not allq:
        print(f"no questions loaded from {a.fbbench} -- run describe_schema() "
              f"and pin the field names first", file=sys.stderr)
        return 1

    qs = stratified(allq, cfg["n_questions"], cfg["seed"])
    dist = collections.Counter(q.domain for q in qs)
    print(f"screening set: {len(qs)} questions  {dict(dist)}")

    paths = [out_dir / f"screen_{s['name']}.jsonl" for s in specs]

    if not a.report_only:
        for spec in specs:
            print(f"\n=== {spec['name']}  ({spec['fp16_gb']} GB fp16) ===")
            try:
                run_model(spec, qs, cfg, out_dir, a.frames, a.mock)
            except Exception as e:                       # noqa: BLE001
                # one model failing must not lose the others
                print(f"  FAILED: {type(e).__name__}: {e}")

    report(paths, cfg["n_questions"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
