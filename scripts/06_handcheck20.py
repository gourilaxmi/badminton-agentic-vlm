#!/usr/bin/env python3
"""Hand-check 20 answers against the key (HANDOVER step 4, D4, ~1 h).

Rule 10 sanity in human form: before any number goes in the deck, a person
opens 20 clips and confirms the gold label is actually right and the question
is answerable at all. Priorities from the e03 error analysis:

  - detailed hitting count, where Qwen3 dropped 22 points vs Qwen2.5
  - Position, where every model sits below the always-A floor
  - wrong predictions first: a wrong pred on a bad key is not a model failure

The pick is deterministic (seed 0), stratified 5 per domain, so every
teammate reviews the same 20. Output: results/handcheck20.md -- print it or
work through it on screen; fill the "verdict / notes" column.

Usage:  python scripts/06_handcheck20.py [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bavlm.eval.fbbench import load_questions           # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "e06_full_Qwen3-VL-8B-Instruct.jsonl"
OUT = ROOT / "results" / "handcheck20.md"
PER_DOMAIN = 5
# sub-task priorities from the e03 error analysis (e06 confirms them)
PRIORITY = {"detailed hitting count": 0, "hitting localization": 1,
            "hitting prediction": 2, "moving recognition": 3}


def pick(rows: list[dict], rng: random.Random) -> list[dict]:
    by_domain: dict[str, list[dict]] = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r)
    chosen: list[dict] = []
    for domain in ("count", "action", "position", "cognition"):
        pool = by_domain[domain]
        # wrong predictions first, priority sub-tasks first, then stable shuffle
        pool = sorted(pool, key=lambda r: (PRIORITY.get(r["subtask"], 9), r["qid"]))
        wrong = [r for r in pool if r["pred"] != r["gold"]]
        right = [r for r in pool if r["pred"] == r["gold"]]
        rng.shuffle(wrong)
        rng.shuffle(right)
        take = (wrong[:PER_DOMAIN - 1] + right[:1]) if len(wrong) >= PER_DOMAIN - 1 \
            else wrong + right[:PER_DOMAIN - len(wrong)]
        chosen += sorted(take, key=lambda r: (PRIORITY.get(r["subtask"], 9), r["qid"]))
    return chosen


def conf(row: dict) -> str:
    lp = row.get("choice_logprobs") or {}
    if not isinstance(lp, dict) or row["pred"] is None:
        return "-"
    try:
        return f"{max(float(v) for v in lp.values()):.2f}"
    except (TypeError, ValueError):
        return "-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in RESULTS.open() if l.strip()]
    chosen = pick(rows, random.Random(args.seed))
    assert len(chosen) == 4 * PER_DOMAIN, f"pick returned {len(chosen)}"

    questions = {q.qid: q for q in load_questions(ROOT / "data" / "fbbench")}
    clip_url = ("https://huggingface.co/datasets/Moujuruo/FineBadmintonBenchmark/"
                "resolve/main/finebadmintonbenchmark/hf_video_clips_qa/")

    lines = [
        "# Hand-check: 20 answers from e06 against the key",
        "",
        "Deterministic pick (seed "
        f"{args.seed}), 5 per domain, wrong predictions first, priority "
        "sub-tasks first (detailed hitting count, Position).",
        "",
        "For each row: open the clip, read the question, decide **is the gold "
        "label right?** — then record verdict (gold OK / gold WRONG / question "
        "bad) and any note. Clips: `"
        + clip_url + "<clip_id>.mp4`.",
        "",
        "| # | domain / subtask | qid | question | gold | pred | conf | verdict / notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(chosen, 1):
        q = questions.get(r["qid"])
        text = (q.question if q else "").replace("|", "/").replace("\n", " ")
        if len(text) > 110:
            text = text[:107] + "..."
        lines.append(
            f"| {i} | {r['domain']} / {r['subtask']} | `{r['qid']}` | {text} "
            f"| {r['gold']} | {r['pred'] or '-'} | {conf(r)} | |")

    lines += ["", "## Summary", "",
              "- gold wrong: ", "- question unusable: ",
              "- model error confirmed: ", "- surprising (model wrong, gold arguable): ", ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT.relative_to(ROOT)} ({len(chosen)} questions)")
    n_wrong = sum(r["pred"] != r["gold"] for r in chosen)
    print(f"mix: {n_wrong} wrong predictions, {len(chosen) - n_wrong} correct")
    for r in chosen:
        print(f"  {r['domain']:<10} {r['subtask']:<26} {r['qid']}  "
              f"gold {r['gold']} pred {r['pred']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
