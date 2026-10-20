#!/usr/bin/env python3
"""Review-2 visual results: charts from results/*.csv plus per-clip panels.

    python scripts/07_visual_results.py [--frames data/frame_cache]

Writes PNGs to results/figures/:
  fig_domains.png      e01 / e03 / e06 per domain vs the always-A floor and paper references
  fig_subtasks.png     e06 (all 2,413) per sub-task
  fig_resolution.png   accuracy vs input tokens across 224 / 336 / 448 px
  qual_<clip>.png      real Qwen3-VL-8B (e06) answer on the clip's frames, with option probabilities
  loop_<clip>.png      loop-demo trace on the same frames: which spans each round looked at
The qual/loop panels need the frame cache (frames/<video_uid>/*.jpg) and are skipped without it.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bavlm.eval.fbbench import load_questions  # noqa: E402

RES = ROOT / "results"
OUT = RES / "figures"
DOMAINS = ["count", "action", "position", "cognition"]
Q3 = "Qwen3-VL-8B-Instruct"
GREEN, RED, BLUE, GREY = "#2c7c3f", "#b2182b", "#1f4e79", "#bdbdbd"

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.dpi": 150})


def by_domain(name):
    p = RES / f"{name}_by_domain.csv"
    return pd.read_csv(p).set_index("domain").reindex(DOMAINS) if p.exists() else None


def jsonl(name):
    """Tolerates pretty-printed 'JSONL' (the loop demo writes indented objects)."""
    s, dec, i, rows = (RES / name).read_text(encoding="utf-8"), json.JSONDecoder(), 0, []
    while True:
        while i < len(s) and s[i].isspace():
            i += 1
        if i >= len(s):
            return rows
        obj, i = dec.raw_decode(s, i)
        rows.append(obj)


def fig_domains():
    e06 = pd.DataFrame(jsonl(f"e06_full_{Q3}.jsonl"))
    floor = (e06.gold == "A").groupby(e06.domain).mean().reindex(DOMAINS)
    series = [("Qwen2.5-VL-7B, n=500 (e01)", by_domain("e01_zeroshot"), "#7f7f7f"),
              ("Qwen3-VL-8B, n=500 (e03)", by_domain(f"e03_zeroshot_{Q3}"), "#6baed6"),
              ("Qwen3-VL-8B, all 2,413 (e06)", by_domain(f"e06_full_{Q3}"), BLUE)]
    series = [(lab, df.accuracy, c) for lab, df, c in series if df is not None]
    series.append(("Always-A floor", floor, GREY))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.8 / len(series)
    for i, (lab, s, col) in enumerate(series):
        xs = [x + (i - (len(series) - 1) / 2) * w for x in range(len(DOMAINS))]
        ax.bar(xs, s.values * 100, w, label=lab, color=col, edgecolor="black", linewidth=0.4)
        for x, v in zip(xs, s.values):
            ax.text(x, v * 100 + 0.8, f"{v * 100:.0f}", ha="center", fontsize=7)
    overall = (e06.pred == e06.gold).mean() * 100
    for y, lab, ls in [(overall, f"e06 overall {overall:.1f}%", "-"),
                       (42.06, "paper, fine-tuned 42.1%", "--"), (38.6, "Gemini 2.5 Pro (paper) 38.6%", ":")]:
        ax.axhline(y, color=RED if ls != "-" else BLUE, ls=ls, lw=1)
        ax.text(2.0, y + 0.6, lab, ha="center", fontsize=7, color=RED if ls != "-" else BLUE)
    ax.set_xticks(range(len(DOMAINS)), [d.capitalize() for d in DOMAINS])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 62)
    ax.set_title("FBBench zero-shot, uniform 2 FPS, 448 px, cap 25 frames")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.savefig(OUT / "fig_domains.png")
    plt.close(fig)


def fig_subtasks():
    df = pd.read_csv(RES / f"e06_full_{Q3}_by_subtask.csv")
    rows = pd.DataFrame(jsonl(f"e06_full_{Q3}.jsonl"))
    floor = (rows.gold == "A").groupby(rows.subtask).mean()
    df = df.sort_values(["domain", "accuracy"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ys = range(len(df))
    cols = {d: c for d, c in zip(DOMAINS, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"])}
    ax.barh(ys, df.accuracy * 100, color=[cols[d] for d in df.domain], edgecolor="black", linewidth=0.4)
    ax.scatter(df.subtask.map(floor) * 100, ys, marker="|", s=200, color="black", zorder=3,
               label="always-A floor")
    for y, (a, n) in enumerate(zip(df.accuracy, df.n)):
        ax.text(a * 100 + 0.8, y, f"{a * 100:.1f}%  (n={n})", va="center", fontsize=7)
    ax.set_yticks(ys, [f"{d[:4]} · {s}" for d, s in zip(df.domain, df.subtask)], fontsize=8)
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 75)
    ax.set_title("e06: Qwen3-VL-8B zero-shot on all 2,413 FBBench questions, by sub-task")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(OUT / "fig_subtasks.png")
    plt.close(fig)


def fig_resolution():
    pts = [(s, by_domain(f"e05_resolution_{Q3}@{s}")) for s in (224, 336)]
    pts.append((448, by_domain(f"e03_zeroshot_{Q3}")))
    pts += [(720, df) for df in [by_domain(f"e08_res720_{Q3}")] if df is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    for d, mk in zip(DOMAINS, "osD^"):
        xs = [df.loc[d, "input_tokens"] for _, df in pts]
        ys = [df.loc[d, "accuracy"] * 100 for _, df in pts]
        line, = ax.plot(xs, ys, marker=mk, label=d.capitalize())
        for (side, _), x, y in zip(pts, xs, ys):
            ax.annotate(f"{side}px", (x, y), textcoords="offset points", xytext=(3, 4), fontsize=7,
                        color=line.get_color())
    ax.set_xlabel("Mean input tokens per question")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Resolution sweep, Qwen3-VL-8B, same 500 questions")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    ax.set_ylim(5, 62)
    fig.savefig(OUT / "fig_resolution.png")
    plt.close(fig)


def _frames(root, clip):
    return sorted((root / clip).glob("*.jpg"))


def _t(p):
    """Timestamp from <idx>_<seconds>.jpg; falls back to 2 FPS."""
    parts = p.stem.split("_")
    return float(parts[1]) if len(parts) > 1 else int(parts[0]) / 2


def _strip(fig, gs_row, frames, ncols, highlight=None):
    for i, p in enumerate(frames):
        ax = fig.add_subplot(gs_row[i // ncols, i % ncols])
        ax.imshow(plt.imread(p))
        ax.set_xticks([])
        ax.set_yticks([])
        col = highlight(_t(p)) if highlight else None
        for sp in ax.spines.values():
            sp.set_visible(bool(col))
            if col:
                sp.set_color(col)
                sp.set_linewidth(3)
        ax.set_title(f"{_t(p):.1f}s", fontsize=7, pad=2)


def qual_panels(root):
    qs = {q.qid: q for q in load_questions(ROOT / "data" / "fbbench")}
    preds = {r["qid"]: r for r in jsonl(f"e06_full_{Q3}.jsonl")}
    for tr in jsonl("loop_demo_traces.jsonl"):
        qid, clip = tr["question_id"], tr["clip_id"]
        frames = _frames(root, clip)
        if not frames or qid not in preds:
            continue
        q, r = qs[qid], preds[qid]
        frames = frames[::max(1, math.ceil(len(frames) / 12))][:12]
        ncols = min(6, len(frames))
        nrows = math.ceil(len(frames) / ncols)
        letters = list("ABCDEFGH"[:len(q.options)])

        # ---- real model answer
        fig = plt.figure(figsize=(12, 1.25 * nrows + 2.4))
        gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[1] * nrows + [1.4], hspace=0.25)
        _strip(fig, gs, frames, ncols)
        ok = r["pred"] == r["gold"]
        fig.suptitle(f"{clip} · {q.subtask} — Qwen3-VL-8B zero-shot (e06): "
                     f"pred {r['pred']} / gold {r['gold']}  {'CORRECT' if ok else 'WRONG'}",
                     color=GREEN if ok else RED, fontsize=11, y=0.995)
        ax = fig.add_subplot(gs[nrows, :ncols // 2 + 1])
        ax.axis("off")
        txt = "Q: " + textwrap.fill(q.question, 70) + "\n" + "\n".join(
            textwrap.shorten(f"{L}. {o}", 80) for L, o in zip(letters, q.options))
        ax.text(0, 1, txt, va="top", fontsize=7.5, family="monospace")
        ax = fig.add_subplot(gs[nrows, ncols // 2 + 1:])
        lp = r.get("choice_logprobs") or {}
        z = sum(math.exp(v) for v in lp.values()) or 1
        probs = [math.exp(lp.get(L, -1e9)) / z * 100 for L in letters]
        cols = [GREEN if L == r["gold"] else (RED if L == r["pred"] else GREY) for L in letters]
        ax.bar(letters, probs, color=cols, edgecolor="black", linewidth=0.4)
        ax.set_ylim(0, 105)
        ax.set_ylabel("P(option) %", fontsize=8)
        ax.set_title("green = gold, red = wrong pick", fontsize=8)
        fig.savefig(OUT / f"qual_{clip}.png")
        plt.close(fig)

        # ---- loop demo trace on the same frames
        all_frames = _frames(root, clip)
        spans = {r_["round"]: [(s["start"], s["end"]) for s in r_["plan"]["spans"]] for r_ in tr["rounds"]}
        rcol = {1: "#fdae61", 2: "#2b83ba", 3: "#7b3294"}

        def hl(t):
            for rn in sorted(spans, reverse=True):
                if any(a <= t <= b for a, b in spans[rn]):
                    return rcol.get(rn, "black")
            return None
        shown = all_frames[::max(1, math.ceil(len(all_frames) / 12))][:12]
        fig = plt.figure(figsize=(12, 1.25 * nrows + 2.2))
        gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[1] * nrows + [1.7], hspace=0.25)
        _strip(fig, gs, shown, ncols, hl)
        fig.suptitle(f"Agent loop trace on {clip} ({q.subtask}) — MOCK agent, qualitative, no accuracy claimed",
                     fontsize=11, y=0.995)
        ax = fig.add_subplot(gs[nrows, :])
        ax.axis("off")
        ev = {e["eid"]: e for e in tr["evidence"]}
        lines = []
        for r_ in tr["rounds"]:
            p = r_["plan"]
            sp = ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in spans[r_["round"]])
            v = "sufficient" if r_["sufficient"] else f"INSUFFICIENT: {r_['missing']}"
            lines.append(f"Round {r_['round']}  plan: '{p['what']}'  [{sp}] @ {p['fps']} fps, "
                         f"{p['resolution']} res, {r_['frames_this_round']} frames  ->  {v} "
                         f"(conf {r_['confidence']:.2f})")
            for eid in r_["evidence_ids"]:
                e = ev[eid]
                lines.append(f"    {eid} [{e['span']['start']:.2f}-{e['span']['end']:.2f}s] "
                             f"{e.get('hit_type') or ''} {e['description']} ({e['court_region']})")
        lines.append(f"Answer {tr['answer']}  cites {', '.join(tr['cited'])}   "
                     f"(frame border colour = round that looked at it: orange R1, blue R2)")
        ax.text(0, 1, "\n".join(textwrap.shorten(s, 150) for s in lines), va="top", fontsize=7.5,
                family="monospace")
        fig.savefig(OUT / f"loop_{clip}.png")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", default=str(ROOT / "data" / "frame_cache"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_domains()
    fig_subtasks()
    fig_resolution()
    root = pathlib.Path(args.frames)
    if root.exists():
        qual_panels(root)
    else:
        print(f"no frame cache at {root}: per-clip panels skipped")
    for p in sorted(OUT.glob("*.png")):
        print("wrote", p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
