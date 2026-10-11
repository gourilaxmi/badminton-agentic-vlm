# Kickoff — paste this as the first message in the new Claude session

Attach these files alongside it: `HANDOFF.md`, `RUNBOOK.md`, `CLAUDE.md`,
`badminton-agentic-vlm.zip`, `architecture.png`.

---

```
I'm working on my final-year BTP at IIIT Kottayam. Review 2 is Wed 21 Oct 2026,
technical freeze Sun 18 Oct — so 7 build days from today.

I've attached four documents:
- CLAUDE.md     — project context; put this in the repo root
- HANDOFF.md    — full brief: novelty, papers, datasets, rules
- RUNBOOK.md    — setup and all code, inlined
- badminton-agentic-vlm.zip — the repo scaffold, 47 files, tests pass

Read CLAUDE.md and HANDOFF.md before anything else. The design is settled and
was defended at a review panel — I need implementation, not a redesign.

Situation:
- No A100. Running on Kaggle, GPU T4 x2, 30 GPU-hours/week.
- Team of four: branches gouri, amal, vishal, sabeeh.
- Datasets are public and Apache-2.0, no request form.
- Nothing is built yet beyond the scaffold.

Review 2 needs exactly four things:
1. results/dataset_report.md — clip counts, splits, zero train/test
   contamination, the real label vocabulary from the JSON
2. Zero-shot FBBench accuracy for Qwen2.5-VL-7B, reported per-domain
   (Count / Action / Position / Cognition)
3. Inference cost — seconds and input tokens per query
4. Working repo, tests green, every run in results/MANIFEST.md

Do NOT build the agentic loop. That is Review 3 (January).

Start here, in this order:
1. Help me create the GitHub repo from the scaffold with four branches
2. Add CLAUDE.md to the repo root
3. Walk me through scripts/00_verify_dataset.py — the clip-overlap check
   between Finebadminton-20K and FBBench is blocking everything else

One rule I want enforced throughout: mock mode must pass on my laptop before
anything runs on Kaggle. Oracle 100%, random ~25%. If either is off, the
scorer is broken and any GPU time is wasted.
```

---

## Why `CLAUDE.md` rather than re-pasting context

Claude Code reads `CLAUDE.md` from the repo root automatically at the start of
every session. Commit it once and every future session — yours and your
teammates' — starts with the project rules, the dataset facts, the traps, and
the current sprint already loaded. No re-pasting, no drift between what
different people's sessions believe.

Update it when something changes: a pinned schema, a new gotcha, the sprint
rolling over to Review 3.
