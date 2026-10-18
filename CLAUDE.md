# CLAUDE.md

Project context, read automatically at the start of every Claude Code session.
Keep it current — it is the shortest path to a useful session.

Deeper references in this repo: `START_HERE.md` (index and first steps),
`docs/HANDOFF.md` (full project brief), `docs/NOVELTY.md` (contributions),
`docs/ROADMAP.md` (day-by-day plan), `RUNBOOK.md` (setup and all code).

---

## What this is

Final-year BTP, IIIT Kottayam. **VLM-Grounded Agentic Reasoning for Explainable
Action and Error Recognition in Dynamic Video Sequences.**

Badminton video understanding. We build a query-conditioned agentic perception
loop and test two things nobody has: whether an agent allocates a fixed
observation budget better than a learned query-agnostic selector, and whether
the evidence it cites is what actually drove its answer.

Team of four. One A100 is **not** available — we run on Kaggle (GPU T4 × 2).

| Milestone | Date | Scope |
|---|---|---|
| Review 1 | done | Problem, literature, design |
| **Review 2** | **Wed 21 Oct 2026**, freeze Sun 18 Oct | Data verified · backbone screening table · zero-shot numbers · resolution sweep · loop demo |
| Review 3 | mid-Jan 2027 | The agentic loop and the headline result |
| Review 4 | mid-Mar 2027 | Faithfulness study, ablations, write-up |

**Current sprint is Review 2. Do not build the agentic loop yet.**

---

## Datasets

Apache-2.0, already public, no request form.

```
Moujuruo/Finebadminton-20K        training corpus — develop here
Moujuruo/FineBadmintonBenchmark   FBBench — final numbers only
```

FineBadminton: the paper describes 120 matches → **3,215 rally clips**, 33,325
strokes; the HF **release** (`Finebadminton-20K`, 70 matches) carries **2,066
rallies / 20,757 hits**, mean rally **10.4 s**, frame rate **not constant**
(~30 fps × 1,370 rallies, ~25 fps × 696 — convert indices with each rally's own
`fps`). 720p+, overhead-behind camera. Three annotation levels: Foundational
Actions (11 hit types, 20 subtypes, 9 court regions) · Tactical Semantics ·
Decision Evaluation (quality 1–7 plus free-text reasons).

FBBench: the current HF release (@`ff86093a`) has **2,413 multiple-choice QA
pairs only** — the paper's 150 open-ended questions are missing, so rule 12
cannot run yet. It **ships one clip per question** (`video_uid`), so eval needs
no 20K download. Four domains × three subcategories: **Count, Action, Position,
Cognition**.

Reference numbers to beat, from the FineBadminton paper (ACM MM 2025):

| System | Overall MCQ |
|---|---|
| 7B open-source zero-shot | < 25% |
| Gemini 2.5 Pro + their sampling | 38.6% |
| Qwen2.5-VL-7B + their pipeline, fine-tuned | **42.06%** |

Their baseline is **uniform 2 FPS ≈ 25 frames per rally**. That is our frame
budget too.

---

## The twelve rules

Breaking any of these invalidates results. They outrank any implementation
preference.

1. **FBBench is touched only for final numbers.** Develop on FineBadminton-20K.
2. **Verify no clip overlap** between the two before any fine-tuning.
3. If creating a split, split **by match**, not by clip.
4. **Equal frame budget in every comparison.** The claim is *same frames, chosen
   better* — never *more frames*.
5. **Report per-domain, never overall alone.**
6. Primary efficiency metric is **input tokens and frame count**, not wall-clock.
7. **Reproduce the hit-centric baseline ourselves**; do not quote theirs.
8. Evaluate the agentic loop **training-free first**.
9. `temperature=0`, fixed seeds.
10. **Every reported number has a row in `results/MANIFEST.md`.** No row, no report.
11. Train MCQ and open-ended as **separate objectives** when fine-tuning.
12. Report **both ROUGE and BERTScore** for open-ended output.

---

## Things that are easy to get wrong

| Trap | Why it matters |
|---|---|
| Calling FineBadminton's sampling "a heuristic" | It is a **trained** module (VideoMAE → temporal encoder → hit probability, plus a FastRCNN-guided ReSampler). The accurate and stronger criticism is that it is **query-agnostic** |
| Saying LoRA reduces inference cost | LoRA reduces **training** cost. At inference it costs the same as the base model |
| Claiming the memory graph improves FBBench accuracy | FBBench is per-clip; memory answers cross-rally questions. Evaluated separately |
| Decoding video inside a training or eval loop | Dominates wall-clock. Extract frames **once** to disk |
| Hand-parsing `"The answer is (B) short serve"` | Silent scoring bugs. Use guided decoding |
| Varying the prompt between models | Then you are measuring prompts, not models. Freeze it |
| Defaulting an unparseable answer to "A" | Inflates accuracy by the always-A rate. Return `None` |
| Extracting frames for all 3,215 clips | Only extract clips FBBench references |

---

## Code layout

What is working vs stubbed is tracked in `STATUS.md`. Every stub raises
`NotImplementedError` naming its roadmap phase. Kaggle kernels live in
`kernels/` and are pushed with `kaggle kernels push -p <dir>`.

---

## How we work

**Branches:** `gouri`, `amal`, `vishal`, `sabeeh` — one per person, each mapped
to a file area so we rarely touch the same file. Merge to `main` daily at
standup. `main` must always run; the Kaggle notebook clones it.

**Mock mode first.** `src/bavlm/models/backend.py` has mock backends so the
whole pipeline runs on a laptop with no GPU:

```python
backend = build_backend({"mock": "oracle"}, answer_key=key)   # must score 100%
backend = build_backend({"mock": "random"}, answer_key=key)   # must score ~25%
```

**If oracle is not 100% and random is not ~25%, stop and fix the scorer.** A
real run would otherwise produce a number that looks fine and is wrong. Never
burn Kaggle quota on a pipeline that has not passed this.

**Kaggle:** GPU **T4 × 2** (Qwen2.5-VL-7B is 16.6 GB in fp16 — it does not fit
on one 16 GB card). Load with `device_map="auto"`, `torch_dtype=torch.float16`
(T4 is sm_75, no bf16), `attn_implementation="sdpa"` (flash-attn-2 needs
sm_80+). Frames come from an attached Kaggle Dataset, never re-downloaded.
Results are pushed to GitHub, not left in `/kaggle/working`.

---

## Conventions

- Python 3.10+, type hints on public functions, dataclasses over dicts for
  anything that crosses a module boundary
- Docstrings explain **why**, not what — especially where a choice is
  methodological rather than technical
- Results are JSONL, written incrementally with `flush()`, resumable by skipping
  IDs already present
- Commits: `<area>: <what changed>` e.g. `sampling: add hit-centric anchors`
- `python -m pytest tests/ -q` must pass before any merge to `main`

---

## What a good session looks like

Ask which roadmap phase the task belongs to. Check the twelve rules before
proposing an evaluation change. Prefer making mock mode pass over reaching for a
GPU. When a number looks surprising, suspect the harness before the model.
