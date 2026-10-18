# Project status

Where the project stands and how to pick it up. Read this after
`START_HERE.md`. Who did what is in `docs/WORKLOG.md`; what the data actually
contains is in `results/dataset_report.md`.

**Last updated:** Mon 12 Oct 2026 · **Review 2:** Wed 21 Oct · **Freeze:** Sun 18 Oct

**Picking this up? Start with `docs/HANDOVER.md`**: what is done, and the next steps in order.

---

## Gates

| Gate | Due | State |
|---|---|---|
| **G1** dataset report, zero overlap, schema pinned | Mon 12 Oct | **Done** Mon 12 Oct. FBBench's 20 source matches (published with the 20K release) are held out; training uses the other 50, split by match in `configs/split_20k.json`. Label vocabulary matches the paper. See `results/dataset_report.md`. A frame-fingerprint spot check before fine-tuning is still advised |
| **G2** one question answered end to end on Kaggle | Wed 14 Oct | **Done** Sun 11 Oct: 10-question smoke test on T4 × 2 (below) |
| **G3** ≥500 questions scored, screening table | Fri 16 Oct | **Done** Sun 11 Oct: e01, e02, e03 (below) |

---

## What works today

| Piece | Where | Verified by |
|---|---|---|
| FBBench loader, pinned schema, per-domain + sub-task tags | `src/bavlm/eval/fbbench.py` | `tests/test_fbbench_real.py`: 2,413 questions load; oracle 100%, random 23.8%, refuse 0% in every domain |
| Real model backend (Qwen2.5-VL, Qwen3-VL), scored MCQ answers, token/frame stats | `src/bavlm/models/backend.py` (`HFBackend`) | `tests/test_hf_backend.py` on tiny Qwen2.5-VL and Qwen3-VL, CPU; e01–e05 on Kaggle |
| Gemini reference backend, enum-constrained answers | `src/bavlm/models/backend.py` (`GeminiBackend`) | `tests/test_gemini_backend.py` with a fake client |
| 20K overlap check, match split, label vocabulary | `src/bavlm/data/splits.py`, `scripts/00_verify_dataset.py` | `tests/test_splits.py` (synthetic + real release) |
| Frame cache: 2 FPS, short side 448, all FBBench clips | Kaggle kernel `gourilakshmi05/finebadminton-frame-cache` (`kernels/frame-cache/`) | v2 test run: 20 clips, 0 errors |
| Zero-shot GPU run: several models and resolutions in one session | Kaggle kernel `gourilakshmi05/fbbench-zero-shot` (`kernels/fbbench-zeroshot/`) | e01–e05, 0 errors since v7 |
| Mock backends | `src/bavlm/models/backend.py` | `tests/test_mock_backend.py` |

`python -m pytest tests/ -q` → 50 passed (41 + 9 skipped on a fresh clone without `data/`).

**Still stubs** (each raises `NotImplementedError` naming its phase):
`data/loader.py`, `eval/metrics.py`, `sampling/hit_centric.py`, `agent/{observer,reflector,loop}.py`,
`tools/{pose,tracker}.py`, `scripts/0{1,1b,2,3,4}_*.py` (the Kaggle notebook replaces 01 and 01b).

---

## What we learned about the data

Details and numbers are in `results/dataset_report.md`.

- **FBBench release (HF revision `ff86093a`):** **2,413 MCQ only.** The
  paper's 150 open-ended questions are missing, so rule 12 can't run yet.
- **Domains:** the domain is the `spatial` field. Count has 413 questions,
  Action 650, Position 600, Cognition 750, with 3 sub-tasks each.
- **Clips:** FBBench **ships its own clips**, one per question, named by
  `video_uid`. The evaluation cache needs no Finebadminton-20K download.
- **Clip length:** clips are **short** (2–5 s in the first 20) at 30 fps, so
  2 FPS gives about 10 frames, not the paper's ~25 for full rallies.
- **Answer key:** always-A scores 26.2%. That is the floor to beat.

---

## How to proceed

### Every session

```bash
git checkout <your-branch> && git pull origin main
python -m pytest tests/ -q        # must pass before any GPU time
```

### Get the data locally (question JSONs only, ~4 MB)

```bash
hf download Moujuruo/FineBadmintonBenchmark --repo-type dataset \
    --include "finebadmintonbenchmark/*.json" --local-dir data/fbbench
```

Without this, `tests/test_fbbench_real.py` skips.

### Kaggle

All Kaggle work is pushed from the repo with the CLI. Edit the notebook in
`kernels/`, commit, then push. Log in once with `kaggle auth login`, and
delete any old `~/.kaggle/kaggle.json`, which otherwise takes precedence
over OAuth.

```bash
# frame cache (CPU). Already built; re-run only to change FPS or SIZE
kaggle kernels push -p kernels/frame-cache

# zero-shot run (T4 x2): set EXPERIMENT / N_QUESTIONS / BRANCH in cell 1 first
kaggle kernels push -p kernels/fbbench-zeroshot
kaggle kernels status gourilakshmi05/fbbench-zero-shot
kaggle kernels output gourilakshmi05/fbbench-zero-shot -p ./out
```

- **Access:** the kernels live on Gouri's account and are private.
  Teammates need to be added as collaborators (kernel page → Share) to see
  or attach them.
- **Getting results to GitHub:** Kaggle passes secrets (`GH_TOKEN`) only to
  runs started in the editor, never to `kaggle kernels push` runs. CLI runs
  therefore use the `bavlm-src` code snapshot and leave results in the kernel
  output: download them, copy to `results/`, `git add -f results/`, commit.
  `kaggle kernels output` only returns the latest version.
- **Refreshing the snapshot:** the snapshot is a copy of `src/`, so refresh
  it after code changes:

```bash
git archive --format=tar HEAD src | tar -x -C <dir>
git rev-parse --short HEAD > <dir>/COMMIT
kaggle datasets version -p <dir> --dir-mode zip -m "src @ <sha>"
```

### Next steps

In order, with exact settings: **`docs/HANDOVER.md`**. In short: full
2,413-question run on Qwen3-VL-8B at 448 · Gemini reference run (editor) ·
720 px cache to test whether more resolution fixes Position · hand-check 20
answers · loop demo for the slide · deck.

### Decisions the team owes

- **Rule 1 vs the roadmap.** The sweep runs on FBBench, which rule 1
  reserves for final numbers.
- **Frame budget.** Recommended: 2 FPS for Review 2 to match the paper,
  then a denser cache for Review 3 so the agent has frames to choose
  between.
- **Owners.** Fill in the D1–D4 owners in `docs/ROADMAP.md`.

---

## Kaggle smoke test (G2)

Kernel `gourilakshmi05/fbbench-zero-shot` v3, Sun 11 Oct 2026, code
snapshot `13a4d62`, Qwen2.5-VL-7B-Instruct fp16, uniform 2 FPS, short side 448.
Result files are in `results/e00_smoke_*` (local only, gitignored); manifest row added.

- **Pipeline:** 2,413/2,413 questions have cached frames. Mock check on Kaggle:
  oracle 100%, random 23.8%, refuse 0%.
- **Answers look real:** predictions B 5 · D 3 · C 1 · A 1 (gold D 3 · B 3 ·
  A 2 · C 2). No single-letter collapse. Each prediction is the argmax of the
  choice log-probs, which sum to 1 over the options.
- **Cost:** 6–19 frames, mean 11.1; mean 2,728 input tokens. Visual tokens
  are ~448 per frame *pair* (Qwen merges 2 frames temporally), so odd frame
  counts pay for a padding frame. After the first question (155 s, model
  warm-up) it runs at 2–10 s per question.
- **6/10 correct.** n=10, one question per sub-task: this is a plumbing check,
  not a result. Do not quote it.

**Fixes needed to get here** (all in `kernels/fbbench-zeroshot/`):

1. Kaggle stores the frame cache's 100k-file output as one 6.9 GB
   `_output_.zip` and mounts it unextracted. Cell 8 now unpacks it to `/tmp`
   (86–120 s per run). Publishing the cache as a Kaggle Dataset would remove
   this step.
2. The no-token snapshot fallback pointed `sys.path` one level too deep.
3. Kaggle created the kernel slug from its title: `fbbench-zero-shot`, not
   `fbbench-zeroshot`. Metadata and docs now use the real slug.

**Frame budget:** the longest clip is 1,222 s (2,446 frames at 2 FPS). The
kernel now caps every clip at `MAX_FRAMES = 25`, the `budget_frames` in
`configs/default.yaml`, so long rallies are subsampled uniformly to the same
budget as every other method (rule 4).

## Full FBBench run: Qwen3-VL-8B, all 2,413 questions (e06)

Fri 17 Oct 2026, 16:01–18:32 UTC, on `gourilakshmi05/fbbench-zero-shot` (own config copy, run in the editor), snapshot `3c641a3`, fp16 on T4 × 2, uniform 2 FPS, short side 448, cap 25 frames, **every question in the release**. **0 errors.** 147.2 min.

| Domain | n | Accuracy | Always-A floor | Input tokens | s/q |
|---|---|---|---|---|---|
| Count | 413 | **40.7%** | 28.6% | 3,862 | 5.1 |
| Action | 650 | 24.5% | 25.1% | 2,246 | 3.1 |
| Position | 600 | 20.2% | 23.8% | 2,095 | 2.7 |
| Cognition | 750 | **48.5%** | 27.9% | 3,213 | 4.2 |

Overall 33.7% (812/2,413) against an always-A floor of 26.2%. Cost: 14.0 frames, 2,786 input tokens, 3.66 s per question. Predictions are spread (C 882 · B 669 · A 441 · D 421) against gold A 633 · C 617 · B 594 · D 569 — a mild C/B lean, no single-letter collapse.

- **Consistent with e03** (same backbone and settings, first 500 questions in the same order): Count 40.7 vs 41.1, Action 24.5 vs 27.8, Position 20.2 vs 19.0, Cognition 48.5 vs 50.0 — every domain within ~3.3 pts, so the 500-question numbers were representative.
- **Above its always-A floor on Count (+12.1 pts) and Cognition (+20.6). Below its floor on Action (24.5 vs 25.1) and Position (20.2 vs 23.8): always answering A would score higher than the model on Position.** Uniform 2 FPS at 448 px does not let a zero-shot 8B read stroke or court events — that is the deck's motivation for the query-conditioned observation of Review 3.
- Against the FineBadminton paper's overall MCQ: below Gemini 2.5 Pro + their sampling (38.6%) and below their fine-tuned Qwen2.5-VL-7B pipeline (42.06%). Expected for a training-free zero-shot run — the gap is what Review 3's agentic loop targets without fine-tuning.
- Per-sub-task table: `results/e06_full_Qwen3-VL-8B-Instruct_by_subtask.csv`. Weakest: hitting prediction 17.0% (6.0 frames' worth of information); strongest: point reason 59.6%.

## Resolution sweep: Qwen3-VL-8B at 224 / 336 / 448 (e05, kernel v10)

Mon 12 Oct, snapshot `666507b`. Same 500 questions, same 25-frame cap; frames
downscaled at load from the 448 cache (448 = e03). **0 errors.** 336: 22.9 min,
224: 10.5 min.

| Short side | Count | Action | Position | Cognition | Input tokens | Visual tokens | s/q |
|---|---|---|---|---|---|---|---|
| 224 | 37.9% | 29.4% | 14.3% | 49.2% | **872** | 649 | **1.26** |
| 336 | 41.1% | 27.8% | **12.7%** | 48.4% | 1,692 | 1,468 | 2.75 |
| **448** (e03) | 41.1% | 27.8% | **19.0%** | 50.0% | 2,928 | 2,705 | 4.17 |

Paired against 448 (McNemar): only Position at 336 differs (p = 0.02); every
other cell p ≥ 0.18. Always-A on Position is 18.3%.

- **Count, Action and Cognition do not depend on resolution** between 224 and
  448. 224 px costs **3.4× fewer input tokens** for the same answers there.
- **Position is the only resolution-sensitive domain, and it gets worse as
  frames shrink** (19.0 → 12.7 / 14.3%): hitting prediction 16.7 → 7.1 / 9.5,
  moving recognition 26.2 → 19.0. Even 448 is only at chance.
- **Decision for Review 2: keep 448** for the reported numbers. The open
  question is whether **more** resolution lifts Position; that needs a 720 px
  frame cache (next steps).
- For the agentic loop (~3× the calls per question), 224 is the cheap option
  for Count / Action / Cognition questions; resolution could be chosen per
  question by the planner, which is itself an observation-budget decision.

## Chosen backbone: Qwen3-VL-8B, 500 questions (e03, kernel v9)

Sun 11 Oct, snapshot `6ec245b`, fp16 on T4 × 2, uniform 2 FPS, short side 448,
cap 25 frames. The same 500 questions as e01, in the same order. **0 errors.**
34.7 min.

| Domain | n | Qwen2.5-VL-7B (e01) | **Qwen3-VL-8B** | Δ pts | p (McNemar) | Always-A | Tokens 7B → Q3 |
|---|---|---|---|---|---|---|---|
| Count | 124 | 41.9% | 41.1% | −0.8 | 1.00 | 25.8% | 5,001 → 4,012 |
| Action | 126 | 21.4% | **27.8%** | +6.3 | 0.24 | 23.0% | 2,909 → 2,348 |
| Position | 126 | 19.8% | 19.0% | −0.8 | 1.00 | 18.3% | 2,542 → 2,046 |
| Cognition | 124 | 33.9% | **50.0%** | **+16.1** | **0.002** | 26.6% | 4,107 → 3,330 |

Paired over the same questions (McNemar, exact): Qwen3-VL-8B is right on 85
that Qwen2.5-VL-7B gets wrong and wrong on 59 that it gets right.

- **Qwen3-VL-8B is the backbone for Review 2.** It is never meaningfully worse
  on any domain, is significantly better on Cognition, uses 19% fewer input
  tokens on every domain and takes 22% less time per question.
- **Cognition** gains come from point reason (+27 pts) and hitting comment
  (+21).
- **Count** is a wash overall but hides a swap: round count +20 pts, detailed
  hitting count −22. Worth a look when hand-checking answers.
- **Position is still at chance** (19.0% vs 18.3% always-A). The backbone
  change did not touch it; the resolution sweep is next.
- Predictions lean to C (181) and B (158) against a gold split of 133/133/117/117.

## Backbone screen, 150 questions (e02, kernel v8)

Sun 11 Oct, snapshot `6ec245b`, fp16 on T4 × 2, uniform 2 FPS, short side 448,
cap 25 frames. Same 150 questions (round-robin, ~37 per domain), same frames,
same prompt for every model. Qwen2.5-VL-7B is the first 150 rows of e01.

| Model | Count | Action | Position | Cognition | Errors | Input tokens | s/q |
|---|---|---|---|---|---|---|---|
| **Qwen3-VL-8B** | 29.7% | **30.8%** | 18.9% | **43.2%** | 0 | **3,057** | 4.9 |
| Qwen2.5-VL-7B | 24.3% | 23.1% | **24.3%** | 35.1% | 0 | 3,793 | 6.2 |
| Qwen2.5-VL-3B | 24.3% | 17.9% | 18.9% | 16.2% | **46 non-finite** | 3,793 | 3.8 |

Always-A on these 150: 22.7%. Mean 15.5 frames for every model (rule 4).

- **Qwen3-VL-8B leads** on Action and Cognition, uses 19% fewer input tokens
  (350 vs 448 visual tokens per frame pair) and runs faster. It is the current
  pick for the main backbone.
- **Position is at or below chance for all three.** Court position is not
  readable from 448-px frames by any of them; this is the case for the
  resolution sweep, not for a different backbone.
- **Qwen2.5-VL-3B is not a valid entry**: 46/150 scores overflowed in fp16
  (bf16-trained weights on a T4) and count as wrong. Do not rank it.
- With ~37 questions per domain, one question is ~2.7 points; only the
  Cognition and Action gaps are larger than noise. Confirm the pick on the
  500 questions before freezing.

## Zero-shot baseline: Qwen2.5-VL-7B, 500 questions (e01, kernel v7)

Sun 11 Oct, snapshot `39c567a`, fp16 on T4 × 2, uniform 2 FPS, short side 448,
cap 25 frames. **0 errors** (the GQA fix removed every OOM). 44.7 min.

| Domain | n | Accuracy | Frames | Input tokens | s/q |
|---|---|---|---|---|---|
| Count | 124 | **41.9%** | 21.1 | 5,001 | 7.2 |
| Action | 126 | **21.4%** | 11.7 | 2,909 | 5.0 |
| Position | 126 | **19.8%** | 10.3 | 2,542 | 3.5 |
| Cognition | 124 | **33.9%** | 16.4 | 4,107 | 5.7 |

Always-A on these 500: 23.4%. Predictions are spread (C 141, B 141, D 124,
A 94). Action and Position sit at or below chance: the model cannot yet read
stroke type or court position from 448-px frames. Weakest sub-tasks: action
prediction and hitting prediction (11.9%). Per-sub-task table:
`results/e01_zeroshot_by_subtask.csv` (local, from the kernel output).

The 150-question backbone screen uses the first 150 of the same round-robin
order, so this run's first 150 rows are Qwen2.5-VL-7B's screening entry.

## First 500-question run (e01, kernel v6): invalid

Sun 11 Oct, snapshot `5f28597`. **138/500 questions hit CUDA OOM** and were
scored wrong, so its 19.6% overall must not be reported. The OOMs were every
input of 4,885+ tokens (long rallies: round count, detailed hitting count,
point reason, round comment); every input up to 4,768 tokens ran.

Cause, measured with a GPU probe kernel (`gourilakshmi05/sdpa-probe`): with no
attention mask, transformers passes `enable_gqa=True` to SDPA. On a T4 only
the math kernel accepts that, and it materialises the attention matrix:
+5.8 GiB at 4.8k tokens, OOM (21.6 GiB) at 14.4k. Repeating the kv heads first
lets the memory-efficient kernel run: +0.1 GiB at 14.4k.
`avoid_gqa_math_fallback` in `backend.py` does this on pre-Ampere GPUs.

On the 362 answered questions, accuracy was 18–33% by domain against an
always-A rate of 22.4% on the same subset. That is consistent with the
paper's "< 25%" for 7B zero-shot, but it is a biased subset; wait for the
rerun.
