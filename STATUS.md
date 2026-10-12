# Project status

Where the project stands and how to pick it up. Read this after
`START_HERE.md`. Who did what is in `docs/WORKLOG.md`; what the data actually
contains is in `results/dataset_report.md`.

**Last updated:** Sun 11 Oct 2026 (Day 0 of Sprint A) · **Review 2:** Wed 21 Oct · **Freeze:** Sun 18 Oct

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
| Real model backend (Qwen2.5-VL), scored MCQ answers, token/frame stats | `src/bavlm/models/backend.py` (`HFBackend`) | `tests/test_hf_backend.py` on a tiny Qwen2.5-VL, CPU |
| Frame cache: 2 FPS, short side 448, all FBBench clips | Kaggle kernel `gourilakshmi05/finebadminton-frame-cache` (`kernels/frame-cache/`) | v2 test run: 20 clips, 0 errors |
| Zero-shot GPU run | Kaggle kernel `gourilakshmi05/fbbench-zero-shot` (`kernels/fbbench-zeroshot/`) | Kaggle smoke test, 10 questions, v3 (11 Oct) |
| Mock backends | `src/bavlm/models/backend.py` | `tests/test_mock_backend.py` |

`python -m pytest tests/ -q` → 26 passed.

**Still stubs** (each raises `NotImplementedError` naming its phase):
`data/loader.py`, `eval/metrics.py`, `sampling/hit_centric.py`, `agent/{observer,reflector,loop}.py`,
`tools/{pose,tracker}.py`, `scripts/0{1,2,3,4}_*.py`.

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
- **Getting results to GitHub:** attach the `GH_TOKEN` secret to
  `fbbench-zeroshot` once, in the kernel editor (Add-ons → Secrets).
  Results then push to GitHub automatically. Without it, the kernel runs
  from the `bavlm-src` code snapshot and results stay in the kernel output.
- **Refreshing the snapshot:** the snapshot is a copy of `src/`, so refresh
  it after code changes:

```bash
git archive --format=tar HEAD src | tar -x -C <dir>
git rev-parse --short HEAD > <dir>/COMMIT
kaggle datasets version -p <dir> --dir-mode zip -m "src @ <sha>"
```

### Next steps, in roadmap order (Review 2 freeze: Sun 18 Oct)

1. **e03:** done. Qwen3-VL-8B confirmed as the backbone (above).
2. **e04, Gemini reference:** `kernels/fbbench-gemini`, CPU, same 500
   questions. Needs the `GEMINI_API_KEY` secret and a run started in the
   editor (Save Version → Save & Run All); CLI-started runs get no secrets.
3. **Resolution sweep:** `SHORT_SIDE` = 224 / 336 / None on the chosen
   backbone. Position is at chance for every model so far; this tests whether
   resolution is why.
4. **More backbones (optional):** InternVL3.5-8B, VideoLLaMA3-7B. Only if
   e03 is weak; each needs its own video input path.
5. **Loop demo:** the Review 2 slide shows the planned loop on one rally.
   Do not build the loop itself before Review 3.
6. **G1 overlap check:** done (`scripts/00_verify_dataset.py`). Before
   fine-tuning: frame-fingerprint spot check against the 50 training matches.
7. **Housekeeping:** set `BRANCH = "main"` in the kernels; secrets do not
   reach CLI runs, so results are copied from the kernel output by hand.

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
