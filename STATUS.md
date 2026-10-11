# Project status

Where the project stands and how to pick it up. Read this after
`START_HERE.md`. Who did what is in `docs/WORKLOG.md`; what the data actually
contains is in `results/dataset_report.md`.

**Last updated:** Sun 11 Oct 2026 (Day 0 of Sprint A) · **Review 2:** Wed 21 Oct · **Freeze:** Sun 18 Oct

---

## Gates

| Gate | Due | State |
|---|---|---|
| **G1** dataset report, zero overlap, schema pinned | Mon 12 Oct | **Partial.** Report and schema done. The overlap check is not done: FBBench renames its clips, so it needs Finebadminton-20K plus a content fingerprint (open item 1 in the work log) |
| **G2** one question answered end to end on Kaggle | Wed 14 Oct | Code ready and tested on CPU; Kaggle smoke test pending (below) |
| **G3** ≥500 questions scored, screening table | Fri 16 Oct | Not started |

---

## What works today

| Piece | Where | Verified by |
|---|---|---|
| FBBench loader, pinned schema, per-domain + sub-task tags | `src/bavlm/eval/fbbench.py` | `tests/test_fbbench_real.py`: 2,413 questions load; oracle 100%, random 23.8%, refuse 0% in every domain |
| Real model backend (Qwen2.5-VL), scored MCQ answers, token/frame stats | `src/bavlm/models/backend.py` (`HFBackend`) | `tests/test_hf_backend.py` on a tiny Qwen2.5-VL, CPU |
| Frame cache: 2 FPS, short side 448, all FBBench clips | Kaggle kernel `gourilakshmi05/finebadminton-frame-cache` (`kernels/frame-cache/`) | v2 test run: 20 clips, 0 errors |
| Zero-shot GPU run | Kaggle kernel `gourilakshmi05/fbbench-zeroshot` (`kernels/fbbench-zeroshot/`) | Full dry run on CPU with the tiny model; Kaggle smoke test pending |
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
kaggle kernels status gourilakshmi05/fbbench-zeroshot
kaggle kernels output gourilakshmi05/fbbench-zeroshot -p ./out
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

### Next steps, in roadmap order

1. **G2:** confirm the 10-question smoke test result (below). Check the
   answers, the token counts and the prediction distribution by hand. A
   model answering one letter for everything is a red flag.
2. **Merge `gouri` → `main`.** Then set `BRANCH = "main"` in
   `kernels/fbbench-zeroshot` cell 1.
3. **Backbone sweep (Day 4):** add loaders for the other five families
   (open items 6–7 in the work log).
4. **Full run (Day 5):** `EXPERIMENT = "e01_zeroshot"`, `N_QUESTIONS = 500`.
   **Resolution sweep:** `SHORT_SIDE` = 224 / 336 / None at 50 questions
   each.
5. **G1 overlap check:** open item 1, needs a decision on downloading
   Finebadminton-20K.

### Decisions the team owes

- **Rule 1 vs the roadmap.** The sweep runs on FBBench, which rule 1
  reserves for final numbers.
- **Frame budget.** Recommended: 2 FPS for Review 2 to match the paper,
  then a denser cache for Review 3 so the agent has frames to choose
  between.
- **Owners.** Fill in the D1–D4 owners in `docs/ROADMAP.md`.

---

## Kaggle smoke test (G2)

_Pending: filled in after the run._
