# Handover: next steps to Review 2

For the teammate picking up from Gouri on Mon 12 Oct 2026.
**Review 2: Wed 21 Oct. Technical freeze: Sun 18 Oct.**

Read first: `CLAUDE.md` (the twelve rules), then `STATUS.md` (every result
with its table), then `results/dataset_report.md` (what the data actually is).
Every number below has a row in `results/MANIFEST.md`.

---

## Where things stand

| Done | Result | Where |
|---|---|---|
| Frame cache: all 2,413 FBBench clips, 2 FPS, short side 448 | Kaggle kernel `finebadminton-frame-cache` | `kernels/frame-cache/` |
| G1: overlap check, 20K split, label vocabulary | FBBench's 20 source matches held out; 50 left, split by match (45 train / 5 val) | `results/dataset_report.md`, `configs/split_20k.json` |
| G2: smoke test on Kaggle T4 × 2 | Pipeline works end to end | `STATUS.md` |
| e01: Qwen2.5-VL-7B, 500 q | Count 41.9 · Action 21.4 · Position 19.8 · Cognition 33.9 | `STATUS.md` |
| e02: backbone screen, 150 q | Qwen3-VL-8B best; Qwen2.5-VL-3B invalid (fp16 overflow) | `STATUS.md` |
| e03: **Qwen3-VL-8B, 500 q (chosen backbone)** | Count 41.1 · Action 27.8 · Position 19.0 · **Cognition 50.0**; 19% fewer tokens | `STATUS.md` |
| e05: resolution sweep 224 / 336 / 448 | Flat except Position, which drops below chance at lower resolution. **Keep 448** | `STATUS.md` |
| e06: **Qwen3-VL-8B, all 2,413 questions** | Count 40.7 · Action 24.5 · Position 20.2 (below its 23.8 floor) · Cognition 48.5, overall 33.7; **0 errors** | `STATUS.md`, `results/e06_full_*` |
| Gemini backend + CPU kernel | Code done and tested; **not run yet** (step 2) | `kernels/fbbench-gemini/` |
| Gates G1, G2, G3 | All met | `STATUS.md` |

The 500 questions: the 2,413 FBBench MCQs taken round-robin over the 12
sub-tasks (about 42 each, about 125 per domain), each from a different clip.
The order is deterministic, so every run uses the same questions.

Always-A floors on these 500: Count 25.8 · Action 23.0 · Position 18.3 · Cognition 26.6.

---

## Setup (once, about 30 min)

```bash
git clone https://github.com/gourilaxmi/badminton-agentic-vlm && cd badminton-agentic-vlm
git checkout -b <your-branch> origin/main
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows path; bin/ on Linux/Mac
python -m pytest tests/ -q                          # data tests skip until step below
hf download Moujuruo/FineBadmintonBenchmark --repo-type dataset \
    --include "finebadmintonbenchmark/*.json" --local-dir data/fbbench
hf download Moujuruo/Finebadminton-20K --repo-type dataset \
    --include "finebadminton-20K/*.json" "*.txt" --local-dir data/fb20k
python -m pytest tests/ -q                          # now all 50 should pass
```

**Kaggle.** The kernels and the frame cache are on Gouri's account
(`gourilakshmi05`). Ask her to add you as a collaborator on
`finebadminton-frame-cache`, `fbbench-zero-shot`, `fbbench-gemini` and the
`bavlm-src` dataset (each page → Share). To run on **your** GPU quota, change
the `id` in a kernel's `kernel-metadata.json` to `<your-username>/<slug>`
(keep the `kernel_sources` / `dataset_sources` pointing at Gouri's) and push
with `kaggle kernels push -p kernels/<dir>`.

**Three Kaggle facts that cost us hours:**
- **Secrets (`GH_TOKEN`, `GEMINI_API_KEY`) never reach runs started with
  `kaggle kernels push`**, only runs started in the editor (Save Version →
  Save & Run All). So CLI runs use the `bavlm-src` code snapshot and never
  push results. Download results by hand:
  `kaggle kernels output <user>/fbbench-zero-shot -p out --file-pattern "(results/.*|.*\.log)$"`.
- **Refresh `bavlm-src` after any change under `src/`** (commands in `STATUS.md` → Kaggle).
- On Windows, run the Kaggle CLI with `PYTHONUTF8=1`.

---

## Next steps, in order

### 1. Full FBBench run: all 2,413 questions (about 3 h GPU, one session) — **DONE (e06, Sat 18 Oct)**

The number that can sit next to the paper's. The paper reports overall MCQ
accuracy on all questions; we have only 500 so far.

In `kernels/fbbench-zeroshot/fbbench_zeroshot.ipynb`, config cell:
```python
EXPERIMENT  = "e06_full"
MODELS      = ["Qwen/Qwen3-VL-8B-Instruct"]
N_QUESTIONS = None          # all 2,413
SHORT_SIDES = [None]        # 448
```
Push, wait, download, copy the files into `results/` and commit them with
`git add -f results/` (the folder is gitignored; `kaggle kernels output`
only ever returns the **latest** version, so commit before the next run).
Then add a `STATUS.md` section and a `MANIFEST.md` row (copy e03's format). Report **per domain** (rule 5) next to the
always-A floor for that domain; overall only alongside.

### 2. Gemini reference run (e04): needs Gouri's account or your own key

`kernels/fbbench-gemini/` is ready: same 500 questions and frames, answers
constrained to one letter, CPU only (no GPU quota), roughly a few dollars on
`gemini-2.5-pro`. It **must be started from the Kaggle editor** with the
`GEMINI_API_KEY` secret ticked and Accelerator = None. Gouri has the key on
her account; on yours, add your own from aistudio.google.com. Record it as a
**reference row**, not a backbone candidate (it cannot be fine-tuned or
inspected for the faithfulness study).

### 3. Does more resolution fix Position? (720 px cache)

Position is at chance at 448 and gets worse at 336 / 224 (`STATUS.md`, e05),
so the open question is whether **more** pixels help.
- Copy `kernels/frame-cache/` to a new kernel (new `id`, e.g.
  `<user>/finebadminton-frame-cache-720`) with `SIZE = 720`. Do **not**
  re-run the original: the zero-shot kernel reads its output. CPU only,
  about 1–2 h. Only Position clips are needed if time is short.
- Point the zero-shot kernel's `kernel_sources` at the new cache and run
  `MODELS = ["Qwen/Qwen3-VL-8B-Instruct"]`, `SHORT_SIDES = [None]`, the same
  500 questions. Expect about 2.5× the tokens of 448 (~7k per question).
  The OOM fix (`avoid_gqa_math_fallback`) should cover it; watch for `oom`
  errors in the log.
- If Position stays at chance even at 720, that is a finding for the deck:
  the bottleneck is not pixels, which motivates the agent's query-conditioned
  look (zoom, crop, pose) in Review 3.

### 3b. Denser frames within the same 25-frame budget (Count, Action)

Error analysis of e03: frames are 0.5 s apart at 2 FPS, so most racket
contacts fall between frames. The model under-counts detailed hitting count
73% of the time (over-counts 2%), and confuses shots that look alike in a still
frame. Yet Action and Position clips use only ~12 and ~10 of the 25 allowed
frames, because the clips are short. Test: a frame cache at **6 FPS**
(`kernels/frame-cache`, copied to a new id, `FPS = 6`), still capped at 25
frames, so no comparison uses more frames than the paper's budget (rule 4).
Run the same 500 questions. If Count and Action rise, it is evidence that
*when* the frames are taken matters, which is the agent's premise.

### 4. Hand-check 20 scored answers (D4, about 1 h, laptop)

From `results/e03_zeroshot_Qwen3-VL-8B-Instruct.jsonl` (in the repo: every
run's per-question JSONL is committed under `results/`). Pick 20 across
domains, open the clip, check gold and prediction by eye. Start with
**detailed hitting count**, which fell 22 points from Qwen2.5 to Qwen3 while
round count rose 20. Note anything wrong with the question or key.

### 5. Loop demo for the slide (D3, 2–3.5 h, laptop, mock mode) — **DONE (Sat 18 Oct)**

**DONE (Sat 18 Oct):** `python scripts/05_loop_demo.py` runs 10 questions
(round-robin over the 12 sub-tasks) through plan → evidence → reflection →
round 2 against the deterministic mock. Output: `results/loop_demo_traces.jsonl`
(all 10 traces) and `results/loop_demo_slide.md` (one cognition trace formatted
for the slide). The real loop code is in `agent/{observer,reflector,loop}.py`;
the mock in `agent/mock.py` stands in for judgement only, so nothing here
supports an accuracy claim. Tests: `tests/test_agent_loop.py` (budget clamp,
span/frame selection, round cap, eid/citation integrity).

Original note: one worked trace, **qualitative only, claim no accuracy.** Do
not build the full loop evaluation before Review 3 (`scripts/03_run_agent.py`
remains a phase-3 stub).

### 6. Small fixes

- `CLAUDE.md` "Datasets": the 20K release is **not** a consistent 25 FPS
  (1,370 rallies at 30 FPS) and its mean rally is **10.4 s**, not 12.4 s; the
  12.4 s figure is for the full 3,215-rally dataset. Also "2,563 QA pairs,
  150 open-ended": this release has only the 2,413 MCQ.
- Read FineBadminton §6 and record exactly how they fine-tuned (LoRA or
  full, epochs, LR, data). We only know: Qwen2.5-VL-7B, ~60k QA pairs (10%
  MCQ), with their hit-centric sampler → 42.06%.

### 7. Deck (Mon 19 – Wed 21)

Review 2 deliverables (`docs/ROADMAP.md`, bottom), with current status:

| # | Deliverable | Status |
|---|---|---|
| 1 | Dataset report, zero contamination, label vocabulary | **Done** |
| 2 | Backbone screening table | **Done** (+ Gemini row from step 2) |
| 3 | Zero-shot accuracy on the chosen backbone, per domain | **Done**, full run e06 on all 2,413 (overall 33.7%) |
| 4 | Resolution sweep, accuracy vs tokens | **Done** (+ 720 from step 3) |
| 5 | Inference cost: seconds and tokens per query | **Done** |
| 6 | Loop demonstration | **Done** (mock, 10 clips; `results/loop_demo_slide.md`) |
| 7 | Repo green, manifest complete, fresh-clone reproducible | **Done** (fresh clone: 41 pass, 9 skip without data) |
| 8 | Two slides of honest risk | To write |

Honest risks worth a slide: Position at chance for every model and
resolution; overlap rests on the publisher's list until a frame-fingerprint
check; the 150 open-ended questions are missing from the release (rule 12
cannot run); backbone choice made on FBBench (rule 1, team decision);
fp16-only T4s broke Qwen2.5-VL-3B.

---

## Not for Review 2

- **Fine-tuning.** Rule 8: the agentic loop is evaluated training-free first,
  so any gain is from observation strategy, not dataset fitting. LoRA is a
  Review 4 ablation. Before it: build QA training pairs from the 45 train
  matches, and run a frame-fingerprint overlap spot check.
- **Hit-centric baseline** (`sampling/hit_centric.py`, rule 7): phase 2.
- **More backbones** (InternVL3.5-8B, VideoLLaMA3-7B): only if there is
  spare time; Qwen3-VL-8B is a clear choice.
