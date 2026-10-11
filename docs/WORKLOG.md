# Work log

Who did what, so work is not duplicated and every change has an owner. Add
your own section; keep entries short and link the commit. The **Open** list
at the bottom is work nobody has claimed yet: put your name on an item
before starting it.

Roadmap owners (D1–D4) are not yet assigned in `docs/ROADMAP.md`. Agree them
at standup and fill that table in.

---

## Gouri

### Sun 11 Oct 2026 (Day 0) · branch `gouri`

| Commit | What | Roadmap item |
|---|---|---|
| `9a3c84a` | Frame-cache Kaggle kernel (`kernels/frame-cache/`), first version with a diagnostic mode that dumped the real FBBench schema | Day 2 · extraction |
| `3a64eac` | **Pinned the FBBench schema** in `src/bavlm/eval/fbbench.py`: `video_uid`, `question.question_en`, `options_en`, `answer`, domain = `spatial`, sub-task from file name. Strict loading. Fixed the doubled option letters (`A. A. 8`) and UTF-8 reading. Added `tests/test_fbbench_real.py` (mock backends on all 2,413 real questions) and `results/dataset_report.md` | Day 1 · schema pin, dataset report · Day 3 · mock passes |
| `2d0a0d9` | Frame cache rebuilt from **FBBench's own bundled clips** (no Finebadminton-20K needed), pinned to HF revision `ff86093a`. Revision recorded in `results/MANIFEST.md` | Day 1 · revision hashes · Day 2 · extraction |
| `13a4d62` | **`HFBackend` wired** for Qwen2.5-VL: answers scored from letter log-probs (guided decoding), real fps passed to the processor, tokens and frames recorded per call. `tests/test_hf_backend.py` runs it on a tiny model on CPU. **Zero-shot notebook moved to `kernels/fbbench-zeroshot/`**, now attaches the frame cache directly and pushes results with `git add -f` | Day 2 · VLMBackend real path, scorer + guided decoding · Day 3 · notebook wired |

**On Kaggle (account `gourilakshmi05`, all private):**

| Resource | What | State |
|---|---|---|
| kernel `finebadminton-frame-cache` | 2 FPS, short side 448, all 2,413 FBBench clips | v3 full run (see `STATUS.md`) |
| dataset `bavlm-src` | code snapshot of `src/` at `13a4d62` for the GPU kernel when no `GH_TOKEN` is attached | created |
| kernel `fbbench-zeroshot` | Qwen2.5-VL-7B, T4 × 2, 10-question smoke test (gate G2) | see `STATUS.md` |

**Found along the way** (details in `results/dataset_report.md`):

- **FBBench release:** it has 2,413 MCQ, one clip per question. The paper's
  150 open-ended questions are missing.
- **Label quirk:** `pointer_recog_updated` is mislabelled
  `temporal: "round comment"`, so the sub-task is taken from the file name.
- **Clip length:** FBBench clips are short (2–5 s in the first 20) and run at
  30 fps. At 2 FPS that is about 10 frames per clip, not 25.
- **Rule 2:** the overlap check can't work on filenames, because FBBench
  renames clips. An approach is suggested in the report; nothing is
  implemented.
- **Answer key:** always-A scores 26.2%. That is the per-domain floor to
  beat, not 25%.

**Local setup changes:**
- `~/.kaggle/kaggle.json` renamed to `kaggle.json.bak`, so the Kaggle CLI
  uses OAuth (`kaggle auth login`).
- `torchvision` (CPU build) added to `.venv` for the backend tests.

---

## Amal

_(nothing logged yet)_

## Vishal

_(nothing logged yet)_

## Sabeeh

_(nothing logged yet)_

---

## Open: unclaimed work

Put your name next to an item before you start it. Ordered by the roadmap.

| # | Task | Roadmap | Notes |
|---|---|---|---|
| 1 | **Clip-overlap check, rule 2 (gate G1)** | Day 1 · D1 | Needs Finebadminton-20K (~30 GB, unverified). Filenames won't work; content hash then perceptual frame fingerprint, see `results/dataset_report.md`. Required before fine-tuning; G1 is blocked on it |
| 2 | Record the Finebadminton-20K revision in `results/MANIFEST.md` | Day 1 · D1 | Do it when downloading for #1 |
| 3 | Label vocabulary: 11 hit types, 20 subtypes, 9 regions | Day 1 · D2 | From Finebadminton-20K annotations; `src/bavlm/data/splits.py:label_vocabulary` |
| 4 | Count clips with Decision Evaluation annotations | Day 1 · D3 | Finebadminton-20K |
| 5 | Find the 150 open-ended FBBench questions | — | Not in the HF release. Ask the authors or check the paper's GitHub. Rule 12 needs them |
| 6 | Extra backbone loaders in `HFBackend`: Qwen3-VL-8B, VideoLLaMA3-7B, InternVL3.5-8B, Qwen2.5-VL-3B, SmolVLM2-2.2B | Day 4 · D2 | `HFBackend` is Qwen2.5-VL-tested only. Each family may need its own input handling; `configs/backbones.yaml` |
| 7 | Update `scripts/01b_backbone_sweep.py` to use `choices=` scoring and log tokens | Day 4 · D2 | Copy the pattern in `kernels/fbbench-zeroshot` cell 6 |
| 8 | Planner / observer / reflector prompts | Day 2 · D3 | `src/bavlm/agent/prompts/` |
| 9 | Loop demo on 10 clips | Day 6 · D3 | Qualitative only |
| 10 | Update `RUNBOOK.md` | — | Still contains the old auto-detecting `fbbench.py` and the laptop-extraction flow |
| 11 | Decide: rule 1 vs the roadmap | Team | The sweep runs on FBBench, which rule 1 reserves for final numbers. Option: fixed screening subset, final numbers on the rest |
| 12 | Decide: frame budget | Team | 2 FPS (paper protocol, ~10 frames on these clips) vs a fixed count. Recommended: 2 FPS for Review 2; a denser cache for Review 3 |
