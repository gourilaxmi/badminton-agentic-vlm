# START HERE

Everything for the project is in this repo. This page says what each file is and
what to do first.

**Review 2: Wed 21 Oct 2026 · technical freeze Sun 18 Oct.**

---

## Right now, in order

```bash
# 1. repo, four branches, CI, pushed to GitHub  (~3 min)
bash setup_repo.sh <your-github-username>

# 2. START THE DOWNLOADS. Tonight. They run unattended and the whole
#    schedule hangs off them finishing.
hf download Moujuruo/FineBadmintonBenchmark --repo-type dataset --local-dir data/fbbench
hf download Moujuruo/Finebadminton-20K      --repo-type dataset --local-dir data/finebadminton-20k

# 3. Claude Code — reads CLAUDE.md automatically
claude
```

Then paste the kickoff message from `docs/KICKOFF.md`.

While the downloads run: Kaggle phone verification, API token, and the
`GH_TOKEN` secret. All of it is in `docs/SETUP_KAGGLE.md` and none of it needs
the data.

---

## What is in here

### Read first

| File | What it is |
|---|---|
| **`CLAUDE.md`** | Project context. Claude Code reads this automatically at the start of **every** session — yours and your teammates'. Keep it current |
| **`docs/HANDOFF.md`** | The full brief: novelty, all 11 papers with venues, datasets, hardware, architecture, the twelve rules, stock answers for panel questions |
| **`docs/ROADMAP.md`** | Day-by-day plan to Review 2, then Sprints B and C to March |
| **`RUNBOOK.md`** | Complete setup with every line of code inlined. The reference when something breaks |

### Setup guides

| File | What it is |
|---|---|
| `docs/SETUP_CLAUDE.md` | Installing and using Claude Code for this project |
| `docs/SETUP_KAGGLE.md` | Kaggle step by step — phone verification, secrets, frame cache, T4 × 2 |
| `docs/KICKOFF.md` | The message to paste into a fresh Claude session |

### Reference

| File | What it is |
|---|---|
| `docs/NOVELTY.md` | The four contributions, ranked, with falsifiable predictions |
| `docs/architecture.png` | System diagram. Orange dashed = ours, navy = adapted from prior work |

### Code that works today

| File | What it does |
|---|---|
| `src/bavlm/models/backend.py` | VLM backend with six mock modes. **Oracle must score 100%, random ~25%** |
| `src/bavlm/eval/fbbench.py` | FBBench loader, the frozen prompt, answer extraction, per-domain scoring |
| `src/bavlm/eval/faithfulness.py` | Counterfactual ablation (Sprint C, but written) |
| `src/bavlm/data/schema.py` | Plan / Evidence / Reflection / Trace |
| `src/bavlm/data/splits.py` | The clip-overlap check |
| `src/bavlm/memory/graph.py` | Player tendency graph, SQLite |
| `src/bavlm/sampling/uniform.py` | The 2 FPS baseline sampler |
| `scripts/00_verify_dataset.py` | **Run this first.** Overlap check and label vocabulary |
| `scripts/extract_frames.py` | Frame cache builder — laptop, CPU, once |
| `scripts/01b_backbone_sweep.py` | Screen six backbones on 150 stratified questions |
| `notebooks/fbbench_zeroshot_kaggle.ipynb` | Upload to Kaggle, edit three config lines, run |
| `configs/backbones.yaml` | The six candidate models with notes on each |

### Stubs, each naming its roadmap phase

`data/loader.py` · `sampling/hit_centric.py` · `agent/{observer,reflector,loop}.py` ·
`tools/{pose,tracker}.py` · `eval/metrics.py` · `scripts/0{1,2,3,4}_*.py`

---

## The five things that are easy to get wrong

Full table in `CLAUDE.md`. These five cost the most:

1. **Their sampling is not "a heuristic."** It is a trained module — VideoMAE,
   temporal encoder, FastRCNN-guided ReSampler. The accurate and stronger
   criticism is that it is **query-agnostic**.
2. **LoRA does not reduce inference cost.** It reduces *training* cost.
3. **The memory graph does not improve FBBench accuracy.** FBBench is per-clip.
4. **Never default an unparseable answer to "A".** Return `None`, or you inherit
   the always-A rate as free accuracy.
5. **Extract frames once, to disk.** Decoding video inside a loop will dominate
   your wall-clock.

---

## The three gates

| Gate | Date | Condition |
|---|---|---|
| **G1** | Mon 12 Oct | `dataset_report.md` committed, **zero clip overlap confirmed**, schema pinned |
| **G2** | Wed 14 Oct | **One FBBench question answered end to end and scored, on Kaggle** |
| **G3** | Fri 16 Oct | ≥500 questions scored, screening table complete |

G1 is blocking. If clips appear in both the training corpus and the benchmark,
every later number is void — and you want to know that tomorrow, not in January.

---

## Daily rhythm

```bash
git checkout <your-branch>    # gouri / amal / vishal / sabeeh
git pull origin main
claude
```

15-minute standup. Merge to `main` at every standup. `main` must always run —
the Kaggle notebook clones it.

---

## Before any GPU time

```bash
python -m pytest tests/ -q
python tests/test_mock_backend.py
```

Oracle 100%, random ~25%, refuse 0%. If any of those is off, the scorer is
broken and a real run would produce a number that looks fine and is wrong.
