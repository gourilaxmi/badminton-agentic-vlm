# Roadmap — three reviews to submission

| Milestone | Date | Working time | Theme |
|---|---|---|---|
| Review 1 | done | — | Problem, literature, design |
| **Review 2** | **Wed 21 Oct 2026** | build 11–18 Oct, deck 19–21 | **Full scope: screening table, zero-shot numbers, loop demo** |
| **Review 3** | **mid-Jan 2027** | 12.3 weeks calendar, ~9 effective | **The headline result** |
| **Review 4 (final)** | **mid-Mar 2027** | 8.4 weeks | **Full study, ablations, write-up** |

Pin the exact Review 3 and Review 4 dates with the guide and update this table.

Sprint B spans end-semester exams and the winter break. Budget **nine** effective
weeks, not twelve, and put the risky work at the front of it.

---

## Team assignment

Four people, stable ownership across the whole project so knowledge compounds.

| | Owner | Owns |
|---|---|---|
| **D1** | ___ | Data and infrastructure — dataset integrity, frame cache, repo hygiene, run manifest |
| **D2** | ___ | Models and inference — VLM backend, LoRA, GPU scheduling |
| **D3** | ___ | Agent — planner, observer, reflector, prompts, loop |
| **D4** | ___ | Evaluation — FBBench scoring, baseline reproduction, faithfulness, ablations |

The A100 is contended from Sprint B onward. Queue jobs, do not share interactively.

---
---

# SPRINT A → Review 2 (Wed 21 Oct 2026)

**Work starts Sun 11 Oct. Technical freeze Sun 18 Oct. Deck 19–21 Oct.**

**Full scope — nothing cut.** Six backbones screened, the winner run at full
size, resolution sweep, loop demonstration, reproducibility check. The estimate
is 17–35 hours of effort across four people, which is under 1.5 h per person per
day. It fits. What does not fit is starting the downloads late — see Day 0.

**Theme: prove the pipeline runs end to end and produces real numbers.**

---

## Why this is achievable

| | |
|---|---|
| Total effort | 17–35 h |
| Across 4 people | 4–9 h each |
| Per person per day | **under 1.5 h** |
| Kaggle GPU needed | 5–8 h (quota is 30 h/week) |
| **Sequential waiting that nobody can parallelise** | **5.8–11 h** |

The constraint is the critical path, not the effort:

```
download -> extract -> upload -> screen 6 models -> full run on winner
 (1-3 h)    (25 m)     (70 m)      (3.5-5.4 h)        (1.7-2.8 h)
```

Nothing after the first arrow starts until the download finishes.

---

## Working agreements

- **15-minute standup daily, same time.** Done / doing / blocked.
- **Blocked more than 4 hours → escalate.** Do not sit on it overnight.
- **Push every day.** Work on one laptop does not exist.
- **Merge to `main` at every standup.** Four branches diverging for a week will
  conflict on 18 October, the worst possible day.
- Check the institute calendar for holidays in this window and re-plan now.

---

## Day 0 · Sun 11 Oct — setup, and start the downloads

Everything here is unblocking work. None of it needs the data.

- [ ] **D1** `bash setup_repo.sh` — repo, four branches, CI, pushed
- [ ] **D1** **Start both dataset downloads. Tonight. They run unattended.**
- [ ] **All** Install Claude Code, log in on the work account, confirm it reads `CLAUDE.md`
- [ ] **D2** Kaggle phone verification · API token · `GH_TOKEN` secret
- [ ] **D2** Local venv, `pip install -r requirements.txt`, `pytest` green
- [ ] **D3** Read FineBadminton §6: their frame budget, resolution, scoring method
- [ ] **D4** Read `docs/HANDOFF.md` §9 — the twelve rules — and brief the team

> Starting the download tonight is worth a full day. It is the first link in a
> strictly ordered chain and it needs no supervision.

---

## Day 1 · Mon 12 Oct — dataset integrity

- [x] **D1** `python scripts/00_verify_dataset.py` → commit `results/dataset_report.md` — *report written by hand (gouri, 11 Oct); FBBench part only, see STATUS.md*
- [x] **D1** **Clip-overlap check between Finebadminton-20K and FBBench** — *by match: FBBench's 20 source matches held out, 50 left for training (gouri, 12 Oct)*
- [x] **D1** Record HF revision hashes in `results/MANIFEST.md` — *both done (gouri, 11–12 Oct)*
- [x] **D2** `describe_schema()` on the real data → **pin `_FIELD_CANDIDATES` in `fbbench.py`** — *now `FIELDS` + `SUBTASKS` (gouri, 11 Oct)*
- [x] **D2** Dump the real label vocabulary — 11 hit types, 20 subtypes, 9 regions — *confirmed, plus a few noisy labels (gouri, 12 Oct)*
- [ ] **D3** Count clips carrying Decision Evaluation annotations
- [ ] **D4** Multiple-choice prompt v1, frozen and committed

> ### Gate G1 — end of Mon 12 Oct
> `dataset_report.md` committed, **zero overlap confirmed**, schema pinned.
> If clips appear in both sets, stop and resolve before anything else.

---

## Day 2 · Tue 13 Oct — frames and scorer

- [x] **D1** `extract_frames.py --limit 10` → inspect what landed on disk — *Kaggle kernel `kernels/frame-cache`, LIMIT=20 (gouri, 11 Oct)*
- [ ] **D1** Full extraction **at `--size 448`** (downscale at load time; you
      cannot upscale later, and re-extracting costs 25 min plus a 70 min re-upload)
- [x] **D1** Upload as a Kaggle Dataset — *not needed: the GPU kernel attaches the frame-cache kernel output (gouri, 11 Oct)*
- [x] **D2** `VLMBackend` real path: Qwen2.5-VL loader wired — *gouri, 11 Oct*
- [x] **D4** FBBench scorer + guided decoding + per-domain tagging — *letter log-prob scoring (gouri, 11 Oct)*
- [ ] **D3** Planner / observer / reflector prompts drafted (for Day 6)

---

## Day 3 · Wed 14 Oct — mock passes, then first GPU contact

- [x] **D4** **Mock mode: oracle 100%, random ~25%, refuse 0%** — *on all 2,413 real questions (gouri, 11 Oct)*
- [x] **D2** Notebook wired: clone, secrets, attach dataset, push results — *`kernels/fbbench-zeroshot` (gouri, 11 Oct)*
- [ ] **D2** Kaggle smoke test — ten questions end to end on T4 × 2
- [ ] **D1** `results/MANIFEST.md` format agreed and in use

> ### Gate G2 — end of Wed 14 Oct
> **One FBBench question answered end to end and scored correctly, on Kaggle.**
> The single most important checkpoint. Everything after this is scaling it up.
> If it slips past Thu 15, drop the backbone sweep to three models.

---

## Day 4 · Thu 15 Oct — backbone sweep

Objective 1 of the project. The output table is itself a Review 2 deliverable.

- [ ] **D2** `scripts/01b_backbone_sweep.py`, 150 stratified questions each:
      Qwen2.5-VL-7B · Qwen3-VL-8B · VideoLLaMA3-7B · InternVL3.5-8B ·
      Qwen2.5-VL-3B · SmolVLM2-2.2B
- [ ] **D2** Record seconds and input tokens per model, not only accuracy
- [ ] **D4** Per-domain breakdown; read the separability warning the script prints
- [ ] **D4** Pick the winner — **on practical grounds if the top two are within
      ~15 points**, and say so

> Each family needs its own loader. Wire Qwen2.5-VL first, confirm, then add the
> others — roughly 20 lines each. **If one family fights you, drop it** rather
> than losing a day. Five models is a fine table; four is fine too.

> Speed matters more than it looks: the agentic loop makes ~3× the calls per
> question, so a backbone 40% faster at equal accuracy is better for Review 3.

---

## Day 5 · Fri 16 Oct — the full run

- [ ] **D2** 500 questions on the winning backbone
- [ ] **D2** Resolution sweep: 224 / 336 / 448, 50 questions each → accuracy vs tokens
- [ ] **D4** Hand-check 20 scored answers against the key
- [ ] **D1** Every run in the manifest

> ### Gate G3 — end of Fri 16 Oct
> ≥500 questions scored on ≥1 backbone, per-domain table complete, screening
> table complete.

---

## Day 6 · Sat 17 Oct — demonstration and robustness

- [ ] **D3** Planner → observer → reflector loop, **10 clips only**, traces printed
- [ ] **D3** One trace formatted for a slide: plan → evidence → reflection → round 2
- [ ] **D1** Reproducibility: fresh clone, follow the README, reproduce one table
- [ ] **D4** Second backbone at 500 questions, if quota and time allow

> The loop demo is **qualitative**. Ten clips means nothing statistically — show
> the trace, claim no accuracy. It is the largest single line item at 2–3.5 h
> interactive, and the only genuinely new code in the sprint. If one thing
> overruns, it will be this.

---

## Day 7 · Sun 18 Oct — freeze

- [ ] Re-run anything that looked suspicious
- [ ] Final tables assembled
- [ ] Manifest complete, repo clean, tests green, `main` merged
- [ ] **Technical freeze, end of day. No new experiments.**

---

## Deck · Mon 19 – Wed 21 Oct

- **Mon 19** — D4 builds results slides; D3 updates the architecture slide to
  separate what is **built** from what is **planned**
- **Tue 20** — Full dry run twice, timed. Fix what breaks
- **Wed 21** — Review

---

## Descope ladder

Cut in this order, and say so in the deck. A smaller honest result beats a
larger broken one.

1. SmolVLM2 and Qwen2.5-VL-3B from the sweep → four models
2. Second backbone at full size → one full run
3. Resolution sweep → fix at 224 and move on
4. Loop demo → drop entirely
5. Reproducibility check → push to Sprint B
6. 500 questions → 250, stratified

**Never cut:** dataset verification, the overlap check, the run manifest. Those
three are what make every number from January and March trustworthy.

---

## Review 2 deliverables — the full set

1. `results/dataset_report.md` — clip counts, splits, **zero contamination**, label vocabulary
2. **Backbone screening table** — 6 models × 4 domains × accuracy, with seconds and tokens
3. **Zero-shot FBBench accuracy at full size** on the chosen backbone, per-domain
4. **Resolution sweep** — accuracy vs input tokens, justifying the chosen `max_pixels`
5. Inference cost baseline — seconds and input tokens per query
6. **Loop demonstration** — one worked trace on a slide, qualitative
7. Working repo, tests green, every run in `results/MANIFEST.md`, reproducible from a fresh clone
8. Two slides of honest risk — what is unverified, what could still break

---

## Sanity expectations

| Check | Expected | If not |
|---|---|---|
| Oracle | 100% | Plumbing broken — stop |
| Random | ~25% | Scorer broken — stop |
| 7B zero-shot, real | **< 25%** | Much higher → extraction bug or leakage. Much lower → frames not loading |
| Action domain | **Worst of the four** | Expected. Uniform 2 FPS steps over contact |

A rally is 12.4 s at 25 FPS ≈ 310 frames. Uniform 2 FPS gives 25 frames, one
every 0.5 s. A stroke's visible motion lasts 0.12–0.32 s, so the sampler
regularly steps straight over the moment that answers the question.

Say exactly that in the deck: *"uniform sampling gives the model the rally's
shape but not its detail, and our per-domain breakdown shows where that hurts."*
A weak number becomes the motivation for Review 3.


---
---

# SPRINT B → Review 3 (mid-Jan 2027)

**Theme: the headline result. Does query-conditioned observation beat the query-agnostic
baseline at an equal frame budget?**

This is the project. It must land here, not in March — if the loop does not beat the
baseline, Sprint C is the time you need to reframe around efficiency and faithfulness.

### Block B1 · weeks 1–3 (late Oct – mid Nov) — reproduce the baseline

- [ ] **D4** Hit-event detection: VideoMAE features → temporal encoder → per-frame hit probability
- [ ] **D4** Anchor frames plus query frames sampled from inter-hit segments
- [ ] **D4** Evaluate at the identical frame budget as the uniform baseline
- [ ] **D2** Coordinate-guided condensation — optional, only if B1 finishes early

> **Gate.** Hit-centric must beat uniform on **Action** tasks at equal frame count.
> If it does not, the reproduction is wrong. Fix it before comparing anything to it.

### Block B2 · weeks 3–7 (mid Nov – mid Dec) — the agentic loop, training-free

- [ ] **D3** Planner: query → `(what, where, how)` within the frame budget
- [ ] **D3** Observer: executes a plan, returns structured time-stamped evidence
- [ ] **D3** Reflector: sufficiency score, citation selection, stated gap
- [ ] **D3** Loop orchestration, max 3 rounds, full `Trace` written per question
- [ ] **D2** Adapter/role prompt management on one shared backbone
- [ ] **D4** Per-domain evaluation against both baselines

Training-free deliberately. If the loop improves accuracy with no fine-tuning, the gain
is unambiguously from observation strategy rather than from fitting the dataset.

### Block B3 · weeks 7–9 (Jan) — first faithfulness numbers

- [ ] **D4** Evidence ablation harness, text-level
- [ ] **D4** Random-ablation control
- [ ] **D4** Run on a stratified sample of ~150 questions
- [ ] **D1** Determinism: temperature 0, fixed seeds, or averaged repeats

### Review 3 deliverables

The comparison table, populated:

| Method | Count | Action | Position | Cognition | Overall | Frames | Time/query |
|---|---|---|---|---|---|---|---|
| Uniform 2 FPS | | | | | | ~25 | |
| Hit-centric (reproduced) | | | | | | ~25 | |
| Agentic loop (ours) | | | | | | ~25 | |

Plus:
1. Component ablation — planner on/off, reflector on/off
2. First faithfulness score with the random-ablation control
3. A worked example: one rally, the plan, the evidence, the answer, the citation

> **The prediction to check.** Gains largest on Count and Cognition, near parity on
> Action, because hit-centric sampling is already near-optimal there. If that pattern
> appears, you have mechanistic evidence rather than just a bigger number — say so.

> **Decision point at Review 3.** If the loop does not beat the baseline on accuracy,
> pivot the framing to efficiency plus faithfulness and say so openly. Both are still
> real contributions. Do not spend Sprint C chasing an accuracy win that is not there.

---
---

# SPRINT C → Review 4, final (mid-Mar 2027)

**Theme: complete the study, close the contributions, write it up.**

### Block C1 · weeks 1–3 (mid Jan – early Feb) — faithfulness in full

- [ ] **D4** Sufficiency test: fresh model, cited evidence only
- [ ] **D4** Video-level ablation on a subset (stronger than text-level)
- [ ] **D4** Reflector calibration: confidence vs. correctness
- [ ] **D4** Faithfulness by task domain, and reflector on/off

> The result worth hoping for: reflection improves accuracy while *reducing*
> faithfulness, because the model gets better at constructing post-hoc justification.
> If it appears, it is a genuine and cautionary finding — lead with it.

### Block C2 · weeks 2–4 (parallel) — memory graph and scouting

- [ ] **D1** Ingestion from agent output into the SQLite graph
- [ ] **D1** Tendency queries by player, court region, shot type
- [ ] **D3** Scouting report generator, every claim linked to supporting rallies
- [ ] **D3** Simple viewer — rally timeline plus court overlay

Remember the scoping: memory does **not** contribute to FBBench accuracy. It is evaluated
separately, through the demonstration.

### Block C3 · weeks 3–5 (Feb) — pose tool, or cut it

- [ ] **D2** RTMPose or ViTPose wrapper returning keypoints for a frame range
- [ ] **D3** Planner action space extended to request pose evidence
- [ ] **D4** Ablation: does tool access help, and on which domains?

> **Cut gate, 14 Feb.** If pose is not showing a measurable effect by then, drop it and
> report it as future work. It is a second perception pipeline to maintain and the loop
> is the core claim. A clean three-contribution project beats a ragged four.

### Block C4 · weeks 4–6 (late Feb) — LoRA ablation

IntentVLM's config: LoRA via PEFT, per-device batch 1, gradient accumulation 8, 2 epochs,
LR 2e-5, warmup 0.1, AdamW, bf16.

- [ ] **D2** Separate adapters per role, hot-swapped on one backbone
- [ ] **D2** **Train MCQ and open-ended as separate objectives.** IntentVLM saw ROUGE-1
      fall from 19.71 to 12.89 when fine-tuning collapsed constrained selection into
      unconstrained generation. FBBench's 150 open-ended questions carry your explanation
      contribution
- [ ] **D4** Report both ROUGE and BERTScore — they disagreed sharply there

Question it answers: does domain adaptation add on top of better observation, or do the
two overlap? Either answer is reportable.

### Block C5 · weeks 6–8 (Mar) — write-up

- [ ] **D4** All ablations complete and tabulated
- [ ] **D3** Failure taxonomy — AVP found 76% of their errors were observer perception
      failures; yours will differ and the difference is informative
- [ ] **D1** Reproducibility pass: fresh clone, follow the README, reproduce one table
- [ ] **All** Limitations stated plainly; final report and deck

### Review 4 deliverables

1. Full comparison table with all ablations
2. Faithfulness study: necessity, control, sufficiency, calibration, per-domain
3. Memory graph with a generated scouting report
4. LoRA ablation
5. Failure analysis
6. Report, deck, and a repo someone else can run

---
---

## Risk register

| Risk | Detect by | Response |
|---|---|---|
| Train/test contamination | Week 1, Sprint A | Exclude overlapping clips, document it |
| Zero-shot accuracy near random | Week 3, Sprint A | Debug frame loading and answer extraction before concluding anything about the model |
| Baseline reproduction fails | Block B1 gate | Fix it; a wrong control makes every comparison meaningless |
| Loop does not beat baseline | Review 3 | Pivot framing to efficiency plus faithfulness; both stand alone |
| Pose adds nothing | 14 Feb cut gate | Drop it, report as future work |
| Sprint B shorter than planned | Ongoing | Risky work at the front; B3 is the compressible block |
| A100 contention | Sprint B onward | Queue jobs, no interactive sharing |

## Standing rules

- **FBBench is touched only for final numbers.** All development on FineBadminton-20K.
- **Report per-domain, never overall alone.** The mechanism shows up in the breakdown.
- **Equal frame budget in every comparison.** The claim is "same frames, chosen better".
- **Every reported number has a row in `results/MANIFEST.md`.** No row, no report.
- One branch per block: `phase/b2-agentic-loop`. `main` always runnable.
