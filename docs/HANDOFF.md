# BTP Project Handoff Brief

**Read this first. It is self-contained — everything needed to pick up the project is here.**

Prepared 7 Oct 2026 · for a fresh Claude session or a new team member

---

## 0 · How to use this document

You are joining a final-year B.Tech project (BTP) that is already designed and scoped.
The literature review is done, the architecture is settled, the datasets are acquired.
**What remains is implementation.**

- §1–§3 — what the project is and why. Read once.
- §4–§6 — papers, datasets, hardware. Reference material.
- §7–§9 — architecture and the rules that govern evaluation. **Read carefully; most
  mistakes come from breaking §9.**
- §10 — the current sprint. **This is what to work on today.**
- §11–§12 — repo layout and known traps.

Do not redesign the project. If something here seems wrong, say so, but the design was
settled across many hours of discussion and defended at a review panel.

---

## 1 · Project identity

| | |
|---|---|
| **Title** | VLM-Grounded Agentic Reasoning for Explainable Action and Error Recognition in Dynamic Video Sequences |
| **Institution** | Indian Institute of Information Technology Kottayam |
| **Team** | Amal V Pratish (2023BCS0004) · Gourilakshmi S (2023BCS0021) · Mohammed Sabeeh K (2023BCS0108) · Jaladi Keerthi Vishaal (2023BCS0022) |
| **Guide** | Dr Jeena Thomas |
| **Domain** | Badminton video understanding — sports analytics |
| **Hardware** | 1 × NVIDIA A100 (assume 40 GB unless confirmed otherwise) |
| **Review 1** | Completed — problem, literature, design |
| **Review 2** | **Wed 21 Oct 2026** — build freeze Sun 18 Oct |
| **Review 3** | mid-Jan 2027 |
| **Review 4 (final)** | mid-Mar 2027 |

### History worth knowing

The project began as explainable procedural mistake detection in egocentric cooking
videos (CaptainCook4D). The guide directed it toward sports analytics. Two earlier sports
framings were abandoned:

- **Biomechanics / injury prevention** — dropped because no public badminton dataset
  carries biomechanical fault labels, and the team cannot recruit expert coaches to
  annotate them.
- **Surgical phase recognition (Cholec80)** — dropped for the same reason: the guide
  noted it would require expert verification the team cannot obtain.

**The lesson that shaped the final design: evaluation labels must come from the dataset,
not from people the team has to recruit.** Every contribution in the current design is
measurable without human annotators. Do not propose anything that reintroduces that
dependency.

---

## 2 · One-paragraph pitch

Agentic video systems emit a reasoning trace — a plan, a set of evidence, a justification
— and every paper that produces one treats that legibility as an interpretability
guarantee. None has tested it. We build a query-conditioned agentic perception loop for
fine-grained badminton, a setting where evidence is sparse, fast and spatially
localisable, and use it to ask two questions the field has not: can an agent allocate a
fixed observation budget better than a learned query-agnostic selector, and is the
evidence it cites what actually drove its answer?

---

## 3 · Novelty — four contributions, ranked

### 3.1 Faithfulness of agentic reasoning traces — **strongest**

**Claim.** Cited evidence in an agentic video system is measurably more load-bearing than
uncited evidence — or it is not, and that is itself a finding.

**Why novel.** Active Video Perception, Deep Video Discovery, VideoAgent and CASTLE all
emit reasoning traces and all treat them as interpretable by construction. None tests it.
Turpin et al. (NeurIPS 2023) showed language-model reasoning frequently does not drive the
output; whether this holds for agentic *video* systems is unasked, and matters more here
because the trace is the entire interpretability claim.

**Protocol — fully automatic, no annotators:**

| Measure | Method |
|---|---|
| Necessity | Delete each cited evidence item, re-run, record whether the answer changes |
| Control | Delete an equal quantity of *uncited* evidence from the same clip |
| **Faithfulness score** | **flip-rate(cited) − flip-rate(random)** |
| Sufficiency | Give a fresh model only the cited evidence; does it reach the same answer? |
| Calibration | Does the reflector's confidence track correctness? |

A gap near zero means citations are arbitrary — the explanation points at nothing in
particular even if it reads well.

**The result worth hoping for:** that the reflector improves accuracy while *reducing*
faithfulness, because reflection teaches the model to construct better post-hoc
justification. That would be a genuine, cautionary finding for the whole
agentic-interpretability direction.

### 3.2 Query-conditioned observation for fine-grained sports

**Claim.** Given the same frame budget (~25 frames per rally, matching the 2 FPS
baseline), allocating frames by query type outperforms allocating them by hit detection.

**Why it is not already done.** FineBadminton's Hit-Centric Keyframe Selection is a
**trained** module — VideoMAE features → attention pooling → transformer temporal encoder
→ per-frame hit probability, then a FastRCNN-guided Perceiver ReSampler. It is good, and
it is **query-agnostic**: identical visual tokens whether the question is a counting task
or a causal one.

> **Never call their method "a simple heuristic."** It is a trained module. The accurate
> and stronger criticism is that it is query-agnostic.

FBBench's four domains have conflicting evidence needs:

| Domain | Evidence needed | Hit-centric sampling |
|---|---|---|
| Count | broad temporal coverage, low detail | over-concentrates at contact |
| Action | dense frames at contact | near-optimal |
| Position | spatial detail, moderate spread | partial |
| Cognition | full-rally causal context | over-concentrates at contact |

Their own reported failure modes — subtle action discrimination and complex multi-step
reasoning — sit at opposite ends of that table.

**Falsifiable prediction:** gains largest on Count and Cognition, near parity on Action.
Reporting per-domain turns a higher number into mechanistic evidence.

### 3.3 Pose-grounded evidence inside an agentic loop

Agentic video systems (AVP, DVD) work on raw frames only. Pose-grounded systems
(LLaVA-Pose, MistSense, BioCoach) are single-pass, not agentic. No system lets an agent
*decide* to invoke a pose estimator when its own evidence is insufficient.

In this design pose is **a tool the observer calls**, never a competing pipeline. That
framing also answers "why VLM and not MediaPipe" cleanly — see §13.

### 3.4 Longitudinal player memory

A temporal knowledge graph keyed on **player**, not video, accumulating tendencies across
rallies to support opponent scouting.

> **Critical scoping note.** This does **not** contribute to FBBench accuracy. FBBench
> asks about individual clips; memory answers cross-rally questions. It is evaluated
> separately via the scouting demonstration. Claiming it improves the benchmark score
> would be wrong and a reviewer would catch it.

### 3.5 Explicitly NOT claimed

- Not that the system improves match outcomes (needs a longitudinal study).
- Not that pose estimation beats a purpose-built pose model.
- Not that VLMs are good at badminton — published evidence says they are poor. The claim
  is **relative**, against a reproduced baseline at an equal frame budget.

---

## 4 · Papers

### 4.1 Core — implement from these

| # | Paper | Venue | Used for | When |
|---|---|---|---|---|
| 1 | **FineBadminton: A Multi-Level Dataset for Fine-Grained Badminton Video Understanding** — He, Liu, Ma, Liu, Ma, Wu | ACM Multimedia 2025, Dublin, pp. 12776–12783, DOI 10.1145/3746027.3758218 · arXiv:2508.07554 | Dataset, benchmark, baseline to reproduce, the 42.06% target | **Review 2** |
| 2 | **Active Video Perception: Iterative Evidence Seeking for Agentic Long Video Understanding** — Wang, Zhou, Wang, Li, Xiong, Savarese, Bansal, Ryoo, Niebles (Salesforce AI Research / UNC) | **CVPR 2026 — Findings track** · arXiv:2512.05774 | The plan–observe–reflect architecture | Review 3 |
| 3 | **Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding** — Zhang, Jia, Guo, Li, Li, Li, Lu (Microsoft Research) | NeurIPS 2025, Main Conference Track · arXiv:2505.18079 | The costly passive baseline argued against | Review 3 |
| 4 | **IntentVLM: Open-Vocabulary Intention Recognition through Forward–Inverse Modeling with Video-Language Models** — Rahimi, Grislain, Jacquet Crétides, Sigaud, Chetouani | **Preprint, submitted to ICMI '26 — not published** · arXiv:2604.24002 | LoRA recipe, two-stage decomposition, fine-tuning warning | Review 4 |

> AVP is **Findings track**, not main conference. IntentVLM is a **submitted preprint**.
> Cite both accurately — overclaiming venue is an easy thing for a panel to check.

### 4.2 Supporting — cited, not implemented from

| # | Paper | Venue | Role |
|---|---|---|---|
| 5 | **ShuttleSet** — Wang, Huang, Ik, Peng | ACM SIGKDD 2023, pp. 5126–5136, DOI 10.1145/3580305.3599906 · arXiv:2306.04948 | Badminton task vocabulary; 36,492 strokes, 18 shot classes. Annotation-only, videos sourced separately. Possible generalisation test |
| 6 | **VideoBadminton** — Li, Chiu, Huang, Sun, Ku | IEEE Int. Conf. Big Data 2024 · arXiv:2403.12385 | 7,822 clips, 18 action categories, 145 min, self-recorded. Fallback dataset |
| 7 | **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al. | ICLR 2022 · arXiv:2106.09685 | The adaptation method |
| 8 | **Qwen3-VL Technical Report** — Bai et al. | arXiv:2511.21631 | Backbone config |
| 9 | **LLaVA-Pose** — Zhang, Hussain, An, Shouno | Sensors, vol. 25 no. 16, 2025 | Pose-grounded VLM precedent. Static/image-based, no temporal modelling |
| 10 | **MistSense: Versatile Online Detection of Procedural and Execution Mistakes** — Patsch, Wu, Zakour, Salihu, Steinbach | IEEE/CVF ICCV 2025 | Pose + mistake detection + LLM explanation. Hand pose only, single-pass |
| 11 | **From 3D Pose to Prose: Biomechanics-Grounded Vision-Language Coaching (BioCoach)** — Ji, Shen, Zhu, Kong, Liu | IEEE/CVF CVPR 2026 | 3D kinematics + language feedback. Predefined pipeline, not agentic |

### 4.3 Conceptual grounding for the faithfulness work

| Paper | Venue | Why |
|---|---|---|
| **Language Models Don't Always Say What They Think** — Turpin et al. | NeurIPS 2023 · arXiv:2305.04388 | The core motivation: stated reasoning ≠ actual reasoning |
| **Towards Faithfully Interpretable NLP Systems** — Jacovi & Goldberg | ACL 2020 · arXiv:2004.03685 | Definition of faithfulness; read before designing the evaluation |
| **Grad-CAM** — Selvaraju et al. | IJCV 2020 · arXiv:1610.02391 | The XAI baseline being improved on |

### 4.4 Optional background

SportR (ICLR 2026, arXiv:2511.06499) · SPORTU (ICLR 2025, arXiv:2410.08474) ·
MotionBench (arXiv:2501.02955) · TemporalBench (arXiv:2410.10818) ·
FAVOR-Bench (arXiv:2503.14935) · ReAct (ICLR 2023, arXiv:2210.03629) ·
Reflexion (NeurIPS 2023, arXiv:2303.11366)

---

## 5 · Datasets

### 5.1 Primary — already acquired, Apache-2.0

```bash
hf download Moujuruo/Finebadminton-20K      --repo-type dataset --local-dir data/finebadminton-20k
hf download Moujuruo/FineBadmintonBenchmark --repo-type dataset --local-dir data/fbbench
```

Mirrors may also exist under `iLearn-Lab/`. Project page:
`https://ilearn-lab.github.io/MM25-FineBadminton/` (a showcase viewer, not a download
portal). Apache-2.0 — permissive, no request form, attribution required.

### 5.2 FineBadminton structure

| | |
|---|---|
| Source | BWF official YouTube footage — Olympics, World Championships, Open series |
| Matches | 120 singles matches, past five years |
| **Clips** | **3,215 rally clips — already segmented, no cutting needed** |
| Strokes | 33,325 |
| Average rally | **12.4 seconds** |
| Frame rate | **consistent 25 FPS** |
| Resolution | 720p or higher |
| Camera | predominantly overhead from behind the court; small side-angle subset |
| Total footage | ~11 hours |

**Three-level annotation hierarchy:**

| Level | Content |
|---|---|
| Foundational Actions | 11 primary hit types, 20 subtypes; shuttle position in 9 court regions; hitter identity |
| Tactical Semantics | 3 player-action categories, 9 strategic classifications, 6 shot-characteristic types, 3 last-hit outcome classes |
| Decision Evaluation | Shot quality 1–7; hitting comments ~37.4 words; rally-level Overall Evaluation ~63 words, Scoring Reason ~41 words, Losing Reason ~44 words |

The Decision Evaluation level supplies **supervised natural-language explanation data** —
this is what makes explanation generation trainable and evaluable.

**Annotation pipeline (theirs):** TrackNetV3 for shuttle tracking, FastRCNN for court
boundaries and player keypoints, VideoMAE fine-tuned for hit-event detection, MLLM
proposals refined by humans.

### 5.3 FBBench — the benchmark

- **2,563 QA pairs: 2,413 multiple-choice + 150 open-ended**
- Open-ended confined to the three Cognition subcategories
- **Four domains × three subcategories = 12 task types**, all requiring multi-frame reasoning

| Domain | What it tests |
|---|---|
| **Count** | Quantify discrete events — strokes per rally, rallies per game |
| **Action** | Stroke classification, plausible-return prediction, temporal localisation |
| **Position** | Player localisation, movement tracking, shuttle landing prediction |
| **Cognition** | Shot quality evaluation, strategic inference, rally outcome — causal inference over extended evidence |

### 5.4 Secondary

- **ShuttleSet / ShuttleSet22** — `github.com/wywyWang/CoachAI-Projects`. Annotations only; source videos must be obtained separately.
- **VideoBadminton** — 7,822 clips, 145 min. Small, clean fallback.

---

## 6 · Hardware and stack

Single A100. Assume 40 GB unless confirmed.

| Task | Feasible | Notes |
|---|---|---|
| Inference, 7–8B VLM, bf16 | Yes | ~16 GB weights + KV cache |
| LoRA fine-tune 7–8B, rank 16, bs=1 + grad-accum 8 | Yes | ~28–34 GB with gradient checkpointing, bf16, AdamW 8-bit |
| **Full fine-tune 7B** | **No** | LoRA/QLoRA only — and that is the correct choice regardless |
| Two 8B models resident simultaneously | No | Hot-swap LoRA adapters on one shared base model |

**Stack:** PyTorch · transformers · vLLM (serving) · PEFT (LoRA) · LangGraph
(orchestration, Review 3) · Decord or PyAV (video I/O) · SQLite (memory graph) ·
optionally `lmms-eval`.

### 6.1 Inference cost — what actually helps

> **LoRA does NOT reduce inference cost.** It reduces *training* cost. At inference a
> LoRA model costs essentially the same as the base model. Do not say otherwise at a
> review.

What does reduce it, in order of impact:

1. **Visual token count** — the dominant cost. For Qwen2.5-VL, tokens per frame ≈
   (H/28)×(W/28); at 448×448 that is ~256 per frame, so 25 frames ≈ 6,400 visual tokens
   before any text. Control with `max_pixels` on the processor. **Run a resolution sweep
   on ~50 questions before the full run.**
2. **Frames-first prompt ordering + group by clip.** Put frames *before* the question,
   sort the run by clip, enable vLLM prefix caching. Every question after the first on a
   given clip then hits the cache for the whole visual prefix. Largest throughput win
   available, costs one `sorted()`.
3. **Guided decoding** (`guided_choice=["A","B","C","D"]`) — drops `max_tokens` to ~2 and
   **eliminates answer-extraction bugs entirely**. The second benefit matters more.
4. `temperature=0`, fixed seed — required for the faithfulness work anyway.
5. Set `--max-model-len` to actual need, not the default.
6. Incremental JSONL writes + skip-done-IDs on restart.

**Do not quantize for reported numbers.** AWQ/GPTQ INT4 changes accuracy; bf16 fits
comfortably. FP8 is not available on A100 (that is H100).

**Profile before optimising.** Split one 50-question run into frame loading / vision
encode + prefill / decode. If frame loading dominates, vLLM will not help at all — the
fix is the frame cache.

---

## 7 · Architecture

Input: a rally clip (~12.4 s, 25 FPS, ~310 frames available) and a question.
**Frame budget: ~25 frames**, matching the uniform 2 FPS baseline.

```
rally clip V  +  query Q
        ↓
┌──── QUERY-CONDITIONED AGENTIC LOOP (≤ 3 rounds) ────┐
│  PLANNER   reads Q, classifies the evidence needed,  │
│            allocates the frame budget                │
│     ↓ plan (what · where · how)                      │
│  OBSERVER  runs the VLM on exactly those frames,     │
│            returns structured time-stamped evidence  │
│            ↕ tools: frame sampler · pose (HPE) ·     │
│                     shuttle & court tracker          │
│     ↓ evidence E                                     │
│  REFLECTOR scores sufficiency; halts, or names what  │
│            is missing and triggers re-planning ──────┤
└──────────────────────────────────────────────────────┘
        ↓ sufficient
  ANSWER A + CITED EVIDENCE E
        ↓
  ┌─────────────┬──────────────┬─────────────────┐
  FAITHFULNESS   MEMORY GRAPH   EXPLANATION LAYER
  (ablation)     (SQLite,       (rally timeline +
                 per player)     court overlay)
```

**Planner emits** `(what, where, how)`: the sub-question, temporal spans (+ optional
spatial crop), and sampling density within the budget.

**Observer returns structured evidence**, not prose:
`{(t_start, t_end), court_region, hit_type, description, confidence}`.
The structure is what makes each item individually ablatable later — a design decision,
not a convenience.

**Reflector** outputs sufficiency, confidence, citations, and the gap if insufficient.
Cap at 3 rounds — AVP found gains saturate there.

### 7.1 Why the planner matters — the arithmetic

| | |
|---|---|
| Rally at 25 FPS | ~310 frames available |
| Uniform 2 FPS | 25 frames — one every **0.5 s** |
| A stroke's visible motion | **0.12–0.32 s** |

A 0.5-second gap can step straight over a stroke. Two ways to spend 25 frames: spread
them (context, no contact) or cluster at the hit (contact, no setup). FineBadminton always
clusters. The planner chooses per question. **Same 25 frames, different distribution** —
that is the entire contribution.

---

## 8 · Reference numbers

From the FineBadminton paper, on FBBench:

| System | Overall MCQ accuracy |
|---|---|
| 7B open-source, zero-shot | **< 25%** |
| GPT-4.1 | third of the commercial models |
| Doubao 1.5 Pro | second |
| Gemini 2.5 Pro + their sampling strategy | **38.6%** |
| **Qwen2.5-VL-7B + their full pipeline, fine-tuned** | **42.06%** — best overall, beat every commercial model |

Their baseline comparison: **uniform sampling at 2 FPS**.
Their fine-tuning set: ~60,000 QA pairs, 10% MCQ.
Their named failure modes: deep domain knowledge, subtle action discrimination, complex
reasoning.

**Implications:** the benchmark is far from solved; domain adaptation beats raw scale, so
one A100 suffices; a fine-tuned 7B is the target to match.

From AVP: 5.7% over the best agentic baseline at 18.4% of inference time and 12.4% of
input tokens; 76% of its errors were observer perception failures.

From Deep Video Discovery: 790.5 s and ~1.07M input tokens per video on LVBench.

From IntentVLM: fine-tuning **degraded** open-ended performance — ROUGE-1 fell 19.71 →
12.89 on Qwen3-VL 4B — while BERTScore barely moved (88.68 → 85.82).

---

## 9 · Evaluation discipline — the non-negotiable rules

Breaking any of these invalidates results. They matter more than any implementation
detail in this document.

1. **FBBench is touched only for final numbers.** All development, prompt iteration and
   LoRA training happen on FineBadminton-20K. Tuning against the benchmark inflates
   numbers and voids comparison to published results.
2. **Verify there is no clip overlap between Finebadminton-20K and FBBench before any
   fine-tuning.** If clips appear in both, training contaminates evaluation.
3. **If creating a split, split by match, not by clip.** Otherwise the model memorises
   player appearance, court and broadcast style instead of learning stroke semantics.
4. **Equal frame budget in every comparison.** The claim is *"same frames, chosen
   better"* — never *"we use more frames."*
5. **Report per-domain, never overall alone.** The mechanism shows up in the breakdown.
6. **Primary efficiency metric is input tokens and frame count**, not wall-clock. Those
   are hardware- and serving-stack-independent. Report wall-clock as secondary with the
   hardware and stack stated — otherwise using vLLM when the baseline paper did not makes
   the comparison apples-to-oranges.
7. **Reproduce the hit-centric baseline yourself.** Quoting their number and comparing
   against your own setup is not a valid control.
8. **Evaluate the agentic loop training-free first.** If it improves accuracy with no
   fine-tuning, the gain is unambiguously from observation strategy. LoRA then becomes an
   ablation showing the two are complementary.
9. **`temperature=0`, fixed seeds** — or average over repeats. Otherwise the faithfulness
   evaluation measures sampling noise.
10. **Every reported number has a row in `results/MANIFEST.md`.** No row, no report.
11. **Train MCQ and open-ended as separate objectives** when fine-tuning, with separate
    adapters. See the IntentVLM warning in §8.
12. **Report both ROUGE and BERTScore** for open-ended output — they disagreed sharply in
    IntentVLM, and reporting one would have hidden a real regression.

---

## 10 · CURRENT SPRINT — Review 2, build freeze Sun 18 Oct

**Work started Tue 6 Oct. 13 build days, then 3 days for the deck.**

**Theme: prove the pipeline runs end to end and produces a real number.**
Do not attempt the full agentic loop. Review 2 succeeds with verified data, a working
inference harness and a zero-shot table.

### 10.1 Task list

```
[ ] Push the repo scaffold to GitHub
[ ] Start the dataset download (runs unattended overnight)
[ ] Read FineBadminton §6 — note their frame budget, resolution, scoring method
[ ] Set up the A100 environment (CUDA, torch, transformers, vLLM)
[ ] Load Qwen2.5-VL-7B, generate one caption from one clip
[ ] Run scripts/00_verify_dataset.py
[ ] CHECK: are training clips and benchmark clips different clips?   ← BLOCKING
[ ] Dump the real label vocabulary from the JSON
[ ] Extract ~25 frames per clip to disk — only clips FBBench references
[ ] Write the uniform 2 FPS sampler
[ ] Write the multiple-choice prompt (freeze it; never vary between models)
[ ] Write the answer scorer + guided decoding
[ ] Get ONE question answered end to end and scored correctly
[ ] Resolution sweep: 2-3 settings × ~50 questions, pick max_pixels
[ ] Run 500 questions on Qwen2.5-VL-7B
[ ] Run 500 questions on a second backbone (if time)
[ ] Break results down by Count / Action / Position / Cognition
[ ] Log seconds and input tokens per query
[ ] Record every run in results/MANIFEST.md
[ ] Hand-check 20 scored answers against the key
[ ] STRETCH: run the loop on 10 clips, print traces, show one on a slide
[ ] Build the deck; dry run twice
```

### 10.2 Gates

| Gate | Date | Condition |
|---|---|---|
| **G1** | Thu 8 Oct | `dataset_report.md` committed **and** one caption from one real clip |
| **G2** | Mon 12 Oct | **One FBBench question answered end to end and scored correctly** — the critical checkpoint |
| **G3** | Fri 16 Oct | ≥500 questions scored for ≥1 model, per-domain table exists |
| Freeze | Sun 18 Oct | No new experiments |

### 10.3 Sanity check before believing any number

7B zero-shot should land **near or below 25%** overall. Much higher suggests an
answer-extraction bug or leakage. Much lower suggests frames are not loading.
Investigate either before reporting.

**Expect Action-type questions to score poorly.** That is not a broken pipeline — it is
uniform sampling missing contact, and it is the motivation for Review 3. Say so
explicitly in the deck: *"uniform sampling gives the model the rally's shape but not its
detail, and our per-domain breakdown shows exactly where that hurts."*

### 10.4 Descope ladder — cut in this order if behind

1. Second backbone → Qwen2.5-VL-7B alone
2. 500 questions → 250, stratified
3. The loop demo → drop
4. All four domains → overall plus the two best-covered
5. Reproducibility check → push to the next sprint

**Never cut:** dataset verification, the overlap check, the run manifest.

### 10.5 Review 2 deliverables

1. `results/dataset_report.md` — clip counts, splits, zero contamination, label vocabulary
2. Zero-shot FBBench accuracy for ≥1 backbone, **per-domain**
3. Inference cost baseline — seconds and input tokens per query
4. Working repo, tests green, every run in the manifest
5. Two slides of honest risk

---

## 11 · Repository

`badminton-agentic-vlm` — scaffold exists, 47 files, tests passing.

```
configs/      default.yaml, models/, experiments/
docs/         NOVELTY.md, ROADMAP.md, architecture.png
scripts/      00_verify_dataset · 01_zeroshot_benchmark · 02_reproduce_baseline
              03_run_agent · 04_faithfulness
src/bavlm/
  data/       schema.py (WORKING) · splits.py (WORKING) · loader.py (stub)
  sampling/   uniform.py (WORKING) · hit_centric.py (stub)
  agent/      planner.py (parsing WORKING) · observer.py · reflector.py · loop.py (stubs)
              prompts/ (planner, observer, reflector — written)
  tools/      pose.py · tracker.py (stubs)
  memory/     graph.py (WORKING)
  eval/       faithfulness.py (WORKING) · fbbench.py · metrics.py (stubs)
results/      MANIFEST.md
tests/        test_schema.py (4 passing)
```

Every stub raises `NotImplementedError` with its roadmap phase named.
Branch per phase: `phase/<n>-<slug>`. `main` always runnable.
Code MIT; the dataset is Apache-2.0 and is **not** redistributed in the repo.

---

## 12 · Known traps

| Trap | Consequence | Avoid by |
|---|---|---|
| Decoding video inside a training or eval loop | Dominates wall-clock; can cost days | Extract frames once to an on-disk cache |
| Calling FineBadminton's sampling "a heuristic" | Factually wrong; a panel member who read the paper will notice | Say **query-agnostic** — a stronger criticism anyway |
| Saying LoRA reduces inference cost | Wrong; LoRA reduces *training* cost | See §6.1 |
| Claiming memory improves FBBench accuracy | Wrong; FBBench is per-clip | See §3.4 |
| Hand-parsing `"The answer is (B) short serve"` | Silent scoring bugs → wrong numbers that look fine | Guided decoding |
| Varying the prompt between models | You measure prompts, not models | Freeze the prompt; put it in the appendix |
| Fine-tuning before the training-free run | Cannot separate observation strategy from dataset fitting | §9 rule 8 |
| Naive fine-tuning on mixed MCQ + open-ended | Improves MCQ, wrecks open-ended generation | Separate objectives and adapters |
| Extracting frames for all 3,215 clips | Wasted hours | Only clips FBBench references |
| Reporting overall accuracy alone | Hides the mechanism | Per-domain, always |

---

## 13 · Stock answers for panel questions

**"Why a VLM and not MediaPipe / OpenPose?"**
Pose estimators give coordinates, not meaning. Every semantic layer on top must be
hand-coded as brittle geometric rules — closed-set, and producing no explanation a coach
can read. A VLM gives open vocabulary, contextual grounding (the same knee angle is a
fault in one stroke and correct in another), and natural language as a first-class output.
**Crucially: pose is a tool the agent calls, not a competitor.** We are not claiming a VLM
localises joints better than ViTPose — we are claiming the *reasoning layer* must be a VLM.

**"Why agentic rather than a single pass?"**
A rally has ~30 strokes but only two or three matter for any question, and the evidence
is multi-scale: contact needs 2–3 frames at high resolution, rally context needs the
whole clip coarsely. A fixed policy is wrong for one of them. AVP found 76% of errors
under a coarse fixed policy were perception failures. Reflection also lets the system
check a claim before it reaches a coach.

**"VLM or MLLM?"**
The same thing — two names from two communities. The title says VLM, so use VLM
throughout with one line: *"we use VLM and MLLM interchangeably, following common usage."*

**"How will you validate explanations without expert coaches?"**
Counterfactual ablation — remove the cited evidence, re-run, check whether the answer
changes, against a random-ablation control. Fully automatic.

**"Will VLMs even work on this?"**
Published results show they underperform — that is our motivation. Our first experiment
establishes the zero-shot baseline, and our contribution is measured as improvement over
a reproduced query-agnostic baseline at an equal frame budget, not as absolute accuracy.

**"How is this different from FineBadminton?"**
Their keyframe selection is trained but query-agnostic — the same visual tokens regardless
of the question. Ours is query-conditioned and iterative, with reflection triggering
re-observation when evidence is insufficient. Their own benchmark has 12 task types with
contradictory evidence requirements, which a single allocation cannot serve.

---

## 14 · Glossary

| Term | Meaning here |
|---|---|
| **FBBench** | FineBadmintonBenchmark — 2,563 QA pairs, the evaluation set |
| **Frame budget** | ~25 frames per rally, fixed, matching uniform 2 FPS |
| **Query-agnostic** | Produces the same visual input regardless of the question |
| **Hit-centric selection** | FineBadminton's trained keyframe sampler |
| **Trace** | Full record of one query — plan, evidence, citations, answer |
| **Cited evidence** | The evidence items the answer explicitly depends on |
| **Faithfulness score** | flip-rate(cited) − flip-rate(random uncited) |
| **Per-domain** | Broken down by Count / Action / Position / Cognition |
| **Training-free** | The loop evaluated with no fine-tuning |
