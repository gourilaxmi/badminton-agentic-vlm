# Novelty

## One-paragraph statement

Agentic video systems produce a reasoning trace — a plan, a set of evidence, a
justification — and that legibility is treated as an interpretability guarantee by every
paper that produces one. It has never been tested. We build a query-conditioned agentic
perception loop for fine-grained badminton, a setting where evidence is sparse, fast and
spatially localisable, and we use it to ask two questions that the field has not: can an
agent allocate a fixed observation budget better than a learned query-agnostic selector,
and is the evidence it cites what actually drove its answer?

## Contribution 1 — Query-conditioned observation for fine-grained sports

**Claim.** Given the same frame budget (~25 frames per rally, matching the 2 FPS
baseline), allocating frames according to the query type outperforms allocating them by
hit detection.

**Why it is not already done.** FineBadminton's Hit-Centric Keyframe Selection is a
*trained* module — VideoMAE features, attention pooling, a transformer temporal encoder
producing per-frame hit probability, then a FastRCNN-guided Perceiver ReSampler. It is
good, and it is query-agnostic. Their own benchmark spans four domains whose evidence
needs conflict:

| Domain | Evidence needed | Hit-centric sampling |
|---|---|---|
| Count | broad temporal coverage, low detail | over-concentrates at contact |
| Action | dense frames at contact | near-optimal |
| Position | spatial detail, moderate temporal spread | partial |
| Cognition | full-rally causal context | over-concentrates at contact |

Their reported failure modes — subtle action discrimination and complex multi-step
reasoning — sit at opposite ends of this table. A single allocation cannot serve both.

**Prediction that makes it falsifiable.** Gains should be largest on Count and Cognition
and near parity on Action. Reporting per-domain turns a higher number into mechanistic
evidence.

**Strength: medium.** The plan–observe–reflect mechanism comes from AVP. The setting is
new: AVP operates on hour-long video where evidence is sparse and *slow*; badminton at
25 FPS is sparse and *fast*, and the binding constraint is spatial detail within a
frame budget rather than temporal search across hours.

## Contribution 2 — Faithfulness of agentic reasoning traces

**Claim.** Cited evidence in an agentic video system is measurably more load-bearing
than uncited evidence — or it is not, and that is a finding.

**Why it is novel.** AVP, Deep Video Discovery, VideoAgent and CASTLE all emit reasoning
traces and all treat them as interpretable by construction. None test it. Turpin et al.
(NeurIPS 2023) showed language-model reasoning frequently does not drive the output;
whether that holds for agentic *video* systems is unasked, and it matters more here
because the trace is the entire interpretability claim.

**Protocol.**

| Measure | Method |
|---|---|
| Necessity | Delete each cited evidence item, re-run, record whether the answer changes |
| Control | Delete an equal quantity of *uncited* evidence from the same clip |
| Faithfulness score | flip-rate(cited) − flip-rate(random) |
| Sufficiency | Give a fresh model only the cited evidence; does it reach the same answer? |
| Calibration | Does the reflector's confidence track correctness? |

No human annotators. Fully automatic. This is precisely why it survived when the
biomechanical-validation design did not.

**The result worth hoping for.** That the reflector improves accuracy while *reducing*
faithfulness — because reflection teaches the model to construct better post-hoc
justifications. If that appears, it is a genuine and cautionary finding for the whole
agentic-interpretability direction.

**Strength: high.** Unasked question, clean protocol, generalises past badminton.

## Contribution 3 — Pose-grounded evidence inside an agentic loop

**Claim.** Agentic video systems (AVP, DVD) operate on raw frames only; pose-grounded
systems (LLaVA-Pose, MistSense, BioCoach) are single-pass. No system lets an agent
*decide* to call a pose estimator when the VLM's own evidence is insufficient.

In our loop, pose is a **tool the observer invokes**, not a competing pipeline. When the
reflector reports that stroke identity is unresolved, the planner can request
keypoint evidence for a narrow window rather than simply more frames.

**Strength: medium.** This is the Gap 4 argument from Review 1, made concrete.

## Contribution 4 — Longitudinal player memory

A temporal knowledge graph keyed on **player**, not video, accumulating tendencies
across rallies and matches to support opponent scouting.

**Important scoping note.** This does **not** contribute to FBBench accuracy. FBBench
asks about individual clips; memory answers cross-rally questions. It is evaluated
separately through the scouting demonstration and a player-tendency analysis. Claiming
it improves the benchmark score would be wrong.

**Strength: low as novelty, high as application value.** This is the societal-impact
story — analytical capability currently confined to elite programmes with dedicated
analysts — not the research claim.

## What we are explicitly not claiming

- Not that the system improves match outcomes. That needs a longitudinal study.
- Not that pose estimation is more accurate than a purpose-built pose model.
- Not that VLMs are good at badminton. The published evidence says they are poor;
  our claim is relative, against a reproduced baseline at an equal frame budget.
