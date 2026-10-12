# Dataset report

What the released data actually contains, as opposed to what the paper
describes. Update this whenever a check below is re-run or a release changes.

Last checked: 2026-10-11.

---

## FBBench — `Moujuruo/FineBadmintonBenchmark`

| | |
|---|---|
| HF revision | `ff86093a51efdf9a243a0635704952d404e99e52` |
| Size | 10.76 GB (10.75 GB of it video) |
| Question files | 12 × `finebadmintonbenchmark/benchmark_*.json`, 3.84 MB total |
| Records | **2,413**, all multiple-choice, 4 options each |
| Clips | **2,413** `hf_video_clips_qa/video_NNNNNN.mp4`, shipped with the benchmark |
| Clip key | `video_uid` (`video_000001` … `video_002413`, contiguous) |
| Clips per question | exactly one: every `video_uid` is used by exactly one question |

Frames for FBBench can therefore be extracted from this repo alone. The
Finebadminton-20K download is not needed to build the evaluation frame cache.

### Missing: the 150 open-ended questions

The paper reports 2,563 QA pairs: 2,413 MCQ and 150 open-ended. **This release
has the 2,413 MCQ only.** No file has free-text answers; even the
`round comment` and `hitting comment` tasks are 4-option MCQ here, and the
README does not mention open-ended questions. Until they are found, rule 12
(ROUGE and BERTScore on open-ended output) has nothing to run on, and any
"overall" figure is over MCQ only.

### Record schema

Identical keys in all 12 files, plus two task-specific extras:

```json
{
  "video_uid": "video_000651",
  "duration": "short",
  "spatial": "count",
  "temporal": "hitting count",
  "question": {"question_zh": "...", "question_en": "Calculate the total number of strokes ..."},
  "options_zh": ["A. 8", "B. 2", "C. 0", "D. 4"],
  "options_en": ["A. 8", "B. 2", "C. 0", "D. 4"],
  "answer": "B"
}
```

- `roundcount` adds `start_score`, `end_score`; `select_detail_count_questions`
  adds `count`, `hit_type`.
- Options carry their own `"A. "` prefix. The loader strips it, so the prompt
  doesn't show `A. A. 8`.
- Pinned in `src/bavlm/eval/fbbench.py` (`FIELDS`, `SUBTASKS`); verified by
  `tests/test_fbbench_real.py`.

### Task taxonomy

**`spatial` is the domain field** (rule 5). It has exactly the paper's four values.
Each domain has three sub-tasks, one per file:

| Domain (`spatial`) | Sub-task (from file) | `temporal` value | File | n |
|---|---|---|---|---|
| count | hitting count | hitting count | `benchmark_hitcount` | 200 |
| count | detailed hitting count | detailed hitting count | `benchmark_select_detail_count_questions` | 130 |
| count | round count | round count | `benchmark_roundcount` | 83 |
| **count total** | | | | **413** |
| action | action classification | action classification | `benchmark_action_classification` | 200 |
| action | action localization | action localization | `benchmark_action_localization` | 200 |
| action | action prediction | action prediction | `benchmark_action_prediction` | 250 |
| **action total** | | | | **650** |
| position | hitting localization | hitting localization | `benchmark_hitting_localization_new` | 200 |
| position | hitting prediction | hitting prediction | `benchmark_hitting_prediction` | 200 |
| position | moving recognition | moving recognization | `benchmark_moving_recognization` | 200 |
| **position total** | | | | **600** |
| cognition | hitting comment | hitting comment | `benchmark_hitting_comment2_updated` | 250 |
| cognition | round comment | round comment | `benchmark_round_comment_updated` | 250 |
| cognition | point reason | **round comment** ⚠ | `benchmark_pointer_recog_updated` | 250 |
| **cognition total** | | | | **750** |
| **all** | | | | **2,413** |

⚠ `pointer_recog_updated` is labelled `temporal: "round comment"`, the same as
`round_comment_updated`, but it asks a different question ("analyze the main
reason why the scoring side scored"). **Sub-task is taken from the file name,
not `temporal`**, otherwise two sub-tasks merge into one 500-question bucket.

`duration` is `short` / `mid` / `long` in 11 files (823 / 789 / 551), but an
**integer 61–202** in all 250 `hitting_comment2_updated` records. The unit is
unknown (seconds? frames at 25 fps would be 2.4–8 s). Do not slice results by
duration until this is resolved.

| Domain | short | mid | long | int |
|---|---|---|---|---|
| count | 42 | 116 | 255 | — |
| action | 359 | 276 | 15 | — |
| position | 351 | 207 | 42 | — |
| cognition | 71 | 190 | 239 | 250 |

### Answer balance and chance floors

| | A | B | C | D |
|---|---|---|---|---|
| all | 633 | 594 | 617 | 569 |

Always-A scores 26.2% overall (count 28.6, action 25.1, position 23.8,
cognition 27.9). **That is the floor to beat per domain, not 25%.** A model
that collapses to one letter will look slightly above chance in count and
cognition.

### Harness check (mock mode, all 2,413 real questions)

| Mock | overall | count | action | position | cognition | expected |
|---|---|---|---|---|---|---|
| oracle | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100% |
| random (seed 7) | 23.8% | 26.4% | 23.4% | 24.8% | 22.0% | ~25% |
| always_a | 26.2% | 28.6% | 25.1% | 23.8% | 27.9% | = share of A |
| refuse | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0% |

Reproduce: `python tests/test_fbbench_real.py`.

---

<!-- 20k:begin -->
## Finebadminton-20K — `Moujuruo/Finebadminton-20K`

Generated by `scripts/00_verify_dataset.py`; do not edit by hand.

| | |
|---|---|
| HF revision | `72cae601e08c83ee8ec7e19f8fd7850e9158af7d` |
| Contents | 70 full-match videos (43.4 GB) · one annotation JSON per match (26 MB) |
| Rallies / hits | **2,066** rallies · **20,757** hits (the full dataset has 3,215 / 33,325) |
| Mean rally | 10.4 s |
| Frame rate | **not constant**: ~30 fps × 1,370 rallies, ~25 fps × 696 rallies. Convert frame indices with each rally's own `fps` |
| Rally key | `video` + `start_frame`–`end_frame` (no clip IDs) |

## Rule 2 — overlap check: clean, by holding out FBBench's matches

FBBench clips are renamed `video_NNNNNN` and carry no source video or frame
range, so comparing IDs reports zero overlap whatever the truth (the original
`check_overlap` did exactly that). The 20K release publishes
`benchmark_source_videos.txt`: the **20 match videos FBBench was built from**,
all 20 present in this release.

**Every rally from those 20 matches is held out of training** (609 rallies,
29%). What remains cannot contain an FBBench rally, provided the
publisher's list is complete. Training uses only the other 50 matches.

| Split (by match, rule 3, seed 0) | Matches | Rallies |
|---|---|---|
| train | 45 | 1,289 |
| val | 5 | 168 |
| held out (FBBench source) | 20 | 609 |

The split is fixed in `configs/split_20k.json`. Val matches: 0013, 0027, 0036, 0038, 0071.

**Residual risk:** the result rests on the publisher's list. Before
fine-tuning, a frame-fingerprint spot check (perceptual hash of FBBench clip
frames against the 50 training matches) would confirm it independently.
That needs the training videos, so it is a phase-2 task on Kaggle.

## Label vocabulary (every hit, all 70 matches)

**hit type**: 11 values (+ 27 noisy entries: `''` 27)

push shot 4,734 · kill 3,015 · net shot 2,274 · block 2,243 · serve 2,065 · drive 1,816 · drop shot 1,372 · clear 1,368 · net lift 705 · cross-court net shot 678 · net kill 460

**subtype**: 20 values (+ 11 noisy entries: `'、'` 3, `'劈吊'` 2, `'拨腰'` 1, `'开始'` 1, `'快筛'` 1, `'上网'` 1, `'抽高'` 1, `'天天'` 1)

flat lift 3,225 · jump smash 1,785 · short serve 1,720 · spinning net 1,103 · attacking clear 827 · full smash 763 · slice smash 648 · high lift 631 · common smash 620 · slice drop shot 590 · high drive 385 · flat drive 382 · reverse slice drop shot 355 · high serve 334 · blocked drop shot 298 · stick smash 219 · stop drop shot 138 · flick serve 105 · high block 79 · continuous net kills 37

**ball area**: 9 values (+ 2 noisy entries: `'中钱场'` 1, `''` 1)

mid front court 4,669 · mid court 2,812 · left front court 2,422 · right front court 2,400 · left mid court 1,877 · right mid court 1,677 · left back court 1,657 · right back court 1,634 · mid back court 1,607

**hitter**: 2 values (+ 1 noisy entries: `None` 1)

top 10,379 · bottom 10,377

**quality**: 7 values

4 8,546 · 5 4,258 · 3 2,693 · 2 1,626 · 6 1,579 · 7 1,131 · 1 924

**player actions**: 14 values

forehand 12,522 · backhand 8,190 · passive 6,102 · intercept 3,397 · defensive 2,702 · move to the net 2,308 · To create depth 1,916 · a high net early shot 1,670 · hesitation 1,300 · deception 951 · seamlessly 829 · turnaround 320 · Control the net 224 · air-shot 11

**shot characteristics**: 9 values

cross-court 12,807 · straight 6,662 · over head 4,769 · body hit 605 · Net fault 525 · wide placement 504 · out 460 · Repeat shot 349 · passing shot 240

The paper's counts hold: **11 hit types, 20 subtypes, 9 court regions**,
quality 1–7. The release adds a few blank, Chinese or punctuation labels
(listed above); they are flagged by `is_clean_label`, never remapped. Quality
is stored as both `4` and `"4"`; `label_vocabulary` merges them.
<!-- 20k:end -->
