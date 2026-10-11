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

## Rule 2 — overlap check is blocked on filenames

Rule 2 requires verifying that no FBBench clip appears in the Finebadminton-20K
training corpus before fine-tuning. **FBBench renames its clips to
`video_NNNNNN`** and ships no mapping back to the source match or rally, so a
filename or stem comparison will report zero overlap whether or not there is
any. Treat that result as meaningless.

Not implemented yet. Suggested approach, in order of cost:

1. **Exact content hash (SHA-256 of the file).** Cheap and conclusive *if* the
   clip was copied byte-for-byte. It misses any re-encode, trim or resize, so a
   zero here does **not** prove there is no overlap. Run it first to catch the
   easy cases.
2. **Frame fingerprint (the real check).** For each clip, decode a few frames
   at fixed relative positions (e.g. 10/30/50/70/90 % of duration). Downscale
   each to a small grayscale image and take a perceptual hash (pHash/dHash, 64
   bits), giving one short signature per clip. Then:
   - Index all Finebadminton-20K signatures, and for each FBBench clip find the
     nearest by Hamming distance.
   - Flag pairs under a threshold for manual review. Calibrate the threshold on
     clip pairs we know are the same: one clip re-encoded against itself.
   - Because FBBench clips may be **sub-spans** of a 20K rally (QA clips can be
     shorter than rallies), match frames against a sliding window over the 20K
     clip's frames, not only fixed relative positions.

   This is robust to re-encoding and resizing, and it reuses the frame cache
   we are building anyway, so 20K frames would be decoded only once.
3. **Use the result to split, not just to report.** Any 20K **match** with a
   hit is held out of training entirely (rule 3: split by match, not clip).

The overlap check needs Finebadminton-20K frames, so it is a phase-2 task
before fine-tuning. It does not block zero-shot evaluation (Review 2).

---

## Finebadminton-20K — `Moujuruo/Finebadminton-20K`

Not yet inspected. Size estimated at ~30 GB (not verified). Needed for development and
training, and for the rule-2 check above; not needed for FBBench evaluation
frames.
