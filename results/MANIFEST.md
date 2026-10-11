# Run manifest

One row per reported run. A number without a row here does not go in the report.

| Date | Experiment | Model | Sampling | Frames | Split | Commit | Result file |
|---|---|---|---|---|---|---|---|
| 2026-10-11 09:50 UTC | e00_smoke (plumbing only, n=10, not a result) | Qwen/Qwen2.5-VL-7B-Instruct | uniform 2 FPS, short side 448 | 11.1 avg | FBBench n=10 | 13a4d62 | e00_smoke_Qwen2.5-VL-7B-Instruct.jsonl |
| 2026-10-11 11:53 UTC | e01_zeroshot **INVALID: 138/500 OOM, do not report** | Qwen/Qwen2.5-VL-7B-Instruct | uniform 2 FPS, short side 448, cap 25 | 14.8 avg | FBBench n=500 | 5f28597 | e01_zeroshot_Qwen2.5-VL-7B-Instruct.jsonl (v6) |
| 2026-10-11 16:09 UTC | e01_zeroshot | Qwen/Qwen2.5-VL-7B-Instruct | uniform 2 FPS, short side 448, cap 25 | 14.8 avg | FBBench n=500 | 39c567a | e01_zeroshot_Qwen2.5-VL-7B-Instruct.jsonl (v7) |

## Dataset revisions

Record the exact HuggingFace revision hashes so results stay reproducible.

- `Moujuruo/Finebadminton-20K` @ `<revision>`
- `Moujuruo/FineBadmintonBenchmark` @ `ff86093a51efdf9a243a0635704952d404e99e52` (checked 2026-10-11)
