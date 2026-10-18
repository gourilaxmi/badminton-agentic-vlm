# Run manifest

One row per reported run. A number without a row here does not go in the report.

| Date | Experiment | Model | Sampling | Frames | Split | Commit | Result file |
|---|---|---|---|---|---|---|---|
| 2026-10-11 09:50 UTC | e00_smoke (plumbing only, n=10, not a result) | Qwen/Qwen2.5-VL-7B-Instruct | uniform 2 FPS, short side 448 | 11.1 avg | FBBench n=10 | 13a4d62 | e00_smoke_Qwen2.5-VL-7B-Instruct.jsonl |
| 2026-10-11 11:53 UTC | e01_zeroshot **INVALID: 138/500 OOM, do not report** | Qwen/Qwen2.5-VL-7B-Instruct | uniform 2 FPS, short side 448, cap 25 | 14.8 avg | FBBench n=500 | 5f28597 | e01_zeroshot_Qwen2.5-VL-7B-Instruct.jsonl (v6) |
| 2026-10-11 16:09 UTC | e01_zeroshot | Qwen/Qwen2.5-VL-7B-Instruct | uniform 2 FPS, short side 448, cap 25 | 14.8 avg | FBBench n=500 | 39c567a | e01_zeroshot_Qwen2.5-VL-7B-Instruct.jsonl (v7) |
| 2026-10-11 16:35 UTC | e02_screen (backbone screen) | Qwen/Qwen3-VL-8B-Instruct | uniform 2 FPS, short side 448, cap 25 | 15.5 avg | FBBench n=150 | 6ec245b | e02_screen_Qwen3-VL-8B-Instruct.jsonl (v8) |
| 2026-10-11 16:35 UTC | e02_screen **INVALID for ranking: 46/150 fp16 non-finite** | Qwen/Qwen2.5-VL-3B-Instruct | uniform 2 FPS, short side 448, cap 25 | 15.5 avg | FBBench n=150 | 6ec245b | e02_screen_Qwen2.5-VL-3B-Instruct.jsonl (v8) |
| 2026-10-11 16:09 UTC | e02_screen (first 150 rows of e01 v7, same questions) | Qwen/Qwen2.5-VL-7B-Instruct | uniform 2 FPS, short side 448, cap 25 | 15.5 avg | FBBench n=150 | 39c567a | e01_zeroshot_Qwen2.5-VL-7B-Instruct.jsonl rows 1-150 |
| 2026-10-11 21:01 UTC | e03_zeroshot | Qwen/Qwen3-VL-8B-Instruct | uniform 2 FPS, short side 448, cap 25 | 14.8 avg | FBBench n=500 | 6ec245b | e03_zeroshot_Qwen3-VL-8B-Instruct.jsonl (v9) |
| 2026-10-12 18:12 UTC | e05_resolution | Qwen/Qwen3-VL-8B-Instruct | uniform 2 FPS, short side 336, cap 25 | 14.8 avg | FBBench n=500 | 666507b | e05_resolution_Qwen3-VL-8B-Instruct@336.jsonl (v10) |
| 2026-10-12 18:12 UTC | e05_resolution | Qwen/Qwen3-VL-8B-Instruct | uniform 2 FPS, short side 224, cap 25 | 14.8 avg | FBBench n=500 | 666507b | e05_resolution_Qwen3-VL-8B-Instruct@224.jsonl (v10) |
| 2026-10-18 09:13 UTC | e06_full | Qwen/Qwen3-VL-8B-Instruct | uniform 2 FPS, short side 448, cap 25 | 14.0 avg | FBBench n=2413 | 3c641a3 | e06_full_Qwen3-VL-8B-Instruct.jsonl |

## Dataset revisions

Record the exact HuggingFace revision hashes so results stay reproducible.

- `Moujuruo/Finebadminton-20K` @ `72cae601e08c83ee8ec7e19f8fd7850e9158af7d` (annotations only, checked 2026-10-12)
- `Moujuruo/FineBadmintonBenchmark` @ `ff86093a51efdf9a243a0635704952d404e99e52` (checked 2026-10-11)
