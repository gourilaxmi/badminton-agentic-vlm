# Kaggle setup — step by step

From nothing to a running FBBench benchmark. Allow about 2 hours, most of it
waiting for downloads.

---

## The memory problem, first

Qwen2.5-VL-7B-Instruct is **8.29 B parameters ≈ 16.6 GB in fp16**. A single
Kaggle GPU has 16 GB. **It does not fit.**

| Accelerator | VRAM | Fits a 7B? |
|---|---|---|
| GPU P100 | 16 GB single | **No** in fp16 — only with 4-bit quantization |
| **GPU T4 × 2** | **16 + 16 = 32 GB** | **Yes**, sharded with `device_map="auto"` |
| TPU VM v3-8 | — | Not worth the porting effort |

**Choose `GPU T4 x2`.** That is what "high memory" means on Kaggle — there is no
separate RAM toggle; memory comes with the accelerator. System RAM is ~29 GB
either way.

Fallbacks if T4 × 2 is unavailable:

| Option | Trade-off |
|---|---|
| P100 + 4-bit NF4 (~4.7 GB) | Works, but quantization changes accuracy — state it if you report the number |
| Qwen2.5-VL-**3B** (~7.5 GB fp16) | Fits one P100 easily. Weaker comparison: the 42.06% reference used the 7B |

---

## Step 1 · Kaggle account — verify your phone

**This blocks everything and is easy to miss.**

1. kaggle.com → Settings → **Phone Verification**
2. Verify.

Without it you cannot enable **GPU** or **Internet** in notebooks, and the
notebook needs both. Do this first; verification is instant but discovering the
requirement three steps later is not.

While there, note your quota: **30 GPU-hours per week**, resetting Saturday.

---

## Step 2 · GitHub repo and token

**2a. Create the repo** (on your laptop, from the scaffold):

```bash
unzip badminton-agentic-vlm.zip
cd badminton-agentic-vlm
# drop in the updated files: backend.py, ROADMAP.md, architecture.png, HANDOFF.md
git init
git add .
git commit -m "scaffold: structure, roadmap, mock backend"
gh repo create badminton-agentic-vlm --private --source=. --push
```

**2b. Create a scoped token:**

github.com → Settings → Developer settings → **Fine-grained personal access
tokens** → Generate new token

| Field | Value |
|---|---|
| Repository access | **Only select repositories** → `badminton-agentic-vlm` |
| Permissions → Contents | **Read and write** |
| Expiry | 90 days (covers to Review 3) |

Copy it now — GitHub shows it once.

> Use a fine-grained token, not a classic one. A classic token carries your
> whole account; this one can touch a single repo.

---

## Step 3 · Kaggle Secrets

1. kaggle.com → **Create → Notebook**
2. Right-hand panel → **Add-ons → Secrets** → *Attach secret* → *Create new*
3. Label `GH_TOKEN`, value = the token from step 2b
4. Toggle it **attached** for this notebook

Secrets never appear in output unless you print them. The notebook also scrubs
the token from any subprocess output, so a failed `git push` cannot leak it into
a saved version.

---

## Step 4 · Build the frame cache (laptop, CPU, ~1 hour)

The single biggest saving. Video decoding is CPU work — doing it inside a GPU
session burns quota on something a laptop does for free.

```bash
hf download Moujuruo/FineBadmintonBenchmark --repo-type dataset --local-dir data/fbbench
hf download Moujuruo/Finebadminton-20K      --repo-type dataset --local-dir data/finebadminton-20k

python scripts/extract_frames.py \
    --fbbench data/fbbench \
    --clips   data/finebadminton-20k \
    --out     ./frame_cache \
    --fps 2 --size 224
```

Extract **only clips FBBench references**, not all 3,215.

Expected size: 25 frames × ~15 KB ≈ 375 KB per clip, so roughly **1.2 GB** at
224 px (about 5 GB at 448).

---

## Step 5 · Upload the cache as a Kaggle Dataset

```bash
pip install kaggle
# token from kaggle.com → Settings → API → Create New Token
# save the downloaded kaggle.json to ~/.kaggle/kaggle.json, chmod 600

kaggle datasets init -p ./frame_cache
```

Edit the generated `dataset-metadata.json`:

```json
{
  "title": "FineBadminton frames 2fps 224",
  "id": "<your-kaggle-username>/finebadminton-frames",
  "licenses": [{"name": "other"}]
}
```

```bash
kaggle datasets create -p ./frame_cache --dir-mode zip
```

Upload takes 10–30 minutes. Do it once; every run afterwards attaches it in
seconds.

---

## Step 6 · Notebook settings

Upload `fbbench_zeroshot_kaggle.ipynb` (File → Import Notebook), then in the
right-hand panel:

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 x2** |
| Internet | **On** |
| Persistence | Variables and Files |
| Environment | Always use latest |

Then **Add data → Your Datasets →** attach `finebadminton-frames`.
It mounts read-only at `/kaggle/input/finebadminton-frames`.

---

## Step 7 · Config and multi-GPU

Edit cell 2 of the notebook:

```python
GH_USER  = "your-github-username"
GH_REPO  = "badminton-agentic-vlm"
FRAME_DS = "/kaggle/input/finebadminton-frames"
```

For T4 × 2, the model loads sharded across both cards:

```python
from transformers import Qwen2_5_VLForConditionalGeneration

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,     # T4 is sm_75 — no bf16
    device_map="auto",             # shards 16.6 GB across the two cards
    attn_implementation="sdpa",    # flash-attn-2 needs sm_80+, do not use here
)
```

Three things that will bite otherwise:

- **`float16`, not `bfloat16`.** T4 and P100 are pre-Ampere. Watch for NaNs in
  long generations; that is the cause if you see them.
- **`attn_implementation="sdpa"`.** FlashAttention-2 requires sm_80+ and will
  fail to build.
- **`device_map="auto"`** is what makes 7B fit at all. Without it you get an OOM
  on card 0.

---

## Step 8 · Run in order

1. **Cells 1–5 first.** Cell 5 is the mock check — oracle must be 100%, random
   ~25%. **If either is off, stop.** Fix the scorer before spending GPU time;
   the real run would produce a number that looks fine and is wrong.
2. **Cell 6** — the benchmark. Resumable; it skips anything already in the
   JSONL, so a timeout costs nothing.
3. **Cells 7–8** — score per domain, push to GitHub.
4. **Save Version** before closing the tab.

---

## What a 500-question run costs

| | |
|---|---|
| Model download (first run only) | 10–15 min |
| 500 questions, 7B, 25 frames | 1.5–3 h on T4 × 2 |
| Resolution sweep, 3 × 50 | ~30 min |
| **Total for Review 2** | **4–6 h of a 30 h weekly quota** |

Comfortable. Keep a few hours spare for the second backbone.

---

## If something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | 7B on one card | `device_map="auto"`; confirm T4 × 2 is selected |
| GPU option greyed out | Phone not verified | Step 1 |
| `git push` fails, 403 | Token lacks Contents: write, or wrong repo scope | Re-issue the token |
| No internet / pip fails | Internet toggle off | Right panel → Internet → On |
| NaNs in generation | fp16 overflow | Shorter `max_new_tokens`; guided decoding makes it ~2 |
| `flash_attn` build error | sm_75 unsupported | `attn_implementation="sdpa"` |
| Oracle ≠ 100% | ID mismatch or shuffled options | **Stop. Fix before any GPU run.** |
| Session dies mid-run | 12 h limit or idle timeout | Re-run cell 6 — it resumes |

---

## Order of operations

```
Phone verify  →  GitHub repo + token  →  Kaggle secret
                                              ↓
         Extract frames (laptop)  →  Upload as Dataset
                                              ↓
              Notebook settings  →  Attach data  →  Run
```

Steps 1–3 take twenty minutes and unblock everything. Do them tonight; start the
frame extraction before bed so the upload is ready in the morning.
