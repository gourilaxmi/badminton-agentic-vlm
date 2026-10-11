# BTP Runbook — local machine + Kaggle GPU

Everything needed to go from an empty laptop to a scored FBBench run, with all
code inlined.

**Project:** VLM-Grounded Agentic Reasoning for Explainable Action and Error
Recognition in Dynamic Video Sequences · IIIT Kottayam
**Target:** Review 2, Wed 21 Oct 2026 — build freeze Sun 18 Oct

---

## 0 · How the pieces fit

You write code on your laptop, GitHub holds the shared state, Kaggle supplies
the GPU. Nothing heavy ever runs on the laptop, and nothing is lost when a
Kaggle session dies.

```
   LAPTOP                      GITHUB                     KAGGLE
   ------                      ------                     ------
   write code         push  ->  main  <- clone      run on GPU T4 x2
   mock-mode tests                     ->            benchmark
   extract frames                      <-            push results
        |                                                  ^
        |  upload once as a Kaggle Dataset                 |
        +--------------------------------------------------+
```

Three rules that follow from this picture:

1. **Frame extraction happens on the laptop.** Decoding video is CPU work;
   doing it inside a GPU session burns quota on something a laptop does free.
2. **Results are pushed to GitHub, not left in `/kaggle/working`.** Kaggle output
   survives only if you remember to Save Version. GitHub always survives.
3. **Mock mode runs on the laptop before anything touches Kaggle.** A Kaggle job
   takes a minute or two just to start; twenty debug round-trips is an hour of
   waiting and quota spent on bugs you could have caught in seconds.

---

## 1 · The memory question, answered first

Qwen2.5-VL-7B-Instruct is **8.29 B parameters ≈ 16.6 GB in fp16**.

| Accelerator | VRAM | Fits a 7B? |
|---|---|---|
| GPU P100 | 16 GB single | **No** in fp16 — only with 4-bit quantization |
| **GPU T4 × 2** | **16 + 16 = 32 GB** | **Yes**, sharded with `device_map="auto"` |

**Select `GPU T4 x2`.** On Kaggle there is no separate "high RAM" switch —
memory comes with the accelerator. System RAM is ~29 GB either way.

Fallbacks:

| Option | Trade-off |
|---|---|
| P100 + 4-bit NF4 (~4.7 GB) | Works; quantization shifts accuracy, so state it if you report the number |
| Qwen2.5-VL-**3B** (~7.5 GB fp16) | Fits one P100 easily. Weaker comparison — the 42.06% reference used the 7B |

---
---

# PART A · Local machine setup

## A1 · Prerequisites

```bash
# check what you already have
git --version          # need 2.x
python3 --version      # need 3.10+
```

Install if missing: [git-scm.com/downloads](https://git-scm.com/downloads) ·
[python.org/downloads](https://python.org/downloads)

Then the real GitHub CLI, for creating the repo:

```bash
# macOS
brew install gh
# Windows
winget install GitHub.cli
# Debian/Ubuntu
sudo apt install gh
```

## A2 · Create the repository

```bash
unzip badminton-agentic-vlm.zip
cd badminton-agentic-vlm

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest tests/ -q         # must pass before you commit anything

git init
git add .
git commit -m "scaffold: structure, roadmap, mock backend"

gh auth login                      # browser flow, one time
gh repo create badminton-agentic-vlm --private --source=. --push
```

## A3 · Four branches

```bash
for b in gouri amal vishal sabeeh; do
  git branch "$b" main
  git push -u origin "$b"
done
```

Map each branch to a file area so you rarely edit the same file:

| Branch | Owns | Mostly touches |
|---|---|---|
| ___ | Data & infra | `data/`, `scripts/00_*`, frame cache, manifest |
| ___ | Models & inference | `models/backend.py`, `scripts/01_*` |
| ___ | Agent & prompts | `agent/`, `agent/prompts/` |
| ___ | Evaluation | `eval/`, scorer, per-domain breakdown |

**Merge to `main` daily**, at the end of each standup:

```bash
git checkout main && git pull && git merge <branch> && git push
```

Four branches left to diverge for a week will conflict badly on 18 October —
the worst possible day. On GitHub → Settings → Branches, at minimum **block
force pushes** on `main`. The Kaggle notebook clones `main`, so keep it green.

## A4 · GitHub token for Kaggle

github.com → Settings → Developer settings → **Fine-grained personal access
tokens** → Generate new token

| Field | Value |
|---|---|
| Repository access | **Only select repositories** → `badminton-agentic-vlm` |
| Permissions → Contents | **Read and write** |
| Expiry | 90 days — covers to Review 3 |

Copy it now; GitHub shows it once. Use a fine-grained token, not a classic one:
a classic token carries your whole account, this one touches a single repo.

---
---

# PART B · Kaggle setup

## B1 · Verify your phone — this blocks everything

kaggle.com → Settings → **Phone Verification**

Without it, **GPU and Internet are both unavailable** in notebooks, and you need
both. It takes a minute, but discovering the requirement three steps later costs
an evening.

While there, note the quota: **30 GPU-hours per week**, resetting Saturday.

## B2 · Kaggle API token (for uploading the frame cache)

kaggle.com → Settings → API → **Create New Token**. Downloads `kaggle.json`.

```bash
pip install kaggle
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## B3 · Kaggle Secret for GitHub

1. kaggle.com → **Create → Notebook**
2. Right panel → **Add-ons → Secrets** → *Attach secret* → *Create new*
3. Label `GH_TOKEN`, value = the token from A4
4. Toggle it **attached**

Secrets never appear in output unless printed. The notebook also scrubs the
token from subprocess output, so a failed `git push` cannot leak it into a saved
version.

---
---

# PART C · All the code

Three modules and one notebook. Each is also shipped as a standalone file —
drop them in rather than copying from here.

## C1 · `src/bavlm/models/backend.py`

The mock backend is what lets the entire pipeline be developed on a laptop. Two
modes matter most: **oracle** must score 100% (if not, the plumbing is broken)
and **random** must score ~25% (if not, the scorer is broken).

```python
"""VLM backend with mock modes.

Mock mode exists so the ENTIRE pipeline -- frame loading, prompting, answer
extraction, scoring, per-domain breakdown, the manifest, later the agentic loop
and the faithfulness ablation -- can be developed and tested on a laptop with no
GPU, no model download and no Kaggle quota spent.

The two modes that matter most:

    oracle  -> always returns the correct answer. The scorer MUST report 100%.
               Anything less means the plumbing is broken: question IDs not
               matching the answer key, a shuffled option order, an off-by-one.
               This catches more bugs than any other single check.

    random  -> uniform random option. The scorer MUST report ~25% on 4-option
               questions. Much higher means the scorer is marking wrong answers
               correct. Much lower means it is marking correct ones wrong.

Run both before you ever touch a GPU. If oracle is not 100% and random is not
near chance, your numbers would have been wrong and you would not have known.
"""
from __future__ import annotations

import random
import re
from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class VLMBackend(Protocol):
    """One interface, so experiments compare observation strategies rather than
    plumbing. Every backend -- mock, HuggingFace, vLLM, an API model -- answers
    to this."""

    name: str

    def generate(self, frames: Sequence, prompt: str, **kw) -> str:
        """frames may be paths, arrays or PIL images; mock backends ignore them."""
        ...


# --------------------------------------------------------------------------- mock

class MockBackend:
    """No model, no GPU. Returns a plausibly-shaped answer immediately.

    mode:
      oracle        correct answer every time              -> expect 100%
      random        uniform over the options               -> expect ~1/n
      noisy         correct with probability `accuracy`    -> expect ~accuracy
      always_a      always the first option                -> reveals key skew
      malformed     verbose prose around the answer        -> tests extraction
      refuse        unhelpful text, no option letter       -> tests the fallback
    """

    def __init__(self, mode: str = "random", answer_key: dict[str, str] | None = None,
                 accuracy: float = 0.4, seed: int = 0, n_options: int = 4) -> None:
        self.name = f"mock:{mode}"
        self.mode = mode
        self.key = answer_key or {}
        self.accuracy = accuracy
        self.n_options = n_options
        self.rng = random.Random(seed)
        self.calls = 0

        if mode in ("oracle", "noisy") and not self.key:
            raise ValueError(f"mode '{mode}' needs answer_key={{qid: letter}}")

    # -- helpers ----------------------------------------------------------
    def _letters(self) -> list[str]:
        return [chr(ord("A") + i) for i in range(self.n_options)]

    def _wrong(self, correct: str) -> str:
        pool = [l for l in self._letters() if l != correct]
        return self.rng.choice(pool) if pool else correct

    # -- the interface ----------------------------------------------------
    def generate(self, frames: Sequence, prompt: str, qid: str | None = None, **kw) -> str:
        self.calls += 1

        if self.mode == "always_a":
            return "A"

        if self.mode == "refuse":
            return "I cannot determine this from the frames provided."

        if self.mode == "random":
            return self.rng.choice(self._letters())

        correct = self.key.get(qid or "", "A")

        if self.mode == "oracle":
            return correct

        if self.mode == "noisy":
            return correct if self.rng.random() < self.accuracy else self._wrong(correct)

        if self.mode == "malformed":
            # the shape a real VLM actually produces when not constrained
            return (f"Looking at the sequence of frames, the player appears to "
                    f"execute the stroke in question. The answer is "
                    f"({correct}) based on the racket position.")

        raise ValueError(f"unknown mock mode: {self.mode}")


# --------------------------------------------------------------------------- real

class HFBackend:
    """Real model. Loaded lazily so importing this module costs nothing."""

    def __init__(self, hf_id: str, dtype: str = "bfloat16", max_frames: int = 32) -> None:
        self.name = hf_id
        self.hf_id = hf_id
        self.dtype = dtype          # NOTE: use "float16" on Kaggle P100/T4 -- no bf16
        self.max_frames = max_frames
        self._model = None
        self._proc = None

    def _load(self):
        raise NotImplementedError("roadmap phase 1 -- wire transformers or vLLM here")

    def generate(self, frames: Sequence, prompt: str, **kw) -> str:
        if self._model is None:
            self._load()
        raise NotImplementedError("roadmap phase 1")


# --------------------------------------------------------------------------- factory

def build_backend(cfg: dict, answer_key: dict[str, str] | None = None) -> VLMBackend:
    """One switch between mock and real, so no caller needs to know which it has.

        backend = build_backend({"mock": "oracle"}, answer_key=key)
        backend = build_backend({"hf_id": "Qwen/Qwen2.5-VL-7B-Instruct"})
    """
    if cfg.get("mock"):
        return MockBackend(
            mode=cfg["mock"],
            answer_key=answer_key,
            accuracy=cfg.get("mock_accuracy", 0.4),
            seed=cfg.get("seed", 0),
            n_options=cfg.get("n_options", 4),
        )
    return HFBackend(
        hf_id=cfg["hf_id"],
        dtype=cfg.get("dtype", "bfloat16"),
        max_frames=cfg.get("max_frames", 32),
    )


# --------------------------------------------------------------------------- extraction

_LETTER = re.compile(r"\b([A-H])\b")

def extract_choice(text: str, n_options: int = 4) -> str | None:
    """Pull an option letter out of free-form model output.

    Guided decoding makes this unnecessary for the real runs -- which is the
    point of using it. Keep this for the open-ended questions, for API models
    that ignore constraints, and so `mock=malformed` has something to exercise.

    Returns None when no option is present, so a refusal scores as wrong rather
    than silently defaulting to 'A' and inflating the number.
    """
    valid = {chr(ord("A") + i) for i in range(n_options)}

    stripped = text.strip()
    if stripped[:1].upper() in valid and len(stripped) <= 2:
        return stripped[:1].upper()

    for pat in (r"answer\s*(?:is|:)\s*\(?([A-H])\)?",
                r"^\s*\(?([A-H])\)?[\.\):]"):
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()

    hits = [m.group(1).upper() for m in _LETTER.finditer(text.upper())
            if m.group(1).upper() in valid]
    return hits[-1] if hits else None
```

## C2 · `src/bavlm/eval/fbbench.py`

Loading, the frozen prompt, answer extraction, per-domain scoring.

Field names are auto-detected because the exact FBBench schema is unknown until
you dump it. **Run `describe_schema()` once on the real data and pin them** —
auto-detection is a day-one convenience, not something to rely on for reported
numbers.

```python
"""FBBench loading, prompting and scoring.

The exact field names in the released JSON are not known until phase 0 dumps
them, so the loader auto-detects across a list of candidates and reports what it
found. **Run `Question.describe_schema()` once on the real data and pin the
field names** -- auto-detection is a convenience for day one, not something to
rely on for reported numbers.

Design decisions that matter downstream:

* Frames go BEFORE the question in the prompt. Questions are then grouped by
  clip so prefix caching hits the whole visual prefix for every question after
  the first on that clip. Question-first ordering gets none of that.
* `extract_choice` returns None rather than defaulting to "A", so a refusal
  scores as wrong instead of silently inheriting the always-A rate.
"""
from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Iterator, Sequence

DOMAINS = ("count", "action", "position", "cognition")

_FIELD_CANDIDATES = {
    "qid":      ("qid", "question_id", "id", "index"),
    "clip_id":  ("clip_id", "video_id", "video", "clip", "vid", "video_name"),
    "question": ("question", "query", "prompt", "text"),
    "options":  ("options", "choices", "candidates", "answers"),
    "answer":   ("answer", "label", "gt", "ground_truth", "correct"),
    "domain":   ("domain", "task", "task_type", "category", "type"),
}


def _pick(record: dict, key: str):
    for cand in _FIELD_CANDIDATES[key]:
        if cand in record:
            return record[cand]
    return None


def _norm_domain(raw) -> str | None:
    """FBBench task types map onto four domains; names vary (T1..T12, 'Count',
    'count_stroke', ...). Pin this once the real values are known."""
    if raw is None:
        return None
    s = str(raw).lower()
    for d in DOMAINS:
        if d in s:
            return d
    return None


@dataclass
class Question:
    qid: str
    clip_id: str
    question: str
    options: list[str]
    answer: str                       # option LETTER, e.g. "B"
    domain: str | None = None
    raw: dict = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- prompt
    @property
    def prompt(self) -> str:
        """Frozen. Use the identical prompt for every model and every setting --
        otherwise you are comparing prompts, not models."""
        lines = [
            "You are analysing a badminton rally. Answer using only what is "
            "visible in these frames.",
            "",
            f"Question: {self.question}",
            "",
        ]
        for i, opt in enumerate(self.options):
            lines.append(f"{chr(ord('A') + i)}. {opt}")
        lines += ["", "Reply with a single letter and nothing else."]
        return "\n".join(lines)

    @property
    def n_options(self) -> int:
        return len(self.options)

    # ---------------------------------------------------------------- frames
    def load_frames(self, cache_root: str | pathlib.Path | None) -> list[str]:
        """Paths to this clip's cached frames, in temporal order.

        Returns paths rather than decoded arrays so the backend decides how to
        load them -- and so mock mode costs nothing at all.
        """
        if cache_root is None:
            return []
        d = pathlib.Path(cache_root) / self.clip_id
        return [str(p) for p in sorted(d.glob("*.jpg"))]


# -------------------------------------------------------------------- loading

def _iter_records(root: pathlib.Path) -> Iterator[dict]:
    for p in sorted(list(root.rglob("*.json")) + list(root.rglob("*.jsonl"))):
        if p.name == "manifest.json":
            continue
        try:
            text = p.read_text()
        except OSError:
            continue
        try:
            records = ([json.loads(l) for l in text.splitlines() if l.strip()]
                       if p.suffix == ".jsonl" else json.loads(text))
        except json.JSONDecodeError:
            continue
        if isinstance(records, dict):
            records = records.get("data", records.get("questions", [records]))
        for r in records:
            if isinstance(r, dict):
                yield r


def load_questions(root: str | pathlib.Path = "data/fbbench",
                   limit: int | None = None,
                   domains: Sequence[str] | None = None) -> list[Question]:
    out: list[Question] = []
    for i, r in enumerate(_iter_records(pathlib.Path(root))):
        q_text = _pick(r, "question")
        opts = _pick(r, "options")
        ans = _pick(r, "answer")
        if not (q_text and opts and ans is not None):
            continue                                  # open-ended or malformed

        opts = list(opts) if isinstance(opts, (list, tuple)) else [opts]

        # the answer may be a letter, an index, or the option text itself
        a = str(ans).strip()
        if len(a) == 1 and a.upper().isalpha():
            letter = a.upper()
        elif a.isdigit():
            letter = chr(ord("A") + int(a))
        elif a in opts:
            letter = chr(ord("A") + opts.index(a))
        else:
            continue

        dom = _norm_domain(_pick(r, "domain"))
        if domains and dom not in domains:
            continue

        out.append(Question(
            qid=str(_pick(r, "qid") or f"q{i}"),
            clip_id=pathlib.Path(str(_pick(r, "clip_id") or "unknown")).stem,
            question=str(q_text),
            options=[str(o) for o in opts],
            answer=letter,
            domain=dom,
            raw=r,
        ))
        if limit and len(out) >= limit:
            break
    return out


def describe_schema(root: str | pathlib.Path = "data/fbbench", n: int = 3) -> None:
    """Run this ONCE on the real data, then pin the field names above."""
    for i, r in enumerate(_iter_records(pathlib.Path(root))):
        if i >= n:
            break
        print(f"--- record {i} ---")
        for k, v in r.items():
            s = str(v)
            print(f"  {k:16} {type(v).__name__:6} {s[:70]}{'...' if len(s) > 70 else ''}")


# ------------------------------------------------------------------ extraction

_LETTER = re.compile(r"\b([A-H])\b")


def extract_choice(text: str, n_options: int = 4) -> str | None:
    """Pull an option letter from free-form output.

    Guided decoding makes this unnecessary for the real MCQ runs -- that is the
    point of using it. Kept for API models that ignore constraints, for the
    open-ended split, and so mock mode 'malformed' has something to exercise.

    Returns None when no option is present: a refusal must score as wrong, not
    quietly default to 'A'.
    """
    valid = {chr(ord("A") + i) for i in range(n_options)}
    if not text:
        return None

    stripped = text.strip()
    if stripped[:1].upper() in valid and len(stripped) <= 2:
        return stripped[:1].upper()

    for pat in (r"answer\s*(?:is|:)\s*\(?([A-H])\)?",
                r"^\s*\(?([A-H])\)?[\.\):]"):
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()

    hits = [m.group(1).upper() for m in _LETTER.finditer(text.upper())
            if m.group(1).upper() in valid]
    return hits[-1] if hits else None


# --------------------------------------------------------------------- scoring

def score(records: Sequence[dict]) -> dict:
    """records: [{qid, domain, gold, pred, seconds}, ...]"""
    import collections

    by_dom: dict[str, list[bool]] = collections.defaultdict(list)
    secs: dict[str, list[float]] = collections.defaultdict(list)
    correct = unparsed = 0

    for r in records:
        ok = r.get("pred") == r.get("gold")
        correct += ok
        unparsed += r.get("pred") is None
        d = r.get("domain") or "unknown"
        by_dom[d].append(ok)
        if r.get("seconds") is not None:
            secs[d].append(r["seconds"])

    n = max(len(records), 1)
    return {
        "n": len(records),
        "overall": correct / n,
        "unparsed": unparsed,
        "by_domain": {
            d: {"n": len(v),
                "accuracy": round(sum(v) / len(v), 4),
                "sec_per_q": round(sum(secs[d]) / len(secs[d]), 3) if secs[d] else None}
            for d, v in sorted(by_dom.items())
        },
    }
```

## C3 · `scripts/extract_frames.py`

Runs on the laptop, once. Only clips FBBench references, ~1.2 GB at 224 px.

```python
#!/usr/bin/env python3
"""Extract frames once, on CPU, so GPU quota is never spent decoding video.

Only clips that FBBench actually references are extracted -- not all 3,215.
Output is a flat per-clip folder of JPEGs plus a manifest, sized to upload as a
Kaggle Dataset (~1.2 GB at 224 px).

    python scripts/extract_frames.py \
        --fbbench data/fbbench \
        --clips   data/finebadminton-20k \
        --out     ./frame_cache \
        --fps 2 --size 224

Resumable: clips already extracted are skipped, so a killed run costs nothing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


# --------------------------------------------------------------------- discovery

def referenced_clip_ids(fbbench_root: pathlib.Path) -> set[str]:
    """Every clip id mentioned anywhere in the benchmark files.

    Field names are guessed because the exact FBBench schema is not known until
    you dump it (roadmap phase 0). Widen CANDIDATES after you have looked at a
    real record -- see scripts/00_verify_dataset.py output.
    """
    CANDIDATES = ("clip_id", "video_id", "video", "clip", "vid", "id", "video_name")
    ids: set[str] = set()

    for p in list(fbbench_root.rglob("*.json")) + list(fbbench_root.rglob("*.jsonl")):
        try:
            text = p.read_text()
        except OSError:
            continue
        records = ([json.loads(l) for l in text.splitlines() if l.strip()]
                   if p.suffix == ".jsonl" else json.loads(text))
        if isinstance(records, dict):
            records = records.get("data", records.get("questions", [records]))
        for r in records:
            if not isinstance(r, dict):
                continue
            for k in CANDIDATES:
                v = r.get(k)
                if isinstance(v, (str, int)):
                    ids.add(pathlib.Path(str(v)).stem)
                    break
    return ids


def index_videos(clips_root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Map clip id -> video path, keyed on filename stem."""
    return {p.stem: p for p in clips_root.rglob("*") if p.suffix.lower() in VIDEO_EXT}


# --------------------------------------------------------------------- extraction

def extract_one(video: pathlib.Path, out_dir: pathlib.Path,
                fps: float, size: int, quality: int = 88) -> tuple[str, int, str]:
    """Return (clip_id, n_frames, status)."""
    clip_id = video.stem
    dest = out_dir / clip_id

    if dest.exists() and any(dest.glob("*.jpg")):
        return clip_id, len(list(dest.glob("*.jpg"))), "skipped"

    try:
        import decord
        from PIL import Image
        import numpy as np

        vr = decord.VideoReader(str(video), num_threads=1)
        src_fps = float(vr.get_avg_fps()) or 25.0
        step = max(1, round(src_fps / fps))
        idx = list(range(0, len(vr), step))
        if not idx:
            return clip_id, 0, "empty"

        batch = vr.get_batch(idx).asnumpy()
        dest.mkdir(parents=True, exist_ok=True)

        for n, (frame, src_i) in enumerate(zip(batch, idx)):
            im = Image.fromarray(frame)
            w, h = im.size
            scale = size / min(w, h)                      # short side -> `size`
            im = im.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
            # timestamp in the filename: the sampler needs it, and it makes the
            # cache self-describing if the manifest is ever lost
            t_ms = round(src_i / src_fps * 1000)
            im.save(dest / f"{n:03d}_{t_ms:07d}.jpg", quality=quality)

        return clip_id, len(idx), "ok"

    except Exception as e:                                 # noqa: BLE001
        return clip_id, 0, f"error: {type(e).__name__}: {e}"


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbbench", required=True, type=pathlib.Path)
    ap.add_argument("--clips", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--size", type=int, default=224, help="short-side pixels")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="test on N clips first")
    a = ap.parse_args()

    for d in (a.fbbench, a.clips):
        if not d.exists():
            print(f"missing: {d}", file=sys.stderr)
            return 1

    wanted = referenced_clip_ids(a.fbbench)
    have = index_videos(a.clips)
    print(f"FBBench references {len(wanted)} clips; {len(have)} videos on disk")

    todo = [have[c] for c in sorted(wanted) if c in have]
    missing = sorted(wanted - set(have))
    if missing:
        print(f"WARNING {len(missing)} referenced clips not found, e.g. {missing[:5]}")
    if a.limit:
        todo = todo[: a.limit]
    print(f"extracting {len(todo)} clips at {a.fps} fps, short side {a.size}px")

    a.out.mkdir(parents=True, exist_ok=True)
    results, errors = [], []

    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(extract_one, v, a.out, a.fps, a.size): v for v in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            cid, n, status = fut.result()
            results.append({"clip_id": cid, "frames": n, "status": status})
            if status.startswith("error"):
                errors.append((cid, status))
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")

    (a.out / "manifest.json").write_text(json.dumps({
        "fps": a.fps, "size": a.size,
        "clips": len(results),
        "frames": sum(r["frames"] for r in results),
        "missing_videos": missing,
        "results": results,
    }, indent=1))

    total = sum(r["frames"] for r in results)
    mb = sum(f.stat().st_size for f in a.out.rglob("*.jpg")) / 1e6
    print(f"\ndone: {len(results)} clips, {total} frames, {mb:.0f} MB")
    if errors:
        print(f"{len(errors)} failed, first few: {errors[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## C4 · The Kaggle notebook

Upload `fbbench_zeroshot_kaggle.ipynb` directly, or build the cells from this.

### Cell 1 — config

```python
GH_USER = "your-github-username"
GH_REPO = "badminton-agentic-vlm"
BRANCH  = "main"
FRAME_DS = "/kaggle/input/finebadminton-frames"

EXPERIMENT  = "e01_zeroshot"
MODEL_ID    = "Qwen/Qwen2.5-VL-7B-Instruct"
N_QUESTIONS = 500
```

### Cell 2 — secrets and repo clone

```python
import os, subprocess, pathlib, datetime
from kaggle_secrets import UserSecretsClient

GH_TOKEN = UserSecretsClient().get_secret("GH_TOKEN")
WORK = pathlib.Path("/kaggle/working")
REPO = WORK / GH_REPO

def sh(cmd, cwd=None, check=True):
    r = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str),
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr).replace(GH_TOKEN, "***")   # never leak the token
    print(out[-3000:])
    if check and r.returncode != 0:
        raise RuntimeError(f"failed: {cmd}")
    return r

AUTH_URL = f"https://{GH_TOKEN}@github.com/{GH_USER}/{GH_REPO}.git"

if REPO.exists():
    sh(["git", "fetch", "origin"], cwd=REPO)
    sh(["git", "reset", "--hard", f"origin/{BRANCH}"], cwd=REPO)
else:
    sh(["git", "clone", "--depth", "1", "-b", BRANCH, AUTH_URL, str(REPO)])

sh(["git", "config", "user.email", "btp@iiitkottayam.ac.in"], cwd=REPO)
sh(["git", "config", "user.name",  "BTP Kaggle Runner"], cwd=REPO)
```

### Cell 3 — environment and dtype

```python
!pip install -q --upgrade "transformers>=4.51" accelerate qwen-vl-utils 2>&1 | tail -2

import torch
p = torch.cuda.get_device_properties(0)
print(f"{p.name}  {p.total_memory/1e9:.1f} GB  sm_{p.major}{p.minor}")
print("devices:", torch.cuda.device_count())

# P100 = sm_60, T4 = sm_75, A100 = sm_80. bf16 needs sm_80+.
DTYPE = torch.bfloat16 if p.major >= 8 else torch.float16
print("dtype:", DTYPE)
```

### Cell 4 — the model, sharded across both T4s

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,              # float16 on T4/P100 — pre-Ampere, no bf16
    device_map="auto",              # shards 16.6 GB across the two cards
    attn_implementation="sdpa",     # flash-attn-2 needs sm_80+, do not use here
)
processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=224*224)
model.eval()
print("loaded across:", set(model.hf_device_map.values()))
```

Three things that will otherwise bite:

- **`float16`, not `bfloat16`** — T4 and P100 are pre-Ampere. Watch for NaNs in
  long generations; that is the cause if you see them.
- **`attn_implementation="sdpa"`** — FlashAttention-2 requires sm_80+ and fails
  to build on T4.
- **`device_map="auto"`** is what makes 7B fit at all. Without it you OOM on
  card 0.

### Cell 5 — mock check, before any GPU time

```python
import sys
sys.path.insert(0, str(REPO / "src"))
from bavlm.models.backend import build_backend
from bavlm.eval.fbbench import load_questions, extract_choice, score

questions = load_questions(REPO.parent / "fbbench", limit=200)
key = {q.qid: q.answer for q in questions}

for mode, expect in [("oracle", "100%"), ("random", "~25%")]:
    be = build_backend({"mock": mode, "seed": 0}, answer_key=key)
    recs = [{"qid": q.qid, "domain": q.domain, "gold": q.answer,
             "pred": extract_choice(be.generate([], q.prompt, qid=q.qid))}
            for q in questions]
    print(f"{mode:8} {score(recs)['overall']:6.1%}   expected {expect}")
```

**If oracle is not 100% and random is not near 25%, stop.** Fix the scorer
before spending GPU time — the real run would otherwise produce a number that
looks fine and is wrong.

### Cell 6 — the run, resumable

```python
import json, time
from PIL import Image

RESULTS = REPO / "results" / f"{EXPERIMENT}_{MODEL_ID.split('/')[-1]}.jsonl"
RESULTS.parent.mkdir(parents=True, exist_ok=True)

done = set()
if RESULTS.exists():
    done = {json.loads(l)["qid"] for l in open(RESULTS) if l.strip()}
    print(f"resuming, {len(done)} already done")

todo = [q for q in load_questions(limit=N_QUESTIONS) if q.qid not in done]
todo.sort(key=lambda q: q.clip_id)      # group by clip -> prefix cache hits

def answer(q):
    frames = [Image.open(p) for p in q.load_frames(FRAME_DS)]
    msgs = [{"role": "user", "content":
              [{"type": "image", "image": f} for f in frames] +
              [{"type": "text", "text": q.prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    gen = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(gen, skip_special_tokens=True), inputs.input_ids.shape[1]

t0 = time.time()
with open(RESULTS, "a") as fh:
    for i, q in enumerate(todo, 1):
        t = time.time()
        raw, n_tok = answer(q)
        fh.write(json.dumps({
            "qid": q.qid, "clip_id": q.clip_id, "domain": q.domain,
            "gold": q.answer, "raw": raw, "pred": extract_choice(raw, q.n_options),
            "seconds": round(time.time() - t, 3), "input_tokens": n_tok,
        }) + "\n")
        fh.flush()                       # survive a kill at any moment
        if i % 25 == 0:
            print(f"{i}/{len(todo)}  {(time.time()-t0)/60:.1f} min")
```

### Cell 7 — score

```python
records = [json.loads(l) for l in open(RESULTS) if l.strip()]
s = score(records)

print(f"{'domain':12} {'n':>5} {'acc':>8} {'sec/q':>8}")
for d, v in s["by_domain"].items():
    print(f"{d:12} {v['n']:5} {v['accuracy']:8.3f} {str(v['sec_per_q']):>8}")
print(f"\nOVERALL {s['overall']:.1%}  n={s['n']}  unparsed={s['unparsed']}")

# Reference points, FineBadminton (ACM MM 2025):
#   7B zero-shot        < 25%
#   Gemini 2.5 Pro + S    38.6%
#   Qwen2.5-VL-7B ft      42.06%
```

### Cell 8 — push results to GitHub

```python
import pandas as pd

stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                     capture_output=True, text=True).stdout.strip()

with open(REPO / "results" / "MANIFEST.md", "a") as fh:
    fh.write(f"| {stamp} | {EXPERIMENT} | {MODEL_ID} | uniform 2 FPS | "
             f"{s['n']} | FBBench | {sha} | {RESULTS.name} | {s['overall']:.3f} |\n")

pd.DataFrame(s["by_domain"]).T.to_csv(REPO / "results" / f"{EXPERIMENT}_by_domain.csv")

sh(["git", "add", "results/"], cwd=REPO)
sh(["git", "commit", "-m",
    f"results: {EXPERIMENT} n={s['n']} acc={s['overall']:.3f}"], cwd=REPO, check=False)
sh(["git", "push", "origin", BRANCH], cwd=REPO)
print("pushed")
```

---
---

# PART D · Run order

## D1 · Laptop — frame cache

```bash
hf download Moujuruo/FineBadmintonBenchmark --repo-type dataset --local-dir data/fbbench
hf download Moujuruo/Finebadminton-20K      --repo-type dataset --local-dir data/finebadminton-20k

# ALWAYS test on ten clips first
python scripts/extract_frames.py --fbbench data/fbbench --clips data/finebadminton-20k \
    --out ./frame_cache --fps 2 --size 224 --limit 10
ls frame_cache/*/ | head

# then the full run, ~1 hour
python scripts/extract_frames.py --fbbench data/fbbench --clips data/finebadminton-20k \
    --out ./frame_cache --fps 2 --size 224
```

## D2 · Laptop — pin the schema

```python
from bavlm.eval.fbbench import describe_schema
describe_schema("data/fbbench", n=3)
```

Look at the real field names and **pin `_FIELD_CANDIDATES`** in `fbbench.py`.
Auto-detection is for day one only.

## D3 · Laptop — mock mode must pass

```bash
python -m pytest tests/ -q
python tests/test_mock_backend.py
```

Oracle 100%, random ~25%. **Do not proceed otherwise.**

## D4 · Upload the frame cache

```bash
kaggle datasets init -p ./frame_cache
```

Edit `dataset-metadata.json`:

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

10–30 minutes. Once only.

## D5 · Kaggle — notebook settings

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 x2** |
| Internet | **On** |
| Persistence | Variables and Files |
| Environment | Always use latest |

**Add data → Your Datasets →** attach `finebadminton-frames`. It mounts
read-only at `/kaggle/input/finebadminton-frames`.

## D6 · Kaggle — run

1. Cells 1–5. **Cell 5 must pass before you continue.**
2. Cell 6 — the benchmark. Resumable; a timeout costs nothing.
3. Cells 7–8 — score, push.
4. **Save Version** before closing the tab.

## D7 · Cost of a 500-question run

| | |
|---|---|
| Model download, first run only | 10–15 min |
| 500 questions, 7B, 25 frames | 1.5–3 h on T4 × 2 |
| Resolution sweep, 3 × 50 | ~30 min |
| **Total for Review 2** | **4–6 h of a 30 h weekly quota** |

---
---

# PART E · Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | 7B on one card | `device_map="auto"`; confirm **T4 x2** selected |
| GPU option greyed out | Phone not verified | Part B1 |
| `git push` → 403 | Token lacks Contents: write, or wrong repo | Re-issue the token (A4) |
| `pip install` fails | Internet off | Right panel → Internet → On |
| `flash_attn` build error | T4 is sm_75 | `attn_implementation="sdpa"` |
| NaNs in generation | fp16 overflow | `max_new_tokens=8`; guided decoding |
| **Oracle ≠ 100%** | **ID mismatch, shuffled options, off-by-one** | **Stop. Fix before any GPU run.** |
| Random ≫ 25% | Scorer marks wrong answers correct | Check `extract_choice` and the gold letter |
| `unparsed` is high | Model ignoring the format instruction | Guided decoding, or widen `extract_choice` |
| Session dies mid-run | 12 h limit or idle timeout | Re-run cell 6 — it resumes |
| `load_questions` returns 0 | Field names differ | `describe_schema()`, then pin them |
| Frames list empty | `clip_id` mismatch between FBBench and cache folders | Compare `frame_cache/` names to `q.clip_id` |

---

## Sanity expectations

| Check | Expected | If not |
|---|---|---|
| Oracle | 100% | Plumbing broken — stop |
| Random | ~25% | Scorer broken — stop |
| 7B zero-shot, real | **< 25%** | Much higher → extraction bug or leakage. Much lower → frames not loading |
| Action domain | **Worst of the four** | Expected: uniform 2 FPS steps over contact. This is the motivation for Review 3, not a bug |

A rally is 12.4 s at 25 FPS ≈ 310 frames. Uniform 2 FPS gives 25 frames, one
every 0.5 s. A stroke's visible motion lasts 0.12–0.32 s — so the sampler
regularly steps straight over the moment that answers the question.

Say exactly that in the Review 2 deck: *"uniform sampling gives the model the
rally's shape but not its detail, and our per-domain breakdown shows where that
hurts."* A weak number becomes the motivation for the next phase.

---

## The twelve rules that govern every number

1. FBBench is touched only for final numbers; develop on FineBadminton-20K
2. Verify no clip overlap between the two before any fine-tuning
3. If creating a split, split **by match**, not by clip
4. Equal frame budget in every comparison
5. Report **per-domain**, never overall alone
6. Primary efficiency metric is input tokens and frame count, not wall-clock
7. Reproduce the hit-centric baseline yourself; do not quote theirs
8. Evaluate the agentic loop **training-free first**
9. `temperature=0`, fixed seeds
10. Every reported number has a row in `results/MANIFEST.md`
11. Train MCQ and open-ended as separate objectives when fine-tuning
12. Report both ROUGE and BERTScore for open-ended output
