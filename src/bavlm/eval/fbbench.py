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
