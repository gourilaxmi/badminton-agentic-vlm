"""FBBench loading, prompting and scoring.

The schema is pinned to the HF release `Moujuruo/FineBadmintonBenchmark` @
ff86093a (see results/dataset_report.md). Loading is strict: a record that does
not match raises, because a silently skipped question shrinks the denominator
and inflates accuracy.

Design decisions that matter downstream:

* Frames go BEFORE the question in the prompt, so prefix caching can reuse the
  visual prefix. In this release every question has its own clip, so the
  saving only appears if a later release shares clips across questions.
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

# Pinned field paths (tuples walk nested dicts). English is what we prompt with.
FIELDS = {
    "clip_id":  ("video_uid",),
    "question": ("question", "question_en"),
    "options":  ("options_en",),
    "answer":   ("answer",),
    "domain":   ("spatial",),
}
# Kept reachable via Question.question_zh / options_zh; never used for prompting.
FIELDS_ZH = {
    "question": ("question", "question_zh"),
    "options":  ("options_zh",),
}

# File stem -> (domain, subtask). The subtask comes from the FILE, not the
# `temporal` field: pointer_recog_updated is labelled temporal="round comment",
# the same as round_comment_updated, although it asks why the point was won.
# The domain here is cross-checked against each record's `spatial`.
SUBTASKS = {
    "benchmark_hitcount":                      ("count",     "hitting count"),
    "benchmark_roundcount":                    ("count",     "round count"),
    "benchmark_select_detail_count_questions": ("count",     "detailed hitting count"),
    "benchmark_action_classification":         ("action",    "action classification"),
    "benchmark_action_localization":           ("action",    "action localization"),
    "benchmark_action_prediction":             ("action",    "action prediction"),
    "benchmark_hitting_localization_new":      ("position",  "hitting localization"),
    "benchmark_hitting_prediction":            ("position",  "hitting prediction"),
    "benchmark_moving_recognization":          ("position",  "moving recognition"),
    "benchmark_hitting_comment2_updated":      ("cognition", "hitting comment"),
    "benchmark_round_comment_updated":         ("cognition", "round comment"),
    "benchmark_pointer_recog_updated":         ("cognition", "point reason"),
}

_OPTION_PREFIX = re.compile(r"^\s*([A-H])\s*[\.\):]\s*")


def _get(record: dict, path: tuple[str, ...]):
    """Follow a pinned field path; KeyError names the missing field."""
    v = record
    for k in path:
        if not isinstance(v, dict) or k not in v:
            raise KeyError(".".join(path))
        v = v[k]
    return v


def _strip_letters(options: list) -> list[str]:
    """Options ship as "A. 8"; the prompt adds its own letters, so strip theirs.

    The shipped letter must match the position -- if it does not, the gold
    letter would point at a different option than the one the model sees.
    """
    out = []
    for i, o in enumerate(options):
        m = _OPTION_PREFIX.match(str(o))
        if not m or m.group(1) != chr(ord("A") + i):
            raise ValueError(f"option {i} has no matching letter prefix: {str(o)[:40]!r}")
        out.append(str(o)[m.end():])
    return out


@dataclass
class Question:
    qid: str
    clip_id: str
    question: str
    options: list[str]
    answer: str                       # option LETTER, e.g. "B"
    domain: str | None = None
    subtask: str | None = None        # one of three per domain, from SUBTASKS
    duration: str | int | None = None # short/mid/long; an int in hitting_comment2 (unit unknown)
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def question_zh(self) -> str | None:
        try:
            return str(_get(self.raw, FIELDS_ZH["question"]))
        except KeyError:
            return None

    @property
    def options_zh(self) -> list[str] | None:
        try:
            return list(_get(self.raw, FIELDS_ZH["options"]))
        except KeyError:
            return None

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

def _benchmark_files(root: pathlib.Path) -> list[pathlib.Path]:
    """The 12 benchmark_*.json files. Anything else under root (HF's .cache
    metadata, a frame manifest) is not a question file and is never read."""
    files = sorted(p for p in root.rglob("benchmark_*.json") if ".cache" not in p.parts)
    unknown = [p.name for p in files if p.stem not in SUBTASKS]
    if unknown:
        raise ValueError(f"benchmark files not in SUBTASKS, pin them first: {unknown}")
    return files


def _iter_records(root: pathlib.Path) -> Iterator[tuple[pathlib.Path, int, dict]]:
    for p in _benchmark_files(root):
        records = json.loads(p.read_text(encoding="utf-8"))   # zh fields: never the locale default
        if not isinstance(records, list):
            raise ValueError(f"{p.name}: expected a list of records, got {type(records).__name__}")
        for i, r in enumerate(records):
            yield p, i, r


def load_questions(root: str | pathlib.Path = "data/fbbench",
                   limit: int | None = None,
                   domains: Sequence[str] | None = None) -> list[Question]:
    """Every MCQ in the release, in file order. Raises on any schema mismatch."""
    out: list[Question] = []
    for p, i, r in _iter_records(pathlib.Path(root)):
        where = f"{p.name}[{i}]"
        file_domain, subtask = SUBTASKS[p.stem]
        try:
            dom = str(_get(r, FIELDS["domain"]))
            options = _strip_letters(list(_get(r, FIELDS["options"])))
            answer = str(_get(r, FIELDS["answer"])).strip()
            clip_id = str(_get(r, FIELDS["clip_id"]))
            question = str(_get(r, FIELDS["question"]))
        except (KeyError, ValueError) as e:
            raise ValueError(f"{where}: {e}") from e

        if dom not in DOMAINS or dom != file_domain:
            raise ValueError(f"{where}: spatial={dom!r}, expected {file_domain!r}")
        if answer not in {chr(ord("A") + k) for k in range(len(options))}:
            raise ValueError(f"{where}: answer {answer!r} is not one of {len(options)} option letters")

        if domains and dom not in domains:
            continue
        out.append(Question(
            # one clip per question in this release; the subtask prefix keeps
            # qids unique if a later release reuses a clip across subtasks
            qid=f"{p.stem.removeprefix('benchmark_')}/{clip_id}",
            clip_id=clip_id,
            question=question,
            options=options,
            answer=answer,
            domain=dom,
            subtask=subtask,
            duration=r.get("duration"),
            raw=r,
        ))
        if limit and len(out) >= limit:
            break
    return out


def describe_schema(root: str | pathlib.Path = "data/fbbench", n: int = 1) -> None:
    """Print the first n raw records of each file -- for checking a new release."""
    seen: dict[str, int] = {}
    for p, _, r in _iter_records(pathlib.Path(root)):
        if seen.get(p.name, 0) >= n:
            continue
        seen[p.name] = seen.get(p.name, 0) + 1
        print(f"--- {p.name} ---")
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
