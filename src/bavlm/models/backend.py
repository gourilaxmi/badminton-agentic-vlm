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
