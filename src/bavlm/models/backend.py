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

_TS_MS = re.compile(r"_(\d+)\.jpe?g$", re.IGNORECASE)


def frames_fps(paths: Sequence, default: float) -> float:
    """Sampling rate of a frame list, from the `<i>_<ms>.jpg` cache names.

    Qwen2.5-VL puts absolute time into its position encoding, so the processor
    must be told the real rate. Left unset, transformers assumes 24 fps and a
    5 s clip at 2 FPS reads to the model as 0.4 s.
    """
    ts = [int(m.group(1)) for p in paths if (m := _TS_MS.search(str(p)))]
    if len(ts) == len(paths) and len(ts) > 1 and ts[-1] > ts[0]:
        return (len(ts) - 1) / ((ts[-1] - ts[0]) / 1000)
    return default


class HFBackend:
    """Real model via transformers. Loaded lazily so importing this costs nothing.

    Tested with Qwen2.5-VL on transformers 5.19.0; the Kaggle notebook pins that
    version, because video-processor defaults (fps, pixel caps) have changed
    between releases and silently change what the model sees.

    MCQ answers are scored, not parsed: one forward pass, then the highest
    next-token logit among the option letters. That is guided decoding at its
    simplest -- deterministic (rule 9), never unparseable, no extra library.
    Free-form generation is kept for open-ended questions.

    Every call records `last_stats` (input and visual tokens, frames used) so
    efficiency is reported in tokens and frames, not wall-clock (rule 6).
    """

    def __init__(self, hf_id: str, dtype: str = "float16", max_frames: int = 64,
                 fps: float = 2.0, short_side: int | None = None,
                 device_map: str = "auto", attn_implementation: str = "sdpa",
                 max_memory: dict | None = None) -> None:
        self.name = hf_id
        self.hf_id = hf_id
        self.dtype = dtype          # T4 is sm_75: float16. bf16 needs sm_80+
        self.max_frames = max_frames
        self.fps = fps              # fallback when frame names carry no timestamps
        self.short_side = short_side  # resolution sweep: downscale at load, never upscale
        self.device_map = device_map
        self.attn_implementation = attn_implementation
        # "auto" fills cuda:0 first, leaving no room there for attention on a long
        # clip; a per-GPU cap spreads the weights so each card keeps headroom
        self.max_memory = max_memory
        self._model = None
        self._proc = None
        self.last_stats: dict = {}

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._proc = AutoProcessor.from_pretrained(self.hf_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.hf_id,
            dtype=getattr(torch, self.dtype),
            device_map=self.device_map,
            attn_implementation=self.attn_implementation,
            **({"max_memory": self.max_memory} if self.max_memory else {}),
        ).eval()
        tok = self._proc.tokenizer
        self._video_pad = tok.convert_tokens_to_ids("<|video_pad|>")

    def _letter_ids(self, letters: Sequence[str]) -> list[int]:
        ids = []
        for L in letters:
            enc = self._proc.tokenizer.encode(L, add_special_tokens=False)
            if len(enc) != 1:
                raise ValueError(f"option letter {L!r} is not a single token: {enc}")
            ids.append(enc[0])
        return ids

    def _frames(self, frames: Sequence) -> tuple[list, list]:
        """Load paths to RGB images; cap the count uniformly, never silently."""
        from PIL import Image

        if len(frames) > self.max_frames:
            step = len(frames) / self.max_frames
            frames = [frames[int(i * step)] for i in range(self.max_frames)]
        imgs = []
        for f in frames:
            im = Image.open(f).convert("RGB") if isinstance(f, (str, bytes)) or hasattr(f, "__fspath__") else f
            if self.short_side and min(im.size) > self.short_side:
                s = self.short_side / min(im.size)
                im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BICUBIC)
            imgs.append(im)
        return list(frames), imgs

    def _inputs(self, frames: Sequence, prompt: str):
        used, imgs = self._frames(frames)
        content = ([{"type": "video"}] if imgs else []) + [{"type": "text", "text": prompt}]
        text = self._proc.apply_chat_template([{"role": "user", "content": content}],
                                              tokenize=False, add_generation_prompt=True)
        kw: dict = {}
        if imgs:
            fps = frames_fps(used, self.fps)
            kw = {"videos": [imgs], "do_sample_frames": False, "cap_pixels_per_frame": True,
                  "video_metadata": [{"total_num_frames": len(imgs), "fps": fps,
                                      "frames_indices": list(range(len(imgs)))}]}
        inputs = self._proc(text=[text], return_tensors="pt", **kw).to(self._model.device)
        ids = inputs["input_ids"]
        self.last_stats = {
            "n_frames_in": len(frames),
            "n_frames": len(imgs),
            "input_tokens": int(ids.shape[1]),
            "visual_tokens": int((ids == self._video_pad).sum()),
            "frame_wh": list(imgs[0].size) if imgs else None,
        }
        if imgs:
            self.last_stats["fps"] = round(kw["video_metadata"][0]["fps"], 3)
        return inputs

    def generate(self, frames: Sequence, prompt: str, choices: Sequence[str] | None = None,
                 max_new_tokens: int = 64, **kw) -> str:
        """With `choices` (e.g. "ABCD"), return the best-scoring letter.
        Without, greedy free-form generation."""
        import torch

        if self._model is None:
            self._load()
        inputs = self._inputs(frames, prompt)

        with torch.inference_mode():
            if choices:
                ids = self._letter_ids(choices)
                logits = self._model(**inputs, logits_to_keep=1).logits[0, -1].float()
                lp = torch.log_softmax(logits, dim=-1)[ids]
                self.last_stats["choice_logprobs"] = {c: round(float(v), 4) for c, v in zip(choices, lp)}
                return choices[int(lp.argmax())]

            out = self._model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
            new = out[0, inputs["input_ids"].shape[1]:]
            self.last_stats["output_tokens"] = int(new.shape[0])
            return self._proc.tokenizer.decode(new, skip_special_tokens=True)


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
        dtype=cfg.get("dtype", "float16"),
        max_frames=cfg.get("max_frames", 64),
        fps=cfg.get("fps", 2.0),
        short_side=cfg.get("short_side"),
        device_map=cfg.get("device_map", "auto"),
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
        max_memory=cfg.get("max_memory"),
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
