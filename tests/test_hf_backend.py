"""HFBackend on a laptop: the real code path, on a tiny random Qwen2.5-VL.

The tiny model's answers are meaningless; what is tested is the plumbing --
frames reach the processor with the right fps, option letters are single
tokens, scoring returns a valid letter, and the stats are recorded.

Skipped without torch/torchvision/transformers, or if the tiny model cannot be
downloaded (~10 MB, once).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from bavlm.models.backend import HFBackend, build_backend, frames_fps  # noqa: E402

TINY = "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"


def _write_frames(d: pathlib.Path, n: int, ms_step: int = 500) -> list[str]:
    from PIL import Image
    paths = []
    for i in range(n):
        p = d / f"{i:03d}_{i * ms_step:07d}.jpg"
        Image.new("RGB", (796, 448), (40 * (i % 6), 120, 60)).save(p)
        paths.append(str(p))
    return paths


# ------------------------------------------------------------------ no model

def test_fps_from_cache_names():
    names = [f"x/{i:03d}_{i * 500:07d}.jpg" for i in range(10)]
    assert frames_fps(names, default=7.0) == pytest.approx(2.0)


def test_fps_falls_back_without_timestamps():
    assert frames_fps(["a.jpg", "b.jpg"], default=2.0) == 2.0
    assert frames_fps(["x/000_0000000.jpg"], default=2.0) == 2.0


def test_factory_defaults_suit_a_t4():
    be = build_backend({"hf_id": "some/model"})
    assert isinstance(be, HFBackend)
    assert be.dtype == "float16" and be.attn_implementation == "sdpa"
    assert be._model is None                      # lazy: nothing loaded yet


# ------------------------------------------------------------------ tiny model

@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("transformers")
    be = HFBackend(TINY, dtype="float32", device_map=None)
    try:
        be._load()
    except OSError as e:                          # offline, or the repo moved
        pytest.skip(f"tiny model unavailable: {e}")
    return be


def test_scored_answer_is_a_valid_letter(tiny, tmp_path):
    frames = _write_frames(tmp_path, 10)
    out = tiny.generate(frames, "Which shot? Reply with a single letter.", choices="ABCD")
    assert out in "ABCD"
    s = tiny.last_stats
    assert s["n_frames"] == 10 and s["fps"] == pytest.approx(2.0)
    assert 0 < s["visual_tokens"] < s["input_tokens"]
    assert set(s["choice_logprobs"]) == set("ABCD")


def test_scoring_is_deterministic(tiny, tmp_path):
    frames = _write_frames(tmp_path, 6)
    a = tiny.generate(frames, "Q?", choices="ABCD")
    la = tiny.last_stats["choice_logprobs"]
    b = tiny.generate(frames, "Q?", choices="ABCD")
    assert a == b and la == tiny.last_stats["choice_logprobs"]


def test_processor_receives_the_real_fps(tiny, tmp_path):
    # the bug this guards: unset fps -> transformers assumes 24 -> wrong time encoding
    frames = _write_frames(tmp_path, 8)
    inputs = tiny._inputs(frames, "Q?")
    assert float(inputs["second_per_grid_ts"][0]) == pytest.approx(2 / 2.0)   # temporal_patch / fps


def test_max_frames_caps_uniformly_and_reports_it(tmp_path, tiny):
    tiny.max_frames = 4
    try:
        tiny.generate(_write_frames(tmp_path, 12), "Q?", choices="ABCD")
        assert tiny.last_stats["n_frames_in"] == 12 and tiny.last_stats["n_frames"] == 4
    finally:
        tiny.max_frames = 64


def test_short_side_downscales_never_upscales(tiny, tmp_path):
    frames = _write_frames(tmp_path, 4)
    tiny.short_side = 224
    try:
        tiny.generate(frames, "Q?", choices="ABCD")
        assert min(tiny.last_stats["frame_wh"]) == 224
        tiny.short_side = 1000
        tiny.generate(frames, "Q?", choices="ABCD")
        assert min(tiny.last_stats["frame_wh"]) == 448
    finally:
        tiny.short_side = None


def test_text_only_and_free_generation(tiny):
    out = tiny.generate([], "Say hi.", max_new_tokens=4)
    assert isinstance(out, str)
    assert tiny.last_stats["visual_tokens"] == 0 and tiny.last_stats["output_tokens"] <= 4
