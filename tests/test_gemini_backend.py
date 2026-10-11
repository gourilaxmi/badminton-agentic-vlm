"""GeminiBackend without a network: a fake client stands in for the API.

What is tested is our side of the contract -- the same frames and resolution
as HFBackend, timestamps before each frame, an enum schema restricting the
reply to the option letters, temperature 0, token counts taken from the usage
metadata, and every failure scored as None rather than a guessed letter.
"""
import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

pytest.importorskip("google.genai")

from bavlm.models.backend import GeminiBackend, build_backend  # noqa: E402


def _write_frames(d: pathlib.Path, n: int, ms_step: int = 500) -> list[str]:
    from PIL import Image
    paths = []
    for i in range(n):
        p = d / f"{i:03d}_{i * ms_step:07d}.jpg"
        Image.new("RGB", (796, 448), (40 * (i % 6), 120, 60)).save(p)
        paths.append(str(p))
    return paths


class FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


class FakeClient:
    """Records each request; replies with `reply`, after raising `fail` errors."""

    def __init__(self, reply="B", fail=()):
        self.reply, self.fail, self.calls = reply, list(fail), []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        if self.fail:
            raise FakeAPIError(self.fail.pop(0))
        n_img = sum(1 for p in contents if p.inline_data)
        usage = SimpleNamespace(
            prompt_token_count=258 * n_img + 40, candidates_token_count=1, thoughts_token_count=None,
            prompt_tokens_details=[SimpleNamespace(modality="MediaModality.IMAGE", token_count=258 * n_img),
                                   SimpleNamespace(modality="MediaModality.TEXT", token_count=40)])
        return SimpleNamespace(text=self.reply, usage_metadata=usage)


def _backend(client, **kw):
    return GeminiBackend("gemini-test", client=client, **kw)


def test_factory_routes_gemini_by_name():
    be = build_backend({"hf_id": "gemini-2.5-flash", "max_frames": 25, "short_side": 224})
    assert isinstance(be, GeminiBackend)
    assert be.max_frames == 25 and be.short_side == 224


def test_mcq_reply_is_constrained_to_the_letters(tmp_path):
    fc = FakeClient("C")
    be = _backend(fc)
    assert be.generate(_write_frames(tmp_path, 6), "Which shot?", choices="ABCD") == "C"
    cfg = fc.calls[0].config
    assert cfg.temperature == 0 and cfg.seed == 0
    assert cfg.response_mime_type == "text/x.enum"
    assert cfg.response_schema.enum == list("ABCD")


def test_same_frames_as_local_models_with_timestamps(tmp_path):
    fc = FakeClient()
    be = _backend(fc, max_frames=4, short_side=224)
    be.generate(_write_frames(tmp_path, 12), "Q?", choices="ABCD")
    parts = fc.calls[0].contents
    texts = [p.text for p in parts if p.text is not None]
    assert texts[:-1] == ["<0.0 seconds>", "<1.5 seconds>", "<3.0 seconds>", "<4.5 seconds>"]
    assert texts[-1] == "Q?"                                          # prompt unchanged, last
    from io import BytesIO

    from PIL import Image
    img = Image.open(BytesIO(next(p for p in parts if p.inline_data).inline_data.data))
    assert min(img.size) == 224                                       # same resize as HFBackend
    s = be.last_stats
    assert s["n_frames_in"] == 12 and s["n_frames"] == 4 and s["fps"] == pytest.approx(2 / 3, abs=1e-3)   # 4 of 12 at 2 FPS


def test_tokens_come_from_usage_metadata(tmp_path):
    be = _backend(FakeClient())
    be.generate(_write_frames(tmp_path, 5), "Q?", choices="ABCD")
    assert be.last_stats["visual_tokens"] == 5 * 258
    assert be.last_stats["input_tokens"] == 5 * 258 + 40


def test_rate_limit_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    fc = FakeClient("A", fail=[429, 503])
    be = _backend(fc)
    assert be.generate(_write_frames(tmp_path, 3), "Q?", choices="ABCD") == "A"
    assert len(fc.calls) == 3 and "error" not in be.last_stats


@pytest.mark.parametrize("fail,reply,err", [([400], "A", "api:400"),        # not retryable
                                            ([429] * 3, "A", "api:429"),    # retries exhausted
                                            ([], "", "unparsed"),           # blocked / empty
                                            ([], "E", "unparsed")])         # outside the options
def test_failures_score_none_never_a_letter(tmp_path, monkeypatch, fail, reply, err):
    monkeypatch.setattr("time.sleep", lambda s: None)
    be = _backend(FakeClient(reply, fail=fail), retries=2)
    assert be.generate(_write_frames(tmp_path, 3), "Q?", choices="ABCD") is None
    assert be.last_stats["error"] == err
    assert "input_tokens" in be.last_stats and "visual_tokens" in be.last_stats   # kernel reads both


def test_free_generation_without_choices(tmp_path):
    fc = FakeClient("a smash")
    be = _backend(fc)
    assert be.generate(_write_frames(tmp_path, 2), "Describe.", max_new_tokens=16) == "a smash"
    assert fc.calls[0].config.max_output_tokens == 16
    assert fc.calls[0].config.response_schema is None
