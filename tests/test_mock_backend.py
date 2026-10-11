"""The check that must pass before any GPU time is spent.

oracle -> 100%  : if not, the plumbing is broken (ids not matching the answer
                  key, shuffled options, an off-by-one). Catches more bugs than
                  any other single check.
random -> ~25%  : if not, the scorer is broken.
refuse ->   0%  : a non-answer must score zero, never silently default to "A".

Run directly for a readable table:  python tests/test_mock_backend.py
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from bavlm.eval.fbbench import extract_choice          # noqa: E402
from bavlm.models.backend import build_backend         # noqa: E402

_rng = random.Random(1)
KEY = {f"q{i}": _rng.choice("ABCD") for i in range(400)}


def _accuracy(mode: str, **extra) -> float:
    be = build_backend({"mock": mode, "seed": 7, **extra}, answer_key=KEY)
    hits = sum(extract_choice(be.generate([], "prompt", qid=q)) == gold
               for q, gold in KEY.items())
    return hits / len(KEY)


def test_oracle_is_perfect():
    assert _accuracy("oracle") == 1.0


def test_random_is_near_chance():
    assert 0.18 < _accuracy("random") < 0.33


def test_noisy_tracks_its_target():
    assert abs(_accuracy("noisy", mock_accuracy=0.70) - 0.70) < 0.08


def test_malformed_output_is_still_extracted():
    assert _accuracy("malformed") == 1.0


def test_refusal_scores_zero_and_does_not_default_to_a():
    assert _accuracy("refuse") == 0.0


def test_extract_choice_returns_none_when_no_option_present():
    assert extract_choice("I cannot determine this from the frames.") is None


if __name__ == "__main__":
    checks = [
        ("oracle", _accuracy("oracle"), "must be 100%"),
        ("random", _accuracy("random"), "must be ~25%"),
        ("noisy .70", _accuracy("noisy", mock_accuracy=0.70), "must be ~70%"),
        ("always_a", _accuracy("always_a"), "= share of 'A' in the key"),
        ("malformed", _accuracy("malformed"), "must be 100% (extractor works)"),
        ("refuse", _accuracy("refuse"), "must be 0% (no silent default)"),
    ]
    for name, acc, note in checks:
        print(f"{name:10} {acc:6.1%}   <- {note}")
