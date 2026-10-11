"""Mock backends against the REAL benchmark JSONs, through the real code path:
load_questions -> Question.prompt -> backend -> extract_choice -> score.

test_mock_backend.py uses a synthetic key, so it cannot catch a loader bug
(wrong field, shifted options, dropped records). This file can.

Skipped when data/fbbench is absent. Fetch the JSONs (~4 MB, no video) with:
    huggingface-cli download Moujuruo/FineBadmintonBenchmark --repo-type dataset \
        --include "finebadmintonbenchmark/*.json" --local-dir data/fbbench

Run directly for a per-domain table:  python tests/test_fbbench_real.py
"""
import collections
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from bavlm.eval.fbbench import DOMAINS, SUBTASKS, extract_choice, load_questions, score  # noqa: E402
from bavlm.models.backend import build_backend                                         # noqa: E402

DATA = ROOT / "data" / "fbbench"
pytestmark = pytest.mark.skipif(not any(DATA.rglob("benchmark_*.json")),
                                reason="FBBench JSONs not downloaded to data/fbbench")

# Counts from the HF release @ ff86093a. A change means a new release or a loader bug.
EXPECTED = {"count": 413, "action": 650, "position": 600, "cognition": 750}


@pytest.fixture(scope="module")
def questions():
    return load_questions(DATA)


def _run(questions, mode: str) -> dict:
    key = {q.qid: q.answer for q in questions}
    be = build_backend({"mock": mode, "seed": 7}, answer_key=key)
    recs = [{"qid": q.qid, "domain": q.domain, "gold": q.answer,
             "pred": extract_choice(be.generate([], q.prompt, qid=q.qid), q.n_options)}
            for q in questions]
    return score(recs)


def test_every_record_loads(questions):
    assert len(questions) == 2413
    assert collections.Counter(q.domain for q in questions) == EXPECTED
    assert len({q.qid for q in questions}) == len(questions)


def test_three_subtasks_per_domain(questions):
    by_dom = collections.defaultdict(set)
    for q in questions:
        by_dom[q.domain].add(q.subtask)
    assert {d: len(s) for d, s in by_dom.items()} == {d: 3 for d in DOMAINS}
    assert sum(len(s) for s in by_dom.values()) == len(SUBTASKS)


def test_options_render_with_single_letters(questions):
    # the release prefixes options "A. ..."; the prompt must not double it
    for q in questions:
        assert len(q.options) == 4
        assert "A. A." not in q.prompt and "\nA. " in q.prompt


def test_chinese_fields_reachable(questions):
    q = questions[0]
    assert q.question_zh and q.options_zh and len(q.options_zh) == len(q.options)


def test_oracle_is_perfect_in_every_domain(questions):
    s = _run(questions, "oracle")
    assert s["overall"] == 1.0 and s["unparsed"] == 0
    assert all(v["accuracy"] == 1.0 for v in s["by_domain"].values())


def test_random_is_near_chance_in_every_domain(questions):
    s = _run(questions, "random")
    assert 0.22 < s["overall"] < 0.28
    for d, v in s["by_domain"].items():
        assert 0.18 < v["accuracy"] < 0.32, (d, v)


def test_refusal_scores_zero(questions):
    s = _run(questions, "refuse")
    assert s["overall"] == 0.0 and s["unparsed"] == len(questions)


if __name__ == "__main__":
    qs = load_questions(DATA)
    print(f"{len(qs)} questions")
    for mode, note in (("oracle", "must be 100%"), ("random", "must be ~25%"),
                       ("always_a", "= share of A per domain"), ("refuse", "must be 0%")):
        s = _run(qs, mode)
        doms = "  ".join(f"{d[:5]} {s['by_domain'][d]['accuracy']:6.1%}" for d in DOMAINS)
        print(f"{mode:9} overall {s['overall']:6.1%}   {doms}   <- {note}")
