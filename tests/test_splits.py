"""Overlap check and match split, on a synthetic release and on the real one.

The real-data tests skip unless the 20K annotations are downloaded:
    hf download Moujuruo/Finebadminton-20K --repo-type dataset \\
        --include "finebadminton-20K/*.json" "*.txt" --local-dir data/fb20k
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from bavlm.data.splits import (benchmark_videos, check_overlap, is_clean_label,  # noqa: E402
                               label_vocabulary, load_rallies, match_split)

REAL = pathlib.Path(__file__).resolve().parents[1] / "data" / "fb20k"


def _release(tmp: pathlib.Path, n_videos=8, bench=("0001.mp4", "0002.mp4")) -> pathlib.Path:
    ann = tmp / "finebadminton-20K"
    ann.mkdir(parents=True)
    for v in range(1, n_videos + 1):
        name = f"{v:04d}.mp4"
        rallies = [{"video": name, "fps": 30.0, "start_frame": 100 * k, "end_frame": 100 * k + 59,
                    "hitting": [{"Foundational Actions Level": {"hit type": "serve", "subtype": ["short serve"],
                                                               "hitter": "top", "ball area": "mid court"},
                                 "Tactical Semantics Level": {"player actions": ["forehand"]},
                                 "Decision Evaluation Level": {"quality": "4" if k else 4}}]}
                   for k in range(3)]
        (ann / f"{v:04d}_updated.json").write_text(json.dumps(rallies))
    (tmp / "benchmark_source_videos.txt").write_text("\n".join(bench) + "\n")
    return tmp


def test_benchmark_matches_are_held_out(tmp_path):
    rep = check_overlap(_release(tmp_path))
    assert rep["videos"] == 8 and rep["rallies"] == 24
    assert rep["held_out_rallies"] == 6 and rep["train_rallies"] == 18 and rep["train_videos"] == 6
    assert rep["bench_videos_missing"] == []


def test_split_is_by_match_and_never_touches_the_benchmark(tmp_path):
    root = _release(tmp_path)
    rallies, bench = load_rallies(root), benchmark_videos(root)
    s = match_split(rallies, bench, val_matches=2, seed=0)
    tr, va = {r.video for r in s["train"]}, {r.video for r in s["val"]}
    assert not (tr & va) and not ((tr | va) & bench)          # rule 2 and rule 3
    assert len(va) == 2 and len(tr) == 4
    assert match_split(rallies, bench, val_matches=2, seed=0) == s    # same split for everyone


def test_rally_seconds_use_each_videos_fps(tmp_path):
    r = load_rallies(_release(tmp_path))[0]
    assert r.seconds == pytest.approx(60 / 30.0) and r.rally_id == "0001:0-59"


def test_vocabulary_normalises_quality_types(tmp_path):
    v = label_vocabulary(load_rallies(_release(tmp_path)))
    assert v["quality"] == {4: 24}                            # 4 and "4" are one label
    assert v["hit type"] == {"serve": 24}


def test_clean_label():
    assert is_clean_label("mid back court")
    assert not any(is_clean_label(x) for x in ("", None, "、", "劈吊", "中钱场"))


# ------------------------------------------------------------------ real release

@pytest.fixture(scope="module")
def real():
    if not (REAL / "benchmark_source_videos.txt").exists():
        pytest.skip("Finebadminton-20K annotations not downloaded")
    return load_rallies(REAL), benchmark_videos(REAL)


def test_real_release_matches_its_readme(real):
    rallies, bench = real
    assert len({r.video for r in rallies}) == 70 and len(rallies) == 2066
    assert sum(len(r.hits) for r in rallies) == 20757
    assert len(bench) == 20 and bench <= {r.video for r in rallies}


def test_real_vocabulary_has_the_papers_label_counts(real):
    v = label_vocabulary(real[0])
    clean = {k: {x for x in c if is_clean_label(x)} for k, c in v.items()}
    assert len(clean["hit type"]) == 11
    assert len(clean["subtype"]) == 20
    assert len(clean["ball area"]) == 9
    assert set(v["quality"]) == set(range(1, 8))
